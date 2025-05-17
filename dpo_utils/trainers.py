import torch
import os
import json
import random
from datetime import datetime
from torch import nn
from torch.utils.data import DataLoader
# from configs import TrainingConfig
from tqdm import tqdm
from torch.nn import functional as F
# import matplotlib.pyplot as plt
import numpy as np
from torch.optim.lr_scheduler import LRScheduler

from optimizer.AdamW_PMA import AdamW_PMA
from optimizer.Lion_PMA import Lion_PMA


class Trainer:

    def __init__(self) -> None:
        self.model = None
        self.optimizer = None
        random.seed(1)

    def save_hyperparams(self, hp):
        if not os.path.exists(f'./runs/{self.run_name}'):
            os.makedirs(f'./runs/{self.run_name}')

        with open(f'./runs/{self.run_name}/hyperparams.json', 'w') as fp:
            json.dump(hp, fp, indent=4)

    def save_metrics(self, metrics):
        if not os.path.exists(f'./runs/{self.run_name}'):
            os.makedirs(f'./runs/{self.run_name}')
        with open(f'./runs/{self.run_name}/metrics.json', 'w') as fp:
            json.dump(metrics, fp, indent=4)

    def save_states(self, step, is_last=False):
        # pass
        if not os.path.exists(f'./runs/{self.run_name}'):
            os.makedirs(f'./runs/{self.run_name}')
        file_name = f'{self.run_name}_final.pt' if is_last else f'{self.run_name}_step{step}.pt'
        torch.save(
            {
                'step': step,
                'model_state_dict':
                    self.model.state_dict(),  # Save the unoptimized model
                'optimizer_state_dict': self.optimizer.state_dict(),
            },
            f'./runs/{self.run_name}/{file_name}')
    


class DPOTrainer(Trainer):
    def __init__(self, args, device, model: nn.Module, ref_model: nn.Module,
                 train_dataset, test_dataset, beta=0.1) -> None:
        super().__init__()
        self.cfg = args
        self.device = device
        self.ref_model = ref_model
        self.beta = beta
        self.run_name = f"sft_{args.exp_name}_{datetime.now().strftime('%Y%m%d%H%M')}"

        self.train_dataloader = DataLoader(train_dataset,
                       batch_size=args.batch_size,
                       num_workers=min(6,args.batch_size),
                       shuffle=True,
                       pin_memory=True)
        # self.train_dataloader = iter(DataLoader(train_dataset,
        #                batch_size=cfg.batch_size,
        #                num_workers=min(6,cfg.batch_size),
        #                shuffle=True,
        #                pin_memory=True))
        self.test_dataloader = DataLoader(test_dataset,
                       batch_size=args.batch_size,
                       num_workers=min(6,args.batch_size),
                       pin_memory=True)
        self.model = model
        self.dtype = torch.float16

        hp = {
            "dtype": str(self.dtype),
            "train_dataset": type(train_dataset).__name__,
            "train_dataset_len": len(train_dataset),
            "test_dataset": type(test_dataset).__name__,
            "test_dataset_len": len(test_dataset),
            **args.dict(),
        }
        self.save_hyperparams(hp)

        self.loss_list = []
        self.eval_loss_list = []
        self.eval_acc_list = []

        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.dpo_lr)
        if self.cfg.optimizer == 'adam_pma':
            self.optimizer = AdamW_PMA(self.model.parameters(), lr=self.cfg.dpo_lr, accumulate_steps=self.cfg.eval_interval)
        elif self.cfg.optimizer == 'SGD':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.cfg.dpo_lr, momentum=0.9, weight_decay=0.01)
        else:
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.cfg.dpo_lr)
        # if self.cfg.scheduler:
        #     # self.scheduler = WarmUpScheduler(self.optimizer, warmup_steps=10)
        #     self.scheduler = WarmUpScheduler(self.optimizer, warmup_steps=150)
        #     # self.scheduler = PMA(self.optimizer, cumulate_steps=self.cfg.eval_interval)
        # else:
        #     self.scheduler = None
        self.scheduler = None
        

    def dpo_loss(self, logits, ref_logits, x, B, T, C):
        logits, ref_logits = logits.view(B, 2, T, C), ref_logits.view(B, 2, T, C)
        x = x.view(B, 2, T)
        logits_pos, logits_neg = torch.split(logits, 1, dim=1)
        ref_logits_pos, ref_logits_neg = torch.split(ref_logits, 1, dim=1)
        x_pos, x_neg = torch.split(x, 1, dim=1)
        logits_pos, logits_neg = logits_pos.squeeze(1), logits_neg.squeeze(1)
        ref_logits_pos, ref_logits_neg = ref_logits_pos.squeeze(1), ref_logits_neg.squeeze(1)
        x_pos, x_neg = x_pos.squeeze(1), x_neg.squeeze(1)

        logits_pos, logits_neg = torch.log_softmax(logits_pos, dim=-1), torch.log_softmax(logits_neg, dim=-1)
        ref_logits_pos, ref_logits_neg = torch.log_softmax(ref_logits_pos, dim=-1), torch.log_softmax(ref_logits_neg, dim=-1)

        logits_pos = torch.gather(logits_pos[:,:-1], dim=-1, index=x_pos[:,1:].unsqueeze(-1)).squeeze(-1)
        logits_neg = torch.gather(logits_neg[:,:-1], dim=-1, index=x_neg[:,1:].unsqueeze(-1)).squeeze(-1)
        ref_logits_pos = torch.gather(ref_logits_pos[:,:-1], dim=-1, index=x_pos[:,1:].unsqueeze(-1)).squeeze(-1)
        ref_logits_neg = torch.gather(ref_logits_neg[:,:-1], dim=-1, index=x_neg[:,1:].unsqueeze(-1)).squeeze(-1)
        
        pi_logits = logits_pos - logits_neg
        if self.cfg.loss_clamp:
            pi_logits = torch.clamp(pi_logits, None, self.cfg.loss_clamp_value)
        pi_ref_logits = ref_logits_pos - ref_logits_neg

        # loss = -F.logsigmoid(self.beta * pi_logits) - F.logsigmoid(-self.beta * pi_ref_logits)
        loss = -F.logsigmoid(self.beta * pi_logits-self.beta * pi_ref_logits)
        # loss = -F.logsigmoid(self.beta * torch.clamp(pi_logits - pi_ref_logits, None, 5)) # TODO clip on pi_logits
        return loss.mean()
    
    def conservative_dpo_loss(self, logits, ref_logits, x, B, T, C):
        # loss = self.dpo_loss(logits, ref_logits, x, B, T, C)
        # conservative_regularizer = 0
        logits, ref_logits = logits.view(B, 2, T, C), ref_logits.view(B, 2, T, C)
        x = x.view(B, 2, T)
        logits_pos, logits_neg = torch.split(logits, 1, dim=1)
        ref_logits_pos, ref_logits_neg = torch.split(ref_logits, 1, dim=1)
        x_pos, x_neg = torch.split(x, 1, dim=1)
        logits_pos, logits_neg = logits_pos.squeeze(1), logits_neg.squeeze(1)
        ref_logits_pos, ref_logits_neg = ref_logits_pos.squeeze(1), ref_logits_neg.squeeze(1)
        x_pos, x_neg = x_pos.squeeze(1), x_neg.squeeze(1)
        
        logits_pos, logits_neg = torch.log_softmax(logits_pos, dim=-1), torch.log_softmax(logits_neg, dim=-1)
        ref_logits_pos, ref_logits_neg = torch.log_softmax(ref_logits_pos, dim=-1), torch.log_softmax(ref_logits_neg, dim=-1)

        selected_logits_pos = torch.gather(logits_pos[:,:-1], dim=-1, index=x_pos[:,1:].unsqueeze(-1)).squeeze(-1)
        selected_logits_neg = torch.gather(logits_neg[:,:-1], dim=-1, index=x_neg[:,1:].unsqueeze(-1)).squeeze(-1)
        selected_ref_logits_pos = torch.gather(ref_logits_pos[:,:-1], dim=-1, index=x_pos[:,1:].unsqueeze(-1)).squeeze(-1)
        selected_ref_logits_neg = torch.gather(ref_logits_neg[:,:-1], dim=-1, index=x_neg[:,1:].unsqueeze(-1)).squeeze(-1)

        pi_logits = selected_logits_pos - selected_logits_neg
        if self.cfg.loss_clamp:
            pi_logits = torch.clamp(pi_logits, None, self.cfg.loss_clamp_value)
        pi_ref_logits = selected_ref_logits_pos - selected_ref_logits_neg

        loss = -F.logsigmoid(self.beta * pi_logits-self.beta * pi_ref_logits)
        
        # conservative_regularizer = F.logsigmoid((self.beta * logits_pos[:,:-1]-self.beta * ref_logits_pos[:,:-1])) + F.logsigmoid((self.beta * logits_neg[:,:-1]-self.beta * ref_logits_neg[:,:-1]))
        # conservative_regularizer = conservative_regularizer.sum(dim=-1) - F.logsigmoid(self.beta * (selected_logits_pos - selected_ref_logits_pos)) - F.logsigmoid(self.beta * (selected_logits_neg - selected_ref_logits_neg))
        # conservative_regularizer /= (C-2)
        conservative_regularizer = F.sigmoid((self.beta * logits_pos[:,:-1]-self.beta * ref_logits_pos[:,:-1])) + F.sigmoid((self.beta * logits_neg[:,:-1]-self.beta * ref_logits_neg[:,:-1]))
        conservative_regularizer = conservative_regularizer.sum(dim=-1) - F.sigmoid(self.beta * (selected_logits_pos - selected_ref_logits_pos)) - F.sigmoid(self.beta * (selected_logits_neg - selected_ref_logits_neg))
        conservative_regularizer /= (C-2)

        return loss.mean() + self.cfg.conservative_coeff * conservative_regularizer.mean()

    def contrastive_loss(self, logits, ref_logits, B, T, C):
        logits, ref_logits = logits.view(B, 2, T, C), ref_logits.view(B, 2, T, C)
        logits_pos, logits_neg = torch.split(logits, 1, dim=1)
        ref_logits_pos, ref_logits_neg = torch.split(ref_logits, 1, dim=1)
        logits_pos, logits_neg = logits_pos.squeeze(1), logits_neg.squeeze(1)
        ref_logits_pos, ref_logits_neg = ref_logits_pos.squeeze(1), ref_logits_neg.squeeze(1)

        logits_pos, logits_neg = torch.softmax(logits_pos, dim=-1), torch.softmax(logits_neg, dim=-1)
        ref_logits_pos, ref_logits_neg = torch.softmax(ref_logits_pos, dim=-1), torch.softmax(ref_logits_neg, dim=-1)

        pi_logits = torch.log(logits_pos) + torch.log(logits_neg)
        pi_ref_logits = torch.log(ref_logits_pos) + torch.log(ref_logits_neg)

        similarity = F.cosine_similarity(pi_logits, pi_ref_logits, dim=-1)
        if self.cfg.contrastive_clamp:
            similarity = torch.clamp(similarity, None, self.cfg.contrastive_clamp_value)

        distance = F.logsigmoid(self.beta * pi_logits) + F.logsigmoid(self.beta * pi_ref_logits)
        return distance.mean() + similarity.mean()
    
    @torch.no_grad()
    def compute_accuracy(self, logits, x, B,T,C):
        logits, x = logits.view(B, 2, T, C), x.view(B, 2, T)
        logits, x = logits[:,:,:-1,:], x[:,:,1:]
        logits_pos, logits_neg = torch.split(logits, 1, dim=1)
        logits_pos, logits_neg = logits_pos.squeeze(1), logits_neg.squeeze(1)
        x_pos, x_neg = torch.split(x, 1, dim=1)
        x_pos, x_neg = x_pos.squeeze(1), x_neg.squeeze(1)
        pred_prob_pos = torch.softmax(logits_pos.squeeze(1), dim=-1)
        pred_prob_neg = torch.softmax(logits_neg.squeeze(1), dim=-1)
        pred_prob_pos = torch.gather(pred_prob_pos, dim=-1, index=x_pos.unsqueeze(-1)).squeeze(-1)
        pred_prob_neg = torch.gather(pred_prob_neg, dim=-1, index=x_neg.unsqueeze(-1)).squeeze(-1)
        # sentence_prob_pos = torch.prod(pred_prob_pos, dim=-1)
        # sentence_prob_neg = torch.prod(pred_prob_neg, dim=-1)
        sentence_log_prob_pos = torch.sum(torch.log(pred_prob_pos), dim=-1).squeeze()
        sentence_log_prob_neg = torch.sum(torch.log(pred_prob_neg), dim=-1).squeeze()

        return (sentence_log_prob_pos > sentence_log_prob_neg).float().mean().item()
        # return (sentence_log_prob_pos < sentence_log_prob_neg).float().mean().item()
        
    @torch.no_grad()
    def evaluate(self, eval_loader=None):
        if eval_loader is None:
            eval_loader = self.test_dataloader
        self.model.eval()

        loss_list, acc_list, B_list = [], [], []
        eval_step_count=0
        for x, attn_mask in eval_loader:
            # eval_step_count += 1
            # if eval_step_count > self.cfg.eval_set_size:
            #     break
            # print(x.size())
            B,_, T = x.size()
            x, attn_mask = x.reshape(-1, T), attn_mask.reshape(-1, T)
            x, attn_mask = x.to(self.device), attn_mask.to(self.device)
            logits = self.model(x, attention_mask=attn_mask)
            _, _, C = logits.shape
            logits = logits.view(-1,C)
            ref_logits = self.ref_model(x, attention_mask=attn_mask).view(-1,C)

            loss = self.dpo_loss(logits, ref_logits, x, B, T, C)
            loss_list.append(loss.item())
            acc = self.compute_accuracy(logits, x, B, T, C)
            acc_list.append(acc)
            B_list.append(B)
        self.model.train()
        return np.mean(np.array(loss_list)), np.sum(np.array(acc_list)*np.array(B_list))/np.sum(np.array(B_list))
    
    @torch.no_grad()
    def batch_evaluate(self, batched_x):
        self.model.eval()

        B,_, T = batched_x.size()
        batched_x = batched_x.reshape(-1, T)
        batched_x = batched_x.to(self.device)
        logits = self.model(batched_x)
        _, _, C = logits.shape
        logits = logits.view(-1,C)
        ref_logits = self.ref_model(batched_x).view(-1,C)

        loss = self.dpo_loss(logits, ref_logits, batched_x, B, T, C)
        acc = self.compute_accuracy(logits, batched_x, B, T, C)
        self.model.train()
        return loss.item(), acc

    def plot_figure(self):
        if not os.path.exists(f'./runs/{self.run_name}'):
            os.makedirs(f'./runs/{self.run_name}')
        plt.figure()
        plt.plot(self.loss_list)
        plt.xlabel('steps')
        plt.ylabel('train_loss')
        plt.savefig(f'./runs/{self.run_name}/train_loss.png')
        plt.close()

        plt.figure()
        plt.plot(self.eval_loss_list)
        plt.xlabel('steps')
        plt.ylabel('eval_loss')
        plt.savefig(f'./runs/{self.run_name}/eval_loss.png')
        plt.close()

        plt.figure()
        plt.plot(self.eval_acc_list)
        plt.xlabel('steps')
        plt.ylabel('eval_acc')
        plt.savefig(f'./runs/{self.run_name}/eval_acc.png')
        plt.close()

    def fit(self):
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.actor_lr, betas=(self.cfg.adam_beta1, self.cfg.adam_beta2))
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.dpo_lr, betas=(self.cfg.adam_beta1, self.cfg.adam_beta2))
        # self.scheduler = WarmUpScheduler(self.optimizer, warmup_steps=150)
        # criterion = torch.nn.CrossEntropyLoss()

        self.model.to(self.device)
        self.ref_model.to(self.device)
        cummulated_iterations = 0
        eval_interval = 0
        epoch_loss_list = []
        self.optimizer.zero_grad()
        steps_count = 0
        cummulated_steps = 0
        flag=False
        
        for epoch in tqdm(range(self.cfg.total_epochs)):
            for x,attn_mask in tqdm(self.train_dataloader):
                steps_count += 1
                # if steps_count > self.cfg.max_steps:
                #     steps_count = 0
                #     break
                B,_, T = x.size()
                x, attn_mask = x.reshape(-1, T), attn_mask.reshape(-1, T)
                x, attn_mask = x.to(self.device), attn_mask.to(self.device)

                logits = self.model(x, attention_mask=attn_mask)
                _, _, C = logits.shape
                logits = logits.view(-1,C)
                with torch.no_grad():
                    ref_logits = self.ref_model(x, attention_mask=attn_mask).view(-1,C)

                if self.cfg.conservative_loss:
                    loss = self.conservative_dpo_loss(logits, ref_logits, x, B, T, C)
                elif self.cfg.contrastive_loss:
                    loss = self.contrastive_loss(logits, ref_logits, B, T, C)
                else:
                    loss = self.dpo_loss(logits, ref_logits, x, B, T, C)

                epoch_loss_list.append(loss.item())
                loss.backward()

                cummulated_iterations += 1
                if cummulated_iterations >= self.cfg.max_cumulate_iter:
                    cummulated_steps += 1
                    cummulated_iterations = 0

                    if self.cfg.gradient_clipping:
                        nn.utils.clip_grad_value_(self.model.parameters(), clip_value=self.cfg.grad_value_clip*self.cfg.max_cumulate_iter)
                    # nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0*self.cfg.max_cumulate_iter, norm_type=2)
            
                    self.optimizer.step()
                    if self.scheduler is not None:
                        self.scheduler.step()
                    self.optimizer.zero_grad()
                    self.loss_list.append(np.mean(np.array(epoch_loss_list)))
                    epoch_loss_list = []

                    eval_interval += 1
                    if eval_interval >= self.cfg.eval_interval:
                        # flag=True
                        eval_interval = 0
                        current_eval_loss, current_eval_acc = self.evaluate()
                        # current_eval_loss, current_eval_acc = 0,0
                        
                        if len(self.eval_loss_list)>0 and current_eval_acc > max(self.eval_acc_list):
                            self.save_states(cummulated_steps, is_last=True)
                            if self.cfg.update_ref_model:
                                ref_state_dict = self.model.state_dict()
                                self.ref_model.load_state_dict(ref_state_dict)
                                self.ref_model.to(self.device)
                    
                        self.eval_loss_list.append(current_eval_loss)
                        self.eval_acc_list.append(current_eval_acc)
                        self.plot_figure()
                        self.save_metrics({'train_loss':self.loss_list, 'eval_loss':self.eval_loss_list, 'eval_acc':self.eval_acc_list})
                        self.model.train()
                    
                    # if self.cfg.max_cumulate_iter < 512 and steps_count % 100 == 0:
                    #     self.eval_loss_list.append(current_eval_loss)
                    #     self.eval_acc_list.append(current_eval_acc)
                    #     self.plot_figure()
                    #     self.save_metrics({'train_loss':self.loss_list, 'eval_loss':self.eval_loss_list, 'eval_acc':self.eval_acc_list})
                    #     self.model.train()
                    # elif self.cfg.max_cumulate_iter >= 512:
                    #     self.eval_loss_list.append(current_eval_loss)
                    #     self.eval_acc_list.append(current_eval_acc)
                    #     self.plot_figure()
                    #     self.save_metrics({'train_loss':self.loss_list, 'eval_loss':self.eval_loss_list, 'eval_acc':self.eval_acc_list})
                    #     self.model.train()
                    # self.eval_loss_list.append(current_eval_loss)
                    # self.eval_acc_list.append(current_eval_acc)
                    # self.plot_figure()
                    # self.save_metrics({'train_loss':self.loss_list, 'eval_loss':self.eval_loss_list, 'eval_acc':self.eval_acc_list})
