# CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES=2,4,5,7

export HF_HOME="~/.cache/huggingface"

# DeepSpeed Team
BASE_PATH="./"
DATA_PATH="Dahoas/full-hh-rlhf"
ZERO_STAGE=0
REF_ZERO_STAGE=0

LOG_PATH="${BASE_PATH}/log"
SEED=1234
LR=2e-6
ALGORITHM="lion"


# MODEL_NAME="./models/Qwen2-0_5B"
# PER_DEVICE_BATCH_SIZE=2
# GRADIENT_ACCUMULATION_STEPS=1
# OUTPUT="${LOG_PATH}/dpo-qwen2-0_5b-full/dpo-${ALGORITHM}-${PER_DEVICE_BATCH_SIZE}-${GRADIENT_ACCUMULATION_STEPS}"

MODEL_NAME="./models/deepseek-coder-1_3b"
PER_DEVICE_BATCH_SIZE=2
GRADIENT_ACCUMULATION_STEPS=16
OUTPUT="${LOG_PATH}/dpo-deepseek-coder-1_3b-full/dpo-${ALGORITHM}-${PER_DEVICE_BATCH_SIZE}-${GRADIENT_ACCUMULATION_STEPS}"

# MODEL_NAME="./models/mamba-130m"
# PER_DEVICE_BATCH_SIZE=8
# GRADIENT_ACCUMULATION_STEPS=4
# OUTPUT="${LOG_PATH}/dpo-mamba-130m-full/dpo-${ALGORITHM}-${PER_DEVICE_BATCH_SIZE}-${GRADIENT_ACCUMULATION_STEPS}"

mkdir -p $OUTPUT

nohup deepspeed  --master_port 26700 ./dpo.py \
   --data_path $DATA_PATH \
   --data_output_path "${BASE_PATH}/tmp_dpo/" \
   --data_split 2,0,8 \
   --model_name_or_path $MODEL_NAME \
   --tokenizer_path $MODEL_NAME \
   --per_device_train_batch_size $PER_DEVICE_BATCH_SIZE \
   --per_device_eval_batch_size 4 \
   --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
   --max_seq_len 2048 \
   --learning_rate $LR \
   --weight_decay 0. \
   --num_train_epochs 3 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 10 \
   --seed $SEED \
   --actor_zero_stage $ZERO_STAGE \
   --reference_zero_stage $REF_ZERO_STAGE \
   --optimizer $ALGORITHM \
   --deepspeed \
   --dtype bf16 \
   --gradient_checkpointing \
   --output_dir $OUTPUT \
   --enable_tensorboard \
   --print_loss \
   --flash_attn True\
   --save_model \
   --eval_interval 20 \
   &> $OUTPUT/training.log
