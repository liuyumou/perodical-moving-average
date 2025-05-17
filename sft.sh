# CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES=2,4,5,7

export HF_HOME="~/.cache/huggingface"

# DeepSpeed Team
BASE_PATH="./"
DATA_PATH="shibing624/sharegpt_gpt4"


ZERO_STAGE=0

LOG_PATH="${BASE_PATH}/log"
SEED=1234
LR=2e-6
ALGORITHM="adamw"

# MODEL_NAME="./models/Qwen2-0_5B"
# PER_DEVICE_BATCH_SIZE=4
# GRADIENT_ACCUMULATION_STEPS=8
# OUTPUT="${LOG_PATH}/sft-qwen2-0_5b-full/sft-${ALGORITHM}-${PER_DEVICE_BATCH_SIZE}-${GRADIENT_ACCUMULATION_STEPS}"
# USE_FLASH_ATTN=$true

MODEL_NAME="./models/deepseek-coder-1_3b"
PER_DEVICE_BATCH_SIZE=8
GRADIENT_ACCUMULATION_STEPS=4
OUTPUT="${LOG_PATH}/sft-deepseek-coder-1_3b-full/sft-${ALGORITHM}-${PER_DEVICE_BATCH_SIZE}-${GRADIENT_ACCUMULATION_STEPS}"
USE_FLASH_ATTN=$true

# MODEL_NAME="./models/openelm-1_1b"
# PER_DEVICE_BATCH_SIZE=2
# GRADIENT_ACCUMULATION_STEPS=16
# OUTPUT="${LOG_PATH}/sft-openelm-1_1b-full/sft-${ALGORITHM}-${PER_DEVICE_BATCH_SIZE}-${GRADIENT_ACCUMULATION_STEPS}"
# USE_FLASH_ATTN=$true

# MODEL_NAME="./models/mamba-130m"
# PER_DEVICE_BATCH_SIZE=8
# GRADIENT_ACCUMULATION_STEPS=4
# OUTPUT="${LOG_PATH}/sft-mamba-130m-full/sft-${ALGORITHM}-${PER_DEVICE_BATCH_SIZE}-${GRADIENT_ACCUMULATION_STEPS}"
# USE_FLASH_ATTN=$false

mkdir -p $OUTPUT

nohup deepspeed  --master_port 26700 ./sft.py \
   --data_path $DATA_PATH \
   --data_output_path "${BASE_PATH}/tmp/" \
   --data_split 4,6,0 \
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
   --zero_stage $ZERO_STAGE \
   --optimizer $ALGORITHM \
   --deepspeed \
   --dtype fp32 \
   --gradient_checkpointing \
   --output_dir $OUTPUT \
   --enable_tensorboard \
   --print_loss \
   --flash_attn True\
   --save_model \
   --eval_interval 20 \
   &> $OUTPUT/training.log
