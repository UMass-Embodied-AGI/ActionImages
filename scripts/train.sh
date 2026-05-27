export TOKENIZERS_PARALLELISM=false

NUM_GPUS=${1:-8}

# Distributed training configuration
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}

# Set distributed training environment variables
export OMP_NUM_THREADS=1

export WANDB_PROJECT="actionimages"
# export WANDB_API_KEY="YOUR_WANDB_API_KEY"

# model_id="Wan-AI/Wan2.1-T2V-1.3B"
model_id="Wan-AI/Wan2.2-TI2V-5B"
# --init_ckpt_path $init_ckpt_path \ 

# Use torch run for distributed training
torchrun \
    --nnodes=1 \
    --nproc_per_node=$NUM_GPUS \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    train.py \
    --deepspeed ./configs/zero.json \
    --dataset_path ./data \
    --dataset_name rlbench \
    --output_dir ./outputs/rlbench-wan2.2 \
    --height 512 \
    --width 512 \
    --full_param True \
    --num_frames 41 \
    --model_id $model_id \
    --steps_per_epoch 8000 \
    --num_train_epochs 10000 \
    --learning_rate 5e-7 \
    --gradient_accumulation_steps 1 \
    --max_grad_norm 1.0 \
    --use_gradient_checkpointing \
    --dataloader_num_workers 2 \
    --dataloader_prefetch_factor 2 \
    --dataloader_pin_memory True \
    --checkpoint_every_n_steps 250 \
    --checkpoint_save_top_k 5 \
    --checkpoint_monitor "train_loss" \
    --remove_unused_columns False \
    --dataloader_drop_last True \
    --prediction_loss_only True \
    --bf16 True \
    --ddp_find_unused_parameters False \
    --report_to "wandb" \
    --save_safetensors False \
    --per_device_train_batch_size 1 \
    --logging_steps 1 \
    --lr_scheduler_type "constant_with_warmup" \
    --warmup_steps 1000 \
    --seed 42
