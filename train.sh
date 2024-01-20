#! /bin/bash
set -e

# method

adap_base=1
adap_factor=2

# device

# gpu_ids="0,1,2,3"
gpu_ids="4,5,6,7"
num_gpus=4

num_cpu_processors=256
total_num_gpus=8
num_cpu_threads_per_process=$((num_cpu_processors / total_num_gpus))

# model

model_size_b=7
model_name="llemma-${model_size_b}b"
model_name_or_path="EleutherAI/llemma_${model_size_b}b"

# data

data_name="math-instruct-core-diffi-aware"
data_path="tongyx361/MathInstruct-Core-DifficultyAware"

# hyperparameters

model_max_length=512
bs=128
train_bs_per_gpu=$((bs / num_gpus))

num_train_epochs=1 # Data already upsampled.
# Train for 1 epoch for fair comparison with multiple epochs.

lr="2e-5"
weight_decay=0
warmup_ratio=0.03
lr_scheduler_type="cosine"

gradient_acc_steps=$((train_bs_per_gpu / train_bs_per_gpu))

# infrastructure

deepspeed_config_file="configs/stage1.conf"
export OMP_NUM_THREADS=1

torch_compile_backend="inductor"

exp_info="\
${model_name}-${data_name}\
-b${adap_base}-w${adap_factor}\
-bs${bs}-gas${gradient_acc_steps}\
-lr${lr}-wd${weight_decay}\
-warmratio${warmup_ratio}-sched-${lr_scheduler_type}\
-epochs${num_train_epochs}-maxlen${model_max_length}"

# adapted from https://gist.github.com/pacman100/1cb1f17b2f1b3139a63b764263e70b25
# OTHER LAUNCHERS CAN BE USED HERE
launcher="\
accelerate launch \
--mixed_precision bf16 \
--num_machines 1 \
--num_processes ${num_gpus} \
--num_cpu_threads_per_process ${num_cpu_threads_per_process} \
--use_deepspeed \
--gpu_ids ${gpu_ids} \
--rdzv_backend static \
--deepspeed_config_file ${deepspeed_config_file} \
--dynamo_backend ${torch_compile_backend} \
"

program="\
train.py \
--model_name_or_path ${model_name_or_path} \
--data_path ${data_path} \
--output_dir models/${exp_info} \
--model_max_length ${model_max_length} \
--per_device_train_batch_size ${train_bs_per_gpu} \
--per_device_eval_batch_size 4 \
--gradient_checkpointing True \
--gradient_accumulation_steps ${gradient_acc_steps} \
--num_train_epochs ${num_train_epochs} \
--logging_nan_inf_filter False \
--learning_rate ${lr} \
--weight_decay ${weight_decay} \
--warmup_ratio ${warmup_ratio} \
--lr_scheduler_type ${lr_scheduler_type} \
--save_strategy epoch \
--bf16 True \
--tf32 True \
--logging_strategy steps \
--logging_steps 1 \
--deepspeed ${deepspeed_config_file} \
--torch_compile True \
--torch_compile_backend ${torch_compile_backend} \
"

program="\
${program} \
--adap2difficulty True \
--adap_base ${adap_base} \
--adap_factor ${adap_factor}\
"

cmd="${launcher} ${program}"

echo -e "${launcher//--/\\n\\t--}\n${program//--/\\n\\t--}"

bash -c "${cmd}"
