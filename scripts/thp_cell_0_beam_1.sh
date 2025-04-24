#!/bin/bash

# 创建输出日志目录
output_dir="./results/thp_value_cell_0_beam_1/"
data_path="processed_thp_value_cell_0_beam_1.csv"
mkdir -p $output_dir

# 基本配置
model_name=TimeLLM
model_id=THP_CELL_0_BEAM_1
train_epochs=10
learning_rate=0.01
llama_layers=32
num_process=2
batch_size=32
d_model=16
d_ff=32
comment='TimeLLM-BEAM'

# 检查端口是否被占用，如果被占用则终止相应进程
master_port=51111
if lsof -i:$master_port > /dev/null ; then
    echo "Port $master_port is in use. Killing the process..."
    pid=$(lsof -ti:$master_port)
    kill -9 $pid
    sleep 1
    echo "Process using port $master_port has been terminated."
fi

# 噪声测试参数
noise_min=0.01
noise_max=0.05
noise_steps=5

# 缺失率测试参数
missing_min=0.05
missing_max=0.25
missing_steps=5
interpolation="linear"

# 公共配置参数
common_params="--model $model_name \
  --data beam \
  --features M \
  --target value \
  --seq_len 12 \
  --label_len 12 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 3 \
  --dec_in 3 \
  --c_out 3 \
  --batch_size $batch_size \
  --llm_layers $llama_layers \
  --model_id $model_id \
  --data_path $data_path \
  --root_path ./dataset/processed/"

# 训练模型函数
run_training() {
    local task_name=$1
    local train_rate=$2
    local pred_len=$3
    local log_prefix="${output_dir}${task_name}_train${train_rate/./}"
    
    echo "开始训练任务：$task_name，train_rate=$train_rate, pred_len=$pred_len"
    
    nohup accelerate launch --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
      --task_name $task_name \
      --is_training 1 \
      --pred_len $pred_len \
      --train_rate $train_rate \
      --learning_rate $learning_rate \
      --train_epochs $train_epochs \
      --model_comment $comment \
      $common_params \
      > ${log_prefix}.log 2>&1 &
      
    local task_pid=$!
    wait $task_pid
    
    if [ -f ./checkpoints/checkpoint ]; then
        local ckpt_path="${log_prefix}.ckpt"
        cp ./checkpoints/checkpoint $ckpt_path
        rm -rf ./checkpoints/checkpoint
        
        run_tests $task_name $train_rate $pred_len $ckpt_path
    fi
}

# 测试函数
run_tests() {
    local task_name=$1
    local train_rate=$2
    local pred_len=$3
    local ckpt_path=$4
    local test_output="${output_dir}${task_name}_train${train_rate/./}/"
    
    echo "正在对${task_name}任务(train_rate=${train_rate})进行鲁棒性测试..."
    mkdir -p $test_output
    
    # 测试
    accelerate launch --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
      --task_name $task_name \
      --is_training 0 \
      --pred_len $pred_len \
      --train_rate $train_rate \
      --checkpoint_path $ckpt_path \
      --output_dir $test_output \
      $common_params \
      > ${test_output}test.log 2>&1
    
    # 噪声鲁棒性测试
    accelerate launch --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port robustness_test.py \
      --task_name $task_name \
      --pred_len $pred_len \
      --train_rate $train_rate \
      --checkpoint_path $ckpt_path \
      --output_dir $test_output \
      --noise_min $noise_min \
      --noise_max $noise_max \
      --noise_steps $noise_steps \
      $common_params \
      > ${test_output}noisy_test.log 2>&1
    
    # 缺失数据测试
    accelerate launch --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port missing_data_test.py \
      --task_name $task_name \
      --pred_len $pred_len \
      --train_rate $train_rate \
      --checkpoint_path $ckpt_path \
      --output_dir $test_output \
      --missing_min $missing_min \
      --missing_max $missing_max \
      --missing_steps $missing_steps \
      --interpolation $interpolation \
      $common_params \
      > ${test_output}missing_data_test.log 2>&1
}

# 执行任务
# 任务1：长期预测，train_rate=0.8
run_training "long_term_forecast" "0.8" "24"

# 任务2：长期预测，train_rate=0.2
run_training "long_term_forecast" "0.2" "24"

# 任务3：短期预测，train_rate=0.8
run_training "short_term_forecast" "0.8" "6"

# 任务4：短期预测，train_rate=0.2
run_training "short_term_forecast" "0.2" "6"

# 生成日志和报告
nohup python -m scripts.log --log_folder $output_dir &
wait

echo "所有任务和鲁棒性测试已完成，结果保存在 ${output_dir} 目录下"