#!/bin/bash

# 创建输出日志目录
output_dir="./results/thp_value_cell_0_beam_1/"
data_path="processed_thp_value_cell_0_beam_1.csv"
mkdir -p $output_dir

model_name=TimeLLM
model_id=CELL_0_BEAM_1
train_epochs=10
learning_rate=0.01
llama_layers=32

# 检查端口是否被占用，如果被占用则终止相应进程
master_port=51111
if lsof -i:$master_port > /dev/null ; then
    echo "Port $master_port is in use. Killing the process..."
    pid=$(lsof -ti:$master_port)
    kill -9 $pid
    sleep 1
    echo "Process using port $master_port has been terminated."
fi
num_process=2
batch_size=32
d_model=16
d_ff=32

comment='TimeLLM-BEAM'

# 噪声测试参数
noise_min=0.01
noise_max=0.05
noise_steps=5

# 缺失率测试参数
missing_min=0.0
missing_max=0.25
missing_steps=6
interpolation="linear"

# 任务1：long_term_forecast，train_rate 0.8
nohup accelerate launch --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/processed/ \
  --data_path $data_path \
  --model_id $model_id \
  --model $model_name \
  --data beam \
  --features M \
  --target value \
  --seq_len 12 \
  --label_len 12 \
  --pred_len 24 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 3 \
  --dec_in 3 \
  --c_out 3 \
  --batch_size $batch_size \
  --train_rate 0.8 \
  --learning_rate $learning_rate \
  --llm_layers $llama_layers \
  --train_epochs $train_epochs \
  --model_comment $comment \
  > ${output_dir}long_term_train8.log 2>&1 &

# 获取后台进程的PID
task1_pid=$!
wait $task1_pid

# 任务1完成后复制检查点并进行鲁棒性测试
if [ -f ./checkpoints/checkpoint ]; then
  ckpt_path="${output_dir}long_term_train8.ckpt"
  cp ./checkpoints/checkpoint $ckpt_path
  rm -rf ./checkpoints/checkpoint
  
  echo "正在对长期预测任务(train_rate=0.8)进行鲁棒性测试..."
  robustness_output="${output_dir}long_term_train8/"
  mkdir -p $robustness_output
  
  accelerate launch --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port robustness_test.py \
    --model $model_name \
    --task_name long_term_forecast \
    --checkpoint_path $ckpt_path \
    --data beam \
    --model_id $model_id \
    --output_dir $robustness_output \
    --data_path $data_path \
    --root_path ./dataset/processed/ \
    --target value \
    --seq_len 12 \
    --label_len 12 \
    --pred_len 24 \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 3 \
    --dec_in 3 \
    --c_out 3 \
    --batch_size $batch_size \
    --train_rate 0.2 \
    --llm_layers $llama_layers \
    --noise_min $noise_min \
    --noise_max $noise_max \
    --noise_steps $noise_steps \
    > ${robustness_output}noisy_test.log 2>&1

  accelerate launch --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port missing_data_test.py \
    --model $model_name \
    --task_name long_term_forecast \
    --checkpoint_path ${output_dir}long_term_train8.ckpt \
    --data beam \
    --model_id $model_id \
    --output_dir $robustness_output \
    --data_path $data_path \
    --root_path ./dataset/processed/ \
    --target value \
    --seq_len 12 \
    --label_len 12 \
    --pred_len 24 \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 3 \
    --dec_in 3 \
    --c_out 3 \
    --batch_size $batch_size \
    --train_rate 0.8 \
    --llm_layers $llama_layers \
    --missing_min $missing_min \
    --missing_max $missing_max \
    --missing_steps $missing_steps \
    --interpolation $interpolation \
    --visualize_examples \
    > ${robustness_output}missing_data_test.log 2>&1
fi

wait

# 任务2：long_term_forecast，train_rate 0.2
nohup accelerate launch --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/processed/ \
  --data_path $data_path \
  --model_id $model_id \
  --model $model_name \
  --data beam \
  --features M \
  --target value \
  --seq_len 12 \
  --label_len 12 \
  --pred_len 24 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 3 \
  --dec_in 3 \
  --c_out 3 \
  --batch_size $batch_size \
  --train_rate 0.2 \
  --learning_rate $learning_rate \
  --llm_layers $llama_layers \
  --train_epochs $train_epochs \
  --model_comment $comment \
  > ${output_dir}long_term_train2.log 2>&1 &

# 获取后台进程的PID
task2_pid=$!
wait $task2_pid

# 任务2完成后复制检查点并进行鲁棒性测试
if [ -f ./checkpoints/checkpoint ]; then
  ckpt_path="${output_dir}long_term_train2.ckpt"
  cp ./checkpoints/checkpoint $ckpt_path
  rm -rf ./checkpoints/checkpoint
  
  echo "正在对长期预测任务(train_rate=0.2)进行鲁棒性测试..."
  robustness_output="${output_dir}long_term_train2/"
  mkdir -p $robustness_output
  
  accelerate launch --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port robustness_test.py \
    --model $model_name \
    --task_name long_term_forecast \
    --checkpoint_path $ckpt_path \
    --data beam \
    --model_id $model_id \
    --output_dir $robustness_output \
    --data_path $data_path \
    --root_path ./dataset/processed/ \
    --target value \
    --seq_len 12 \
    --label_len 12 \
    --pred_len 24 \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 3 \
    --dec_in 3 \
    --c_out 3 \
    --batch_size $batch_size \
    --train_rate 0.2 \
    --llm_layers $llama_layers \
    --noise_min $noise_min \
    --noise_max $noise_max \
    --noise_steps $noise_steps \
    > ${robustness_output}noisy_test.log 2>&1

  accelerate launch --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port missing_data_test.py \
    --model $model_name \
    --task_name long_term_forecast \
    --checkpoint_path $ckpt_path \
    --data beam \
    --model_id $model_id \
    --output_dir $robustness_output \
    --data_path $data_path \
    --root_path ./dataset/processed/ \
    --target value \
    --seq_len 12 \
    --label_len 12 \
    --pred_len 24 \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 3 \
    --dec_in 3 \
    --c_out 3 \
    --batch_size $batch_size \
    --train_rate 0.8 \
    --llm_layers $llama_layers \
    --missing_min $missing_min \
    --missing_max $missing_max \
    --missing_steps $missing_steps \
    --interpolation $interpolation \
    --visualize_examples \
    > ${robustness_output}missing_data_test.log 2>&1
fi

wait

# 任务3：short_term_forecast，train_rate 0.8
nohup accelerate launch --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ./dataset/processed/ \
  --data_path $data_path \
  --model_id $model_id \
  --model $model_name \
  --data beam \
  --features M \
  --target value \
  --seq_len 12 \
  --label_len 12 \
  --pred_len 6 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 3 \
  --dec_in 3 \
  --c_out 3 \
  --batch_size $batch_size \
  --train_rate 0.8 \
  --learning_rate $learning_rate \
  --llm_layers $llama_layers \
  --train_epochs $train_epochs \
  --model_comment $comment \
  > ${output_dir}short_term_train8.log 2>&1 &

# 获取后台进程的PID
task3_pid=$!
wait $task3_pid

# 任务3完成后复制检查点并进行鲁棒性测试
if [ -f ./checkpoints/checkpoint ]; then
  ckpt_path="${output_dir}short_term_train8.ckpt"
  cp ./checkpoints/checkpoint $ckpt_path
  rm -rf ./checkpoints/checkpoint
  
  echo "正在对短期预测任务(train_rate=0.8)进行鲁棒性测试..."
  robustness_output="${output_dir}short_term_train8/"
  mkdir -p $robustness_output
  
  accelerate launch --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port robustness_test.py \
    --model $model_name \
    --task_name short_term_forecast \
    --checkpoint_path $ckpt_path \
    --data beam \
    --model_id $model_id \
    --output_dir $robustness_output \
    --data_path $data_path \
    --root_path ./dataset/processed/ \
    --target value \
    --seq_len 12 \
    --label_len 12 \
    --pred_len 6 \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 3 \
    --dec_in 3 \
    --c_out 3 \
    --batch_size $batch_size \
    --train_rate 0.2 \
    --llm_layers $llama_layers \
    --noise_min $noise_min \
    --noise_max $noise_max \
    --noise_steps $noise_steps \
    > ${robustness_output}noisy_test.log 2>&1

  accelerate launch --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port missing_data_test.py \
    --model $model_name \
    --task_name short_term_forecast \
    --checkpoint_path $ckpt_path \
    --data beam \
    --model_id $model_id \
    --output_dir $robustness_output \
    --data_path $data_path \
    --root_path ./dataset/processed/ \
    --target value \
    --seq_len 12 \
    --label_len 12 \
    --pred_len 6 \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 3 \
    --dec_in 3 \
    --c_out 3 \
    --batch_size $batch_size \
    --train_rate 0.8 \
    --llm_layers $llama_layers \
    --missing_min $missing_min \
    --missing_max $missing_max \
    --missing_steps $missing_steps \
    --interpolation $interpolation \
    --visualize_examples \
    > ${robustness_output}missing_data_test.log 2>&1
fi

wait

# 任务4：short_term_forecast，train_rate 0.2
nohup accelerate launch --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ./dataset/processed/ \
  --data_path $data_path \
  --model_id $model_id \
  --model $model_name \
  --data beam \
  --features M \
  --target value \
  --seq_len 12 \
  --label_len 12 \
  --pred_len 6 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 3 \
  --dec_in 3 \
  --c_out 3 \
  --batch_size $batch_size \
  --train_rate 0.2 \
  --learning_rate $learning_rate \
  --llm_layers $llama_layers \
  --train_epochs $train_epochs \
  --model_comment $comment \
  > ${output_dir}short_term_train2.log 2>&1 &

# 获取后台进程的PID
task4_pid=$!
wait $task4_pid

# 任务4完成后复制检查点并进行鲁棒性测试
if [ -f ./checkpoints/checkpoint ]; then
  ckpt_path="${output_dir}short_term_train2.ckpt"
  cp ./checkpoints/checkpoint $ckpt_path
  rm -rf ./checkpoints/checkpoint
  
  echo "正在对短期预测任务(train_rate=0.2)进行鲁棒性测试..."
  robustness_output="${output_dir}short_term_train2/"
  mkdir -p $robustness_output
  
  accelerate launch --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port robustness_test.py \
    --model $model_name \
    --task_name short_term_forecast \
    --checkpoint_path $ckpt_path \
    --data beam \
    --model_id $model_id \
    --output_dir $robustness_output \
    --data_path $data_path \
    --root_path ./dataset/processed/ \
    --target value \
    --seq_len 12 \
    --label_len 12 \
    --pred_len 6 \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 3 \
    --dec_in 3 \
    --c_out 3 \
    --batch_size $batch_size \
    --train_rate 0.2 \
    --llm_layers $llama_layers \
    --noise_min $noise_min \
    --noise_max $noise_max \
    --noise_steps $noise_steps \
    > ${robustness_output}noisy_test.log 2>&1

  accelerate launch --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port missing_data_test.py \
    --model $model_name \
    --task_name short_term_forecast \
    --checkpoint_path $ckpt_path \
    --data beam \
    --model_id $model_id \
    --output_dir $robustness_output \
    --data_path $data_path \
    --root_path ./dataset/processed/ \
    --target value \
    --seq_len 12 \
    --label_len 12 \
    --pred_len 6 \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 3 \
    --dec_in 3 \
    --c_out 3 \
    --batch_size $batch_size \
    --train_rate 0.8 \
    --llm_layers $llama_layers \
    --missing_min $missing_min \
    --missing_max $missing_max \
    --missing_steps $missing_steps \
    --interpolation $interpolation \
    --visualize_examples \
    > ${robustness_output}missing_data_test.log 2>&1
fi

wait

nohup python -m scripts.log --log_folder $output_dir &
# rm -rf ${output_dir}*.ckpt
wait
# 生成综合报告
echo "所有任务和鲁棒性测试已完成，结果保存在 ${output_dir} 目录下"
