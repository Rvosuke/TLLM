#!/bin/bash

# 创建输出日志目录
output_dir="./results/transfer_learning/"
mkdir -p $output_dir

model_name=TimeLLM
llama_layers=32
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

# 定义源数据集和目标数据集
# 情景1: thp类数据集迁移 (cell2beam0 -> cell0beam1)
src_thp_data="processed_thp_value_cell_2_beam_0.csv"
target_thp_data="processed_thp_value_cell_0_beam_1.csv"

# 情景2: number类数据集迁移 (cell2beam19 -> cell2beam27)
src_num_data="processed_number_value_cell_2_beam_19.csv"
target_num_data="processed_number_value_cell_2_beam_27.csv"

# 源模型检查点路径
src_thp_ckpt_long="./results/thp_cell_2_beam_0/long_term_forecast_train08.ckpt"
src_thp_ckpt_short="./results/thp_cell_2_beam_0/short_term_forecast_train08.ckpt"
src_num_ckpt_long="./results/number_value_cell_2_beam_19/long_term_forecast_train08.ckpt"
src_num_ckpt_short="./results/number_value_cell_2_beam_19/short_term_forecast_train08.ckpt"

# ============================= THP数据集迁移学习 =============================
# echo "开始THP数据迁移学习测试..."

# 1. 长序列预测 (thp数据集)
echo "THP数据: 长序列预测迁移测试..."
thp_long_output="${output_dir}thp_long/"
mkdir -p $thp_long_output

accelerate launch --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
  --task_name long_term_forecast \
  --is_training 0 \
  --root_path ./dataset/processed/ \
  --data_path $target_thp_data \
  --model_id THP_TRANSFER_LONG \
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
  --llm_layers $llama_layers \
  --checkpoint_path $src_thp_ckpt_long \
  --output_dir $thp_long_output \
  > ${thp_long_output}zero_shot_test.log 2>&1

# 2. 短序列预测 (thp数据集)
echo "THP数据: 短序列预测迁移测试..."
thp_short_output="${output_dir}thp_short/"
mkdir -p $thp_short_output

accelerate launch --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
  --task_name short_term_forecast \
  --is_training 0 \
  --root_path ./dataset/processed/ \
  --data_path $target_thp_data \
  --model_id THP_TRANSFER_SHORT \
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
  --llm_layers $llama_layers \
  --checkpoint_path $src_thp_ckpt_short \
  --output_dir $thp_short_output \
  > ${thp_short_output}zero_shot_test.log 2>&1

# ============================= NUMBER数据集迁移学习 =============================
echo "开始NUMBER数据迁移学习测试..."

# 3. 长序列预测 (number数据集)
echo "NUMBER数据: 长序列预测迁移测试..."
num_long_output="${output_dir}number_long/"
mkdir -p $num_long_output

accelerate launch --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
  --task_name long_term_forecast \
  --is_training 0 \
  --root_path ./dataset/processed/ \
  --data_path $target_num_data \
  --model_id NUMBER_TRANSFER_LONG \
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
  --llm_layers $llama_layers \
  --checkpoint_path $src_num_ckpt_long \
  --output_dir $num_long_output \
  > ${num_long_output}zero_shot_test.log 2>&1

# 4. 短序列预测 (number数据集)
echo "NUMBER数据: 短序列预测迁移测试..."
num_short_output="${output_dir}number_short/"
mkdir -p $num_short_output

accelerate launch --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
  --task_name short_term_forecast \
  --is_training 0 \
  --root_path ./dataset/processed/ \
  --data_path $target_num_data \
  --model_id NUMBER_TRANSFER_SHORT \
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
  --llm_layers $llama_layers \
  --checkpoint_path $src_num_ckpt_short \
  --output_dir $num_short_output \
  > ${num_short_output}zero_shot_test.log 2>&1

echo "所有迁移学习测试完成，结果保存在 ${output_dir} 目录下"