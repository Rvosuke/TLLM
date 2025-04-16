# export CUDA_VISIBLE_DEVICES=0,1
# export DEEPSPEED_NO_MPI_INIT=1

model_name=TimeLLM
train_epochs=10
learning_rate=0.01
llama_layers=32

master_port=16067
num_process=2
batch_size=32
d_model=16
d_ff=32

comment='TimeLLM-BEAM'

accelerate launch --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/processed/ \
  --data_path processed_thp_value_cell_0_beam_1.csv \
  --model_id CELL_0_BEAM_1 \
  --model $model_name \
  --data beam \
  --features M \
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
  --model_comment $comment

accelerate launch --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/processed/ \
  --data_path processed_thp_value_cell_0_beam_1.csv \
  --model_id CELL_0_BEAM_1 \
  --model $model_name \
  --data beam \
  --features M \
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
  --model_comment $comment


accelerate launch --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ./dataset/processed/ \
  --data_path processed_thp_value_cell_0_beam_1.csv \
  --model_id CELL_0_BEAM_1 \
  --model $model_name \
  --data beam \
  --features M \
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
  --model_comment $comment

accelerate launch --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ./dataset/processed/ \
  --data_path processed_thp_value_cell_0_beam_1.csv \
  --model_id CELL_0_BEAM_1 \
  --model $model_name \
  --data beam \
  --features M \
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
  --model_comment $comment
