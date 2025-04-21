export CUDA_VISIBLE_DEVICES=1
export DEEPSPEED_NO_MPI_INIT=1

model_name=TimeLLM
learning_rate=0.01
llama_layers=32

master_port=12345
num_process=1
batch_size=32
d_model=32
d_ff=32

comment='TimeLLM-5GTraffic_0_1'

accelerate launch --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_pretrain.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ours/ \
  --data_path_pretrain processed_beam_0.csv \
  --data_path processed_beam_1.csv \
  --model_id ETTh1_ETTh2_512_96 \
  --model $model_name \
  --data_pretrain beam \
  --data beam \
  --features M \
  --seq_len 12 \
  --label_len 12 \
  --pred_len 24 \
  --factor 3 \
  --enc_in 3 \
  --dec_in 3 \
  --c_out 3 \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --llm_layers $llama_layers \
  --train_epochs 10 \
  --model_comment $comment
