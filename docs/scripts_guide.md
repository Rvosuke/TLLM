# Time-LLM 实验脚本详解

本文档详细解释了Time-LLM项目中各个实验脚本的功能和用法，帮助用户理解不同实验的目的和执行流程。

## 概述

Time-LLM项目包含多个脚本文件，用于执行不同的实验任务：
- 针对不同数据集的训练和测试
- 鲁棒性测试（噪声和缺失数据）
- 迁移学习实验

这些脚本保存在`scripts/`目录下，可以单独运行或通过`run_all.sh`批量执行。

## 主要脚本说明

### 1. TimeLLM_5GTraffic_num_C2B19.sh

此脚本用于对小区2波束19的Number类数据进行训练和测试。

#### 执行内容：
- **长期预测**：使用80%和20%的训练数据，预测未来24个时间步
- **短期预测**：使用80%和20%的训练数据，预测未来6个时间步
- **鲁棒性测试**：对每个训练模型进行噪声和缺失数据的鲁棒性测试

#### 关键参数：
- 数据路径：`processed_number_value_cell_2_beam_19.csv`
- 模型ID：`CELL_2_BEAM_19`
- 噪声测试范围：0.01-0.05
- 缺失数据测试范围：0-0.25

#### 使用方法：
```bash
bash scripts/TimeLLM_5GTraffic_num_C2B19.sh
```

### 2. TimeLLM_5GTraffic_num_C2B27.sh

此脚本用于对小区2波束27的Number类数据进行训练和测试，结构与上一个脚本类似。

#### 执行内容：
- 长期和短期预测任务（不同训练数据比例）
- 针对每个模型的鲁棒性测试

#### 关键参数：
- 数据路径：`processed_number_value_cell_2_beam_27.csv`
- 模型ID：`CELL_2_BEAM_27`

#### 使用方法：
```bash
bash scripts/TimeLLM_5GTraffic_num_C2B27.sh
```

### 3. TimeLLM_5GTraffic_thp_C0B1.sh

此脚本针对小区0波束1的吞吐量(Throughput)数据进行实验。

#### 执行内容：
- 长期和短期预测任务
- 不同训练数据比例下的模型训练
- 鲁棒性测试

#### 关键参数：
- 数据路径：`processed_thp_value_cell_0_beam_1.csv`
- 模型ID：`CELL_0_BEAM_1`

#### 使用方法：
```bash
bash scripts/TimeLLM_5GTraffic_thp_C0B1.sh
```

### 4. TimeLLM_5GTraffic_thp_C2B0.sh

此脚本针对小区2波束0的吞吐量数据进行实验。

#### 执行内容：
- 长期和短期预测任务
- 不同训练数据比例下的模型训练
- 鲁棒性测试

#### 关键参数：
- 数据路径：`processed_thp_value_cell_2_beam_0.csv`
- 模型ID：`CELL_2_BEAM_0`

#### 使用方法：
```bash
bash scripts/TimeLLM_5GTraffic_thp_C2B0.sh
```

### 5. TimeLLM_5GTraffic_transfer.sh

此脚本实现了迁移学习实验，评估模型从一个场景迁移到另一个场景的能力。

#### 执行内容：
- 从小区2波束0迁移到小区0波束1的吞吐量数据迁移学习
- 从小区2波束19迁移到小区2波束27的Number数据迁移学习
- 分别测试长期和短期预测任务

#### 关键参数：
- 源数据集和目标数据集定义
- 源模型检查点路径
- 迁移学习策略参数

#### 使用方法：
```bash
bash scripts/TimeLLM_5GTraffic_transfer.sh
```

## 脚本执行流程详解

以`TimeLLM_5GTraffic_num_C2B19.sh`为例，详细说明执行流程：

1. **参数设置**：
   - 设置输出目录、模型参数和训练配置
   - 检查端口占用情况，避免冲突
   - 设置鲁棒性测试参数（噪声范围、缺失数据比例等）

2. **长期预测训练（80%训练数据）**：
   - 使用Accelerate启动分布式训练
   - 设置任务类型为"long_term_forecast"
   - 训练完成后保存检查点

3. **鲁棒性测试（长期预测模型）**：
   - 对训练好的模型进行噪声鲁棒性测试
   - 测试模型对缺失数据的处理能力
   - 生成可视化结果

4. **长期预测训练（20%训练数据）**：
   - 相同配置，但使用更少的训练数据
   - 评估数据稀缺条件下的模型性能

5. **重复上述过程用于短期预测任务**：
   - 使用更短的预测长度(pred_len=6)
   - 分别测试不同训练数据比例下的性能

6. **结果记录和清理**：
   - 保存训练日志和模型检查点
   - 生成综合报告

## 自定义脚本

如需创建自定义实验脚本，可以参考以下模板：

```bash
#!/bin/bash

# 设置输出目录和数据路径
output_dir="./results/your_experiment_name/"
data_path="your_processed_data.csv"
mkdir -p $output_dir

# 设置模型参数
model_name=TimeLLM  # 可选：TimeLLM, Autoformer, DLinear
model_id=YOUR_MODEL_ID
train_epochs=10
learning_rate=0.01
llama_layers=32  # 语言模型层数

# 设置分布式训练参数
master_port=51111  # 注意避免端口冲突
num_process=2
batch_size=32

# 设置模型维度
d_model=16
d_ff=32

# 设置任务参数（根据需要修改）
seq_len=12      # 输入序列长度
label_len=12    # 标签长度
pred_len=24     # 预测长度（长期预测）
# pred_len=6    # 短期预测

# 设置特征数量（根据数据集的特征数量调整）
enc_in=3
dec_in=3
c_out=3

# 执行训练
accelerate launch --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/processed/ \
  --data_path $data_path \
  --model_id $model_id \
  --model $model_name \
  --data beam \
  --features M \
  --target value \
  --seq_len $seq_len \
  --label_len $label_len \
  --pred_len $pred_len \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in $enc_in \
  --dec_in $dec_in \
  --c_out $c_out \
  --batch_size $batch_size \
  --train_rate 0.8 \
  --learning_rate $learning_rate \
  --llm_layers $llama_layers \
  --train_epochs $train_epochs \
  > ${output_dir}your_experiment.log 2>&1
```

## 注意事项

1. **端口占用**：多个脚本同时运行可能导致端口冲突，请确保每个脚本使用不同的`master_port`。

2. **资源消耗**：语言模型需要较大内存，对于资源有限的环境，可以减少`llm_layers`或使用更小的语言模型。

3. **数据路径**：确保数据文件位于正确的位置（通常是`dataset/processed/`目录）。

4. **并行度**：`num_process`参数应根据可用GPU数量设置，单GPU环境设置为1即可。

5. **混合精度训练**：使用`--mixed_precision bf16`可以显著减少内存使用并加速训练，但某些旧GPU可能不支持。