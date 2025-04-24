## 结果文件说明

`results/` 目录下的各文件夹对应不同数据集上的实验结果。例如，`number_value_cell_2_beam_19/` 表示在名为 *number_value_cell_2_beam_19* 的数据集上进行的测试。

每个数据集文件夹内进一步划分为不同任务设置的子目录，例如：  
`long_term_forecast_train02/` 表示该任务为长期预测（输入序列长度为12，输出序列长度为24），训练集占比为0.2（少样本训练）。短期预测任务中，输出序列长度为6。

### 子目录文件结构说明

在每个具体任务目录下，包含以下类型的文件：

- `long_term_forecast_train02.log`  
  记录模型训练过程的日志信息。日志内容已解析汇总于 `metrics.csv`，并配套绘制了以下可视化图像：  
  - `train_loss_plot.png`: 训练损失曲线（横轴：epoch，纵轴：平均训练损失）  
  - `test_loss_plot.png`: 测试集均方误差（MSE）曲线（横轴：epoch，纵轴：MSE）  
  - `mae_loss_plot.png`: 测试集平均绝对误差（MAE）曲线（横轴：epoch，纵轴：MAE）

- `metrics.csv`  
  按 epoch 记录训练损失、测试集上的 MSE 和 MAE 指标。

- `test.log`  
  完整训练结束后，基于测试集最低 MSE 所选出的最佳权重进行最终测试时的终端输出记录。

- `test_results_{任务名称}.csv` 与 `test_summary.txt`  
  最终测试结果的详细记录及摘要说明。

- `prediction_example_{i}.png`  
  测试阶段的样例预测图（共三幅）。图中：蓝色曲线为输入序列，绿色为真实输出序列，红色虚线为模型预测结果。可根据需要选择其中一幅作为展示样例。

### 鲁棒性测试文件说明

- `noisy_test.log`  
  加噪测试的终端输出记录。

- `robustness_test_{任务名称}.csv`  
  加入不同水平高斯噪声后（噪声水平以方差控制）的测试结果。

- `robustness_test_{任务名称}.png`  
  噪声水平变化下，测试指标随之变化的曲线图。

### 缺失数据测试文件说明

- `missing_data_test.log`  
  缺失值测试的终端输出记录。

- `missing_data_test_{任务名称}.csv`  
  在不同缺失比例下，模型测试结果的记录（缺失部分通过线性插值补全）。

- `missing_data_test_{任务名称}.png`  
  基于上述测试结果绘制的缺失率与指标变化关系图。
