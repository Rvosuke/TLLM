#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from accelerate import Accelerator
from models import Autoformer, DLinear, TimeLLM
from data_provider.data_factory import data_provider
from utils.tools import vali
from torch import nn
from scipy import interpolate

def create_missing_data(batch_x, missing_rate=0.1, mask=None):
    """
    在输入序列中创建缺失值
    
    Args:
        batch_x: 输入序列 [batch_size, seq_len, feature_dim]
        missing_rate: 缺失率，范围0~1
        mask: 可选的预定义掩码，如果为None则随机生成
        
    Returns:
        missing_x: 包含缺失值的序列
        mask: 掩码，1表示保留，0表示缺失
    """
    if mask is None:
        # 创建随机掩码，1表示保留，0表示缺失
        mask = torch.FloatTensor(batch_x.shape).uniform_() > missing_rate
        mask = mask.to(batch_x.device)
    
    # 将缺失位置的值设为NaN
    missing_x = batch_x.clone()
    missing_x[~mask] = float('nan')
    
    return missing_x, mask

def interpolate_missing_values(missing_x):
    """
    使用插值法填充缺失值
    
    Args:
        missing_x: 包含缺失值的序列 [batch_size, seq_len, feature_dim]
        
    Returns:
        interpolated_x: 插值后的序列
    """
    batch_size, seq_len, feature_dim = missing_x.shape
    device = missing_x.device
    
    # 将数据移到CPU进行处理
    x_cpu = missing_x.detach().cpu().numpy()
    interpolated_x = np.zeros_like(x_cpu)
    
    # 为每个batch和每个特征进行插值
    for b in range(batch_size):
        for f in range(feature_dim):
            # 获取当前序列
            series = x_cpu[b, :, f]
            
            # 找出非NaN值的索引
            valid_indices = np.where(~np.isnan(series))[0]
            valid_values = series[valid_indices]
            
            if len(valid_indices) <= 1:  # 如果只有0或1个有效值，无法插值
                interpolated_x[b, :, f] = 0.0  # 填充为0
            else:
                if len(valid_indices) < seq_len:  # 如果有缺失值
                    # 使用线性插值填充缺失值
                    interp_func = interpolate.interp1d(
                        valid_indices, valid_values,
                        kind='linear', bounds_error=False, fill_value=(valid_values[0], valid_values[-1])
                    )
                    all_indices = np.arange(seq_len)
                    interpolated_series = interp_func(all_indices)
                    interpolated_x[b, :, f] = interpolated_series
                else:  # 如果没有缺失值
                    interpolated_x[b, :, f] = series
    
    # 将插值后的数据转回tensor并移回原设备
    return torch.FloatTensor(interpolated_x).to(device)

def evaluate_model_with_missing_data(args, model, test_data, test_loader, missing_rates, accelerator):
    """
    评估模型在不同数据缺失率下的性能
    
    Args:
        args: 参数
        model: 模型
        test_data: 测试数据
        test_loader: 测试数据加载器
        missing_rates: 缺失率列表
        accelerator: Accelerator实例
        
    Returns:
        mse_losses: 不同缺失率下的MSE损失
        mae_losses: 不同缺失率下的MAE损失
    """
    criterion = nn.MSELoss()
    mae_metric = nn.L1Loss()
    
    mse_losses = []
    mae_losses = []
    
    for missing_rate in missing_rates:
        accelerator.print(f"测试数据缺失率: {missing_rate}")
        
        # 如果缺失率为0，使用原始测试数据
        if missing_rate == 0:
            test_loss, test_mae_loss = vali(args, accelerator, model, test_data, test_loader, criterion, mae_metric)
        else:
            # 定义自定义的DataLoader迭代器，用于注入缺失值并插值
            class MissingDataLoader:
                def __init__(self, data_loader, missing_rate):
                    self.data_loader = data_loader
                    self.missing_rate = missing_rate
                    self.data = []
                    
                    # 预处理每个batch，添加缺失值并插值
                    for batch_x, batch_y, batch_x_mark, batch_y_mark in data_loader:
                        # 仅在输入序列中添加缺失值
                        missing_x, _ = create_missing_data(batch_x, missing_rate)
                        # 使用插值填充缺失值
                        interpolated_x = interpolate_missing_values(missing_x)
                        # 保存处理后的batch
                        self.data.append((interpolated_x, batch_y, batch_x_mark, batch_y_mark))
                    
                def __iter__(self):
                    return iter(self.data)
                
                def __len__(self):
                    return len(self.data)
            
            # 创建包含缺失值和插值的数据加载器
            missing_data_loader = MissingDataLoader(test_loader, missing_rate)
            
            # 评估模型在缺失值数据上的性能
            test_loss, test_mae_loss = vali(args, accelerator, model, test_data, missing_data_loader, criterion, mae_metric)
        
        mse_losses.append(test_loss)
        mae_losses.append(test_mae_loss)
        accelerator.print(f"缺失率 {missing_rate}，MSE Loss: {test_loss:.6f}, MAE Loss: {test_mae_loss:.6f}")
    
    return mse_losses, mae_losses

def plot_results(missing_rates, mse_losses, mae_losses, output_dir, model_id, task_name, pred_len, interpolation_method="linear"):
    """
    绘制结果图
    
    Args:
        missing_rates: 缺失率列表
        mse_losses: MSE损失列表
        mae_losses: MAE损失列表
        output_dir: 输出目录
        model_id: 模型ID
        task_name: 任务名称
        pred_len: 预测长度
        interpolation_method: 插值方法
    """
    plt.figure(figsize=(12, 5))
    
    # 绘制MSE损失
    plt.subplot(1, 2, 1)
    plt.plot(missing_rates, mse_losses, 'o-', linewidth=2, markersize=8)
    plt.title(f'MSE Loss vs. Missing Rate\n{model_id} - {task_name} - Pred Len: {pred_len}')
    plt.xlabel('Missing Rate')
    plt.ylabel('MSE Loss')
    plt.grid(True)
    
    # 绘制MAE损失
    plt.subplot(1, 2, 2)
    plt.plot(missing_rates, mae_losses, 'o-', linewidth=2, markersize=8, color='orange')
    plt.title(f'MAE Loss vs. Missing Rate\n{model_id} - {task_name} - Pred Len: {pred_len}')
    plt.xlabel('Missing Rate')
    plt.ylabel('MAE Loss')
    plt.grid(True)
    
    plt.tight_layout()
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存图像
    plt.savefig(os.path.join(output_dir, f'missing_data_test_{model_id}_{task_name}_pred{pred_len}_{interpolation_method}.png'), dpi=300)
    
    # 保存结果到CSV
    results_df = pd.DataFrame({
        'missing_rate': missing_rates,
        'mse_loss': mse_losses,
        'mae_loss': mae_losses
    })
    results_df.to_csv(os.path.join(output_dir, f'missing_data_test_{model_id}_{task_name}_pred{pred_len}_{interpolation_method}.csv'), index=False)
    
    plt.close()

def visualize_missing_data_examples(test_loader, missing_rates, output_dir, num_examples=3):
    """
    可视化展示原始数据、缺失数据和插值后的数据
    
    Args:
        test_loader: 测试数据加载器
        missing_rates: 缺失率列表
        output_dir: 输出目录
        num_examples: 示例数量
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取一些样本
    samples = []
    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
        if i < num_examples:
            samples.append((batch_x[0], batch_y[0], batch_x_mark[0], batch_y_mark[0]))
        else:
            break
    
    for sample_idx, (x, y, x_mark, y_mark) in enumerate(samples):
        plt.figure(figsize=(15, 12))
        
        # 遍历不同的缺失率
        for i, missing_rate in enumerate(missing_rates):
            if missing_rate == 0:
                continue
                
            # 创建具有缺失值的数据
            missing_x, mask = create_missing_data(x.unsqueeze(0), missing_rate)
            missing_x = missing_x.squeeze(0)
            mask = mask.squeeze(0)
            
            # 插值恢复缺失值
            interpolated_x = interpolate_missing_values(missing_x.unsqueeze(0)).squeeze(0)
            
            # 绘制每个特征
            for j in range(x.shape[1]):  # 遍历特征维度
                plt.subplot(len(missing_rates), x.shape[1], i * x.shape[1] + j + 1)
                
                # 将数据转移到CPU
                x_np = x[:, j].detach().cpu().numpy()
                missing_x_np = missing_x[:, j].detach().cpu().numpy()
                interpolated_x_np = interpolated_x[:, j].detach().cpu().numpy()
                
                # 绘制原始数据
                plt.plot(np.arange(len(x_np)), x_np, 'b-', label='Original')
                
                # 标记缺失值位置
                missing_indices = np.where(np.isnan(missing_x_np))[0]
                plt.scatter(missing_indices, x_np[missing_indices], color='red', marker='x', s=50, label='Missing' if j == 0 else None)
                
                # 绘制插值后的数据
                plt.plot(np.arange(len(interpolated_x_np)), interpolated_x_np, 'g--', label='Interpolated' if j == 0 else None)
                
                plt.title(f'Missing Rate: {missing_rate}, Feature: {j}')
                if j == 0:
                    plt.legend()
                plt.grid(True)
                
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'missing_data_example_{sample_idx}.png'), dpi=300)
        plt.close()

def main():
    parser = argparse.ArgumentParser(description='Model Missing Data Robustness Test')
    
    # 必需参数
    parser.add_argument('--model', type=str, required=True, help='模型名称，例如：TimeLLM')
    parser.add_argument('--task_name', type=str, required=True, help='任务名称，例如：long_term_forecast')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='模型检查点路径')
    parser.add_argument('--data', type=str, required=True, help='数据集类型')
    parser.add_argument('--model_id', type=str, required=True, help='模型ID')
    parser.add_argument('--output_dir', type=str, required=True, help='输出目录')
    
    # 数据参数
    parser.add_argument('--root_path', type=str, default='./dataset/processed/', help='数据文件根路径')
    parser.add_argument('--data_path', type=str, required=True, help='数据文件')
    parser.add_argument('--features', type=str, default='M', help='特征类型：M-多变量, S-单变量, MS-多变量预测单变量')
    parser.add_argument('--target', type=str, default='value', help='S或MS任务中的目标特征')
    parser.add_argument('--freq', type=str, default='h', help='时间特征编码频率')
    parser.add_argument('--train_rate', type=float, default=0.2, help='训练数据比例')
    
    # 序列参数
    parser.add_argument('--seq_len', type=int, default=12, help='输入序列长度')
    parser.add_argument('--label_len', type=int, default=12, help='起始令牌长度')
    parser.add_argument('--pred_len', type=int, default=24, help='预测序列长度')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    
    # 模型参数
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=16, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=32, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')
    parser.add_argument('--stride', type=int, default=8, help='stride')
    parser.add_argument('--prompt_domain', type=int, default=0, help='')
    parser.add_argument('--llm_model', type=str, default='BERT', help='LLM model') # LLAMA, GPT2, BERT
    parser.add_argument('--llm_dim', type=int, default='768', help='LLM model dimension')# LLama7b:4096; GPT2-small:768; BERT-base:768
    parser.add_argument('--llm_layers', type=int, default=32, help='LLM层数')

    # 缺失数据测试参数
    parser.add_argument('--missing_min', type=float, default=0.0, help='最小缺失率')
    parser.add_argument('--missing_max', type=float, default=0.5, help='最大缺失率')
    parser.add_argument('--missing_steps', type=int, default=6, help='缺失率步数')
    parser.add_argument('--interpolation', type=str, default='linear', help='插值方法（linear, cubic等）')
    parser.add_argument('--visualize_examples', action='store_true', help='是否可视化缺失值示例')

    # 其他参数
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--align_epochs', type=int, default=10, help='alignment epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--eval_batch_size', type=int, default=8, help='batch size of model evaluation')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--pct_start', type=float, default=0.2, help='pct_start')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    parser.add_argument('--percent', type=int, default=100)
    
    args = parser.parse_args()
    
    # 初始化加速器
    accelerator = Accelerator()
    accelerator.print("使用加速器进行推理，不启用DeepSpeed")
    
    # 加载测试数据
    args.is_training = 0  # 设置为测试模式
    test_data, test_loader = data_provider(args, 'test')
    
    # 初始化模型
    if args.model == 'Autoformer':
        model = Autoformer.Model(args).float()
    elif args.model == 'DLinear':
        model = DLinear.Model(args).float()
    else:
        model = TimeLLM.Model(args).float()
    
    # 加载模型权重
    try:
        # 尝试使用map_location直接加载到当前设备
        state_dict = torch.load(args.checkpoint_path, map_location=accelerator.device)
        model.load_state_dict(state_dict)
        accelerator.print(f"成功从 {args.checkpoint_path} 加载模型权重")
    except Exception as e:
        accelerator.print(f"加载模型出错: {str(e)}")
        try:
            # 尝试加载到CPU并转换数据类型
            state_dict = torch.load(args.checkpoint_path, map_location='cpu')
            model.load_state_dict(state_dict)
            accelerator.print(f"成功从 {args.checkpoint_path} 加载模型权重（使用备选方法）")
        except Exception as e2:
            accelerator.print(f"备选加载方法也失败: {str(e2)}")
            accelerator.print("继续使用未初始化的模型...")
    
    # 将数据和模型移到设备并应用加速器
    test_loader, model = accelerator.prepare(test_loader, model)
    
    # 设置模型为评估模式
    model.eval()
    
    # 检查并打印模型参数类型
    for name, param in model.named_parameters():
        if 'weight' in name:
            accelerator.print(f"模型参数 {name} 的数据类型: {param.dtype}")
            break
    
    # 设置缺失率列表
    missing_rates = np.linspace(args.missing_min, args.missing_max, args.missing_steps)
    
    # 如果需要，可视化一些缺失数据示例
    if args.visualize_examples and accelerator.is_local_main_process:
        visualize_missing_data_examples(
            test_loader, 
            [0.1, 0.3, 0.5], 
            os.path.join(args.output_dir, 'missing_data_examples')
        )
    
    # 评估模型在不同缺失率下的性能
    mse_losses, mae_losses = evaluate_model_with_missing_data(
        args, model, test_data, test_loader, missing_rates, accelerator
    )
    
    # 绘制并保存结果
    if accelerator.is_local_main_process:
        plot_results(
            missing_rates, mse_losses, mae_losses, 
            args.output_dir, args.model_id, args.task_name, args.pred_len,
            args.interpolation
        )
        accelerator.print(f"结果已保存到 {args.output_dir}")

if __name__ == '__main__':
    main()