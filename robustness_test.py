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

def add_noise_to_data(data_loader, noise_level=0.0, accelerator=None):
    """
    添加高斯噪声到数据集，确保噪声类型与模型参数类型一致
    Args:
        data_loader: 数据加载器
        noise_level: 噪声水平（标准差）
        accelerator: Accelerator实例，用于确保数据类型一致
    Returns:
        添加了噪声的数据加载器
    """
    noisy_data_loader = []
    device = accelerator.device if accelerator else torch.device("cpu")
    
    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(data_loader):
        # 确保batch_x已经移动到正确的设备上
        if not batch_x.is_cuda and device.type == 'cuda':
            batch_x = batch_x.to(device)
            
        # 只对输入序列添加噪声（batch_x），保持目标序列不变
        if noise_level > 0:
            # 创建与batch_x相同数据类型的噪声
            dtype = batch_x.dtype
            noise = torch.randn_like(batch_x, dtype=torch.float32) * noise_level
            
            # 确保噪声的数据类型与batch_x一致
            noise = noise.to(dtype=dtype, device=device)
            
            # 添加噪声
            batch_x = batch_x + noise
        
        noisy_data_loader.append((batch_x, batch_y, batch_x_mark, batch_y_mark))
    
    return noisy_data_loader

def evaluate_model_with_noise(args, model, test_data, test_loader, noise_levels, accelerator):
    """
    在不同噪声水平下评估模型性能
    Args:
        args: 参数
        model: 模型
        test_data: 测试数据
        test_loader: 测试数据加载器
        noise_levels: 噪声水平列表
        accelerator: Accelerator实例
    Returns:
        mse_losses: 不同噪声水平下的MSE损失
        mae_losses: 不同噪声水平下的MAE损失
    """
    criterion = nn.MSELoss()
    mae_metric = nn.L1Loss()
    
    mse_losses = []
    mae_losses = []
    
    for noise_level in noise_levels:
        accelerator.print(f"测试噪声水平: {noise_level}")
        
        # 如果噪声水平为0，使用原始测试数据
        if noise_level == 0:
            test_loss, test_mae_loss = vali(args, accelerator, model, test_data, test_loader, criterion, mae_metric)
        else:
            # 添加噪声到测试数据
            noisy_test_loader = add_noise_to_data(test_loader, noise_level, accelerator)
            
            # 定义自定义的DataLoader迭代器
            class NoiseDataLoader:
                def __init__(self, data):
                    self.data = data
                    
                def __iter__(self):
                    return iter(self.data)
                
                def __len__(self):
                    return len(self.data)
            
            noisy_loader = NoiseDataLoader(noisy_test_loader)
            test_loss, test_mae_loss = vali(args, accelerator, model, test_data, noisy_loader, criterion, mae_metric)
        
        mse_losses.append(test_loss)
        mae_losses.append(test_mae_loss)
        accelerator.print(f"噪声水平 {noise_level}，MSE Loss: {test_loss:.6f}, MAE Loss: {test_mae_loss:.6f}")
    
    return mse_losses, mae_losses

def plot_results(noise_levels, mse_losses, mae_losses, output_dir, model_id, task_name, pred_len):
    """
    绘制结果图
    Args:
        noise_levels: 噪声水平列表
        mse_losses: MSE损失列表
        mae_losses: MAE损失列表
        output_dir: 输出目录
        model_id: 模型ID
        task_name: 任务名称
        pred_len: 预测长度
    """
    plt.figure(figsize=(12, 5))
    
    # 绘制MSE损失
    plt.subplot(1, 2, 1)
    plt.plot(noise_levels, mse_losses, 'o-', linewidth=2, markersize=8)
    plt.title(f'MSE Loss vs. Noise Level\n{model_id} - {task_name} - Pred Len: {pred_len}')
    plt.xlabel('Noise Level (σ)')
    plt.ylabel('MSE Loss')
    plt.grid(True)
    
    # 绘制MAE损失
    plt.subplot(1, 2, 2)
    plt.plot(noise_levels, mae_losses, 'o-', linewidth=2, markersize=8, color='orange')
    plt.title(f'MAE Loss vs. Noise Level\n{model_id} - {task_name} - Pred Len: {pred_len}')
    plt.xlabel('Noise Level (σ)')
    plt.ylabel('MAE Loss')
    plt.grid(True)
    
    plt.tight_layout()
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存图像
    plt.savefig(os.path.join(output_dir, f'robustness_test_{model_id}_{task_name}_pred{pred_len}.png'), dpi=300)
    
    # 保存结果到CSV
    results_df = pd.DataFrame({
        'noise_level': noise_levels,
        'mse_loss': mse_losses,
        'mae_loss': mae_losses
    })
    results_df.to_csv(os.path.join(output_dir, f'robustness_test_{model_id}_{task_name}_pred{pred_len}.csv'), index=False)
    
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Model Robustness Test')
    
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

    
    # 噪声测试参数
    parser.add_argument('--noise_min', type=float, default=0.0, help='最小噪声水平')
    parser.add_argument('--noise_max', type=float, default=0.1, help='最大噪声水平')
    parser.add_argument('--noise_steps', type=int, default=11, help='噪声水平步数')

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
    
    # 初始化加速器，不使用DeepSpeed，因为推理不需要优化器
    # 使用简单的加速器设置，避免DeepSpeed错误
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
            
            # 打印模型权重和当前模型的数据类型进行调试
            for param_name, param in state_dict.items():
                if 'weight' in param_name:
                    accelerator.print(f"权重 {param_name} 的数据类型: {param.dtype}")
                    break
            
            for param_name, param in model.named_parameters():
                if 'weight' in param_name:
                    accelerator.print(f"当前模型 {param_name} 的数据类型: {param.dtype}")
                    break
            
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
    
    # 设置噪声水平
    noise_levels = np.linspace(args.noise_min, args.noise_max, args.noise_steps)
    
    # 评估模型在不同噪声水平下的性能
    mse_losses, mae_losses = evaluate_model_with_noise(args, model, test_data, test_loader, noise_levels, accelerator)
    
    # 绘制并保存结果
    if accelerator.is_local_main_process:
        plot_results(noise_levels, mse_losses, mae_losses, args.output_dir, args.model_id, args.task_name, args.pred_len)
        accelerator.print(f"结果已保存到 {args.output_dir}")

if __name__ == '__main__':
    main()