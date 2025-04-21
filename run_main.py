import argparse
import torch
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate import DistributedDataParallelKwargs
from torch import nn, optim
from torch.optim import lr_scheduler
from tqdm import tqdm

from models import Autoformer, DLinear, TimeLLM

from data_provider.data_factory import data_provider
import time
import random
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd

os.environ['CURL_CA_BUNDLE'] = ''
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

from utils.tools import del_files, EarlyStopping, adjust_learning_rate, vali, load_content

parser = argparse.ArgumentParser(description='Time-LLM')

fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

# basic config
parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
                    help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
parser.add_argument('--model_comment', type=str, default='none', help='prefix when saving test results')
parser.add_argument('--model', type=str, required=True, default='Autoformer',
                    help='model name, options: [Autoformer, DLinear]')
parser.add_argument('--seed', type=int, default=2021, help='random seed')

# data loader
parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
parser.add_argument('--root_path', type=str, default='./dataset', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; '
                         'M:multivariate predict multivariate, S: univariate predict univariate, '
                         'MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--loader', type=str, default='modal', help='dataset type')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, '
                         'options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], '
                         'you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
parser.add_argument('--train_rate', type=float, default=0.8, help='train data rate')

# forecasting task
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
parser.add_argument('--label_len', type=int, default=48, help='start token length')
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')

# model define
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


# optimization
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
parser.add_argument('--llm_layers', type=int, default=6)
parser.add_argument('--percent', type=int, default=100)
parser.add_argument('--use_mixed_precision', type=bool, default=True)

# 用于测试模式的参数
parser.add_argument('--checkpoint_path', type=str, default=None, help='模型检查点路径')
parser.add_argument('--output_dir', type=str, default='./results', help='输出目录')
parser.add_argument('--num_examples', type=int, default=3, help='可视化的预测样例数量')
parser.add_argument('--visualize_predictions', action='store_true', help='是否可视化预测结果', default=True)

args = parser.parse_args()
# Check if we should use CPU or GPU with accelerate
if torch.cuda.is_available():
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    deepspeed_plugin = DeepSpeedPlugin(hf_ds_config='./ds_config_zero2.json')
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], deepspeed_plugin=deepspeed_plugin)
    print("Using GPU with DeepSpeed for training")
else:
    # CPU configuration - simpler setup without DeepSpeed
    accelerator = Accelerator()
    print("Using CPU for training")

# 设置数据加载器
train_data, train_loader = data_provider(args, 'train')
test_data, test_loader = data_provider(args, 'test')

# 初始化模型
if args.model == 'Autoformer':
    model = Autoformer.Model(args).float()
elif args.model == 'DLinear':
    model = DLinear.Model(args).float()
else:
    model = TimeLLM.Model(args).float()

# 可视化预测结果函数
def visualize_predictions(args, model, test_data, test_loader, accelerator, output_dir, num_examples=3):
    """
    可视化模型预测结果
    
    Args:
        args: 参数
        model: 模型
        test_data: 测试数据
        test_loader: 测试数据加载器
        accelerator: Accelerator实例
        output_dir: 输出目录
        num_examples: 可视化样例数量
    """
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    
    criterion = nn.MSELoss()
    total_loss = 0
    visualized = 0
    
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            if visualized >= num_examples:
                break
                
            # 模型输入
            batch_x = batch_x.float().to(accelerator.device)
            batch_y = batch_y.float().to(accelerator.device)
            batch_x_mark = batch_x_mark.float().to(accelerator.device)
            batch_y_mark = batch_y_mark.float().to(accelerator.device)
            
            # 准备解码器输入
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float().to(accelerator.device)
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(accelerator.device)
            
            # 模型预测
            if args.output_attention:
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
            else:
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            
            f_dim = -1 if args.features == 'MS' else 0
            outputs = outputs[:, -args.pred_len:, f_dim:]
            batch_y = batch_y[:, -args.pred_len:, f_dim:].to(accelerator.device)
            
            # 计算损失
            pred = outputs.detach()
            true = batch_y.detach()
            loss = criterion(pred, true)
            total_loss += loss.item()
            
            # 可视化样例
            for j in range(min(1, pred.shape[0])):  # 每个batch只取第一个样例
                if visualized >= num_examples:
                    break
                    
                # 创建绘图
                plt.figure(figsize=(15, 8))
                
                # 获取完整序列（输入 + 预测）
                full_x = torch.cat([batch_x[j], batch_y[j]], dim=0).cpu().numpy()
                input_len = batch_x[j].shape[0]
                
                # 绘制每个特征
                for k in range(pred.shape[2]):
                    plt.subplot(pred.shape[2], 1, k+1)
                    
                    # 绘制输入序列
                    x_indices = np.arange(input_len)
                    plt.plot(x_indices, full_x[:input_len, k], 'b-', label='Input')
                    
                    # 绘制真实值
                    y_indices = np.arange(input_len, input_len + args.pred_len)
                    plt.plot(y_indices, true[j, :, k].cpu().numpy(), 'g-', label='True')
                    
                    # 绘制预测值
                    plt.plot(y_indices, pred[j, :, k].cpu().numpy(), 'r--', label='Prediction')
                    
                    # 添加分隔线表示预测开始的位置
                    plt.axvline(x=input_len-1, color='gray', linestyle='--')
                    
                    # 添加标题和图例
                    plt.title(f'Feature {k+1}')
                    plt.legend()
                    
                    if k == 0:
                        plt.title(f'{args.model} - 序列{visualized+1} - Feature {k+1}')
                    if k == pred.shape[2] - 1:
                        plt.xlabel('Time Steps')
                
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'prediction_example_{visualized+1}.png'), dpi=300)
                plt.close()
                
                visualized += 1
    
    # 保存并返回均值损失
    avg_loss = total_loss / (i + 1)
    accelerator.print(f"平均测试损失: {avg_loss:.6f}")
    
    # 保存测试摘要
    summary_file = os.path.join(output_dir, 'test_summary.txt')
    with open(summary_file, 'w') as f:
        f.write(f"Model: {args.model}\n")
        f.write(f"Task: {args.task_name}\n")
        f.write(f"Prediction Length: {args.pred_len}\n")
        f.write(f"Average Test Loss: {avg_loss:.6f}\n")
    
    return avg_loss

if args.is_training:
    # 训练模式
    path = os.path.join(args.checkpoints)  # unique checkpoint saving path
    args.content = load_content(args)
    if not os.path.exists(path) and accelerator.is_local_main_process:
        os.makedirs(path)

    time_now = time.time()

    train_steps = len(train_loader)
    early_stopping = EarlyStopping(accelerator=accelerator, patience=args.patience)

    trained_parameters = []
    for p in model.parameters():
        if p.requires_grad is True:
            trained_parameters.append(p)

    model_optim = optim.Adam(trained_parameters, lr=args.learning_rate)

    if args.lradj == 'COS':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=20, eta_min=1e-8)
    else:
        scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                            steps_per_epoch=train_steps,
                                            pct_start=args.pct_start,
                                            epochs=args.train_epochs,
                                            max_lr=args.learning_rate)

    criterion = nn.MSELoss()
    mae_metric = nn.L1Loss()

    train_loader, test_loader, model, model_optim, scheduler = accelerator.prepare(
        train_loader, test_loader, model, model_optim, scheduler)

    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()

    for epoch in range(args.train_epochs):
        iter_count = 0
        train_loss = []

        model.train()
        epoch_time = time.time()
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(train_loader)):
            iter_count += 1
            model_optim.zero_grad()

            batch_x = batch_x.float().to(accelerator.device)
            batch_y = batch_y.float().to(accelerator.device)
            batch_x_mark = batch_x_mark.float().to(accelerator.device)
            batch_y_mark = batch_y_mark.float().to(accelerator.device)

            # decoder input
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float().to(
                accelerator.device)
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(
                accelerator.device)

            # encoder - decoder
            if args.use_amp:
                with torch.cuda.amp.autocast():
                    if args.output_attention:
                        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if args.features == 'MS' else 0
                    outputs = outputs[:, -args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -args.pred_len:, f_dim:].to(accelerator.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())
            else:
                if args.output_attention:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if args.features == 'MS' else 0
                outputs = outputs[:, -args.pred_len:, f_dim:]
                batch_y = batch_y[:, -args.pred_len:, f_dim:]
                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())

            if (i + 1) % 100 == 0:
                accelerator.print(
                    f"\titers: {i + 1}, epoch: {epoch + 1} | loss: {loss.item():.7f}")
                speed = (time.time() - time_now) / iter_count
                left_time = speed * ((args.train_epochs - epoch) * train_steps - i)
                accelerator.print(f"\tspeed: {speed:.4f}s/iter; left time: {left_time:.4f}s")
                iter_count = 0
                time_now = time.time()

            if args.use_amp:
                scaler.scale(loss).backward()
                scaler.step(model_optim)
                scaler.update()
            else:
                accelerator.backward(loss)
                model_optim.step()

            if args.lradj == 'TST':
                adjust_learning_rate(accelerator, model_optim, scheduler, epoch + 1, args, printout=False)
                scheduler.step()

        accelerator.print(f"Epoch: {epoch + 1} cost time: {time.time() - epoch_time}")
        train_loss = np.average(train_loss)
        test_loss, test_mae_loss = vali(args, accelerator, model, test_data, test_loader, criterion, mae_metric)
        accelerator.print(
            f"Epoch: {epoch + 1} | Train Loss: {train_loss:.7f} Test Loss: {test_loss:.7f} MAE Loss: {test_mae_loss:.7f}")

        early_stopping(test_loss, model, path)
        if early_stopping.early_stop:
            accelerator.print("Early stopping")
            break

        if args.lradj != 'TST':
            if args.lradj == 'COS':
                scheduler.step()
                accelerator.print(f"lr = {model_optim.param_groups[0]['lr']:.10f}")
            else:
                if epoch == 0:
                    args.learning_rate = model_optim.param_groups[0]['lr']
                    accelerator.print(f"lr = {model_optim.param_groups[0]['lr']:.10f}")
                adjust_learning_rate(accelerator, model_optim, scheduler, epoch + 1, args, printout=True)
        else:
            accelerator.print(f"Updating learning rate to {scheduler.get_last_lr()[0]}")

else:
    # 测试模式：加载预训练模型并进行预测
    accelerator.print("进入测试模式，加载预训练模型...")
    
    # 确保输出目录存在
    if args.output_dir is None:
        args.output_dir = f"./results/{args.model_id}_{args.task_name}_{args.pred_len}"
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 准备测试数据加载器
    test_loader = accelerator.prepare(test_loader)
    
    # 加载模型权重
    try:
        # 尝试直接加载到当前设备
        state_dict = torch.load(args.checkpoint_path, map_location=accelerator.device)
        model.load_state_dict(state_dict)
        accelerator.print(f"成功从 {args.checkpoint_path} 加载模型权重")
    except Exception as e:
        accelerator.print(f"加载模型出错: {str(e)}")
        try:
            # 尝试加载到CPU并转换数据类型
            state_dict = torch.load(args.checkpoint_path, map_location='cpu')
            
            # 调试信息
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
    
    # 将模型移动到设备并准备加速
    model = accelerator.prepare(model)
    
    # 设置模型为评估模式
    model.eval()
    
    # 检查并打印模型参数类型
    for name, param in model.named_parameters():
        if 'weight' in name:
            accelerator.print(f"模型参数 {name} 的数据类型: {param.dtype}")
            break
    
    # 设置评估指标
    criterion = nn.MSELoss()
    mae_metric = nn.L1Loss()
    
    # 评估模型性能
    test_loss, test_mae_loss = vali(args, accelerator, model, test_data, test_loader, criterion, mae_metric)
    accelerator.print(f"测试结果 - MSE Loss: {test_loss:.7f}, MAE Loss: {test_mae_loss:.7f}")
    
    # 可视化一些预测结果
    if args.visualize_predictions:
        visualize_predictions(
            args, model, test_data, test_loader, 
            accelerator, args.output_dir, args.num_examples
        )
        accelerator.print(f"预测可视化结果已保存到 {args.output_dir}")
    
    # 保存测试结果摘要
    results_file = os.path.join(args.output_dir, f'test_results_{args.model_id}_{args.task_name}_pred{args.pred_len}.csv')
    results_df = pd.DataFrame({
        'model': [args.model],
        'model_id': [args.model_id],
        'task': [args.task_name],
        'pred_len': [args.pred_len],
        'mse_loss': [test_loss],
        'mae_loss': [test_mae_loss]
    })
    results_df.to_csv(results_file, index=False)
    accelerator.print(f"测试结果已保存到 {results_file}")

accelerator.wait_for_everyone()
if accelerator.is_local_main_process:
    path = './checkpoints'  # unique checkpoint saving path
    # del_files(path)  # delete checkpoint files
    accelerator.print('success delete checkpoints')