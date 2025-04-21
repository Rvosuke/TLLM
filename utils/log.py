import os
import re
import argparse
import matplotlib.pyplot as plt

# 使用argparse定义命令行参数
parser = argparse.ArgumentParser(description="Process log files and generate plots.")
parser.add_argument("--log_folder", type=str, required=True, help="Path to the folder containing log files.")
args = parser.parse_args()

# 获取日志文件所在的目录
log_folder = args.log_folder

# 遍历目录内所有.log文件
for filename in os.listdir(log_folder):
    if filename.endswith(".log"):
        # 构建日志文件的完整路径
        log_file_path = os.path.join(log_folder, filename)
        
        # 为该日志文件创建一个同名的输出目录（去掉扩展名）
        output_dir = os.path.join(log_folder, os.path.splitext(filename)[0])
        os.makedirs(output_dir, exist_ok=True)
        
        # 读取日志文件
        with open(log_file_path, "r") as file:
            log_content = file.read()

        # 正则表达式提取数据：Epoch、Train Loss、Vali Loss、Test Loss 和 MAE Loss
        pattern = r"Epoch: (\d+) \| Train Loss: ([\d.]+) Vali Loss: ([\d.]+) Test Loss: ([\d.]+) MAE Loss: ([\d.]+)"
        matches = re.findall(pattern, log_content)

        # 将提取的数据存储为字典
        data = {
            "epoch": [],
            "train_loss": [],
            "vali_loss": [],
            "test_loss": [],
            "mae_loss": []
        }

        for match in matches:
            epoch, train_loss, vali_loss, test_loss, mae_loss = match
            data["epoch"].append(int(epoch))
            data["train_loss"].append(float(train_loss))
            data["vali_loss"].append(float(vali_loss))
            data["test_loss"].append(float(test_loss))
            data["mae_loss"].append(float(mae_loss))

        # 绘制单独的曲线图并保存到对应输出目录
        metrics = ["train_loss", "vali_loss", "test_loss", "mae_loss"]
        titles = ["Train Loss", "Validation Loss", "Test Loss", "MAE Loss"]

        for metric, title in zip(metrics, titles):
            plt.figure(figsize=(8, 5))
            plt.plot(data["epoch"], data[metric], label=title, marker="o")
            plt.xlabel("Epoch")
            plt.ylabel(title)
            plt.title(f"{title} over Epochs")
            plt.legend()
            plt.grid(True)
            # 保存输出到对应目录中
            output_file = os.path.join(output_dir, f"{metric}_plot.png")
            plt.savefig(output_file)
            plt.close()
        
        # 最终将.log文件也放置到对应的输出目录中
        log_file_output_path = os.path.join(output_dir, filename)
        os.rename(log_file_path, log_file_output_path)
        print(f"Processed {filename} and saved plots to {output_dir}")
        # 另外在输出目录中创建一个.csv文件，来存储提取的数据
        csv_file_path = os.path.join(output_dir, "metrics.csv")
        with open(csv_file_path, "w") as csv_file:
            csv_file.write("epoch,train_loss,vali_loss,test_loss,mae_loss\n")
            for i in range(len(data["epoch"])):
                csv_file.write(f"{data['epoch'][i]},{data['train_loss'][i]},{data['vali_loss'][i]},{data['test_loss'][i]},{data['mae_loss'][i]}\n")
