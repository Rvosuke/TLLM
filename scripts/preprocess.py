import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

def process_file(input_path, output_path):
    df = pd.read_csv(input_path)

    # 日期格式转换
    df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y %H:%M').dt.strftime('%Y/%m/%d %H:%M')

    # 替换 '#N/A' 为 NaN，再填充为0
    df.replace('#N/A', np.nan, inplace=True)
    df.fillna(0, inplace=True)

    df.to_csv(output_path, index=False)
    print(f"数据已处理并保存到 {output_path}")

def main():
    input_folder = "dataset/raw"
    output_folder = "dataset/processed"
    os.makedirs(output_folder, exist_ok=True)

    input_files = [f for f in os.listdir(input_folder) if f.startswith("24h_date") and f.endswith(".csv")]
    for input_file in input_files:
        input_path = os.path.join(input_folder, input_file)
        output_file = input_file.replace("24h_date", "processed")
        output_path = os.path.join(output_folder, output_file)
        try:
            process_file(input_path, output_path)
        except Exception as e:
            print(f"处理文件 {input_file} 时出错: {e}")

if __name__ == "__main__":
    main()
