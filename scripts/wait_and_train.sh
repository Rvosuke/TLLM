#!/bin/bash
# filepath: /home/baizy25/Time-LLM/wait_and_train.sh

# 要监控的PID
TARGET_PID=$1

# 训练脚本路径
TRAIN_SCRIPT=$2

if [ -z "$TARGET_PID" ] || [ -z "$TRAIN_SCRIPT" ]; then
    echo "用法: $0 <进程PID> <训练脚本路径>"
    echo "例如: $0 12345 ./scripts/TimeLLM_ECL.sh"
    exit 1
fi

echo "开始监控进程 PID: $TARGET_PID"
echo "进程结束后将运行: $TRAIN_SCRIPT"

# 监控进程是否存在
while ps -p $TARGET_PID > /dev/null; do
    echo "进程 $TARGET_PID 仍在运行，等待30秒后再次检查..."
    sleep 30
done

echo "进程 $TARGET_PID 已结束，等待GPU资源释放..."
# 等待一段时间确保GPU资源完全释放
sleep 60

echo "开始训练任务..."
# 运行训练脚本
bash $TRAIN_SCRIPT

echo "训练启动完成！"