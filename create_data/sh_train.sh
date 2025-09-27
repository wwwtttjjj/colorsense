#!/bin/bash
set -e  # 出错就停止执行

train_num=500

# 要运行的命令列表
commands=(
    "python main.py --number=$train_num --data_type=train_data"
    "python get_all_json.py --data_type=train"
)

# 遍历执行
for cmd in "${commands[@]}"; do
    echo ">>> Running: $cmd ..."
    eval $cmd
    echo ">>> Finished: $cmd"
    echo "-----------------------"
done
