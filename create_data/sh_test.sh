#!/bin/bash
set -e  # 出错就停止执行

test_num=50

# 要运行的命令列表
commands=(
    "python main.py --number=$test_num --data_type=test_data"
    "python get_all_json.py --data_type=test"
)

# 遍历执行
for cmd in "${commands[@]}"; do
    echo ">>> Running: $cmd ..."
    eval $cmd
    echo ">>> Finished: $cmd"
    echo "-----------------------"
done
