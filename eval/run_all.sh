#!/bin/bash
# run_all.sh
# 作用：按顺序执行多个 .sh 文件，遇到错误会立即停止。

set -e   # 🚨 遇到任何错误立即退出

echo "===== Start batch run ====="

bash onevision.sh
bash llava.sh
bash intervl3_5.sh
bash qwen.sh
bash monv.sh

echo "===== All jobs finished successfully ====="
#nohup bash run_all.sh run.log 2>&1 &

