# 初始化 conda
source ~/miniconda3/etc/profile.d/conda.sh
source gpu_env.sh
DatasetTypes=("geo" "chart")
export HF_ENDPOINT=https://huggingface.co
max_workers=16
# 定义模型和数据集
QwenModels=("Qwen2.5-VL-7B-Instruct" "Qwen2.5-VL-3B-Instruct" "GC-Qwen2_5_3B-model" "GC-Qwen2_5_7B-model")

============= Qwen 任务 =============
conda activate llamafactory

for model in "${QwenModels[@]}"; do
  for dataset in "${DatasetTypes[@]}"; do
    log_name="log/qwen_${model}_${dataset}.log"
    echo "[INFO] 开始跑 $model on $dataset"
    python Qwen2_5.py --model_name=$model --max_workers=$max_workers --data_type=$dataset >> $log_name 2>&1
    echo "[INFO] 完成 $model on $dataset"
  done
done
conda deactivate
echo "[INFO] Qwen2.5 小任务完成 ✅"

conda activate llamafactory

QwenModels_big=("Qwen2.5-VL-72B-Instruct")
for model in "${QwenModels[@]}"; do
  for dataset in "${DatasetTypes[@]}"; do
    log_name="log/qwen_${model}_${dataset}.log"
    echo "[INFO] 开始跑 $model on $dataset"
    python Qwen2_5.py --model_name=$model --max_workers=2 --data_type=$dataset >> $log_name 2>&1
    echo "[INFO] 完成 $model on $dataset"
  done
done
echo "[INFO] Qwen2.5 小任务完成 ✅"
conda deactivate
