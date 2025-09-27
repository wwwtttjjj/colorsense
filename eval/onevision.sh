source ~/miniconda3/etc/profile.d/conda.sh
source gpu_env.sh
DatasetTypes=("geo" "chart")
export HF_ENDPOINT=https://huggingface.co
max_workers=20

conda activate Qwen2_5
# ============= OneVision 任务 =============
OneVisionModels=("LLaVA-One-Vision-1.5-8B-Instruct")

for model in "${OneVisionModels[@]}"; do
  for dataset in "${DatasetTypes[@]}"; do
    log_name="log/Onevision_${model}_${dataset}.log"
    echo "[INFO] 开始跑 $model on $dataset"
    python OneVision.py --model_name=$model --max_workers=$max_workers --data_type=$dataset >> $log_name 2>&1
    echo "[INFO] 完成 $model on $dataset"
  done
done
conda deactivate
echo "[INFO] 全部任务完成 ✅"
#nohup bash onevision.sh run.log 2>&1 &