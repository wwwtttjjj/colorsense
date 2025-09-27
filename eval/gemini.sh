# ============= Gemini 任务 =============
source ~/miniconda3/etc/profile.d/conda.sh
conda activate Qwen2_5

GeminiModels=("gemini-2.0-flash" "gemini-2.5-flash")
DatasetTypes=("geo" "chart")

for model in "${GeminiModels[@]}"; do
  for dataset in "${DatasetTypes[@]}"; do
    log_name="log/Gemini_${model}_${dataset}.log"
    echo "[INFO] 开始跑 $model on $dataset"
    python Gemini.py --model_name=$model --max_workers=5 --data_type=$dataset >> $log_name 2>&1
    echo "[INFO] 完成 $model on $dataset"
  done
done
conda deactivate
echo "[INFO] 全部任务完成 ✅"
#nohup bash gemini.sh run.log 2>&1 &