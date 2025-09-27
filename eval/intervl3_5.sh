source ~/miniconda3/etc/profile.d/conda.sh
source gpu_env.sh
DatasetTypes=("geo" "chart")
export HF_ENDPOINT=https://huggingface.co
#8卡可以3个worker
max_workers=3
============= InternVL 任务 =============
conda activate intervl
InternVLModels=("InternVL3_5-38B-Instruct")

for model in "${InternVLModels[@]}"; do
  for dataset in "${DatasetTypes[@]}"; do
    log_name="log/internvl_${model}_${dataset}.log"
    echo "[INFO] 开始跑 $model on $dataset"
    python InternVL3_5.py --model_name=$model --max_workers=$max_workers --data_type=$dataset >> $log_name 2>&1
    echo "[INFO] 完成 $model on $dataset"
  done
done
conda deactivate
echo "[INFO] 全部任务完成 ✅"
#nohup bash intervl3_5.sh run.log 2>&1 &