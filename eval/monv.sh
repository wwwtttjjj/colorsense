source ~/miniconda3/etc/profile.d/conda.sh
source gpu_env.sh
DatasetTypes=("geo" "chart")
export HF_ENDPOINT=https://huggingface.co
max_workers=2
# ============= Molmo 任务 =============
conda activate NVMO
# MolmoModels=("Molmo-72B-0924")

# for model in "${MolmoModels[@]}"; do
#   for dataset in "${DatasetTypes[@]}"; do
#     log_name="log/Molmo_${model}_${dataset}.log"
#     echo "[INFO] 开始跑 $model on $dataset"
#     python Molmo.py --model_name=$model --max_workers=$max_workers --data_type=$dataset >> $log_name 2>&1
#     echo "[INFO] 完成 $model on $dataset"
#   done
# done

# ============= NVLM 任务 =============
MolmoModels=("NVLM-D-72B")

for model in "${MolmoModels[@]}"; do
  for dataset in "${DatasetTypes[@]}"; do
    log_name="log/NVLM_${model}_${dataset}.log"
    echo "[INFO] 开始跑 $model on $dataset"
    python NVLM.py --model_name=$model --max_workers=$max_workers --data_type=$dataset >> $log_name 2>&1
    echo "[INFO] 完成 $model on $dataset"
  done
done

conda deactivate
echo "[INFO] 全部任务完成 ✅"

# nohup bash monv.sh run.log 2>&1 &
