source ~/miniconda3/etc/profile.d/conda.sh
source gpu_env.sh
DatasetTypes=("geo" "chart")
export HF_ENDPOINT=https://huggingface.co
max_workers=6

============= Llava 任务 =============
conda activate llava
# 需要运行的模型名称列表
LlavaModels=("llava-v1.6-34b")
# 循环执行任务
for model in "${LlavaModels[@]}"; do
  for dataset in "${DatasetTypes[@]}"; do
    log_name="log/llava_${model}_${dataset}.log"
    echo "[INFO] 开始跑 $model on $dataset"
    # 根据你的实际python文件名修改，比如 llava_infer.py 或 main.py
    python Llava.py --model_name=$model --max_workers=$max_workers --data_type=$dataset >> $log_name 2>&1
    echo "[INFO] 完成 $model on $dataset"
  done
done
conda deactivate
# nohup bash llava.sh run.log 2>&1 &
