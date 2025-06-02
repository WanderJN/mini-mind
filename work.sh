
# 创建虚拟环境
# conda config --remove-key channels
# conda create -n mini-deepseek python==3.10 -y
# conda activate mini-deepseek

# 切换工作目录，安库，设置python路径
cd /lpai/volumes/base-copilot-ali-sh/wangjianing1/wjn/minimind
# pip install -r requements.txt
export PYTHONPATH=/lpai/volumes/base-copilot-ali-sh/wangjianing1/wjn/minimind:$PYTHONPATH

# 预训练脚本
# deepspeed trainer/train.py --include 'localhost:0,1,2,3,4,5,6,7' --deepspeed_config=deepspeed_config.json

# 微调脚本
# deepspeed trainer/sft_train.py --include 'localhost:0,1,2,3,4,5,6,7' --deepspeed_config=deepspeed_config.json

# dpo后训练脚本
deepspeed trainer/dpo_train.py --include 'localhost:0,1,2,3,4,5,6,7' --deepspeed_config=deepspeed_config.json

# # CUDA_VISIBLE_DEVICES=0 python trainer/dpo_train.py  # 检查单卡运行程序是否报错