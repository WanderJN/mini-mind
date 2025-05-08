
conda config --remove-key channels
conda create -n mini-deepseek python==3.10 -y

conda activate mini-deepseek


cd /mnt/pfs-guan-ssai/nlu/wangjianing1/wjn/mini-deepseek
pip install -r requements.txt

deepspeed train.py --include 'localhost:0,1,2,3,4,5,6,7' --deepspeed_config=deepspeed_config.json

deepspeed sft_train.py --include 'localhost:0,1,2,3,4,5,6,7' --deepspeed_config=deepspeed_config.json