from tokenizers import decoders, models, pre_tokenizers, trainers, Tokenizer
import os 
import json


# 读取数据
def read_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            yield data['text']


# BPE分词器
tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

# 定义特殊tokens
special_tokens = ['<|endoftext|>', '<|im_start|>', '<|im_end|>']


# 初始化训练器
trainer = trainers.BpeTrainer(
    vocab_size=6400,
    special_tokens = special_tokens,
    show_progress=True,
    initial_alphabet = pre_tokenizers.ByteLevel.alphabet()
    
)

# 流式读取数据，texts是iterator
texts = read_data('./dataset/tokenizer_train.jsonl')

tokenizer.train_from_iterator(texts, trainer)
tokenizer.decoder = decoders.ByteLevel()

tokenizer_dir = "./tokenizer"
os.makedirs(tokenizer_dir, exist_ok=True)
tokenizer.save(os.path.join(tokenizer_dir, "tokenizer.json"))

tokenizer.model.save(tokenizer_dir)


config = {
        "add_bos_token": False,
        "add_eos_token": False,
        "add_prefix_space": False,
        "added_tokens_decoder": {
            "0": {
            "content": "<|endoftext|>",
            "lstrip": False,
            "normalized": False,
            "rstrip": False,
            "single_word": False,
            "special": True
            },
            "1": {
            "content": "<|im_start|>",
            "lstrip": False,
            "normalized": False,
            "rstrip": False,
            "single_word": False,
            "special": True
            },
            "2": {
            "content": "<|im_end|>",
            "lstrip": False,
            "normalized": False,
            "rstrip": False,
            "single_word": False,
            "special": True
            }
        },
        "additional_special_tokens": [],
        "bos_token": "<|im_start|>",
        "clean_up_tokenization_spaces": False,
        "eos_token": "<|im_end|>",
        "legacy": True,
        "model_max_length": 32768,
        "pad_token": "<|endoftext|>",
        "sp_model_kwargs": {},
        "spaces_between_special_tokens": False,
        "tokenizer_class": "PreTrainedTokenizerFast",
        "unk_token": "<|endoftext|>",
        "chat_template": "{% if messages[0]['role'] == 'system' %}{% set system_message = messages[0]['content'] %}{{ '<|im_start|>system\\n' + system_message + '<|im_end|>\\n' }}{% else %}{{ '<|im_start|>system\\nYou are a helpful assistant<|im_end|>\\n' }}{% endif %}{% for message in messages %}{% set content = message['content'] %}{% if message['role'] == 'user' %}{{ '<|im_start|>user\\n' + content + '<|im_end|>\\n<|im_start|>assistant\\n' }}{% elif message['role'] == 'assistant' %}{{ content + '<|im_end|>' + '\\n' }}{% endif %}{% endfor %}"
        }

# 保存配置文件
with open(os.path.join(tokenizer_dir, "tokenizer_config.json"), "w", encoding="utf-8") as config_file:
    json.dump(config, config_file, ensure_ascii=False, indent=4)



# 测试
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("./tokenizer")
print(tokenizer.encode("您好"))
print(tokenizer.encode("<|im_start|>"), tokenizer.encode("<|im_end|>"), tokenizer.encode("<|endoftext|>"))

print(tokenizer.decode(588))

print(tokenizer.vocab_size)