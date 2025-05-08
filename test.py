from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch
from train import LLM, Config


# 加载分词器
tokenizer = AutoTokenizer.from_pretrained('./saves/sft_1024')

# 向AutoConfig注册自定义的Config对象，这样可以如下加载：config = AutoConfig.from_pretrained('./path_to_config', model_type="small_model")
AutoConfig.register("small_model", Config)


# 向AutoModelForCausalLM 中注册一个自定义的模型类 MyModel，并告诉库它与配置类 Config 配套使用
AutoModelForCausalLM.register(Config, LLM)
# model = AutoModelForCausalLM.from_pretrained('./saves/pretrain')
model = AutoModelForCausalLM.from_pretrained('./saves/sft_1024')

# 测试样例
# input_data = [tokenizer.bos_token_id] + tokenizer.encode('1+1等于')

input_data = tokenizer.apply_chat_template([{'role':'user', 'content':'1+1等于几，请详细回答'}])

print(input_data)

for token in model.generate({"input_ids":torch.tensor(input_data).unsqueeze(0), "labels":None}, tokenizer.eos_token_id, max_new_tokens=50, stream=False, temperature=0.0, top_k=8):
    print(tokenizer.decode(token[0]))

