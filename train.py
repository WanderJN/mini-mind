from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer, DefaultDataCollator, DataCollatorForTokenClassification, AutoConfig
from dataset.minimind_dataset import LLMDataset
from models.model_mininind import Config, LLM



if __name__ == '__main__':   

    config = Config()
    model = LLM(config)
    print(f'模型参数量为：{sum(p.numel() for p in model.parameters() if p.requires_grad)}')

    data_collator = DefaultDataCollator()
    tokenizer = AutoTokenizer.from_pretrained("./tokenizer", use_fast=True)
    args = TrainingArguments(output_dir='./results/pretrain', 
                            num_train_epochs=10, 
                            do_train=True, 
                            per_device_train_batch_size=256,
                            gradient_accumulation_steps=8,
                            # max_steps=15000,
                            logging_steps=100,
                            report_to='tensorboard',
                            save_total_limit=5,
                            bf16=True,
                            learning_rate=2e-4,
                            lr_scheduler_type='cosine',
                            dataloader_num_workers=8,
                            dataloader_pin_memory=True,
                            save_safetensors=False,
                            deepspeed="./deepspeed_config.json")          
    dataset = LLMDataset('./data/pretrain_hq.jsonl', tokenizer=tokenizer, max_seq_len=512)
    trainer = Trainer(model=model, args=args, train_dataset=dataset, tokenizer=tokenizer, data_collator=data_collator)
    # 如果是初次训练resume_from_checkpoint为false，接着checkpoint继续训练，为True
    trainer.train(resume_from_checkpoint=False)
    trainer.save_model('./saves/pretrain')
    trainer.save_state()