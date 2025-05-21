# -*- coding: utf-8 -*-
"""
数据准备和模型测试脚本（最终修正版）
功能：
1. 加载tokenizer
2. 手动加载法律问答数据集
3. 对数据集进行tokenization
"""

import os
import json
import torch
from pathlib import Path
from pyexpat import model
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig
from peft import LoraConfig, TaskType, get_peft_model
from transformers import TrainingArguments, Trainer


MODEL_PATH = "deepseekr1-1.5b"
DATA_DIR = Path("LegalQA-all_js")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)
    
def tokenize_function(examples):
    texts = [
        f"输入：{input}\n输出：{output}\n类型：{type}\n标签：{label}" 
        for input, output, type, label in zip(
            examples["input"], 
            examples["output"], 
            examples["type"], 
            examples["label"]
        )
    ]
    
    # 使用tokenizer处理文本
    tokenized = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    tokenized["labels"] = tokenized["input_ids"].clone()
    return tokenized


def main():
    global tokenizer
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda")
    # 初始化tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
    
    # 加载数据集
    print("开始加载数据集...")
    dataset = {
        "train": Dataset.from_list(load_json_file(DATA_DIR/"legal_qa_train_formatted.json")),
        "validation": Dataset.from_list(load_json_file(DATA_DIR/"legal_qa_dev_formatted.json")),
        "test": Dataset.from_list(load_json_file(DATA_DIR/"legal_qa_test_formatted.json"))
    }
    
    print("\n-----------数据集加载成功-----------")
    print(f"训练集: {len(dataset['train'])}条")
    print(f"验证集: {len(dataset['validation'])}条")
    print(f"测试集: {len(dataset['test'])}条")
    
    # Tokenization处理
    print("\n开始Tokenization处理(可能需要几分钟)...")
    tokenized_train = dataset["train"].map(tokenize_function,batched=True,batch_size=1000)  # 增加批处理大小提高效率)
    tokenized_valid = dataset["validation"].map(tokenize_function, batched=True, batch_size=1000)
    tokenized_test = dataset["test"].map(tokenize_function, batched=True, batch_size=1000)

    print("\n -----------Tokenization处理完成!-----------")

    # 量化配置
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH,quantization_config=quantization_config, device_map="auto")
    print("-----------量化加载成功-----------")

    # LoRA微调配置
    lora_config = LoraConfig( r=8,lora_alpha=16,lora_dropout=0.05, task_type=TaskType.CAUSAL_LM)

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters() 
    print("-----------LoRA配置成功-----------")

    # 设置训练参数
    training_args = TrainingArguments(
        output_dir="./finetunedmodels/output",
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=16,
        bf16=False,
        fp16=True,  # CUDA上使用fp16
        logging_steps=100,
        save_strategy="epoch",  
        # save_steps=500,
        eval_strategy="epoch",
        # eval_steps=10,
        learning_rate=4e-5,
        logging_dir="./logs",
        run_name="deepseekr1-1.5bFinetune",
        report_to="none",  # 不使用wandb
        dataloader_pin_memory=True,
        dataloader_num_workers=4,  # 使用多线程加载数据
        load_best_model_at_end=True,
        metric_for_best_model="Best Model",  # 替换为你的评估指标
    )
    print("-----------训练参数设置成功-----------")

    # 初始化 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_valid
    )
    print("-----------训练器设置成功，开始训练-----------")

    # 执行训练
    trainer.train()
    print("-----------训练完成-----------")

    # 保存模型和 tokenizer
    model.save_pretrained("./finetunedmodels/deepseekr1-1.5b")
    tokenizer.save_pretrained("./finetunedmodels/deepseekr1-1.5b")
    print("-----------模型保存成功-----------")


    
    
if __name__ == "__main__":
    # 添加异常处理
    try:
        tokenized_data = main()
    except Exception as e:
        print(f"程序执行出错: {str(e)}")


