# -*- coding: utf-8 -*-
"""
数据准备和模型测试脚本（最终修正版，适配Mac M1/MPS并自动选择设备）
"""

import os
import json
from pathlib import Path
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, TaskType, get_peft_model
from transformers import TrainingArguments, Trainer
import torch

# 设置环境变量
os.environ["LC_ALL"] = "en_US.UTF-8"
os.environ["LANG"] = "en_US.UTF-8"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

MODEL_PATH = "/Users/lixiangyi/Documents/VScode/AI/Model/deepseekr1-1.5b"
DATA_DIR = Path("LegalQA-all_js")

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
    tokenized = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=512
    )
    tokenized["labels"] = tokenized["input_ids"]
    return tokenized

def main():
    global tokenizer

    # 自动选择设备（MPS > CUDA > CPU）
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"当前使用设备: {device}")

    # 初始化tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

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
    tokenized_train = dataset["train"].map(tokenize_function, batched=True, batch_size=1000)
    tokenized_valid = dataset["validation"].map(tokenize_function, batched=True, batch_size=1000)
    tokenized_test = dataset["test"].map(tokenize_function, batched=True, batch_size=1000)

    print("\n-----------Tokenization处理完成!-----------")

    # 加载模型并移动到设备
    print("开始加载模型...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to(device)
    print("-----------模型加载成功-----------")

    # 配置LoRA参数
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        task_type=TaskType.CAUSAL_LM
    )

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
        fp16=False,  # MPS 不支持 fp16
        logging_steps=100,
        save_steps=100,
        eval_steps=10,
        learning_rate=4e-5,
        logging_dir="./logs",
        run_name="deepseekr1-1.5bFinetune",
        report_to="none",  # 禁用wandb等记录器
        dataloader_pin_memory=False  # 必须关闭以兼容MPS
    )

    print("-----------训练参数设置成功-----------")

    # 初始化训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_valid
    )

    print("-----------训练器设置成功,开始训练-----------")

    # 开始训练
    trainer.train()
    print("-----------训练完成-----------")

    # 保存模型和tokenizer
    model.save_pretrained("./finetunedmodels/deepseekr1-1.5b")
    tokenizer.save_pretrained("./finetunedmodels/deepseekr1-1.5b")
    print("-----------模型保存成功-----------")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"程序执行出错: {str(e)}")
