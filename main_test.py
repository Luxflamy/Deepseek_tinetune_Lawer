# -*- coding: utf-8 -*-
"""
纯CPU版本的法律问答模型微调脚本
功能：
1. 完全禁用CUDA/GPU相关功能
2. 优化内存使用
3. 适配CPU训练
"""

import os
import json
from pathlib import Path
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, TaskType, get_peft_model
from transformers import TrainingArguments, Trainer
import torch

# 强制使用CPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 禁用CUDA
torch.set_default_device("cpu")  # 设置默认设备为CPU

MODEL_PATH = "deepseekr1-1.5b"
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
        max_length=256,  # 减小最大长度节省内存
        return_tensors="pt"
    )
    tokenized["labels"] = tokenized["input_ids"].clone()
    return tokenized

def main():
    global tokenizer
    
    print("当前运行设备: CPU")
    
    # 初始化tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
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
    tokenized_train = dataset["train"].map(
        tokenize_function,
        batched=True,
        batch_size=500,  # 减小批处理大小降低内存压力
        remove_columns=dataset["train"].column_names
    )

    tokenized_valid = dataset["validation"].map(
        tokenize_function, 
        batched=True, 
        batch_size=500,
        remove_columns=dataset["validation"].column_names
    )

    print("\n-----------Tokenization处理完成!-----------")

    # 加载模型(纯CPU)
    print("正在加载模型(CPU模式)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float32,  # CPU上使用fp32
        device_map={"": "cpu"}  # 强制使用CPU
    )
    print("-----------模型加载成功-----------")

    # LoRA微调配置
    lora_config = LoraConfig(
        r=4,  # 减小r值降低内存占用
        lora_alpha=8,
        lora_dropout=0.05,
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "v_proj"]
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    print("-----------LoRA配置成功-----------")
    
    # 训练参数配置(CPU优化)
    training_args = TrainingArguments(
        output_dir="./finetunedmodels/output_cpu",
        num_train_epochs=1,  # 减少epoch数
        per_device_train_batch_size=1,  # CPU上使用更小的batch size
        gradient_accumulation_steps=8,
        fp16=False,  # CPU上禁用fp16
        logging_steps=50,
        save_steps=100,
        eval_steps=50,
        eval_strategy="steps",
        learning_rate=3e-5,  # 使用稍低的学习率
        logging_dir="./logs_cpu",
        report_to="none",
        dataloader_pin_memory=False,
        save_total_limit=1,
        load_best_model_at_end=True,
        label_names=["input_ids"],
        no_cuda=True  # 明确禁用CUDA
    )
    print("-----------训练参数设置成功-----------")

    # 创建Trainer实例
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_valid,
        tokenizer=tokenizer
    )

    # 开始训练
    print("-----------开始训练(CPU)-----------")
    trainer.train()
    print("-----------训练完成-----------")

    # 保存模型
    print("正在保存模型...")
    model.save_pretrained("./finetunedmodels/deepseekr1-1.5b-lora-cpu")
    tokenizer.save_pretrained("./finetunedmodels/deepseekr1-1.5b-lora-cpu")
    print("-----------模型保存成功-----------")

    # 评估模型
    print("正在评估模型...")
    eval_results = trainer.evaluate(tokenized_valid)
    print(f"验证集评估结果: {eval_results}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"程序执行出错: {str(e)}")