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
from pathlib import Path
from pyexpat import model
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig
from peft import LoraConfig, TaskType, get_peft_model
from transformers import TrainingArguments, Trainer

MODEL_PATH = "D:\VScode\咸鱼工作\AI/Model/deepseekr1-1.5b"
DATA_DIR = Path("LegalQA-all_js")

def load_json_file(file_path):
    """加载JSON文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def tokenize_function(examples):
    """Tokenization处理函数"""
    # 创建格式化的文本
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
