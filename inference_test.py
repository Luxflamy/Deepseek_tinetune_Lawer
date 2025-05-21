# -*- coding: utf-8 -*-
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline
from datasets import Dataset
import json

# 加载模型
final_save_path = "./finetunedmodels/deepseekr1-1.5b_full"
model = AutoModelForCausalLM.from_pretrained(final_save_path)
tokenizer = AutoTokenizer.from_pretrained(final_save_path)

# pipline
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
prompt = "根据《中华人民共和国刑法》，非法经营罪是指..."
generated_text = pipe(
    prompt, 
    max_new_tokens=1000, 
    truncation=True, 
    num_return_sequences=1,
    do_sample=True,  # 启用采样
    temperature=0.7,  # 控制随机性
    top_p=0.9,        # 核采样
    pad_token_id=tokenizer.eos_token_id  # 明确设置pad token
)

print("回答：",generated_text[0]["generated_text"])


