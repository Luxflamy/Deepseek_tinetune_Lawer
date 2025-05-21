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
generated_text = pipe(prompt, max_length=1000, num_return_sequences=1)

print("回答：",generated_text[0]["generated_text"])


