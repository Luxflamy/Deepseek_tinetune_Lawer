import pandas as pd
import json
import os
from typing import Literal, Optional

# 定义类型
QuestionType = Literal["qingjing", "gainian"]

def process_all_datasets():
    """批量处理所有数据集"""
    datasets = ["train", "dev", "test"]
    for dataset in datasets:
        input_csv = f"LegalQA-all/LegalQA-all-{dataset}.csv"
        output_json = f"legal_qa_{dataset}_formatted.json"
        
        if not os.path.exists(input_csv):
            print(f"⚠️ 文件不存在: {input_csv}")
            continue
            
        try:
            csv_to_json(input_csv, output_json)
            print(f"✅ 成功转换: {input_csv} -> {output_json}")
        except Exception as e:
            print(f"❌ 处理失败 {input_csv}: {str(e)}")

def csv_to_json(input_csv: str, output_json: str):
    """将CSV文件转换为特定格式的JSON"""
    # 读取CSV文件（强制关键字段为字符串类型）
    df = pd.read_csv(input_csv, dtype={
        "question: subject": str,
        "question: body": str,
        "answer": str,
        "label": "Int64"  # 支持整数和NaN
    }).fillna("")
    
    # 检查必要的列是否存在
    required_columns = ["question: subject", "question: body", "answer"]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"CSV文件中缺少必要列: {col}")
    
    # 初始化结果列表
    result = []
    
    for _, row in df.iterrows():
        # 构造每个问答对
        item = {
            "input": f"{row['question: subject']}\n{row['question: body']}".strip(),
            "output": row["answer"].strip(),
            "type": determine_type(row["question: subject"], row["question: body"]),
            # 如果存在label字段则保留（兼容无label的数据）
            **({"label": int(row["label"])} if "label" in df.columns and pd.notna(row["label"]) else {})
        }
        result.append(item)
    
    # 写入JSON文件
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

def determine_type(subject: str, body: str) -> QuestionType:
    """增强版类型判断逻辑"""
    subject = str(subject).lower()
    body = str(body).lower()
    
    # 概念题特征（法律条文/理论）
    concept_keywords = [
        "根据《", "依据《", "刑法第", "民法典", "法学理论",
        "法律原则", "法学原理", "马克思说", "法律规定", "司法解释"
    ]
    
    # 情景题特征（具体问题解决）
    scenario_keywords = [
        "怎么办", "如何处理", "是否构成", "怎样", "如何认定",
        "是否违法", "是否合法", "应当如何", "是否可以", "怎么解决"
    ]
    
    # 优先检查概念题特征
    if any(keyword in body for keyword in concept_keywords):
        return "gainian"
    
    # 检查情景题特征
    if any(keyword in body for keyword in scenario_keywords):
        return "qingjing"
    
    # 根据问题开头判断
    if subject.startswith(("什么是", "为何", "解释", "定义", "论")):
        return "gainian"
    
    # 默认归类为情景题
    return "qingjing"

if __name__ == "__main__":
    print("开始处理法律问答数据集...")
    process_all_datasets()
    print("所有数据处理完成！")