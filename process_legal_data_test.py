import json
from pathlib import Path

input_dir = Path("LegalQA-all_js")
output_dir = Path("LegalQA-all_js_test")
output_dir.mkdir(exist_ok=True)

# 文件名和对应截取条数
files_limits = {
    "legal_qa_train_formatted.json": 1000,
    "legal_qa_dev_formatted.json": 100,
    "legal_qa_test_formatted.json": 100,
}

for filename, limit in files_limits.items():
    input_path = input_dir / filename
    output_path = output_dir / filename
    
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    subset = data[:limit]
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(subset, f, ensure_ascii=False, indent=2)
    
    print(f"{filename} 已处理，保存前{limit}条到 {output_path}")

print("全部文件处理完成！")
