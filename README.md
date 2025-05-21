# Deepseek_tinetune_Lawer
This project targets the legal vertical and utilizes the DeepSeek large language model for domain adaptation fine-tuning, aiming to improve the accuracy and reliability of the model in professional tasks such as legal text parsing, case analysis, and contract review.

# Deepseek-R1 Fine-tuning for Chinese Legal QA (适配Mac M1)

本项目基于 [Deepseek-R1 1.5B](https://huggingface.co/deepseek-ai/deepseek-llm-1.5b) 模型，对中文法律问答数据集进行微调，支持在 **Mac M1/M2 芯片（MPS）** 环境下运行，适合作为法律 NLP 项目或大语言模型微调教学参考。

## 📂 项目结构

```
.
├── main.py                    # 主脚本：加载模型 + 数据 + 微调（支持MPS）
├── data_prepare.py           # 原始CSV转JSON格式处理
├── process_legal_data.py     # 数据预处理细节脚本
├── main_test.py / test.py    # 测试模型性能/推理脚本
├── LegalQA-all/              # 原始CSV格式法律问答数据
├── LegalQA-all_js/           # JSON格式处理后数据
├── finetunedmodels/          # 微调后模型存储路径
├── requirement.txt           # Python依赖库
├── LICENSE                   # 项目许可证
└── README.md                 # 项目说明文件
```

## 📦 环境依赖

建议使用 Anaconda 创建虚拟环境：

```bash
conda create -n deepseek-law python=3.10
conda activate deepseek-law
```

安装依赖（建议手动控制库版本避免冲突）：

```bash
pip install -r requirement.txt
```

核心依赖包括：

* `transformers >= 4.38`
* `peft >= 0.10`
* `datasets`
* `torch >= 2.1`（支持 MPS）
* `accelerate`

## ✅ 特性

* 🔍 支持 **中文法律问答数据集**
* 🧠 基于 Deepseek-R1 1.5B 模型的指令微调
* 💻 支持 Apple Silicon 芯片（MPS加速）
* 🧪 支持训练、验证、测试阶段评估
* 🔧 支持 LoRA 参数高效微调（PEFT）

## 🧪 快速开始

### 1. 准备数据

项目已提供处理好的 JSON 格式数据（`LegalQA-all_js/`），如需重新处理：

```bash
python data_prepare.py
```

### 2. 运行主训练脚本

```bash
python main.py
```

支持自动检测设备（MPS > CUDA > CPU）。

### 3. 模型输出

训练完成后模型将保存至：

```
./finetunedmodels/deepseekr1-1.5b/
```

## 📊 数据集说明

| 文件名                              | 样本数量      |
| -------------------------------- | --------- |
| legal\_qa\_train\_formatted.json | 约 100,000 |
| legal\_qa\_dev\_formatted.json   | 约 12,000  |
| legal\_qa\_test\_formatted.json  | 约 26,000  |

每条样本包含字段：

* `input`: 用户问题
* `output`: 法律答案
* `type`: 问题类型（例如：刑法、民法）
* `label`: 标签信息

## 📌 注意事项（适配 M1/MPS）

* 不支持 `fp16`（MPS 限制），需设置为 `fp16=False`
* 需关闭 `dataloader_pin_memory=False`
* 微调速度受限于显存，建议减小 `per_device_train_batch_size`

## 🧹 其他工具

* `packages_to_remove.txt`: 用于排查冲突依赖
* `main_test.py` / `test.py`: 评估模型效果或推理测试脚本

## 📄 License

本项目遵循 [MIT License](./LICENSE)。

