from transformers import TrainingArguments

args = TrainingArguments(
    output_dir="./output",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=1,
    per_device_train_batch_size=1
)

print(args)