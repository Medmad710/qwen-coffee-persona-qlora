import json
import random
import os

random.seed(666)

input_path = "data/coffee_dataset.jsonl"
train_path = "data/train.jsonl"
val_path = "data/test.jsonl"

dataset = []
with open(input_path, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            dataset.append(json.loads(line.strip()))

random.shuffle(dataset)

split_idx = int(len(dataset) * 0.9)
train_data = dataset[:split_idx]
val_data = dataset[split_idx:]

with open(train_path, "w", encoding="utf-8") as f:
    for item in train_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

with open(val_path, "w", encoding="utf-8") as f:
    for item in val_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"Общий размер датасета: {len(dataset)}")
print(f"Размер выборки для обучения (Train): {len(train_data)}")
print(f"Размер выборки для валидации/теста (Test): {len(val_data)}")