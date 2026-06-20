import os
import sys

import torch
import yaml
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer
import wandb
wandb.login()

CONFIG_PATH = sys.argv[1] if len(sys.argv) > 1 else "train_config.yaml"

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)



MODEL_NAME = cfg["model"]["name"]
TRAIN_PATH = cfg["data"]["train_path"]
TEST_PATH = cfg["data"]["test_path"]
SYSTEM_PROMPT = cfg["data"]["system_prompt"]
LENGTH_WARNING_THRESHOLD = cfg["data"]["max_length_warning_threshold"]
OUTPUT_DIR = cfg["output"]["output_dir"]

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=cfg["model"]["trust_remote_code"])
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

dataset = load_dataset("json", data_files={"train": TRAIN_PATH, "validation": TEST_PATH})


print("Проверка длины токенизированных примеров")

lengths = []
for split in ["train", "validation"]:
    for inst, resp in zip(dataset[split]["instruction"], dataset[split]["response"]):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": inst},
            {"role": "assistant", "content": resp},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        lengths.append(len(token_ids))

lengths_sorted = sorted(lengths)
n = len(lengths_sorted)

print(f"Всего примеров:  {n}")
print(f"Минимум:  {lengths_sorted[0]}")
print(f"Максимум: {lengths_sorted[-1]}")
print(f"Среднее:  {sum(lengths_sorted) / n:.1f}")
print(f"Медиана : {lengths_sorted[n // 2]}")
print(f"95й перцентиль: {lengths_sorted[int(n * 0.95)]}")
print(f"99й перцентиль : {lengths_sorted[int(n * 0.99)]}")

over_threshold = sum(1 for l in lengths if l > LENGTH_WARNING_THRESHOLD)
print(f"Примеров длиннее {LENGTH_WARNING_THRESHOLD}  токенов {over_threshold} ({over_threshold / n * 100:.1f}%)")

bf16_supported = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
compute_dtype = torch.bfloat16 if bf16_supported else torch.float16

bnb_config = BitsAndBytesConfig(
    load_in_4bit=cfg["quantization"]["load_in_4bit"],
    bnb_4bit_quant_type=cfg["quantization"]["bnb_4bit_quant_type"],
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=cfg["quantization"]["bnb_4bit_use_double_quant"],
)

print("Начало загрузки модели в режиме  4-битного квантования.")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",

    trust_remote_code=cfg["model"]["trust_remote_code"],
)

model.config.use_cache = False
model = prepare_model_for_kbit_training(model)

peft_config = LoraConfig(
    r=cfg["lora"]["r"],
    lora_alpha=cfg["lora"]["lora_alpha"],
    target_modules=cfg["lora"]["target_modules"],
    lora_dropout=cfg["lora"]["lora_dropout"],
    bias=cfg["lora"]["bias"],
    task_type=cfg["lora"]["task_type"],
)
model = get_peft_model(model, peft_config)

model.print_trainable_parameters()


def format_prompts(batch):
    formatted_texts = []
    for inst, resp in zip(batch["instruction"], batch["response"]):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": inst},
            {"role": "assistant", "content": resp},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        formatted_texts.append(text)

    return {"text": formatted_texts}


dataset = dataset.map(
    format_prompts,
    batched=True,
    remove_columns=dataset["train"].column_names,
)

os.environ["WANDB_PROJECT"] = cfg["wandb"]["project"]
os.environ["WANDB_LOG_MODEL"] = "true" if cfg["wandb"]["log_model"] else "false"

training_args = SFTConfig(

    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=cfg["training"]["per_device_train_batch_size"],
    gradient_accumulation_steps=cfg["training"]["gradient_accumulation_steps"],
    gradient_checkpointing=cfg["training"]["gradient_checkpointing"],
    eval_strategy=cfg["training"]["eval_strategy"],
    eval_steps=cfg["training"]["eval_steps"],
    logging_steps=cfg["training"]["logging_steps"],
    learning_rate=cfg["training"]["learning_rate"],
    num_train_epochs=cfg["training"]["num_train_epochs"],
    weight_decay=cfg["training"]["weight_decay"],
    fp16=not bf16_supported,
    bf16=bf16_supported,
    save_strategy=cfg["training"]["save_strategy"],
    report_to=cfg["wandb"]["report_to"],
    run_name=cfg["wandb"]["run_name"],
    max_length=cfg["training"]["max_length"],
    dataset_text_field=cfg["training"]["dataset_text_field"],
    packing=cfg["training"]["packing"],
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    processing_class=tokenizer,
    args=training_args,
)

print("Запуск процесса дообучения модели.")
trainer.train()

trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"Процесс обучения завершен. Модифицированные веса сохранены в директорию {OUTPUT_DIR}")

wandb.finish()