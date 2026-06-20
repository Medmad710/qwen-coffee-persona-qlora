import json
import os
import time

import wandb
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from google import genai
from google.genai import types

INPUT_PATH = "data\\evaluation_results.jsonl"
OUTPUT_REPORT = "metrics_report.json"
PROGRESS_PATH = "metrics_progress.jsonl"
JUDGE_MODEL = "gemini-3.1-flash-lite"
REQUEST_DELAY_SECONDS = 2
MAX_RETRIES = 3
WANDB_PROJECT = "coffee-bot-finetuning"
WANDB_RUN_NAME = "eval-base-vs-tuned"

API_KEY = "YOUR_API_KEY_HERE"

client = genai.Client(api_key=API_KEY)

system_instruction = """
Ты выступаешь в роли строгого эксперта-судьи, оценивающего ответы языковых моделей в роли "Дружелюбный бариста-эксперт в кофейне".
"""

judge_prompt_template = """
Запрос пользователя: "{instruction}"
Эталонный ответ (Ground Truth): "{ground_truth}"

Ответ Модели А (Базовая): "{model_a}"
Ответ Модели Б (Дообученная): "{model_b}"

Оцени ответы по следующим критериям и выставь оценку от 1 до 5 каждому:
1. role_fit: соответствие роли (звучит ли модель как экспертный, но дружелюбный бариста, использует ли кофейный контекст, вежлив ли тон)
2. correctness: корректность информации относительно эталонного ответа

Выведи результат строго в формате JSON со следующими ключами:
- "winner": "Model_A", "Model_B" или "Tie"
- "model_a_scores": {{"role_fit": int, "correctness": int}}
- "model_b_scores": {{"role_fit": int, "correctness": int}}
- "reasoning": краткое сухое объяснение на русском языке, почему выбран этот победитель
"""


def load_jsonl(path):
    items = []
    if not os.path.exists(path):
        return items
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line.strip()))
    return items


def judge_one(item):
    prompt = judge_prompt_template.format(
        instruction=item["instruction"],
        ground_truth=item["response"],
        model_a=item["base_response"],
        model_b=item["tuned_response"],
    )

    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            response = client.models.generate_content(
                model=JUDGE_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.85,
                    response_mime_type="application/json",
                ),
            )
            return json.loads(response.text)
        except Exception as e:
            last_error = e
            time.sleep(REQUEST_DELAY_SECONDS * (attempt + 1))
    raise RuntimeError(f"Не удалось получить оценку судьи: {last_error}")


results = load_jsonl(INPUT_PATH)

run = wandb.init(project=WANDB_PROJECT, name=WANDB_RUN_NAME, job_type="evaluation")

rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)

rouge_table = wandb.Table(columns=[
    "instruction", "model", "rouge1", "rouge2", "rougeL",
])

rouge_aggregates = {"base": {"rouge1": [], "rouge2": [], "rougeL": []},
                     "tuned": {"rouge1": [], "rouge2": [], "rougeL": []}}

for item in results:
    for tag, key in [("base", "base_response"), ("tuned", "tuned_response")]:
        scores = rouge.score(item["response"], item[key])
        rouge_table.add_data(
            item["instruction"], tag,
            scores["rouge1"].fmeasure, scores["rouge2"].fmeasure, scores["rougeL"].fmeasure,
        )
        rouge_aggregates[tag]["rouge1"].append(scores["rouge1"].fmeasure)
        rouge_aggregates[tag]["rouge2"].append(scores["rouge2"].fmeasure)
        rouge_aggregates[tag]["rougeL"].append(scores["rougeL"].fmeasure)

rouge_summary = {}
for tag in ["base", "tuned"]:
    for metric in ["rouge1", "rouge2", "rougeL"]:
        values = rouge_aggregates[tag][metric]
        rouge_summary[f"rouge/{tag}_{metric}_mean"] = sum(values) / len(values)

wandb.log(rouge_summary)
wandb.log({"rouge/per_example": rouge_table})

print("ROUGE рассчитан.")
print(json.dumps(rouge_summary, indent=4))

references = [item["response"] for item in results]
base_candidates = [item["base_response"] for item in results]
tuned_candidates = [item["tuned_response"] for item in results]

base_p, base_r, base_f1 = bert_score(base_candidates, references, lang="ru", verbose=False)
tuned_p, tuned_r, tuned_f1 = bert_score(tuned_candidates, references, lang="ru", verbose=False)

bertscore_table = wandb.Table(columns=[
    "instruction", "model", "precision", "recall", "f1",
])

for i, item in enumerate(results):
    bertscore_table.add_data(item["instruction"], "base", base_p[i].item(), base_r[i].item(), base_f1[i].item())
    bertscore_table.add_data(item["instruction"], "tuned", tuned_p[i].item(), tuned_r[i].item(), tuned_f1[i].item())

bertscore_summary = {
    "bertscore/base_precision_mean": base_p.mean().item(),
    "bertscore/base_recall_mean": base_r.mean().item(),
    "bertscore/base_f1_mean": base_f1.mean().item(),
    "bertscore/tuned_precision_mean": tuned_p.mean().item(),
    "bertscore/tuned_recall_mean": tuned_r.mean().item(),
    "bertscore/tuned_f1_mean": tuned_f1.mean().item(),
}

wandb.log(bertscore_summary)
wandb.log({"bertscore/per_example": bertscore_table})

print("BERTScore рассчитан.")
print(json.dumps(bertscore_summary, indent=4))

already_done = load_jsonl(PROGRESS_PATH)
done_instructions = {item["instruction"] for item in already_done}

win_stats = {"Model_A": 0, "Model_B": 0, "Tie": 0}
for item in already_done:
    win_stats[item["winner"]] += 1

progress_file = open(PROGRESS_PATH, "a", encoding="utf-8")

print("Запуск оценки через Gemini API.")
for idx, item in enumerate(results):
    if item["instruction"] in done_instructions:
        continue

    judge_output = judge_one(item)
    win_stats[judge_output["winner"]] += 1

    record = {
        "instruction": item["instruction"],
        "winner": judge_output["winner"],
        "model_a_scores": judge_output.get("model_a_scores"),
        "model_b_scores": judge_output.get("model_b_scores"),
        "reasoning": judge_output["reasoning"],
    }
    progress_file.write(json.dumps(record, ensure_ascii=False) + "\n")
    progress_file.flush()

    print(f"Оценено {idx + 1}/{len(results)}: {judge_output['winner']}")
    time.sleep(REQUEST_DELAY_SECONDS)

progress_file.close()

evaluated_results = load_jsonl(PROGRESS_PATH)

judge_table = wandb.Table(columns=[
    "instruction", "winner", "model_a_role_fit", "model_a_correctness",
    "model_b_role_fit", "model_b_correctness", "reasoning",
])
for item in evaluated_results:
    a_scores = item.get("model_a_scores") or {}
    b_scores = item.get("model_b_scores") or {}
    judge_table.add_data(
        item["instruction"], item["winner"],
        a_scores.get("role_fit"), a_scores.get("correctness"),
        b_scores.get("role_fit"), b_scores.get("correctness"),
        item["reasoning"],
    )

total_judged = sum(win_stats.values())
judge_summary = {
    "judge/model_a_win_rate": win_stats["Model_A"] / total_judged if total_judged else 0,
    "judge/model_b_win_rate": win_stats["Model_B"] / total_judged if total_judged else 0,
    "judge/tie_rate": win_stats["Tie"] / total_judged if total_judged else 0,
}

wandb.log(judge_summary)
wandb.log({"judge/per_example": judge_table})
wandb.log({"judge/win_stats": win_stats})

report = {
    "rouge": rouge_summary,
    "bertscore": bertscore_summary,
    "judge_stats": win_stats,
    "judge_details": evaluated_results,
}

with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=4)

print("\n=== Итоговая статистика сравнения (LLM-as-a-judge) ===")
print(json.dumps(win_stats, ensure_ascii=False, indent=4))

wandb.finish()