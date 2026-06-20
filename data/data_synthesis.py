import os
import json
import re
import time
from google import genai
from google.genai import types
from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm

API_KEY = "YOUR_API_KEY_HERE"
Number_of_batches = 25

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", API_KEY)

SEED_INSTRUCTIONS = [
    "Привет! Какое у вас есть альтернативное молоко?",
    "Как к вам пройти от метро?",
    "У вас можно посидеть с ноутбуком, есть розетки?",
    "Что такое флэт уайт и чем он отличается от капучино?",
    "Порекомендуй что-нибудь бодрящее, но без молока.",
    "Я пролил ваш раф на ноутбук, что мне делать?!",
    "Жду свой лонг блэк уже 15 минут, сколько можно?",
    "Помоги выбрать десерт к горькому эспрессо.",
    "У вас можно заказать кофе на зерне собственной обжарки?",
    "Есть ли у вас скидки, если прийти со своей термокружкой?"
]


def basic_filter(raw_data):
    clean_list = []
    bot_patterns = [
        r"как языковая модель", r"я всего лишь ии", r"не имею физического тела", 
        r"как искусственный интеллект", r"я не пью кофе"
    ]
    
    print("\n2/3 Запуск базовой фильтрации качества")
    for item in raw_data:
        if not isinstance(item, dict) or "instruction" not in item or "response" not in item:
            continue
            
        inst = item["instruction"].strip()
        resp = item["response"].strip()
        
        if len(inst.split()) < 2 or len(resp.split()) < 5:
            continue  
        combined_text = (inst + " " + resp).lower()
        if any(re.search(pattern, combined_text) for pattern in bot_patterns):
            continue
            
        clean_list.append({"instruction": inst, "response": resp})
        
    print(f"> После базовой фильтрации осталось: {len(clean_list)} из {len(raw_data)}")
    return clean_list


def semantic_deduplication(data, threshold=0.85):
    print("\n3/3 Запуск семантической дедупликации ")
    if not data:
        return data

    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    instructions = [item["instruction"] for item in data]
    embeddings = model.encode(instructions, show_progress_bar=True)

    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    similarity_matrix = np.dot(embeddings, embeddings.T)
    
    keep_indices = []
    dropped_count = 0
    
    for i in range(len(data)):
        is_duplicate = False
        for j in keep_indices:
            if similarity_matrix[i, j] > threshold:
                is_duplicate = True
                dropped_count += 1
                break
        if not is_duplicate:
            keep_indices.append(i)
            
    final_data = [data[idx] for idx in keep_indices]
    print(f"> Удалено семантических дубликатов: {dropped_count}")
    print(f"> Финальный размер датасета: {len(final_data)}")
    return final_data


def main():
    if not GEMINI_API_KEY:
        raise ValueError("API-ключ пустой. Пожалуйста, проверьте переменную GEMINI_API_KEY.")

    client = genai.Client(api_key=GEMINI_API_KEY)
    
    system_instruction = (
        "Ты — опытный дата-инженер ИИ. Твоя задача — сгенерировать обучающий датасет для "
        "LLM-виджета кофейни в формате Instruction-Response. Ответы должны строго имитировать стиль "
        "'Дружелюбный Бариста-Эксперт': теплый тон, легкий кофейный сленг (зерно, альтернатива, "
        "кислинка/плотность, крафт), но при этом давать четкий ответ по ситуации. Избегай канцеляризмов "
        "и фраз вроде 'Я текстовый ИИ'. Выдавай строго валидный JSON список объектов."
    )
    
    raw_dataset = []
    
    print("1/3 Запуск пакетной генерации датасета через Gemini 3.1 Flash-Lite")
    
    

    for batch in range(Number_of_batches):
        print(f"Генерация пакета {batch + 1}/25...")
        
        prompt = f"""
        Используя эти базовые примеры в качестве вдохновения: {json.dumps(SEED_INSTRUCTIONS, ensure_ascii=False)}
        
        Сгенерируй ровно 10 уникальных пар "instruction" (разнообразные вопросы, просьбы, жалобы клиентов кофейни) 
        и "response" (ответы бариста). Обязательно охвати темы: меню, зерно, растительное молоко, атмосфера, 
        жалобы на скорость, казусы (пролил кофе). Не повторяй идеи из прошлых шагов.
        
        Выходной формат должен быть СТРОГО JSON-массивом (без разметки ```json ... ```):
        [
          {{"instruction": "текст", "response": "текст"}},
          ...
        ]
        """
        
        try:
            response = client.models.generate_content(
                model='gemini-3.1-flash-lite',
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.8,
                    response_mime_type="application/json" 
                ),
            )
            
            batch_data = json.loads(response.text)
            if isinstance(batch_data, list):
                raw_dataset.extend(batch_data)
            
            time.sleep(5)
            
        except Exception as e:
            print(f"Ошибка на шаге {batch + 1}: {e}")
            time.sleep(5)
            continue
            
    print(f"-> Успешно получено {len(raw_dataset)} сырых примеров от API")

    if not raw_dataset:
        print("Датасет пуст. Завершение работы.")
        return

    filtered_data = basic_filter(raw_dataset)
    final_dataset = semantic_deduplication(filtered_data, threshold=0.88)
    
    output_path = "data/coffee_dataset.jsonl"
    os.makedirs("data", exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        for item in final_dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    print(f"\n Финальный датасет сохранен в '{output_path}'")


if __name__ == "__main__":
    main()