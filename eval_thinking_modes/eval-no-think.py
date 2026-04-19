import json
import logging
import re
import asyncio
import aiohttp
from tqdm import tqdm
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

API_KEY = "sk-xxx"
BASE_URL = "https://api.gpt.ge/v1/"
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

#can change model name
MODEL_NAME = "gpt-5.1-2025-11-13"
LABELS = ["entailment", "neutral", "contradiction"]

def extract_label_from_text(generated_text: str) -> str:
    clean_text = re.sub(r"[^a-zA-Z\s]", " ", generated_text).strip().lower()
    if not clean_text:
        return "unknown"

    keyword_to_label = [
        ("entailment", ["entailment", "entails", "entail", "ent"]),
        ("neutral", ["neutral", "neutrality", "neut", "neu"]),
        ("contradiction", ["contradiction", "contradict", "contradictory", "cont"]),
    ]

    words = set(clean_text.split())
    for label, keywords in keyword_to_label:
        for kw in keywords:
            if kw in words:
                return label

    for label, keywords in keyword_to_label:
        for kw in keywords:
            if len(kw) >= 3 and kw in clean_text:
                if re.search(rf"\b{re.escape(kw)}\b", clean_text):
                    return label

    return "unknown"

async def infer_label_api(session, premise: str, hypothesis: str, index: int) -> dict:
    prompt = (
        "Examine the pair of sentences and determine if they exhibit entailment, neutral, or contradiction. "
        "Answer with either 'entailment', 'neutral', or 'contradiction':\n"
        f"Premise: {premise}\nHypothesis: {hypothesis}\nAnswer:"
    )

    payload = {
        "model": MODEL_NAME,
        "temperature": 0,
        "max_tokens": 10000,
        "reasoning_effort": "none",
        "messages": [{"role": "user", "content": prompt}],
    }

    generated_text = ""
    for attempt in range(5):
        try:
            async with session.post(
                f"{BASE_URL}chat/completions",
                json=payload,
                headers=HEADERS,
                timeout=aiohttp.ClientTimeout(total=300)
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    generated_text = data['choices'][0]['message']['content'].strip()
                    break
                elif resp.status == 429:
                    await asyncio.sleep(5)
                else:
                    await asyncio.sleep(1)
        except Exception as e:
            logger.warning(f"[{index}] API error (attempt {attempt+1}): {e}")
            await asyncio.sleep(2)
    else:
        generated_text = ""

    pred_label = extract_label_from_text(generated_text)
    return {
        "index": index,
        "premise": premise,
        "hypothesis": hypothesis,
        "generated_text": generated_text,
        "predicted_label": pred_label
    }

async def compute_accuracy_async(data_file: str, output_file: str, detail_output_file: str, max_limits: int = 20):
    with open(data_file, encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    valid_samples = []
    for i, e in enumerate(data):
        label_idx = e.get("label", -1)
        if isinstance(label_idx, int) and 0 <= label_idx < len(LABELS):
            valid_samples.append((i, e, LABELS[label_idx]))
        else:
            logger.warning(f"Skip invalid label at line {i+1}: {label_idx}")

    logger.info(f"Total valid samples: {len(valid_samples)}")

    correct = 0
    total = 0
    unknown_count = 0

    readable_file = detail_output_file.replace(".jsonl", "_readable.txt")

    async with aiohttp.ClientSession() as session:
        all_tasks = [
            infer_label_api(session, e["premise"], e["hypothesis"], i + 1)
            for (i, e, gold) in valid_samples
        ]

        with open(detail_output_file, "w", encoding='utf-8') as f_detail, \
             open(readable_file, "w", encoding='utf-8') as f_readable:

            for i in tqdm(range(0, len(all_tasks), max_limits), desc=os.path.basename(data_file)):
                batch = all_tasks[i:i + max_limits]
                batch_results = await asyncio.gather(*batch)

                for (orig_idx, _, gold_label), result in zip(valid_samples[i:i + max_limits], batch_results):
                    pred = result["predicted_label"]
                    is_correct = (pred == gold_label)

                    if pred == "unknown":
                        unknown_count += 1
                    else:
                        total += 1
                        if is_correct:
                            correct += 1

                    record = {
                        "index": orig_idx + 1,
                        "premise": result["premise"],
                        "hypothesis": result["hypothesis"],
                        "gold_label": gold_label,
                        "predicted_label": pred,
                        "generated_text": result["generated_text"],
                        "correct": is_correct if pred != "unknown" else None
                    }
                    f_detail.write(json.dumps(record, ensure_ascii=False) + "\n")

                    f_readable.write(f"[Sample {orig_idx + 1}]\n")
                    f_readable.write(f"Premise: {result['premise']}\n")
                    f_readable.write(f"Hypothesis: {result['hypothesis']}\n")
                    f_readable.write(f"Gold: {gold_label} | Predicted: {pred}\n")
                    f_readable.write(f"Raw Response: '{result['generated_text']}'\n")
                    if pred != "unknown":
                        f_readable.write(f"Correct: {'Yes' if is_correct else 'No'}\n")
                    else:
                        f_readable.write("Correct: N/A (Unknown prediction)\n")
                    f_readable.write("-" * 80 + "\n\n")

                f_detail.flush()
                f_readable.flush()

    accuracy = correct / total if total > 0 else 0.0
    summary = (
        f"Total Valid Samples: {total}\n"
        f"Correct Predictions: {correct}\n"
        f"Unknown Predictions: {unknown_count}\n"
        f"Accuracy: {accuracy:.4f} ({correct}/{total})"
    )

    logger.info("=== Final Results ===\n" + summary)
    print("\n" + summary)

    with open(output_file, "w", encoding='utf-8') as f:
        f.write(summary + "\n")

    return accuracy


async def main():
    datasets = [
        ("../data/3bias.jsonl", "xxx-no-think-3bias")
    ]

    for data_path, name in datasets:
        await compute_accuracy_async(
            data_file=data_path,
            output_file=f"{name}_accuracy.txt",
            detail_output_file=f"{name}_predictions.jsonl",
            max_limits=50  
        )

if __name__ == "__main__":
    asyncio.run(main())