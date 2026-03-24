"""
eval/datasets/simpleqa.py

SimpleQA: 4,326 short factual questions with a single unambiguous answer.

Source (official): https://openaipublic.blob.core.windows.net/simple-evals/simple_qa_test_set.csv
HuggingFace mirror: openai/simple-evals (if available)

Each sample:
  {
    "id":       str,
    "question": str,
    "answer":   str,
    "topic":    str   (optional category)
  }
"""
from __future__ import annotations
import csv
import io
import os
import urllib.request
from typing import List, Dict, Any, Optional

_SIMPLEQA_CSV_URL = (
    "https://openaipublic.blob.core.windows.net/simple-evals/simple_qa_test_set.csv"
)

# Small synthetic fallback for offline testing
_SYNTHETIC_POOL: List[Dict[str, Any]] = [
    {"question": "What year was the Eiffel Tower built?", "answer": "1889", "topic": "History"},
    {"question": "What is the chemical symbol for gold?", "answer": "Au", "topic": "Chemistry"},
    {"question": "Who painted the Sistine Chapel ceiling?", "answer": "Michelangelo", "topic": "Art"},
    {"question": "What is the largest organ in the human body?", "answer": "Skin", "topic": "Biology"},
    {"question": "What is the speed of sound in air at room temperature (m/s)?", "answer": "343", "topic": "Physics"},
    {"question": "What is the capital of Canada?", "answer": "Ottawa", "topic": "Geography"},
    {"question": "In what year did the Soviet Union dissolve?", "answer": "1991", "topic": "History"},
    {"question": "What programming language was created by Guido van Rossum?", "answer": "Python", "topic": "Technology"},
    {"question": "What is the smallest country in the world by area?", "answer": "Vatican City", "topic": "Geography"},
    {"question": "How many elements are in the periodic table?", "answer": "118", "topic": "Chemistry"},
    {"question": "What is the currency of South Korea?", "answer": "Won (KRW)", "topic": "Economics"},
    {"question": "Who composed the opera 'The Magic Flute'?", "answer": "Wolfgang Amadeus Mozart", "topic": "Music"},
    {"question": "What is the longest bone in the human body?", "answer": "Femur", "topic": "Biology"},
    {"question": "In what city is the Colosseum located?", "answer": "Rome", "topic": "Geography"},
    {"question": "What is the half-life of Carbon-14?", "answer": "5,730 years", "topic": "Physics"},
    {"question": "Who wrote 'Pride and Prejudice'?", "answer": "Jane Austen", "topic": "Literature"},
    {"question": "What is the freezing point of water in Fahrenheit?", "answer": "32°F", "topic": "Physics"},
    {"question": "What country has the most UNESCO World Heritage Sites?", "answer": "Italy", "topic": "Culture"},
    {"question": "What is the atomic mass of oxygen?", "answer": "16 u", "topic": "Chemistry"},
    {"question": "Who invented the printing press?", "answer": "Johannes Gutenberg", "topic": "History"},
    {"question": "What is the largest continent by area?", "answer": "Asia", "topic": "Geography"},
    {"question": "How many symphonies did Beethoven compose?", "answer": "9", "topic": "Music"},
    {"question": "What is the national language of Brazil?", "answer": "Portuguese", "topic": "Language"},
    {"question": "What planet has the most moons?", "answer": "Saturn", "topic": "Astronomy"},
    {"question": "Who developed the polio vaccine?", "answer": "Jonas Salk", "topic": "Medicine"},
    {"question": "What is the chemical formula for table salt?", "answer": "NaCl", "topic": "Chemistry"},
    {"question": "In what year did Neil Armstrong walk on the moon?", "answer": "1969", "topic": "History"},
    {"question": "What is the tallest tree species in the world?", "answer": "Coast Redwood (Sequoia sempervirens)", "topic": "Biology"},
    {"question": "What is the name of the longest wall in the world?", "answer": "Great Wall of China", "topic": "History"},
    {"question": "What is the rarest blood type?", "answer": "AB negative", "topic": "Medicine"},
]


def load_simpleqa(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Load SimpleQA dataset.

    Tries downloading from OpenAI's public blob storage first.
    Falls back to HuggingFace, then local synthetic pool.

    Returns list of dicts:
      { "id": str, "question": str, "answer": str, "topic": str }
    """
    samples = _load_from_url(limit)
    if not samples:
        samples = _load_from_hf(limit)
    if not samples:
        samples = _load_synthetic(limit)
    return samples


def _load_from_url(limit: Optional[int]) -> List[Dict[str, Any]]:
    """Download the official SimpleQA CSV from OpenAI blob storage."""
    try:
        cache_path = os.path.join(os.path.dirname(__file__), ".cache_simpleqa.csv")
        if not os.path.exists(cache_path):
            urllib.request.urlretrieve(_SIMPLEQA_CSV_URL, cache_path)

        out = []
        with open(cache_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if limit and i >= limit:
                    break
                out.append({
                    "id":       str(i),
                    "question": row.get("problem", row.get("question", "")),
                    "answer":   row.get("answer", ""),
                    "topic":    row.get("topic", ""),
                })
        return out
    except Exception:
        return []


def _load_from_hf(limit: Optional[int]) -> List[Dict[str, Any]]:
    try:
        from datasets import load_dataset  # type: ignore
        ds = load_dataset("openai/simple-evals", split="test")
        out = []
        for i, row in enumerate(ds):
            if limit and i >= limit:
                break
            out.append({
                "id":       str(i),
                "question": row.get("problem", row.get("question", "")),
                "answer":   row.get("answer", ""),
                "topic":    row.get("topic", ""),
            })
        return out
    except Exception:
        return []


def _load_synthetic(limit: Optional[int]) -> List[Dict[str, Any]]:
    pool = _SYNTHETIC_POOL.copy()
    if limit:
        pool = pool[:limit]
    return [
        {"id": str(i), "question": item["question"], "answer": item["answer"], "topic": item.get("topic", "")}
        for i, item in enumerate(pool)
    ]
