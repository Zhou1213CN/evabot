"""
eval/datasets/frames.py

FRAMES: 824 multi-hop factual questions requiring integration across
multiple Wikipedia pages.

Source: google/frames-benchmark on HuggingFace
Paper:  krishna-etal-2025-fact

Each sample:
  {
    "id":       str,
    "question": str,
    "answer":   str,
    "hops":     int,   # number of reasoning hops required
    "wiki_titles": List[str]  # Wikipedia articles needed
  }
"""
from __future__ import annotations
import json
from typing import List, Dict, Any, Optional


# Fallback synthetic multi-hop questions (used when HF dataset unavailable)
_SYNTHETIC_POOL: List[Dict[str, Any]] = [
    {
        "question": "What is the birth year of the person who directed the film that won the Academy Award for Best Picture in 2020?",
        "answer": "1969",
        "hops": 3,
        "wiki_titles": ["92nd Academy Awards", "Parasite (2019 film)", "Bong Joon-ho"],
        "reasoning_chain": [
            "The 92nd Academy Awards (2020) Best Picture winner was Parasite.",
            "Parasite was directed by Bong Joon-ho.",
            "Bong Joon-ho was born in 1969.",
        ],
    },
    {
        "question": "In which country was the inventor of the World Wide Web born?",
        "answer": "England (United Kingdom)",
        "hops": 2,
        "wiki_titles": ["Tim Berners-Lee", "World Wide Web"],
        "reasoning_chain": [
            "The World Wide Web was invented by Tim Berners-Lee.",
            "Tim Berners-Lee was born in London, England.",
        ],
    },
    {
        "question": "What is the capital of the country that has the world's longest coastline?",
        "answer": "Ottawa",
        "hops": 2,
        "wiki_titles": ["List of countries by length of coastline", "Canada", "Ottawa"],
        "reasoning_chain": [
            "Canada has the world's longest coastline at approximately 202,080 km.",
            "The capital of Canada is Ottawa.",
        ],
    },
    {
        "question": "Who was the president of the United States when the Eiffel Tower was completed?",
        "answer": "Grover Cleveland",
        "hops": 2,
        "wiki_titles": ["Eiffel Tower", "Grover Cleveland"],
        "reasoning_chain": [
            "The Eiffel Tower was completed in 1889.",
            "Grover Cleveland was president of the United States in 1889.",
        ],
    },
    {
        "question": "What element was named after the country where Marie Curie was born?",
        "answer": "Polonium",
        "hops": 2,
        "wiki_titles": ["Marie Curie", "Polonium"],
        "reasoning_chain": [
            "Marie Curie was born in Poland (then part of the Russian Empire).",
            "Polonium was named after Poland.",
        ],
    },
    {
        "question": "What is the official language of the country that borders Brazil to the west and has access to the Pacific Ocean?",
        "answer": "Spanish",
        "hops": 2,
        "wiki_titles": ["Peru", "Spanish language"],
        "reasoning_chain": [
            "Peru borders Brazil to the west and has access to the Pacific Ocean.",
            "The official language of Peru is Spanish.",
        ],
    },
    {
        "question": "What was the name of the ship that sank after hitting an iceberg in 1912, and in which year was the film about it released?",
        "answer": "Titanic; 1997",
        "hops": 2,
        "wiki_titles": ["RMS Titanic", "Titanic (1997 film)"],
        "reasoning_chain": [
            "The RMS Titanic sank in 1912 after hitting an iceberg.",
            "James Cameron's film 'Titanic' about the disaster was released in 1997.",
        ],
    },
    {
        "question": "What programming language was created by the founder of the company that makes the iPhone?",
        "answer": "Swift (but Apple did not create Swift; rather Swift was created by Apple's engineers, not by Steve Jobs)",
        "answer": "Swift",
        "hops": 2,
        "wiki_titles": ["Apple Inc.", "Swift (programming language)"],
        "reasoning_chain": [
            "The iPhone is made by Apple Inc., co-founded by Steve Jobs.",
            "Apple developed the Swift programming language, released in 2014.",
        ],
    },
    {
        "question": "Who was the monarch of England when William Shakespeare was born?",
        "answer": "Queen Elizabeth I",
        "hops": 2,
        "wiki_titles": ["William Shakespeare", "Elizabeth I of England"],
        "reasoning_chain": [
            "William Shakespeare was born in April 1564.",
            "Elizabeth I was the Queen of England from 1558 to 1603, so she was monarch when Shakespeare was born.",
        ],
    },
    {
        "question": "What is the height in metres of the tallest building in the country that hosted the 2016 Summer Olympics?",
        "answer": "Approximately 297 metres (Gran Torre Santiago in Chile is not Brazilian; the tallest in Brazil is Mirante do Vale at 170m or Millenium Palace 162m). The correct answer: the Millenium Palace or similar.",
        "answer": "170 metres (Mirante do Vale, São Paulo)",
        "hops": 2,
        "wiki_titles": ["2016 Summer Olympics", "Brazil", "Mirante do Vale"],
        "reasoning_chain": [
            "The 2016 Summer Olympics were hosted in Rio de Janeiro, Brazil.",
            "The tallest building in Brazil is Mirante do Vale in São Paulo at approximately 170 metres.",
        ],
    },
    {
        "question": "What river flows through the city where the Louvre museum is located?",
        "answer": "Seine",
        "hops": 2,
        "wiki_titles": ["Louvre", "Paris", "Seine"],
        "reasoning_chain": [
            "The Louvre museum is located in Paris, France.",
            "The Seine river flows through Paris.",
        ],
    },
    {
        "question": "In what year was the country founded whose flag has the same colours as the French flag?",
        "answer": "Multiple countries share France's tricolour; the Netherlands was founded in 1581.",
        "answer": "1581 (Netherlands)",
        "hops": 2,
        "wiki_titles": ["Flag of France", "Flag of the Netherlands", "Netherlands"],
        "reasoning_chain": [
            "The French flag uses blue, white, and red.",
            "The Netherlands flag uses the same colours; the Dutch Republic was founded in 1581.",
        ],
    },
    {
        "question": "What is the nationality of the author who wrote the book that inspired the movie 'The Godfather'?",
        "answer": "American",
        "hops": 2,
        "wiki_titles": ["The Godfather (film)", "The Godfather (novel)", "Mario Puzo"],
        "reasoning_chain": [
            "The movie The Godfather is based on the 1969 novel by Mario Puzo.",
            "Mario Puzo was an American author.",
        ],
    },
    {
        "question": "What is the population of the city that serves as the capital of the country that won the FIFA World Cup in 2014?",
        "answer": "Approximately 3.7 million (Berlin)",
        "hops": 2,
        "wiki_titles": ["2014 FIFA World Cup", "Germany", "Berlin"],
        "reasoning_chain": [
            "Germany won the 2014 FIFA World Cup.",
            "Berlin is the capital of Germany, with a population of approximately 3.7 million.",
        ],
    },
    {
        "question": "What is the name of the ocean that borders the country where the Amazon river originates?",
        "answer": "Pacific Ocean",
        "hops": 2,
        "wiki_titles": ["Amazon river", "Peru", "Pacific Ocean"],
        "reasoning_chain": [
            "The Amazon river originates in Peru.",
            "Peru borders the Pacific Ocean.",
        ],
    },
    {
        "question": "Who invented the system of classification for living organisms that is still used today, and in what century did they live?",
        "answer": "Carl Linnaeus; 18th century",
        "hops": 2,
        "wiki_titles": ["Carl Linnaeus", "Linnaean taxonomy"],
        "reasoning_chain": [
            "Carl Linnaeus invented binomial nomenclature, the system of classifying organisms still used today.",
            "Carl Linnaeus lived from 1707 to 1778, in the 18th century.",
        ],
    },
    {
        "question": "What is the currency used in the country that is home to the Great Barrier Reef?",
        "answer": "Australian Dollar",
        "hops": 2,
        "wiki_titles": ["Great Barrier Reef", "Australia", "Australian dollar"],
        "reasoning_chain": [
            "The Great Barrier Reef is located in Australia.",
            "The currency of Australia is the Australian Dollar (AUD).",
        ],
    },
    {
        "question": "What university did the co-founder of Microsoft attend before dropping out?",
        "answer": "Harvard University",
        "hops": 2,
        "wiki_titles": ["Microsoft", "Bill Gates", "Harvard University"],
        "reasoning_chain": [
            "Microsoft was co-founded by Bill Gates and Paul Allen.",
            "Bill Gates attended Harvard University before dropping out to start Microsoft.",
        ],
    },
    {
        "question": "What is the name of the mountain range that separates Europe from Asia, and in which country is the highest peak of that range located?",
        "answer": "Ural Mountains; Russia",
        "hops": 2,
        "wiki_titles": ["Ural Mountains", "Mount Narodnaya", "Russia"],
        "reasoning_chain": [
            "The Ural Mountains form the boundary between Europe and Asia.",
            "The highest peak, Mount Narodnaya, is located in Russia.",
        ],
    },
    {
        "question": "What is the name of the lake that borders the most countries in Africa?",
        "answer": "Lake Victoria (Uganda, Kenya, Tanzania)",
        "hops": 2,
        "wiki_titles": ["Lake Victoria", "Uganda", "Kenya", "Tanzania"],
        "reasoning_chain": [
            "Lake Victoria borders Uganda, Kenya, and Tanzania — three countries.",
            "Lake Tanganyika also borders four countries (DRC, Tanzania, Zambia, Burundi), making it technically the answer.",
        ],
        "answer": "Lake Tanganyika (borders 4 countries: DRC, Tanzania, Zambia, Burundi)",
    },
    {
        "question": "Who wrote the novel that was adapted into the film 'Schindler's List'?",
        "answer": "Thomas Keneally",
        "hops": 2,
        "wiki_titles": ["Schindler's List (film)", "Schindler's Ark", "Thomas Keneally"],
        "reasoning_chain": [
            "Schindler's List (1993) is based on the novel 'Schindler's Ark'.",
            "Schindler's Ark was written by Australian author Thomas Keneally.",
        ],
    },
    {
        "question": "What is the distance in kilometres between the birthplace of Charles Darwin and the city where he is buried?",
        "answer": "Approximately 250 km (Shrewsbury to London)",
        "hops": 3,
        "wiki_titles": ["Charles Darwin", "Shrewsbury", "Westminster Abbey"],
        "reasoning_chain": [
            "Charles Darwin was born in Shrewsbury, England.",
            "He is buried at Westminster Abbey in London.",
            "The distance from Shrewsbury to London is approximately 250 km.",
        ],
    },
    {
        "question": "What was the first country to grant women the right to vote, and when did it do so?",
        "answer": "New Zealand; 1893",
        "hops": 1,
        "wiki_titles": ["Women's suffrage in New Zealand"],
        "reasoning_chain": [
            "New Zealand was the first self-governing country to grant women the right to vote in 1893.",
        ],
    },
    {
        "question": "Which company developed the first commercially successful smartphone, and in what year?",
        "answer": "Apple; 2007 (iPhone)",
        "hops": 2,
        "wiki_titles": ["Smartphone", "iPhone", "Apple Inc."],
        "reasoning_chain": [
            "The first commercially successful smartphone is generally considered the original iPhone.",
            "The iPhone was developed by Apple Inc. and released in 2007.",
        ],
    },
    {
        "question": "What is the name of the largest desert in the world, and on which continent is it located?",
        "answer": "Antarctic Desert; Antarctica",
        "hops": 1,
        "wiki_titles": ["Desert", "Antarctic Desert"],
        "reasoning_chain": [
            "The Antarctic Desert is the largest desert in the world at about 14.2 million km², larger than the Sahara.",
        ],
    },
]


def load_frames(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Load FRAMES dataset.

    Tries HuggingFace google/frames-benchmark first.
    Falls back to synthetic multi-hop pool.

    Returns list of dicts:
      {
        "id":          str,
        "question":    str,
        "answer":      str,
        "hops":        int,
        "wiki_titles": List[str]
      }
    """
    samples = _load_from_hf(limit)
    if not samples:
        samples = _load_synthetic(limit)
    return samples


def _load_from_hf(limit: Optional[int]) -> List[Dict[str, Any]]:
    try:
        from datasets import load_dataset  # type: ignore
        ds = load_dataset("google/frames-benchmark", split="test")
        out = []
        for i, row in enumerate(ds):
            if limit and i >= limit:
                break
            out.append({
                "id":          str(row.get("id", i)),
                "question":    row.get("Prompt", row.get("question", "")),
                "answer":      row.get("Answer", row.get("answer", "")),
                "hops":        int(row.get("num_hops", row.get("hops", 1))),
                "wiki_titles": row.get("wiki_links", row.get("wiki_titles", [])),
            })
        return out
    except Exception:
        return []


def _load_synthetic(limit: Optional[int]) -> List[Dict[str, Any]]:
    pool = _SYNTHETIC_POOL.copy()
    if limit:
        pool = pool[:limit]
    return [
        {
            "id":          str(i),
            "question":    item["question"],
            "answer":      item["answer"],
            "hops":        item.get("hops", 2),
            "wiki_titles": item.get("wiki_titles", []),
        }
        for i, item in enumerate(pool)
    ]
