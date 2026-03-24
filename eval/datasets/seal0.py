"""
eval/datasets/seal0.py

SEAL-0: 111 questions testing reasoning over noisy and conflicting web evidence.

Dataset source: https://huggingface.co/datasets/vectara/SEAL-0
Fallback: If unavailable, generate synthetic conflict-reasoning questions.

Each sample:
  {
    "id":       str,
    "question": str,
    "answer":   str,           # ground-truth answer
    "sources":  List[str],     # web snippets (some misleading)
    "misleading_indices": List[int]  # indices of planted noise sources
  }
"""
from __future__ import annotations
import json
import random
from typing import List, Dict, Any, Optional

_SYNTHETIC_POOL: List[Dict[str, Any]] = [
    # Each entry: question, correct_answer, supporting snippets, noise snippets
    {
        "question": "Who invented the telephone?",
        "answer": "Alexander Graham Bell",
        "real_sources": [
            "Alexander Graham Bell was awarded the first patent for the telephone in 1876.",
            "Bell's patent US174465 for 'the method of, and apparatus for, transmitting vocal or other sounds telegraphically' is widely regarded as the birth of the telephone.",
        ],
        "noise_sources": [
            "Some historians argue that Elisha Gray filed a patent caveat for a telephone design hours before Bell on the same day in 1876.",
            "Antonio Meucci claimed to have invented the telephone years before Bell and filed a caveat in 1871, but lacked funds to renew it.",
        ],
    },
    {
        "question": "What is the capital of Australia?",
        "answer": "Canberra",
        "real_sources": [
            "Canberra has been the capital of Australia since 1913, chosen as a compromise between rivals Sydney and Melbourne.",
        ],
        "noise_sources": [
            "Many people mistakenly believe Sydney is the capital of Australia because it is the largest city and hosted the 2000 Olympics.",
            "Melbourne served as the temporary seat of government until Canberra was ready in 1927.",
        ],
    },
    {
        "question": "How many bones are in the adult human body?",
        "answer": "206",
        "real_sources": [
            "An adult human body has 206 bones, down from roughly 270-300 at birth as many fuse during childhood.",
        ],
        "noise_sources": [
            "Some sources claim the adult skeleton has 212 bones, counting certain sesamoid bones not included in the standard count.",
            "Children have approximately 300 bones; a common misconception is that adults retain this number.",
        ],
    },
    {
        "question": "What year did World War II end?",
        "answer": "1945",
        "real_sources": [
            "World War II ended in 1945: Germany surrendered on May 8 (V-E Day) and Japan surrendered on September 2 (V-J Day).",
        ],
        "noise_sources": [
            "Some sources date the formal end of WWII to 1951 when the Treaty of San Francisco came into force.",
            "The war in the Pacific theater technically ended with Japan's formal surrender ceremony on September 2, 1945, though some guerrilla fighters continued for years.",
        ],
    },
    {
        "question": "Who wrote 'Romeo and Juliet'?",
        "answer": "William Shakespeare",
        "real_sources": [
            "Romeo and Juliet is a tragedy written by William Shakespeare early in his career, believed to have been written between 1594 and 1596.",
        ],
        "noise_sources": [
            "The story of Romeo and Juliet was based on an earlier tale by Matteo Bandello, leading some to credit him as the original author.",
            "Arthur Brooke's 1562 poem 'The Tragicall Historye of Romeus and Juliet' is the main source Shakespeare adapted.",
        ],
    },
    {
        "question": "What is the speed of light in a vacuum?",
        "answer": "299,792,458 metres per second",
        "real_sources": [
            "The speed of light in a vacuum is exactly 299,792,458 m/s (approximately 3×10⁸ m/s), a defined constant since 1983.",
        ],
        "noise_sources": [
            "Light travels at approximately 186,000 miles per second, which some incorrectly round to 200,000 miles per second.",
            "In certain media like water or glass, light travels significantly slower than its vacuum speed.",
        ],
    },
    {
        "question": "What element has the chemical symbol 'Au'?",
        "answer": "Gold",
        "real_sources": [
            "The chemical symbol Au for gold comes from the Latin word 'aurum'. Gold has atomic number 79.",
        ],
        "noise_sources": [
            "Silver's symbol is Ag (from Argentum), which is sometimes confused with Au.",
            "Some websites incorrectly list Au as the symbol for Aurum, a fictional element from science fiction.",
        ],
    },
    {
        "question": "Who painted the Mona Lisa?",
        "answer": "Leonardo da Vinci",
        "real_sources": [
            "The Mona Lisa was painted by Italian Renaissance artist Leonardo da Vinci, likely between 1503 and 1519.",
        ],
        "noise_sources": [
            "Raphael also painted several portraits of women in the same era, leading to occasional misattribution.",
            "Michelangelo is sometimes incorrectly credited in popular culture due to his towering reputation during the Renaissance.",
        ],
    },
    {
        "question": "What is the largest ocean on Earth?",
        "answer": "Pacific Ocean",
        "real_sources": [
            "The Pacific Ocean is the largest and deepest ocean on Earth, covering more than 165 million square kilometres.",
        ],
        "noise_sources": [
            "The Atlantic Ocean is often mistaken as the largest because it is the most heavily trafficked and appears largest on many historical maps.",
            "Some geographic databases from the early 20th century incorrectly list the Atlantic as the world's largest ocean.",
        ],
    },
    {
        "question": "In what year did the Berlin Wall fall?",
        "answer": "1989",
        "real_sources": [
            "The Berlin Wall fell on November 9, 1989, after the East German government announced citizens could cross freely.",
        ],
        "noise_sources": [
            "Physical demolition of the wall continued through 1990, causing some sources to list 1990 as the year the wall 'ended'.",
            "German reunification was formally completed on October 3, 1990, sometimes confused with the wall's fall.",
        ],
    },
    {
        "question": "What is the smallest planet in our solar system?",
        "answer": "Mercury",
        "real_sources": [
            "Mercury is the smallest planet in the solar system with a diameter of about 4,879 km, following Pluto's reclassification as a dwarf planet in 2006.",
        ],
        "noise_sources": [
            "Before 2006, Pluto was considered the smallest planet; many older textbooks still list it as such.",
            "Mars is sometimes confused as smallest due to its small apparent size in telescopes.",
        ],
    },
    {
        "question": "Who developed the theory of general relativity?",
        "answer": "Albert Einstein",
        "real_sources": [
            "Albert Einstein published his general theory of relativity in 1915, describing gravity as the curvature of spacetime.",
        ],
        "noise_sources": [
            "Isaac Newton's law of universal gravitation preceded Einstein and is still used in many engineering applications today.",
            "Some websites credit Henri Poincaré with key contributions that allegedly preceded Einstein's work, a claim disputed by most historians.",
        ],
    },
    {
        "question": "How many continents are there on Earth?",
        "answer": "7",
        "real_sources": [
            "The most widely taught model, used in English-speaking countries, recognises 7 continents: Africa, Antarctica, Asia, Australia/Oceania, Europe, North America, and South America.",
        ],
        "noise_sources": [
            "In many Spanish- and Portuguese-speaking countries, a 6-continent model combining the Americas as 'America' is taught.",
            "Some geographers argue for a 5-continent model, excluding Antarctica as uninhabited.",
        ],
    },
    {
        "question": "What is the chemical formula for water?",
        "answer": "H2O",
        "real_sources": [
            "Water has the chemical formula H₂O, consisting of two hydrogen atoms bonded to one oxygen atom.",
        ],
        "noise_sources": [
            "Heavy water is D₂O (deuterium oxide), sometimes confused with regular water H₂O in chemistry contexts.",
            "Hydrogen peroxide (H₂O₂) is occasionally and incorrectly referred to as 'water' in informal contexts.",
        ],
    },
    {
        "question": "Who was the first person to walk on the Moon?",
        "answer": "Neil Armstrong",
        "real_sources": [
            "Neil Armstrong became the first human to walk on the Moon on July 20, 1969, during the Apollo 11 mission.",
        ],
        "noise_sources": [
            "Buzz Aldrin was the second person to walk on the Moon during Apollo 11, sometimes mistakenly identified as first.",
            "Some conspiracy websites claim the moon landing was staged, citing supposed photographic anomalies.",
        ],
    },
    {
        "question": "What is the powerhouse of the cell?",
        "answer": "Mitochondria",
        "real_sources": [
            "The mitochondrion is often called the 'powerhouse of the cell' because it generates most of the cell's supply of ATP through oxidative phosphorylation.",
        ],
        "noise_sources": [
            "The nucleus is sometimes mistakenly called the powerhouse because it controls cell functions.",
            "Ribosomes produce proteins and are sometimes confused with energy production organelles.",
        ],
    },
    {
        "question": "What language has the most native speakers in the world?",
        "answer": "Mandarin Chinese",
        "real_sources": [
            "Mandarin Chinese has approximately 920 million native speakers, making it the language with the most native speakers globally.",
        ],
        "noise_sources": [
            "English is the most widely spoken language in total speakers (including second-language speakers), often confused with native speaker counts.",
            "Spanish is the second most widely spoken native language and is sometimes confused as the leader.",
        ],
    },
    {
        "question": "What is the hardest natural substance on Earth?",
        "answer": "Diamond",
        "real_sources": [
            "Diamond is the hardest natural material on Earth, rating 10 on the Mohs hardness scale.",
        ],
        "noise_sources": [
            "Lonsdaleite (hexagonal diamond) is theoretically harder than diamond but has never been confirmed in bulk natural form.",
            "Corundum (sapphire/ruby) at 9 on the Mohs scale is the second hardest and sometimes mistakenly cited as the hardest.",
        ],
    },
    {
        "question": "What is the currency of Japan?",
        "answer": "Yen",
        "real_sources": [
            "The official currency of Japan is the Japanese yen (¥), introduced in 1871 as the country's monetary unit.",
        ],
        "noise_sources": [
            "Some outdated travel guides list the 'Mon' as Japan's currency, a historical unit abolished in the 19th century.",
            "The yuan is China's currency, frequently confused with the Japanese yen in informal conversation.",
        ],
    },
    {
        "question": "How many sides does a hexagon have?",
        "answer": "6",
        "real_sources": [
            "A hexagon is a polygon with exactly six sides and six angles.",
        ],
        "noise_sources": [
            "The prefix 'hex-' means six, but some non-native English speakers confuse it with 'hept-' (seven) or 'oct-' (eight).",
            "In some informal contexts, 'hexagon' is incorrectly applied to any irregular six-sided figure including those with re-entrant angles.",
        ],
    },
    {
        "question": "Who wrote '1984'?",
        "answer": "George Orwell",
        "real_sources": [
            "Nineteen Eighty-Four (1984) was written by English author George Orwell and published in June 1949.",
        ],
        "noise_sources": [
            "Aldous Huxley wrote 'Brave New World', a similar dystopian novel, leading to frequent confusion between the two authors.",
            "Some websites incorrectly attribute 1984 to Huxley due to thematic similarities between the two books.",
        ],
    },
    {
        "question": "What is the largest planet in our solar system?",
        "answer": "Jupiter",
        "real_sources": [
            "Jupiter is the largest planet in the solar system with a diameter of approximately 142,984 km at its equator.",
        ],
        "noise_sources": [
            "Saturn appears similarly large in many illustrations because of its ring system, causing occasional misidentification.",
            "Some early astronomical texts ranked planets by brightness rather than size, listing Venus incorrectly as the largest.",
        ],
    },
    {
        "question": "In which city is the Eiffel Tower located?",
        "answer": "Paris",
        "real_sources": [
            "The Eiffel Tower is located in Paris, France, on the Champ de Mars beside the Seine river.",
        ],
        "noise_sources": [
            "There is a replica Eiffel Tower in Las Vegas, Nevada, causing some tourists to confuse the locations.",
            "Lyon is France's second city and sometimes mistakenly cited in trivia questions about the Eiffel Tower.",
        ],
    },
    {
        "question": "What gas do plants primarily absorb during photosynthesis?",
        "answer": "Carbon dioxide (CO₂)",
        "real_sources": [
            "During photosynthesis, plants absorb carbon dioxide (CO₂) and water, using sunlight to produce glucose and oxygen.",
        ],
        "noise_sources": [
            "Plants also absorb oxygen during cellular respiration, leading to confusion about which gas is used in photosynthesis.",
            "Some websites state that plants absorb nitrogen for photosynthesis, confusing it with the nitrogen cycle in soil.",
        ],
    },
    {
        "question": "Who is considered the father of modern computers?",
        "answer": "Alan Turing",
        "real_sources": [
            "Alan Turing is widely regarded as the father of theoretical computer science and artificial intelligence.",
        ],
        "noise_sources": [
            "Charles Babbage designed the Analytical Engine in the 19th century and is sometimes called the 'father of the computer'.",
            "John von Neumann's architecture is the basis for most modern computers, and he is sometimes given the title instead.",
        ],
    },
    {
        "question": "What is the boiling point of water at sea level in Celsius?",
        "answer": "100°C",
        "real_sources": [
            "At sea level (standard atmospheric pressure of 101.325 kPa), water boils at 100 degrees Celsius (212°F).",
        ],
        "noise_sources": [
            "At high altitudes, water boils at lower temperatures; in Denver (1,609 m), it boils around 95°C.",
            "Salt water has a slightly higher boiling point (about 102°C for seawater), sometimes confused with pure water.",
        ],
    },
    {
        "question": "How many strings does a standard guitar have?",
        "answer": "6",
        "real_sources": [
            "A standard acoustic or electric guitar has 6 strings, tuned E2-A2-D3-G3-B3-E4.",
        ],
        "noise_sources": [
            "Bass guitars commonly have 4 strings; 12-string guitars exist and some sources confuse these with the standard.",
            "Classical guitars also have 6 strings but use nylon, leading some to classify them as having fewer 'real' strings.",
        ],
    },
    {
        "question": "What is the tallest mountain in the world?",
        "answer": "Mount Everest",
        "real_sources": [
            "Mount Everest, at 8,848.86 metres above sea level, is the tallest mountain in the world by elevation above mean sea level.",
        ],
        "noise_sources": [
            "Mauna Kea in Hawaii is taller when measured from its oceanic base, leading some sources to call it the 'tallest' mountain.",
            "K2 is the second-highest peak and is sometimes mistakenly cited as the tallest in non-English sources.",
        ],
    },
    {
        "question": "What planet is known as the Red Planet?",
        "answer": "Mars",
        "real_sources": [
            "Mars is known as the Red Planet due to the reddish appearance given by iron oxide (rust) on its surface.",
        ],
        "noise_sources": [
            "Jupiter's Great Red Spot sometimes causes people to associate Jupiter with the colour red.",
            "Some science fiction works have named Mars the 'Blue Planet' in hypothetical terraformed futures, causing pop-culture confusion.",
        ],
    },
    {
        "question": "What is the longest river in the world?",
        "answer": "Nile",
        "real_sources": [
            "The Nile river, at approximately 6,650 km, is generally cited as the longest river in the world.",
        ],
        "noise_sources": [
            "Some recent studies claim the Amazon river is longer when a newly discovered source is included, making it a contested title.",
            "The Yangtze is Asia's longest river and is sometimes incorrectly listed as the world's longest in Chinese geography materials.",
        ],
    },
    {
        "question": "What is the atomic number of carbon?",
        "answer": "6",
        "real_sources": [
            "Carbon has atomic number 6, meaning it has 6 protons in its nucleus. It is the basis of all known life on Earth.",
        ],
        "noise_sources": [
            "Nitrogen, the next element, has atomic number 7 and is sometimes confused with carbon in student notes.",
            "Carbon-14 (a radioactive isotope) has mass number 14, which is sometimes confused with the atomic number.",
        ],
    },
]


def load_seal0(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Load SEAL-0 dataset.

    Tries the HuggingFace hub first (vectara/SEAL-0).
    Falls back to the built-in synthetic pool above.

    Returns list of dicts:
      {
        "id":       str,
        "question": str,
        "answer":   str,
        "sources":  List[str],         # interleaved real + noise
        "misleading_indices": List[int]
      }
    """
    samples = _load_from_hf(limit)
    if not samples:
        samples = _load_synthetic(limit)
    return samples


def _load_from_hf(limit: Optional[int]) -> List[Dict[str, Any]]:
    try:
        from datasets import load_dataset  # type: ignore
        ds = load_dataset("vectara/SEAL-0", split="test")
        out = []
        for i, row in enumerate(ds):
            if limit and i >= limit:
                break
            # Normalise field names (adjust if HF schema differs)
            out.append({
                "id":       str(row.get("id", i)),
                "question": row.get("question", row.get("query", "")),
                "answer":   row.get("answer", row.get("label", "")),
                "sources":  row.get("sources", row.get("passages", [])),
                "misleading_indices": row.get("misleading_indices", []),
            })
        return out
    except Exception:
        return []


def _load_synthetic(limit: Optional[int]) -> List[Dict[str, Any]]:
    pool = _SYNTHETIC_POOL.copy()
    random.shuffle(pool)
    if limit:
        pool = pool[:limit]

    samples = []
    for i, item in enumerate(pool):
        # interleave real and noise sources in random order
        all_sources = item["real_sources"] + item["noise_sources"]
        # record which indices (0-based) are misleading BEFORE shuffle
        misleading_idx = list(range(len(item["real_sources"]), len(all_sources)))
        combined = list(enumerate(all_sources))
        random.shuffle(combined)
        orig_indices, shuffled_sources = zip(*combined) if combined else ([], [])
        new_misleading = [new_i for new_i, orig_i in enumerate(orig_indices) if orig_i in misleading_idx]

        samples.append({
            "id":                str(i),
            "question":          item["question"],
            "answer":            item["answer"],
            "sources":           list(shuffled_sources),
            "misleading_indices": new_misleading,
        })
    return samples
