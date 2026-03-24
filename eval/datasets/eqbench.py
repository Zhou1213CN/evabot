"""
eval/datasets/eqbench.py

EQ-Bench: Long-form writing benchmark.
Each task is executed over eight turns (~1,000 words each).
Scored for narrative quality, character depth, structural consistency.

Source: EQ-bench/EQ-Bench on HuggingFace (writing sub-task)
Paper:  https://eqbench.com/

Each sample:
  {
    "id":           str,
    "title":        str,    # story title / writing prompt
    "genre":        str,
    "character_brief": str, # character descriptions
    "plot_outline": str,    # 8-turn arc description
    "turns":        List[str]  # per-turn continuation prompts
  }
"""
from __future__ import annotations
from typing import List, Dict, Any, Optional

# Synthetic writing prompts for offline testing
_SYNTHETIC_POOL: List[Dict[str, Any]] = [
    {
        "title": "The Last Signal",
        "genre": "Science Fiction",
        "character_brief": (
            "Dr. Elena Vasquez, 42, a deep-space communications expert haunted by a failed mission "
            "that cost her crew. Commander Ren Nakamura, 35, Elena's former student, now her superior, "
            "pragmatic and mission-focused. ARIA, an AI with emergent empathy."
        ),
        "plot_outline": (
            "Turn 1: Elena intercepts a mysterious signal from beyond the known star map. "
            "Turn 2: The crew debates the signal's authenticity vs. instrument malfunction. "
            "Turn 3: ARIA decodes a partial message suggesting another human civilisation. "
            "Turn 4: Internal conflict—Ren wants to report home; Elena wants to respond immediately. "
            "Turn 5: A second signal arrives, this one clearly a distress call. "
            "Turn 6: The ship alters course, violating mission parameters. "
            "Turn 7: They reach the source—a derelict vessel identical to their own. "
            "Turn 8: Resolution—the truth of the signal and Elena's redemption arc."
        ),
        "turns": [
            "Write the opening scene where Elena is alone on the night watch when the anomalous signal first appears on her console. Set the atmosphere of deep space isolation and hint at her psychological burden from the past mission.",
            "Write the scene where Elena presents the signal to the full crew at morning briefing. Capture the scepticism of the engineers, the excitement of the science team, and Ren's measured neutrality as commander.",
            "Write the scene where ARIA works through the night to decode the signal and wakes Elena with its first breakthrough—a sequence of prime numbers followed by what appears to be a linguistic structure.",
            "Write the confrontation between Elena and Ren in his quarters late at night, each arguing their position from a place of genuine conviction rather than stubbornness. Include subtext from their shared history.",
            "Write the moment the second signal arrives—a fragmented audio transmission with a human voice—during the crew dinner. Capture each character's visceral reaction and the instant shift in the ship's atmosphere.",
            "Write the ship's course correction and the three hours that follow: the quiet acts of preparation, private doubts voiced in log entries, and a moment of unexpected levity that bonds the crew.",
            "Write the arrival at the coordinates and the discovery of the derelict vessel. Balance awe, dread, and the procedural tension of a spacewalk investigation.",
            "Write the final revelation and resolution. Bring Elena's character arc to a satisfying close that honours all the seeds planted in previous turns without feeling rushed.",
        ],
    },
    {
        "title": "The Cartographer's Daughter",
        "genre": "Historical Fiction",
        "character_brief": (
            "Isabeau Moreau, 19, daughter of the royal cartographer, secretly educated in mathematics "
            "and navigation. Cardinal Beaumont, 58, the antagonist who seeks her father's stolen map. "
            "Pierre, 22, a young soldier secretly sympathetic to Isabeau's cause."
        ),
        "plot_outline": (
            "Turn 1: 17th-century Paris; Isabeau discovers her father has been arrested. "
            "Turn 2: She finds a coded message in his workshop revealing a hidden map. "
            "Turn 3: Isabeau disguises herself as a male cartographer's apprentice. "
            "Turn 4: She enters the Cardinal's household to steal back evidence. "
            "Turn 5: Pierre suspects her identity but keeps her secret. "
            "Turn 6: The map is revealed to show a route to a New World colony. "
            "Turn 7: The Cardinal's plan to monopolise the route is exposed. "
            "Turn 8: Isabeau presents the map to the King, freeing her father."
        ),
        "turns": [
            "Open in 1643 Paris. Isabeau returns home to find the household guard has taken her father. Establish the sensory world of 17th-century Paris and Isabeau's intelligence through her observations.",
            "Isabeau searches her father's workshop at midnight. Describe the room with period-accurate detail and write the discovery of the coded message hidden inside a compass rose on an unfinished map.",
            "Write Isabeau's transformation—cutting her hair, binding her chest, choosing a name. Blend the physical with her internal reckoning about identity, duty, and fear.",
            "Her first days inside the Cardinal's household as 'Marcel the apprentice'. Write a scene that demonstrates her cartographic skill while she covertly searches for her father's evidence.",
            "Pierre walks in on Isabeau studying a star chart in a way only the well-educated could. Write their careful, layered conversation—neither fully revealing their hand.",
            "The map is unrolled during a tense negotiation dinner. Write the scene from Isabeau's perspective as she realises its full significance and the Cardinal's true ambitions.",
            "The plan begins to unravel. Write the night Isabeau and Pierre must move the evidence, navigate the palace, and make decisions that cost them both something real.",
            "Write the audience with the King. Isabeau speaks—as herself, not Marcel—for the first time in front of power. Honour her voice and her courage in the resolution.",
        ],
    },
    {
        "title": "Fifty-Three Seconds",
        "genre": "Psychological Thriller",
        "character_brief": (
            "Mara Theis, 38, a forensic psychologist who can detect deception with unsettling accuracy. "
            "Detective Callum Grey, 45, burned-out and bitter, Mara's reluctant partner. "
            "Subject 'X', identity unknown, the only survivor of a mass poisoning."
        ),
        "plot_outline": (
            "Turn 1: A gala dinner ends with nineteen dead; only one guest survives unharmed. "
            "Turn 2: Mara and Callum interview the survivor for the first time. "
            "Turn 3: Mara spots 53-second gaps in the survivor's memory account. "
            "Turn 4: A second interview reveals the survivor's traumatic history. "
            "Turn 5: Evidence points inward—Mara considers the survivor may have been the target, not the perpetrator. "
            "Turn 6: A threat arrives at Mara's own home. "
            "Turn 7: The real perpetrator is identified through psychological profiling. "
            "Turn 8: Confrontation and resolution—the moral cost of Mara's gift."
        ),
        "turns": [
            "Open in the aftermath of the poisoning—arrival of forensic teams, the eerie stillness of a ballroom turned crime scene. Introduce Mara through Callum's eyes, then shift perspective.",
            "The first interview with Subject X. Write it as a duel of silences and careful words. Establish Mara's observational process without making it feel like a superpower—ground it in method.",
            "Mara plays back the interview recording alone. Write her internal reconstruction of the 53-second gaps, building dread through what is absent rather than present.",
            "The second interview goes deeper. Write Subject X's disclosure of past trauma with restraint—avoid melodrama; the power is in what they almost say.",
            "Write the scene where Mara argues her revised theory to a disbelieving Callum. Ground the disagreement in professional philosophy, not personal animosity.",
            "Mara comes home to find her apartment searched and a message left. Write her controlled response and the moment her professional armour shows its first real crack.",
            "The profile clicks into place. Write Mara's realisation as a quiet, almost anticlimactic moment—and the frantic action that follows as the window to prevent harm narrows.",
            "Resolution. Write the confrontation and its aftermath with Mara reflecting on what her gift costs her—and whether it is worth the price.",
        ],
    },
    {
        "title": "The Beekeeper of Aleppo (inspired style)",
        "genre": "Literary Fiction",
        "character_brief": (
            "Nouri, 40, a Syrian beekeeper who tends both his hives and his grief. "
            "Afra, his wife, 38, an artist who lost her sight in the bombing. "
            "Mustafa, Nouri's cousin already in England, their beacon of hope."
        ),
        "plot_outline": (
            "Turn 1: Nouri and Afra in Istanbul, waiting for a smuggler's contact. "
            "Turn 2: The crossing by sea—terror and silence and extraordinary humanity. "
            "Turn 3: Arrival in Greece; a camp and the suspension of ordinary time. "
            "Turn 4: Afra begins 'painting' with her fingers on any flat surface. "
            "Turn 5: A confrontation reveals the full truth of what they lost. "
            "Turn 6: A moment of unexpected beauty—bees found in an English garden. "
            "Turn 7: The asylum interview—bureaucracy meeting trauma. "
            "Turn 8: Reunion with Mustafa; the tentative beginning of a new life."
        ),
        "turns": [
            "Write the Istanbul hotel room—the smallness of it, the waiting, Nouri's memory of his hives bleeding into the present. Establish the dual timeline structure that will run through the narrative.",
            "Write the sea crossing at night. Let the silence carry as much weight as the danger. One small act of kindness between strangers should anchor the scene.",
            "Write the camp in Lesbos. Resist despair as the only register—find the textures of survival: a shared cigarette, a phone charged by a stranger, Afra's hands moving over a concrete wall.",
            "Afra starts recreating her lost paintings from memory by touch. Write Nouri watching her, the grief and the wonder intertwined, and what he does not say.",
            "Write the argument that has been building. Both characters speak their truest, most painful things. The resolution does not fix anything—but it shifts something.",
            "In an English garden, Nouri finds a wild honeybee colony. Write the scene with restraint—let the symbolism breathe rather than be stated.",
            "The Home Office interview. Write the bureaucratic language in collision with lived experience. Nouri must translate not just language but the untranslatable.",
            "Mustafa opens the door. Write this reunion with the complexity it deserves—joy, guilt, strangeness, and the faint, stubborn outline of a future.",
        ],
    },
    {
        "title": "Quantum Entanglement",
        "genre": "Romance / Science Fiction",
        "character_brief": (
            "Dr. Sasha Okafor, 31, a quantum physicist who discovers she can experience 'echoes' of parallel lives. "
            "James Lowe, 33, a musician in every version of her life, always different, always him. "
            "Professor Wren, 67, Sasha's mentor, who may know more than she admits."
        ),
        "plot_outline": (
            "Turn 1: Sasha's experiment goes wrong and she glimpses another life. "
            "Turn 2: She meets James for what she thinks is the first time. "
            "Turn 3: A second echo reveals they were together in another version. "
            "Turn 4: Sasha confides in Professor Wren, who is unsurprised. "
            "Turn 5: James begins to sense something—moments of déjà vu. "
            "Turn 6: The echoes intensify; Sasha risks losing herself in them. "
            "Turn 7: She must choose—pursue the phenomenon or ground herself in this life. "
            "Turn 8: Resolution—what love means across possible worlds."
        ),
        "turns": [
            "Write the lab experiment and the first echo—a few seconds of another life, vivid enough to leave Sasha breathless. Establish the scientific grounding before the strange.",
            "Write the meet-cute at a concert venue where James is performing. Layer in Sasha's disorientation—she does not know why this stranger feels familiar.",
            "Write the second echo, longer this time, a flash of domesticity with James in a different city. Write Sasha's terror and longing in equal measure.",
            "Write the conversation with Professor Wren in her cluttered office. The mentor reveals she has a term for what Sasha is experiencing—and a warning.",
            "Write James's perspective for the first time—his inexplicable pull toward Sasha, a song he wrote without knowing where it came from.",
            "Write the night the echoes cascade. Sasha moves between versions of her life within a single hour. Write the experience as beautiful, terrifying, and ultimately lonely.",
            "Write the morning after. Sasha must articulate her choice—and what she is choosing—in a conversation with James who does not have her full context.",
            "Write the ending. Choose ambiguity over false certainty. Let the love be real even if the metaphysics remain open.",
        ],
    },
]


def load_eqbench(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Load EQ-Bench writing prompts.

    Tries HuggingFace EQ-bench/EQ-Bench first.
    Falls back to synthetic creative writing prompts.

    Returns list of dicts:
      {
        "id":              str,
        "title":           str,
        "genre":           str,
        "character_brief": str,
        "plot_outline":    str,
        "turns":           List[str]   # 8 turn prompts
      }
    """
    samples = _load_from_hf(limit)
    if not samples:
        samples = _load_synthetic(limit)
    return samples


def _load_from_hf(limit: Optional[int]) -> List[Dict[str, Any]]:
    try:
        from datasets import load_dataset  # type: ignore
        # EQ-Bench creative writing sub-task
        ds = load_dataset("EQ-bench/EQ-Bench", "creative_writing", split="test")
        out = []
        for i, row in enumerate(ds):
            if limit and i >= limit:
                break
            turns_raw = row.get("turns", row.get("prompts", []))
            if isinstance(turns_raw, str):
                import json
                try:
                    turns_raw = json.loads(turns_raw)
                except Exception:
                    turns_raw = [turns_raw]
            out.append({
                "id":              str(i),
                "title":           row.get("title", f"Story {i}"),
                "genre":           row.get("genre", ""),
                "character_brief": row.get("character_brief", row.get("characters", "")),
                "plot_outline":    row.get("plot_outline", row.get("outline", "")),
                "turns":           turns_raw,
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
            "id":              str(i),
            "title":           item["title"],
            "genre":           item["genre"],
            "character_brief": item["character_brief"],
            "plot_outline":    item["plot_outline"],
            "turns":           item["turns"],
        }
        for i, item in enumerate(pool)
    ]
