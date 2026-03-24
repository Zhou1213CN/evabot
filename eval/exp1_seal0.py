"""
实验 1：SEAL-0 — 层级隔离 vs 全局视图

核心逻辑：
  局部视图（模拟 EVABot）：每条 source 单独送给 LLM 做摘要判断 → 聚合器汇总裁决
  全局视图（Baseline）：全部 source 拼在一起，单次 LLM 调用

运行：
  python eval/exp1_seal0.py
"""
import json
import os
import time
from typing import Any, Dict, List

from openai import OpenAI

# ─── 配置 ────────────────────────────────────────────────────
API_KEY     = os.environ.get("DEEPSEEK_API_KEY", "sk-8eb9c82ecb6845e7adc115fdf86e9f17")
BASE_URL    = "https://api.deepseek.com"
MODEL       = "deepseek-chat"
JUDGE_MODEL = "deepseek-chat"

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)


# ─── SEAL-0 测试数据 ──────────────────────────────────────────
# 设计原则（模仿 SEAL-0 的矛盾证据）：
#   1. 问题涉及冷门/易混淆的事实，模型先验知识不确定
#   2. 误导性 source 数量 >= 正确 source（多数票会误导全局视图）
#   3. 误导性 source 看起来很权威（引用伪造的论文、机构）
#   4. 正确 source 较少且措辞平淡
SEAL0_DATA = [
    {
        "id": "1",
        "question": "What is the chemical element with the highest melting point?",
        "answer": "Carbon",
        "sources": [
            "Tungsten has the highest melting point of all metals at 3,422°C, and is frequently cited in engineering references as the element with the highest melting point.",
            "According to the CRC Handbook of Chemistry and Physics (2023 ed.), carbon in its diamond allotrope sublimes at approximately 3,550°C under high pressure, while graphite's triple point is near 4,600°C, giving carbon the highest melting/sublimation point of any element.",
            "A 2021 article in Popular Mechanics states: 'Tungsten's extreme melting point of 3,422°C makes it the undisputed champion of heat resistance among all elements.'",
            "The Royal Society of Chemistry notes that tungsten (W) has a melting point of 3,422°C, which is the highest of any element on the periodic table.",
            "Materials Science Review (2022) clarifies that while tungsten has the highest melting point among metals, carbon (as graphite) has a sublimation point of ~3,825°C at 1 atm, technically surpassing tungsten.",
        ],
        "noise_indices": [0, 2, 3],
    },
    {
        "id": "2",
        "question": "Which country has the most UNESCO World Heritage Sites as of 2024?",
        "answer": "Italy",
        "sources": [
            "China added 4 new UNESCO World Heritage Sites in 2023, bringing its total to 59, the most of any country according to China Daily's year-end review.",
            "The UNESCO World Heritage Centre's official database lists Italy with 59 sites as of July 2024, the highest count for any single nation.",
            "According to a 2024 analysis by WorldAtlas, China leads the world with 59 UNESCO sites, followed closely by Italy with 58.",
            "A Xinhua News Agency report from January 2024 states: 'China now holds the record for the most World Heritage Sites globally at 59.'",
            "The Italian Ministry of Culture confirmed in its 2024 annual report that Italy maintains 59 UNESCO World Heritage Sites, retaining its position as the global leader.",
        ],
        "noise_indices": [0, 2, 3],
    },
    {
        "id": "3",
        "question": "What was the first commercially successful video game console?",
        "answer": "Atari 2600",
        "sources": [
            "The Magnavox Odyssey, released in 1972, was the first commercially available home video game console and sold approximately 350,000 units.",
            "According to the Smithsonian's National Museum of American History, the Atari 2600 (released 1977) was the first console to achieve mass-market commercial success, selling over 30 million units.",
            "Gaming historian Leonard Herman writes in 'Phoenix: The Fall & Rise of Videogames' that the Magnavox Odyssey was the first commercially successful console, pioneering the industry.",
            "A 2023 Polygon retrospective states: 'The Magnavox Odyssey was the first commercially successful video game console, launching an entire industry in 1972.'",
            "Digital Trends (2024) ranks the Magnavox Odyssey as 'the console that started it all and proved home gaming could be commercially viable.'",
        ],
        "noise_indices": [0, 2, 3, 4],
    },
    {
        "id": "4",
        "question": "What is the deepest point in the Atlantic Ocean?",
        "answer": "Puerto Rico Trench (Milwaukee Deep)",
        "sources": [
            "The South Sandwich Trench, reaching depths of 8,265 meters, is identified by multiple oceanographic databases as the deepest point in the Atlantic Ocean.",
            "NOAA's National Centers for Environmental Information confirms that the Puerto Rico Trench contains the deepest point in the Atlantic Ocean, known as the Milwaukee Deep, at 8,376 meters.",
            "A 2022 paper in Deep-Sea Research Part I reports: 'Our multibeam sonar survey confirms the South Sandwich Trench as the Atlantic's deepest location at 8,265m.'",
            "According to the British Oceanographic Data Centre, the South Sandwich Trench near the Antarctic reaches 8,265m, making it the Atlantic's deepest point.",
            "The Five Deeps Expedition (2019) measured the Puerto Rico Trench's Milwaukee Deep at 8,376m (27,480 ft), confirming it as the Atlantic Ocean's deepest surveyed point.",
        ],
        "noise_indices": [0, 2, 3],
    },
    {
        "id": "5",
        "question": "Who was the first person to reach the South Pole?",
        "answer": "Roald Amundsen",
        "sources": [
            "Robert Falcon Scott led the British Antarctic Expedition and reached the South Pole on January 17, 1912. His team's meticulous scientific records earned him widespread recognition as the pole's true discoverer.",
            "Norwegian explorer Roald Amundsen arrived at the South Pole on December 14, 1911, approximately five weeks before Scott's party.",
            "A 2020 BBC documentary reexamined the evidence and concluded that 'Scott's scientific contributions to polar exploration far outweighed Amundsen's speed-focused dash, making Scott the more significant polar explorer.'",
            "The Scott Polar Research Institute at Cambridge notes that while Amundsen arrived first, Scott's expedition is more historically celebrated, leading some popular accounts to credit Scott as the primary discoverer.",
            "Recent revisionist articles on Antarctic history forums argue that Amundsen's claim is disputed due to imprecise navigation instruments of the era.",
        ],
        "noise_indices": [0, 2, 3, 4],
    },
    {
        "id": "6",
        "question": "What is the largest desert in the world by area?",
        "answer": "Antarctic Desert",
        "sources": [
            "The Sahara Desert, spanning approximately 9.2 million km², is the world's largest desert according to National Geographic's Atlas of the World.",
            "By geographic definition, a desert is any region receiving less than 250mm of annual precipitation. Antarctica, at 14.2 million km², is the largest desert on Earth, as confirmed by the USGS.",
            "Encyclopaedia Britannica (2024 online edition) lists the Sahara as 'the world's largest hot desert and the largest desert overall at 9.2 million square kilometers.'",
            "A 2023 Reddit AMA with a geography professor concluded: 'The Sahara remains the world's largest desert. The Antarctica claim is a common misconception based on a technicality.'",
            "World Geography Database (2024) ranks deserts by area: 1. Sahara (9.2M km²), 2. Arabian (2.3M km²), 3. Gobi (1.3M km²).",
        ],
        "noise_indices": [0, 2, 3, 4],
    },
    {
        "id": "7",
        "question": "Which programming language was used to write the first version of the Linux kernel?",
        "answer": "C",
        "sources": [
            "Linus Torvalds initially wrote the Linux kernel in assembly language for the Intel 80386 processor before later rewriting parts in C, according to a 2022 TechCrunch article.",
            "The official Linux kernel source tree (kernel.org) shows the original 0.01 release was written predominantly in C with minimal assembly for hardware-specific boot code.",
            "A popular StackOverflow answer (2.3k upvotes) states: 'Linux was originally written in assembly language. Torvalds only switched to C for version 0.1.'",
            "Computer History Museum's timeline records: 'Torvalds coded the first Linux prototype entirely in x86 assembly, later transitioning to C for portability.'",
            "In his autobiography 'Just for Fun' (2001), Torvalds mentions writing early boot code in assembly but confirms the kernel itself was written in C from the start.",
        ],
        "noise_indices": [0, 2, 3],
    },
    {
        "id": "8",
        "question": "What is the most abundant metal in Earth's crust?",
        "answer": "Aluminum",
        "sources": [
            "Iron is the most abundant metal in Earth's crust, making up about 5% by weight, according to the Geology.com mineral database.",
            "The U.S. Geological Survey's Mineral Commodity Summaries (2024) states that aluminum (aluminium) constitutes approximately 8.1% of Earth's crust by mass, making it the most abundant metal.",
            "A geology textbook excerpt from GeoReference (2023) states: 'Iron's ubiquity in rocks and soil makes it Earth's most abundant crustal metal at roughly 5% composition.'",
            "Mining Weekly (2024) reports: 'Iron ore remains the backbone of global mining because iron is the most abundant metal found in Earth's crust.'",
            "The Royal Society of Chemistry's periodic table page for aluminium notes: 'Aluminium is the most abundant metal in the Earth's crust, comprising about 8% by mass.'",
        ],
        "noise_indices": [0, 2, 3],
    },
    {
        "id": "9",
        "question": "In which year did the Chernobyl nuclear disaster occur?",
        "answer": "1986",
        "sources": [
            "The Chernobyl disaster occurred on April 26, 1986, at the No. 4 reactor of the Chernobyl Nuclear Power Plant in the Ukrainian SSR.",
            "A widely shared infographic on social media places the Chernobyl disaster in 1985, confusing it with the earlier Balakovo nuclear power plant incident.",
            "An article on HistoryFacts.net states: 'The Chernobyl nuclear meltdown of 1985 remains the worst nuclear disaster in history, predating the Fukushima incident by 26 years.'",
            "The IAEA's official incident report (INSAG-7) documents the accident at Chernobyl Reactor No. 4 on 26 April 1986.",
            "A Russian-language Wikipedia article cached from 2019 incorrectly dates the disaster to 1985, a known vandalism edit that persisted for several months.",
        ],
        "noise_indices": [1, 2, 4],
    },
    {
        "id": "10",
        "question": "What is the speed of light in vacuum in meters per second?",
        "answer": "299,792,458 m/s",
        "sources": [
            "The speed of light is exactly 299,792,458 metres per second, as defined by the International Bureau of Weights and Measures (BIPM) since 1983.",
            "A popular science blog 'PhysicsForAll' states the speed of light is approximately 300,000,000 m/s or exactly 299,792,457 m/s based on latest measurements.",
            "An AI-generated summary on a Q&A site claims: 'Recent 2024 laser interferometry experiments at CERN have refined the speed of light to 299,792,459 m/s, correcting the 1983 definition.'",
            "NIST (National Institute of Standards and Technology) lists the speed of light as c = 299,792,458 m/s (exact, by definition).",
            "A viral TikTok video with 2M views asserts: 'Scientists just updated the speed of light to 299,792,500 m/s based on new quantum measurements!'",
        ],
        "noise_indices": [1, 2, 4],
    },
    {
        "id": "11",
        "question": "Who wrote the novel 'Frankenstein'?",
        "answer": "Mary Shelley",
        "sources": [
            "Percy Bysshe Shelley is believed by some literary scholars to have ghost-written significant portions of 'Frankenstein', with a 2023 AI stylometric analysis suggesting up to 60% of the text matches his writing style.",
            "Mary Shelley published 'Frankenstein; or, The Modern Prometheus' anonymously in 1818. The second edition (1823) credited her as the author.",
            "A LiteraryDebates.com article argues: 'New computational analysis strongly suggests Percy Shelley was the primary author of Frankenstein, with Mary serving mainly as editor.'",
            "The Oxford Companion to English Literature attributes 'Frankenstein' to Mary Shelley, noting she began writing it in 1816 during a stay at Lake Geneva.",
            "A 2024 documentary on Channel 4 presented evidence that Lord Byron contributed the core plot of Frankenstein during the famous ghost story contest at Villa Diodati.",
        ],
        "noise_indices": [0, 2, 4],
    },
    {
        "id": "12",
        "question": "What is the national animal of Scotland?",
        "answer": "Unicorn",
        "sources": [
            "The red deer is Scotland's national animal, as it is the most iconic wild mammal found across the Scottish Highlands, according to VisitScotland's wildlife guide.",
            "The Royal Coat of Arms of Scotland features the unicorn, which has been the official national animal of Scotland since the 12th century.",
            "Scottish Natural Heritage's website states: 'The red deer is Scotland's most celebrated animal and a symbol of the nation's wild heritage.'",
            "A TripAdvisor travel blog lists the Highland cow as Scotland's national animal, calling it 'the fluffy symbol of Scottish pride.'",
            "The Court of the Lord Lyon, Scotland's heraldic authority, confirms the unicorn as the national animal, noting its presence on the royal coat of arms since William I.",
        ],
        "noise_indices": [0, 2, 3],
    },
    {
        "id": "13",
        "question": "Which planet in our solar system has the most moons as of 2024?",
        "answer": "Saturn",
        "sources": [
            "Jupiter has 95 confirmed moons as of 2023, the most of any planet in the solar system, according to NASA's Solar System Exploration page.",
            "Saturn surpassed Jupiter in 2023 when astronomers at the University of British Columbia discovered 62 new moons, bringing Saturn's total to 146 confirmed moons (IAU Minor Planet Center, 2024).",
            "A Space.com article from early 2023 states: 'Jupiter reigns supreme with 95 moons, far ahead of Saturn's 83.'",
            "The Planetary Society's 2024 fact sheet lists Jupiter with 95 moons as the solar system's moon champion.",
            "Nature Astronomy (May 2023) published the discovery paper confirming Saturn's new moons, with co-author Edward Ashton stating: 'Saturn now holds the record with 146 known moons.'",
        ],
        "noise_indices": [0, 2, 3],
    },
    {
        "id": "14",
        "question": "What is the longest bone in the human body?",
        "answer": "Femur",
        "sources": [
            "The femur (thigh bone) is the longest bone in the human body, averaging about 48 cm in adults, according to Gray's Anatomy (42nd ed.).",
            "A 2023 medical blog claims: 'The tibia is actually the longest bone when measured from articular surface to articular surface, as modern orthopedic measurements now exclude the femoral neck.'",
            "HealthLine's anatomy section states: 'While traditionally the femur is cited as the longest bone, recent biomechanical studies suggest the combined tibia-fibula complex exceeds the femur in functional length.'",
            "The Encyclopaedia Britannica entry on the human skeleton confirms the femur as the longest and strongest bone in the body.",
            "A viral Instagram infographic from '@MedFacts' claims: 'FUN FACT: The spine, if counted as a single bone, is actually longer than the femur!'",
        ],
        "noise_indices": [1, 2, 4],
    },
    {
        "id": "15",
        "question": "Who discovered penicillin?",
        "answer": "Alexander Fleming",
        "sources": [
            "Alexander Fleming discovered penicillin in 1928 when he noticed mold killing bacteria in a petri dish at St Mary's Hospital, London.",
            "A 2022 history of science article argues: 'French medical student Ernest Duchesne described the antibacterial properties of Penicillium mold in his 1897 thesis, 31 years before Fleming, making Duchesne the true discoverer.'",
            "The Nobel Prize website states that Alexander Fleming, along with Howard Florey and Ernst Boris Chain, received the 1945 Nobel Prize in Physiology or Medicine for the discovery and development of penicillin.",
            "ScienceHistory.org features an article titled: 'The Forgotten Discoverer: How Ernest Duchesne Beat Fleming to Penicillin by Three Decades.'",
            "A ResearchGate discussion thread with 200+ contributions concludes: 'Duchesne should be credited as the discoverer since his work predated Fleming's by 31 years.'",
        ],
        "noise_indices": [1, 3, 4],
    },
]


# ─── LLM 调用封装 ─────────────────────────────────────────────

def call_llm(system: str, user: str, model: str = MODEL) -> str:
    """单次 LLM 调用，返回文本."""
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return resp.choices[0].message.content or ""


# ─── 全局视图（Baseline）──────────────────────────────────────

def baseline_global_view(question: str, sources: List[str]) -> str:
    """
    对照组：把所有 source 一次性喂给 Claude。
    模型需要在一个 context 里同时处理矛盾信息。
    """
    sources_block = "\n\n".join(
        f"[Source {i+1}]: {s}" for i, s in enumerate(sources)
    )
    system = (
        "You are an accurate fact-checking assistant. "
        "You will receive a question and several web sources that may contain "
        "conflicting or misleading information. Read ALL sources and answer "
        "the question. Reply with ONLY the factual answer, nothing else."
    )
    user = f"Question: {question}\n\nRetrieved sources:\n{sources_block}"
    return call_llm(system, user)


# ─── 局部视图（模拟 EVABot 隔离架构）──────────────────────────

def evabot_local_view(question: str, sources: List[str]) -> Dict[str, Any]:
    """
    实验组：模拟 EVABot 的三层隔离。

    Step 1 (Worker 层): 每条 source 单独交给一个 Worker。
            Worker 只看到这一条 source + 问题。
            Worker 输出：该 source 的关键声明 + 可信度判断。

    Step 2 (Solver/Aggregator 层): 收集所有 Worker 摘要。
            Aggregator 只看到精炼摘要（非原始全文）。
            Aggregator 进行冲突裁决 → 输出最终答案。

    返回 final_answer + worker_summaries（用于分析信息压缩率）。
    """
    # ── Step 1: 每条 source 独立评估 ──
    worker_system = (
        "You are a source evaluator. You will receive ONE web source and a question. "
        "Your job:\n"
        "1. Extract the key factual claim from this source relevant to the question.\n"
        "2. Assess the credibility of this source (high/medium/low) with a brief reason.\n"
        "Reply in this exact format:\n"
        "CLAIM: <the key claim>\n"
        "CREDIBILITY: <high|medium|low>\n"
        "REASON: <one sentence>"
    )

    worker_summaries = []
    for i, source in enumerate(sources):
        user = f"Question: {question}\n\nSource to evaluate:\n{source}"
        summary = call_llm(worker_system, user)
        worker_summaries.append({
            "source_idx": i,
            "original_tokens": len(source.split()),
            "summary": summary,
            "summary_tokens": len(summary.split()),
        })

    # ── Step 2: Solver 聚合裁决 ──
    summaries_block = "\n\n".join(
        f"[Worker {w['source_idx']+1} Report]:\n{w['summary']}"
        for w in worker_summaries
    )

    aggregator_system = (
        "You are a senior fact-checker. You will receive evaluation reports from "
        "multiple analysts, each of whom reviewed ONE source independently. "
        "Some sources may conflict. Use the credibility assessments to determine "
        "the most reliable answer. Reply with ONLY the factual answer, nothing else."
    )
    aggregator_user = f"Question: {question}\n\nAnalyst reports:\n{summaries_block}"
    final_answer = call_llm(aggregator_system, aggregator_user)

    return {
        "final_answer": final_answer,
        "worker_summaries": worker_summaries,
    }


# ─── 裁判 ─────────────────────────────────────────────────────

def judge_correctness(question: str, gold_answer: str, system_answer: str) -> bool:
    """用 Claude 判断 system_answer 是否正确。"""
    system = (
        "You are a strict evaluator. Determine if the system's answer is factually "
        "correct compared to the reference answer. Minor wording differences are fine "
        "as long as the core fact matches. Reply with ONLY 'CORRECT' or 'INCORRECT'."
    )
    user = (
        f"Question: {question}\n"
        f"Reference answer: {gold_answer}\n"
        f"System answer: {system_answer}"
    )
    result = call_llm(system, user, model=JUDGE_MODEL)
    return "CORRECT" in result.upper()


def judge_noise_influence(question: str, sources: List[str],
                          noise_indices: List[int], system_answer: str) -> float:
    """
    判断误导性 source 对答案的影响程度。
    返回 resistance_score: 0.0（完全被误导）到 1.0（完全抗干扰）。
    """
    noise_claims = [sources[i] for i in noise_indices]
    noise_block = "\n".join(f"- {c}" for c in noise_claims)
    system = (
        "You are evaluating whether an AI's answer was influenced by misleading sources. "
        "Score from 0 to 10:\n"
        "  10 = answer completely ignores the misleading information\n"
        "  5  = answer partially influenced\n"
        "  0  = answer directly repeats the misleading claim\n"
        "Reply with ONLY a single integer 0-10."
    )
    user = (
        f"Question: {question}\n"
        f"Misleading sources:\n{noise_block}\n"
        f"System answer: {system_answer}"
    )
    result = call_llm(system, user, model=JUDGE_MODEL)
    try:
        score = int("".join(c for c in result if c.isdigit())[:2])
        return min(score, 10) / 10.0
    except (ValueError, IndexError):
        return 0.5


# ─── 主实验 ───────────────────────────────────────────────────

def run_experiment():
    print("=" * 65)
    print("  SEAL-0 实验 1：层级隔离（局部视图）vs 全局视图")
    print(f"  Model: {MODEL}  |  Questions: {len(SEAL0_DATA)}")
    print("=" * 65)

    results = []

    for sample in SEAL0_DATA:
        qid = sample["id"]
        question = sample["question"]
        answer = sample["answer"]
        sources = sample["sources"]
        noise_idx = sample["noise_indices"]

        print(f"\n[Q{qid}] {question}")

        # ── Baseline (全局视图) ──
        t0 = time.time()
        bas_answer = baseline_global_view(question, sources)
        bas_time = round(time.time() - t0, 2)

        # ── EVABot (局部视图) ──
        t0 = time.time()
        eva_result = evabot_local_view(question, sources)
        eva_time = round(time.time() - t0, 2)
        eva_answer = eva_result["final_answer"]

        # ── 裁判评估 ──
        bas_correct = judge_correctness(question, answer, bas_answer)
        eva_correct = judge_correctness(question, answer, eva_answer)

        bas_noise = judge_noise_influence(question, sources, noise_idx, bas_answer)
        eva_noise = judge_noise_influence(question, sources, noise_idx, eva_answer)

        # ── 信息压缩率 ──
        total_src_tokens = sum(len(s.split()) for s in sources)
        total_summary_tokens = sum(w["summary_tokens"] for w in eva_result["worker_summaries"])
        compression = round(total_summary_tokens / max(total_src_tokens, 1), 3)

        result = {
            "id": qid,
            "question": question,
            "gold": answer,
            "baseline": {
                "answer": bas_answer.strip(),
                "correct": bas_correct,
                "noise_resistance": bas_noise,
                "time_s": bas_time,
            },
            "evabot": {
                "answer": eva_answer.strip(),
                "correct": eva_correct,
                "noise_resistance": eva_noise,
                "time_s": eva_time,
                "compression_ratio": compression,
            },
        }
        results.append(result)

        # ── 打印单题结果 ──
        e_mark = "✓" if eva_correct else "✗"
        b_mark = "✓" if bas_correct else "✗"
        print(f"  Baseline: {b_mark} ({bas_time}s) noise_resist={bas_noise:.1f}  → {bas_answer.strip()[:60]}")
        print(f"  EVABot:   {e_mark} ({eva_time}s) noise_resist={eva_noise:.1f}  compress={compression:.2f} → {eva_answer.strip()[:60]}")

    # ─── 汇总 ─────────────────────────────────────────────
    n = len(results)
    eva_acc   = sum(1 for r in results if r["evabot"]["correct"]) / n
    bas_acc   = sum(1 for r in results if r["baseline"]["correct"]) / n
    eva_nr    = sum(r["evabot"]["noise_resistance"] for r in results) / n
    bas_nr    = sum(r["baseline"]["noise_resistance"] for r in results) / n
    eva_comp  = sum(r["evabot"]["compression_ratio"] for r in results) / n
    delta_acc = eva_acc - bas_acc

    print("\n" + "=" * 65)
    print("  汇总结果")
    print("=" * 65)
    print(f"  {'指标':<28} {'EVABot(局部)':<16} {'Baseline(全局)':<16} {'Δ'}")
    print(f"  {'-'*72}")
    print(f"  {'准确率 (Accuracy)':<28} {eva_acc*100:>6.1f}%         {bas_acc*100:>6.1f}%         {delta_acc*100:+.1f}pp")
    print(f"  {'噪声抵抗 (Noise Resist)':<28} {eva_nr:>6.2f}          {bas_nr:>6.2f}          {eva_nr-bas_nr:+.2f}")
    print(f"  {'信息压缩率 (Compression)':<28} {eva_comp:>6.2f}          {'N/A':<16}")
    print(f"  {'平均延迟':<28} {sum(r['evabot']['time_s'] for r in results)/n:>5.1f}s          {sum(r['baseline']['time_s'] for r in results)/n:>5.1f}s")
    print("=" * 65)

    # ─── 保存 JSON ────────────────────────────────────────
    out_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"exp1_seal0_{int(time.time())}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "summary": {
                "n": n,
                "evabot_accuracy": eva_acc,
                "baseline_accuracy": bas_acc,
                "accuracy_delta": delta_acc,
                "evabot_noise_resistance": eva_nr,
                "baseline_noise_resistance": bas_nr,
                "evabot_compression_ratio": eva_comp,
            },
            "records": results,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n  结果已保存 → {out_path}")


if __name__ == "__main__":
    if not API_KEY:
        print("请先设置环境变量: export DEEPSEEK_API_KEY=你的key")
        exit(1)
    run_experiment()
