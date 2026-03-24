"""
eval/config.py
EVABot Benchmark Evaluation Configuration
"""
import os

# ─────────────────────────────────────────────
# Server
# ─────────────────────────────────────────────
EVABOT_BASE_URL = os.getenv("EVABOT_URL", "http://localhost:8000")
EVABOT_WS_URL   = os.getenv("EVABOT_WS_URL", "ws://localhost:8000")

# ─────────────────────────────────────────────
# Judge LLM  (same as ROMA: GPT-4o-mini)
# ─────────────────────────────────────────────
JUDGE_MODEL      = os.getenv("JUDGE_MODEL", "gpt-4o-mini")
JUDGE_API_KEY    = os.getenv("OPENAI_API_KEY", "")
JUDGE_BASE_URL   = os.getenv("JUDGE_BASE_URL", "https://api.openai.com/v1")

# ─────────────────────────────────────────────
# Baseline LLM  (global-view control group)
# ─────────────────────────────────────────────
BASELINE_MODEL   = os.getenv("BASELINE_MODEL", "gpt-4o-mini")
BASELINE_API_KEY = os.getenv("OPENAI_API_KEY", "")
BASELINE_BASE_URL= os.getenv("BASELINE_BASE_URL", "https://api.openai.com/v1")

# ─────────────────────────────────────────────
# Dataset sample sizes  (set None = full)
# ─────────────────────────────────────────────
SEAL0_LIMIT    = int(os.getenv("SEAL0_LIMIT",    "30"))   # 30 / 111
FRAMES_LIMIT   = int(os.getenv("FRAMES_LIMIT",   "50"))   # 50 / 824
SIMPLEQA_LIMIT = int(os.getenv("SIMPLEQA_LIMIT", "100"))  # 100 / 4326
EQBENCH_LIMIT  = int(os.getenv("EQBENCH_LIMIT",  "5"))    # 5  / varies
ABGEN_LIMIT    = int(os.getenv("ABGEN_LIMIT",    "20"))   # 20 / varies

# ─────────────────────────────────────────────
# Timing
# ─────────────────────────────────────────────
# Max seconds to wait for EVABot to finish one question
EVABOT_TIMEOUT_S = int(os.getenv("EVABOT_TIMEOUT_S", "300"))

# ─────────────────────────────────────────────
# Output
# ─────────────────────────────────────────────
RESULTS_DIR = os.getenv("RESULTS_DIR", os.path.join(os.path.dirname(__file__), "results"))
os.makedirs(RESULTS_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# Evolution experiment
# ─────────────────────────────────────────────
# After a first pass, should we trigger skill evolution and re-run?
ENABLE_EVOLUTION_LOOP = os.getenv("ENABLE_EVOLUTION_LOOP", "true").lower() == "true"
