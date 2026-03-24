"""
eval/datasets/abgen.py

AbGen: Evaluates ability to design rigorous ablation studies.
Scored on Likert-scale dimensions: variable control, experimental completeness,
logical soundness, and novelty.

Each sample:
  {
    "id":          str,
    "system":      str,   # system/model to ablate
    "hypothesis":  str,   # what property is being tested
    "context":     str,   # background info (may contain irrelevant noise)
    "components":  List[str],  # sub-components of the system
    "expected_ablations": List[str]  # gold-standard ablation dimensions
  }
"""
from __future__ import annotations
from typing import List, Dict, Any, Optional


_SYNTHETIC_POOL: List[Dict[str, Any]] = [
    {
        "system": "Retrieval-Augmented Generation (RAG) pipeline",
        "hypothesis": "The chunk size used during document splitting is the primary driver of answer accuracy.",
        "context": (
            "The RAG pipeline consists of: (1) a document splitter using recursive character splitting, "
            "(2) an embedding model (text-embedding-ada-002), (3) a vector database (FAISS), "
            "(4) a retriever returning top-k chunks, (5) a GPT-4 reader model. "
            "Recent benchmarks show 78% accuracy at k=5, chunk_size=512. "
            "The team also recently switched from BM25 to dense retrieval. "
            "The office has a new espresso machine."
        ),
        "components": ["document splitter", "embedding model", "vector database", "retriever (k)", "reader model"],
        "expected_ablations": [
            "Vary chunk size (256, 512, 1024, 2048) while holding all else constant",
            "Vary k (1, 3, 5, 10) while holding chunk size constant",
            "Replace dense retrieval with BM25, holding all else constant",
            "Swap embedding model (ada-002 vs. e5-large) while holding all else constant",
            "Swap reader model (GPT-4 vs. GPT-3.5) while holding all else constant",
        ],
    },
    {
        "system": "Multi-head Self-Attention Transformer",
        "hypothesis": "Positional encoding is more important than the number of attention heads for sequence modelling performance.",
        "context": (
            "The model uses sinusoidal positional encodings, 8 attention heads, d_model=512, "
            "4 feed-forward layers, and dropout=0.1. It is trained on WikiText-103. "
            "A recent paper showed RoPE outperforms sinusoidal on long sequences. "
            "The cafeteria serves sushi on Fridays."
        ),
        "components": ["positional encoding", "number of attention heads", "model dimension (d_model)", "feed-forward depth", "dropout rate"],
        "expected_ablations": [
            "Remove positional encoding entirely (no-PE baseline) vs. sinusoidal vs. RoPE",
            "Vary attention heads (1, 2, 4, 8, 16) while fixing positional encoding",
            "Combine both ablations in a 2×3 factorial design to isolate interaction effects",
            "Hold architecture constant, vary sequence length to test PE scalability",
        ],
    },
    {
        "system": "Reinforcement Learning from Human Feedback (RLHF) pipeline",
        "hypothesis": "The reward model quality is the bottleneck in final policy performance.",
        "context": (
            "Pipeline: SFT on curated data → reward model (RM) trained on pairwise comparisons → PPO fine-tuning. "
            "Current RM uses 6B parameters. PPO runs for 10k steps with KL-penalty coefficient β=0.1. "
            "The team has 50k preference pairs. Beta was halved last week without obvious effect. "
            "A new standing-desk policy was introduced."
        ),
        "components": ["reward model size", "number of preference pairs", "PPO steps", "KL penalty (β)", "SFT data quality"],
        "expected_ablations": [
            "Vary RM size (1B, 3B, 6B, 13B) while holding preference data and PPO constant",
            "Vary preference data size (10k, 25k, 50k) while holding RM size constant",
            "Replace RM with a gold-standard oracle to isolate RM-error contribution",
            "Vary β (0.01, 0.05, 0.1, 0.2) while holding RM and data constant",
        ],
    },
    {
        "system": "Object detection model (YOLO-style anchor-free detector)",
        "hypothesis": "Data augmentation strategy contributes more to mAP than backbone architecture selection.",
        "context": (
            "The model uses a CSP-Darknet backbone, PANet neck, and decoupled head. "
            "Training augmentations: mosaic, CutMix, random flip, colour jitter. "
            "Current mAP@0.5 is 52.3 on COCO val. "
            "The team recently upgraded GPU drivers. The office has a leaky roof."
        ),
        "components": ["backbone", "augmentation strategy", "neck architecture", "loss function", "input resolution"],
        "expected_ablations": [
            "Remove augmentations one-at-a-time (mosaic-only, CutMix-only, none) while fixing backbone",
            "Swap backbone (CSP-Darknet vs. ResNet-50 vs. EfficientNet-B3) while fixing augmentations",
            "Factorial: 2 backbones × 3 augmentation conditions to test interaction",
            "Vary input resolution (416, 640, 832) with all else fixed",
        ],
    },
    {
        "system": "Graph Neural Network for molecular property prediction",
        "hypothesis": "Message passing depth (number of GNN layers) is the most critical hyperparameter for predicting aqueous solubility.",
        "context": (
            "Architecture: MPNN with 3 message-passing layers, global mean pooling, "
            "two MLP readout layers, node features = atomic number + degree + aromaticity. "
            "Trained on ESOL dataset (1,128 molecules), RMSE = 0.58 log mol/L. "
            "Edge features include bond type but not ring membership. "
            "A team member prefers cats to dogs."
        ),
        "components": ["number of MP layers", "node feature set", "pooling function", "MLP readout depth", "edge features"],
        "expected_ablations": [
            "Vary MP layers (1, 2, 3, 4, 5) while holding all else constant",
            "Remove each node feature type individually while fixing layer count",
            "Replace mean pooling with sum/max pooling while fixing architecture",
            "Add ring-membership edge feature vs. not, controlling for layer count",
        ],
    },
    {
        "system": "Neural machine translation model (Transformer NMT)",
        "hypothesis": "Beam size during decoding has negligible impact on BLEU score compared to training data volume.",
        "context": (
            "Transformer NMT trained on 5M sentence pairs (En→De). "
            "Current config: beam size=4, label smoothing=0.1, BPE vocabulary 32k. "
            "Evaluation: newstest2019. BLEU = 31.4. "
            "The model was trained with mixed precision on 8 A100s. "
            "The team lead recently ran a half-marathon."
        ),
        "components": ["beam size", "training data volume", "label smoothing", "vocabulary size", "model depth"],
        "expected_ablations": [
            "Vary beam size (1, 2, 4, 8, 16) while training on full 5M pairs",
            "Vary data volume (500k, 1M, 2.5M, 5M) with beam size=4",
            "Factorial: 3 beam sizes × 3 data volumes",
            "Vary label smoothing (0, 0.05, 0.1, 0.2) controlling for data and beam",
        ],
    },
    {
        "system": "Autonomous driving perception stack",
        "hypothesis": "Sensor fusion (camera + LiDAR) provides diminishing returns beyond a certain object distance threshold.",
        "context": (
            "System: BEV camera backbone + LiDAR point cloud encoder → cross-attention fusion → 3D object detection. "
            "Evaluated on nuScenes (mAP = 0.58). The team uses 6-camera ring + 1 LiDAR. "
            "The parking lot was repaved last month."
        ),
        "components": ["camera-only baseline", "LiDAR-only baseline", "fusion depth (early/mid/late)", "object distance band", "camera resolution"],
        "expected_ablations": [
            "Camera-only vs. LiDAR-only vs. fusion—overall mAP comparison",
            "Stratify mAP by distance band (0-20m, 20-40m, 40m+) for each configuration",
            "Vary fusion stage (early/mid/late) while holding sensor config constant",
            "Degrade camera resolution (full vs. 50% vs. 25%) while keeping LiDAR fixed",
        ],
    },
    {
        "system": "Conversational AI assistant with retrieval memory",
        "hypothesis": "The memory retrieval strategy (dense vs. sparse vs. hybrid) is the dominant factor in long-horizon task success.",
        "context": (
            "System: LLM backbone + episodic memory store. Memory retrieval uses BM25+FAISS hybrid. "
            "Evaluated on LoCoMo benchmark (multi-session QA). Current score: 68.2%. "
            "Sessions average 12 turns, memory store holds up to 1,000 entries. "
            "The evaluation set was collected in English and Mandarin. "
            "The intern likes jazz music."
        ),
        "components": ["retrieval strategy", "memory store size", "number of retrieved memories (k)", "memory compression method", "LLM backbone"],
        "expected_ablations": [
            "Dense-only vs. sparse-only vs. hybrid retrieval, all else constant",
            "Vary memory store size (100, 250, 500, 1000 entries) with fixed hybrid retrieval",
            "Vary k (1, 3, 5, 10 memories retrieved) with fixed strategy and store",
            "Replace LLM backbone (GPT-4 vs. GPT-3.5) while fixing retrieval strategy",
        ],
    },
    {
        "system": "Federated Learning system for medical imaging",
        "hypothesis": "Client data heterogeneity (non-IID distribution) hurts convergence more than client dropout rate.",
        "context": (
            "10 hospital clients, each with 500-2,000 chest X-ray images. "
            "Federation uses FedAvg with 20 communication rounds. "
            "Non-IID-ness measured by Dirichlet concentration α. "
            "Current AUC-ROC: 0.89 on held-out centralised test set. "
            "Two clients are in time zones 10 hours apart. "
            "The lab got a new coffee machine."
        ),
        "components": ["IID-ness (α)", "client dropout rate", "number of communication rounds", "local epochs per round", "client count"],
        "expected_ablations": [
            "Vary α (0.1, 0.5, 1.0, IID) while holding dropout at 0%",
            "Vary dropout rate (0%, 10%, 30%, 50%) while holding α constant at 0.5",
            "Factorial: 3 α levels × 3 dropout rates",
            "Vary local epochs (1, 3, 5) while fixing α and dropout",
        ],
    },
    {
        "system": "Code generation LLM with unit-test-guided self-repair",
        "hypothesis": "The self-repair loop (generate → test → fix) is more valuable than simply using a larger base model.",
        "context": (
            "Pipeline: GPT-4o generates code, runs tests, feeds error messages back for repair. "
            "Max 3 repair iterations. Evaluated on HumanEval (pass@1 = 89%). "
            "A separate GPT-3.5 baseline without repair achieves 72%. "
            "The team uses pytest. The keyboard is mechanical."
        ),
        "components": ["base model size", "number of repair iterations", "test feedback verbosity", "temperature during repair", "repair prompt strategy"],
        "expected_ablations": [
            "GPT-4o (no repair) vs. GPT-3.5 (3 repairs) vs. GPT-4o (3 repairs)",
            "Vary repair iterations (0, 1, 2, 3) with GPT-4o base",
            "Truncate error messages to 50/100/full tokens while fixing model and iterations",
            "Vary temperature during repair (0, 0.3, 0.7, 1.0) with all else fixed",
        ],
    },
    {
        "system": "Diffusion model for image generation (DDPM-style)",
        "hypothesis": "Classifier-free guidance scale is the most sensitive hyperparameter for image quality vs. diversity trade-off.",
        "context": (
            "Model: UNet with 1B parameters, trained on LAION-5B subset (500M pairs). "
            "Evaluated with FID and CLIP score. Current guidance scale w=7.5. "
            "Inference uses 50 DDIM steps. The team uses FP16 training. "
            "One researcher has strong opinions about spaces vs. tabs."
        ),
        "components": ["guidance scale (w)", "inference steps", "model scale", "training data size", "noise schedule"],
        "expected_ablations": [
            "Vary guidance scale (1, 3, 5, 7.5, 10, 15) with fixed steps and model",
            "Vary inference steps (10, 20, 50, 100, 250) with fixed w=7.5",
            "Evaluate a smaller model (500M params) vs. 1B at fixed w and steps",
            "Plot FID vs. CLIP score for each guidance scale to characterise the trade-off curve",
        ],
    },
    {
        "system": "Speech recognition model (Whisper-style encoder-decoder)",
        "hypothesis": "Language model shallow fusion improves WER more on low-resource languages than on English.",
        "context": (
            "Whisper-large fine-tuned on 4 languages: English, Swahili, Welsh, Punjabi. "
            "LM fusion uses 4-gram models trained on Wikipedia text. "
            "WER baseline: EN=4.2%, SW=18.1%, CY=22.4%, PA=31.5%. "
            "Evaluation on CommonVoice v13. The team uses beam search with beam=5. "
            "The office dog is named Byte."
        ),
        "components": ["LM fusion weight (λ)", "LM training data source", "beam size", "language", "fine-tuning data size"],
        "expected_ablations": [
            "No LM fusion vs. 4-gram fusion: per-language WER delta",
            "Vary λ (0.1, 0.3, 0.5, 0.7) per language independently",
            "Replace 4-gram LM with neural LM (GPT-2) while fixing λ=0.3",
            "Reduce Swahili fine-tuning data to 10%/50%/100% to study low-resource effect independently",
        ],
    },
    {
        "system": "Multi-agent debate framework for fact verification",
        "hypothesis": "Adding a dedicated devil's advocate agent improves claim rejection accuracy more than increasing the number of supporting agents.",
        "context": (
            "Framework: 3 supporter agents + 1 judge agent debate claims for 3 rounds. "
            "Evaluated on PolitiFact (1,000 claims). F1 = 0.74 on the 'False' class. "
            "Agents use GPT-4o. The judge uses majority vote with confidence weighting. "
            "Debate prompts are temperature 0.8. The team recently adopted async Python."
        ),
        "components": ["devil's advocate agent (present/absent)", "number of supporter agents", "debate rounds", "judge aggregation method", "agent temperature"],
        "expected_ablations": [
            "With devil's advocate vs. without (3 supporters → 4 supporters), same total agents",
            "Vary supporter count (1, 2, 3, 4) without devil's advocate",
            "Vary debate rounds (1, 2, 3, 5) with and without devil's advocate",
            "Replace majority-vote judge with LLM meta-judge, controlling for agent config",
        ],
    },
    {
        "system": "Tabular data classification with gradient boosting (XGBoost)",
        "hypothesis": "Feature engineering contributes more to AUC than hyperparameter tuning on this dataset.",
        "context": (
            "Dataset: credit card fraud detection (284k transactions, 0.17% fraud rate). "
            "Current model: XGBoost with 100 trees, max_depth=6, learning_rate=0.1. "
            "AUC = 0.979. Features: PCA-transformed V1-V28 + Amount + Time. "
            "No domain-specific features have been engineered yet. "
            "The data scientist has a lucky rubber duck on their desk."
        ),
        "components": ["feature set", "tree count", "max depth", "learning rate", "class imbalance handling"],
        "expected_ablations": [
            "Baseline features vs. baseline + engineered features (e.g., time-of-day, velocity) with fixed hypers",
            "Grid search on max_depth × learning_rate while fixing the raw feature set",
            "Combine best features + best hypers to test additivity",
            "SMOTE vs. class_weight balancing vs. none, controlling for features and hypers",
        ],
    },
    {
        "system": "Large language model chain-of-thought prompting",
        "hypothesis": "Zero-shot CoT ('think step by step') is as effective as few-shot CoT for mathematical reasoning.",
        "context": (
            "Evaluated on MATH benchmark (Level 3-5 problems). "
            "Few-shot CoT: 8 hand-crafted examples with full solution chains. "
            "Zero-shot CoT: single instruction appended. "
            "Model: GPT-4o. Accuracy: few-shot 74%, zero-shot 71%. "
            "The prompts are in English. The team recently moved to a new office."
        ),
        "components": ["prompting strategy", "number of few-shot examples", "problem difficulty level", "model version", "temperature"],
        "expected_ablations": [
            "No CoT vs. zero-shot CoT vs. few-shot CoT (2, 4, 8 examples) in a single table",
            "Stratify all conditions by difficulty level (3, 4, 5) to test interaction",
            "Vary few-shot example quality (random vs. hand-crafted vs. self-generated) at k=8",
            "Replace GPT-4o with GPT-3.5 across all CoT conditions to test model-dependency",
        ],
    },
    {
        "system": "Recommendation system (two-tower model)",
        "hypothesis": "Hard negative mining is more important than tower depth for retrieval quality.",
        "context": (
            "Two-tower: user encoder (MLP, 3 layers) + item encoder (MLP, 3 layers). "
            "Training: in-batch negatives + 5 hard negatives per positive. "
            "Evaluated on MovieLens-1M (Recall@20 = 0.41). "
            "Item features: title embedding + genre one-hot. User features: watch history embedding. "
            "The team uses PyTorch Lightning. A team member recently adopted a puppy."
        ),
        "components": ["negative sampling strategy", "tower depth (layers)", "user feature set", "item feature set", "embedding dimension"],
        "expected_ablations": [
            "In-batch only vs. 5 hard negatives vs. 10 hard negatives (same tower depth)",
            "Vary tower depth (1, 2, 3, 5 layers) with hard negatives fixed at 5",
            "Remove user watch-history embedding while fixing negatives and depth",
            "Vary embedding dimension (64, 128, 256, 512) with all else fixed",
        ],
    },
    {
        "system": "Video understanding model (clip-based contrastive learning)",
        "hypothesis": "Frame sampling strategy affects action recognition accuracy more than clip duration.",
        "context": (
            "Model: CLIP vision encoder + temporal attention. "
            "Evaluated on Kinetics-400 (top-1 accuracy = 81.2%). "
            "Current: 8 frames sampled uniformly from 8-second clips. "
            "The training batch includes 32 clips. "
            "The GPU cluster has intermittent networking issues. "
            "The team lead recently won a chess tournament."
        ),
        "components": ["frame sampling strategy", "clip duration", "number of frames", "temporal attention depth", "backbone frozen/unfrozen"],
        "expected_ablations": [
            "Uniform sampling vs. dense sampling vs. random sampling (same clip duration)",
            "Vary clip duration (4s, 8s, 16s, 32s) with uniform sampling fixed",
            "Vary frame count (4, 8, 16, 32) with uniform strategy and 8s clips",
            "Freeze backbone vs. fine-tune backbone, controlling for frame and sampling config",
        ],
    },
    {
        "system": "Named entity recognition (NER) model",
        "hypothesis": "Pre-training domain specificity matters more than model size for biomedical NER.",
        "context": (
            "Fine-tuned on BC5CDR (chemical and disease entities). "
            "Backbone: BERT-base (110M) pre-trained on general English. "
            "Current F1: 89.3%. BioBERT (same size, biomedical pre-training) achieves 92.1%. "
            "PubMedBERT-large is available but twice the size. "
            "The annotation team uses a custom Prodigy setup."
        ),
        "components": ["pre-training domain", "model size", "fine-tuning data size", "entity span representation", "CRF vs. linear head"],
        "expected_ablations": [
            "BERT-base (general) vs. BioBERT-base (bio) vs. PubMedBERT-base: domain effect at same size",
            "BioBERT-base vs. PubMedBERT-large: disentangle size from domain by including a large general model",
            "Reduce BC5CDR fine-tuning data (100%, 50%, 25%) for each backbone",
            "CRF head vs. linear classification head controlling for backbone",
        ],
    },
    {
        "system": "Robotic manipulation policy (imitation learning)",
        "hypothesis": "Demonstration quality (expert vs. sub-optimal) is more important than demonstration quantity.",
        "context": (
            "Task: block stacking with 6-DoF arm. "
            "Training data: 500 expert demos + 200 sub-optimal demos. "
            "Policy: Diffusion Policy with UNet backbone. "
            "Success rate: 83% on evaluation rollouts. "
            "The robot makes a characteristic beeping sound when initialising. "
            "Lab temperature is maintained at 22°C."
        ),
        "components": ["demonstration quality", "demonstration quantity", "observation modality", "action horizon", "noise schedule"],
        "expected_ablations": [
            "Expert-only (varying N) vs. sub-optimal-only (varying N) vs. mixed",
            "Vary total demo count (100, 250, 500, 700) using only expert demos",
            "Compare image-only vs. proprioception-only vs. image+proprioception observation",
            "Vary action horizon (1, 4, 8, 16 steps) with fixed expert-only 500 demos",
        ],
    },
    {
        "system": "Summarisation model (abstractive, BART-style)",
        "hypothesis": "Length penalty during beam search has negligible effect on ROUGE scores compared to training with length-controlled targets.",
        "context": (
            "BART-large fine-tuned on CNN/DailyMail. ROUGE-L = 44.6. "
            "Current length penalty α=1.0, beam size=4, max summary length=128 tokens. "
            "Training targets are raw reference summaries (unconstrained length). "
            "The team recently switched to a new tokeniser. "
            "The project manager loves pivot tables."
        ),
        "components": ["length penalty (α)", "beam size", "max summary length", "training target length", "model scale"],
        "expected_ablations": [
            "Vary α (0.5, 1.0, 1.5, 2.0) with fixed beam and max length",
            "Train with length-bucketed targets vs. raw targets, then evaluate at same α",
            "Vary max generation length (64, 128, 256 tokens) at fixed α and model",
            "Vary beam size (1, 2, 4, 8) with fixed α and generation length",
        ],
    },
]


def load_abgen(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Load AbGen dataset (ablation study design tasks).

    AbGen is not widely publicly available; this implementation uses a
    curated synthetic pool derived from the benchmark's description in ROMA.

    Returns list of dicts:
      {
        "id":                  str,
        "system":              str,
        "hypothesis":          str,
        "context":             str,
        "components":          List[str],
        "expected_ablations":  List[str]
      }
    """
    pool = _SYNTHETIC_POOL.copy()
    if limit:
        pool = pool[:limit]
    return [
        {
            "id":                 str(i),
            "system":             item["system"],
            "hypothesis":         item["hypothesis"],
            "context":            item["context"],
            "components":         item["components"],
            "expected_ablations": item["expected_ablations"],
        }
        for i, item in enumerate(pool)
    ]
