#!/usr/bin/env python3
"""AI File Discovery — Find AI/ML files consuming disk space."""

import argparse
import csv
import hashlib
import json
import os
import re
import stat as stat_module
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple

try:
    import pwd
    import grp
    _HAS_POSIX_USERS = True
except ImportError:
    _HAS_POSIX_USERS = False

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Data Model
# ---------------------------------------------------------------------------

class Category(Enum):
    DATA = "data"
    MODEL = "model"
    CONFIG = "config"
    VECTOR = "vector"
    CHECKPOINT = "checkpoint"
    SOURCE = "source"
    DOCUMENT = "document"
    MULTIMEDIA = "multimedia"
    SKILL = "skill"
    AGENT = "agent"
    SECRET = "secret"


CATEGORY_COLORS = {
    Category.DATA: "cyan",
    Category.MODEL: "magenta",
    Category.CONFIG: "green",
    Category.VECTOR: "blue",
    Category.CHECKPOINT: "yellow",
    Category.SOURCE: "bright_white",
    Category.DOCUMENT: "bright_cyan",
    Category.MULTIMEDIA: "bright_magenta",
    Category.SKILL: "bright_yellow",
    Category.AGENT: "bright_green",
    Category.SECRET: "red",
}


@dataclass
class FileResult:
    path: str
    filename: str
    extension: str
    categories: List[Category]
    size_bytes: int
    modified_timestamp: float
    is_directory: bool = False
    detection_method: str = "extension"  # extension | heuristic | pattern
    confidence: str = "high"  # high | medium | low
    risk_level: str = "none"  # none | warning | danger
    risk_reason: str = ""
    owner: str = ""
    group: str = ""
    permissions: str = ""
    permission_alert: str = ""  # "" | "world_readable" | "world_writable" | "group_writable" | "too_open"
    permission_alert_reason: str = ""
    hash: str = ""


@dataclass
class ScanConfig:
    root_path: str
    max_depth: Optional[int] = None
    min_size_bytes: int = 0
    categories: Optional[List[str]] = None
    follow_symlinks: bool = False
    include_hidden: bool = False
    quiet: bool = False
    compute_permissions: bool = False
    compute_hashes: bool = False


@dataclass
class ScanSummary:
    scan_path: str = ""
    scan_time: str = ""
    total_files: int = 0
    total_size: int = 0
    by_category: Dict[str, dict] = field(default_factory=dict)
    by_extension: Dict[str, dict] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    scan_duration: float = 0.0


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def format_size(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} B"
    for unit in ("KB", "MB", "GB", "TB"):
        size_bytes /= 1024
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
    return f"{size_bytes:.1f} PB"


def parse_size(size_str: str) -> int:
    size_str = size_str.strip().upper()
    multipliers = {
        "B": 1, "K": 1024, "KB": 1024,
        "M": 1024 ** 2, "MB": 1024 ** 2,
        "G": 1024 ** 3, "GB": 1024 ** 3,
        "T": 1024 ** 4, "TB": 1024 ** 4,
    }
    match = re.match(r"^(\d+(?:\.\d+)?)\s*([A-Z]{0,2})$", size_str)
    if not match:
        raise ValueError(f"Invalid size format: {size_str!r}. Examples: 10MB, 1GB, 500KB")
    value = float(match.group(1))
    suffix = match.group(2) or "B"
    if suffix not in multipliers:
        raise ValueError(f"Unknown size unit: {suffix!r}")
    return int(value * multipliers[suffix])


def format_timestamp(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def _get_permissions(path: str) -> Tuple[str, str, str]:
    """Return (owner, group, permissions) for a path."""
    try:
        st = os.stat(path)
        perms = stat_module.filemode(st.st_mode)
        if _HAS_POSIX_USERS:
            try:
                owner = pwd.getpwuid(st.st_uid).pw_name
            except KeyError:
                owner = str(st.st_uid)
            try:
                group_name = grp.getgrgid(st.st_gid).gr_name
            except KeyError:
                group_name = str(st.st_gid)
        else:
            owner = str(st.st_uid)
            group_name = str(st.st_gid)
        return (owner, group_name, perms)
    except OSError:
        return ("", "", "")


# Categories where world-readable is a real concern (credentials, keys, prompts)
_CONFIDENTIAL_CATEGORIES = {Category.SECRET, Category.AGENT}

# Categories where group-writable / world-writable is a concern (any AI asset)
_SENSITIVE_CATEGORIES = {
    Category.MODEL, Category.SECRET, Category.CHECKPOINT,
    Category.CONFIG, Category.AGENT, Category.DATA, Category.VECTOR,
}


def _check_permission_alert(
    perms: str, categories: List[Category],
) -> Tuple[str, str]:
    """Check if file permissions are too open for sensitive AI files.

    Returns (alert_code, reason) or ("", "") if OK.

    Severity levels:
      - world_writable:  anyone can modify the file (always flagged)
      - world_readable:  anyone can read secrets/agent prompts (chmod 600)
      - group_writable:  group can modify sensitive AI files
      - too_open:        world-executable on non-source files
    """
    if not perms or len(perms) < 10:
        return ("", "")

    # Parse: drwxrwxrwx  (0=type, 1-3=owner, 4-6=group, 7-9=others)
    others_r = perms[7] == "r"
    others_w = perms[8] == "w"
    others_x = perms[9] not in ("-", "S", "T")
    group_w = perms[5] == "w"

    is_sensitive = bool(set(categories) & _SENSITIVE_CATEGORIES)
    is_confidential = bool(set(categories) & _CONFIDENTIAL_CATEGORIES)

    # World-writable is always dangerous
    if others_w:
        return ("world_writable", "World-writable — anyone on the system can modify this file")

    # World-readable only flagged for truly confidential files (secrets, agent prompts)
    if others_r and is_confidential:
        if Category.SECRET in categories:
            return ("world_readable", "Secret file is world-readable — restrict to owner only (chmod 600)")
        return ("world_readable", "Agent/prompt file is world-readable — consider restricting access (chmod 640)")

    # Group-writable on sensitive AI files
    if group_w and is_sensitive:
        return ("group_writable", "Sensitive AI file is group-writable — any group member can modify it")

    # World-executable on non-source, non-directory files
    is_dir = perms[0] == "d"
    if others_x and not is_dir and Category.SOURCE not in categories and is_sensitive:
        return ("too_open", "Non-source AI file has world-execute permission — unusual, review permissions")

    return ("", "")


def _compute_file_hash(path: str, chunk_size: int = 65536) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return ""


def _compute_dir_hash(path: str) -> str:
    """Compute SHA-256 hash of a directory (hash of sorted file hashes)."""
    file_hashes = []
    try:
        for dirpath, _, filenames in os.walk(path):
            for fname in sorted(filenames):
                fpath = os.path.join(dirpath, fname)
                rel = os.path.relpath(fpath, path)
                fhash = _compute_file_hash(fpath)
                if fhash:
                    file_hashes.append(f"{rel}:{fhash}")
        file_hashes.sort()
        return hashlib.sha256("\n".join(file_hashes).encode()).hexdigest()
    except OSError:
        return ""


# ---------------------------------------------------------------------------
# File Classifier
# ---------------------------------------------------------------------------

class FileClassifier:
    EXTENSION_MAP: Dict[str, List[Category]] = {
        # Data files — tabular & structured
        ".jsonl": [Category.DATA],
        ".csv": [Category.DATA],
        ".tsv": [Category.DATA],
        ".parquet": [Category.DATA],
        ".arrow": [Category.DATA],
        ".feather": [Category.DATA],
        ".hdf5": [Category.DATA],
        ".h5": [Category.DATA],
        ".tfrecord": [Category.DATA],
        ".npy": [Category.DATA],
        ".npz": [Category.DATA],
        # Data files — additional formats (Apache ecosystem, scientific)
        ".avro": [Category.DATA],
        ".orc": [Category.DATA],
        ".petastorm": [Category.DATA],
        ".nc": [Category.DATA],           # NetCDF scientific data
        ".netcdf": [Category.DATA],
        ".zarr": [Category.DATA],          # Zarr chunked arrays
        ".loom": [Category.DATA],          # Single-cell genomics
        ".anndata": [Category.DATA],       # AnnData (scanpy)
        ".beton": [Category.DATA],         # FFCV fast data loading
        ".mindrecord": [Category.DATA],    # MindSpore dataset
        ".recordio": [Category.DATA],      # MXNet RecordIO
        ".idx": [Category.DATA],           # IDX format (MNIST)
        ".shard": [Category.DATA],         # WebDataset shards
        ".tar.gz": [Category.DATA],        # Common dataset archives
        ".wds": [Category.DATA],           # WebDataset
        # Model files — core formats
        ".onnx": [Category.MODEL],
        ".pt": [Category.MODEL],
        ".pth": [Category.MODEL],
        ".safetensors": [Category.MODEL],
        ".gguf": [Category.MODEL],
        ".ggml": [Category.MODEL],
        ".pkl": [Category.MODEL, Category.DATA],
        ".pickle": [Category.MODEL, Category.DATA],
        ".joblib": [Category.MODEL, Category.DATA],
        ".pb": [Category.MODEL, Category.CONFIG],
        ".tflite": [Category.MODEL],
        ".mlmodel": [Category.MODEL],
        ".mar": [Category.MODEL],
        # Model files — additional formats
        ".pte": [Category.MODEL],          # ExecuTorch mobile/edge
        ".ckpt": [Category.MODEL, Category.CHECKPOINT],  # Checkpoint/model hybrid
        ".engine": [Category.MODEL],       # TensorRT engine
        ".plan": [Category.MODEL],         # TensorRT plan
        ".torchscript": [Category.MODEL],  # TorchScript serialized
        ".savedmodel": [Category.MODEL],   # TF SavedModel file
        ".h5model": [Category.MODEL],      # Keras model file
        ".keras": [Category.MODEL],        # Keras v3 native
        ".bento": [Category.MODEL, Category.CONFIG],  # BentoML artifact
        ".paddle": [Category.MODEL],       # PaddlePaddle
        ".pdparams": [Category.MODEL],     # PaddlePaddle params
        ".pdmodel": [Category.MODEL],      # PaddlePaddle model
        ".mnn": [Category.MODEL],          # MNN mobile inference
        ".ncnn": [Category.MODEL],         # NCNN mobile inference
        ".xmodel": [Category.MODEL],       # Vitis AI / Xilinx
        ".om": [Category.MODEL],           # Huawei CANN / MindSpore
        ".air": [Category.MODEL],          # MindSpore AIR format
        ".mindir": [Category.MODEL],       # MindSpore MindIR
        ".ort": [Category.MODEL],          # ONNX Runtime optimized
        ".mlpackage": [Category.MODEL],    # CoreML package
        ".pmml": [Category.MODEL, Category.CONFIG],  # Predictive Model Markup
        # Config + Source (notebooks contain code)
        ".ipynb": [Category.CONFIG, Category.SOURCE],
        # Vector / embedding stores
        ".faiss": [Category.VECTOR],
        ".annoy": [Category.VECTOR],
        ".index": [Category.VECTOR],
        ".lancedb": [Category.VECTOR],
        # Vector — additional stores
        ".hnsw": [Category.VECTOR],        # HNSW index files
        ".usearch": [Category.VECTOR],     # USearch vector index
        ".voy": [Category.VECTOR],         # Voy WASM vector search
        ".ivfpq": [Category.VECTOR],       # FAISS IVF-PQ index
        ".scann": [Category.VECTOR],       # Google ScaNN index
        ".hnswlib": [Category.VECTOR],     # hnswlib index
        ".nmslib": [Category.VECTOR],      # NMSLIB index
        ".milvus": [Category.VECTOR],      # Milvus segment
        # Source code
        ".py": [Category.SOURCE],
        ".js": [Category.SOURCE],
        ".ts": [Category.SOURCE],
        ".jsx": [Category.SOURCE],
        ".tsx": [Category.SOURCE],
        ".java": [Category.SOURCE],
        ".cpp": [Category.SOURCE],
        ".c": [Category.SOURCE],
        ".h": [Category.SOURCE],
        ".go": [Category.SOURCE],
        ".rs": [Category.SOURCE],
        ".rb": [Category.SOURCE],
        ".r": [Category.SOURCE],
        ".jl": [Category.SOURCE],
        ".scala": [Category.SOURCE],
        ".kt": [Category.SOURCE],
        ".swift": [Category.SOURCE],
        ".sh": [Category.SOURCE],
        ".bash": [Category.SOURCE],
        ".lua": [Category.SOURCE],
        # Documents — markup & text
        ".md": [Category.DOCUMENT],
        ".markdown": [Category.DOCUMENT],
        ".rst": [Category.DOCUMENT],
        ".tex": [Category.DOCUMENT],
        ".adoc": [Category.DOCUMENT],
        ".log": [Category.DOCUMENT],
        # Documents — office & publishing
        ".pdf": [Category.DOCUMENT],
        ".docx": [Category.DOCUMENT],
        ".doc": [Category.DOCUMENT],
        ".pptx": [Category.DOCUMENT],
        ".ppt": [Category.DOCUMENT],
        ".xlsx": [Category.DOCUMENT, Category.DATA],
        ".xls": [Category.DOCUMENT, Category.DATA],
        ".odt": [Category.DOCUMENT],
        ".rtf": [Category.DOCUMENT],
        ".epub": [Category.DOCUMENT],
        # Documents — data interchange & web
        ".html": [Category.DOCUMENT],
        ".htm": [Category.DOCUMENT],
        ".xml": [Category.DOCUMENT, Category.CONFIG],
        ".css": [Category.DOCUMENT],
        # Documents — annotations & labels
        ".ann": [Category.DOCUMENT, Category.DATA],
        ".conll": [Category.DOCUMENT, Category.DATA],
        ".brat": [Category.DOCUMENT, Category.DATA],
        ".bio": [Category.DOCUMENT, Category.DATA],
        # Documents — notebooks & literate programming
        ".rmd": [Category.DOCUMENT, Category.SOURCE],
        ".qmd": [Category.DOCUMENT, Category.SOURCE],
        # Multimedia — images (raster)
        ".png": [Category.MULTIMEDIA],
        ".jpg": [Category.MULTIMEDIA],
        ".jpeg": [Category.MULTIMEDIA],
        ".gif": [Category.MULTIMEDIA],
        ".bmp": [Category.MULTIMEDIA],
        ".tiff": [Category.MULTIMEDIA],
        ".tif": [Category.MULTIMEDIA],
        ".webp": [Category.MULTIMEDIA],
        ".ico": [Category.MULTIMEDIA],
        ".heic": [Category.MULTIMEDIA],
        ".heif": [Category.MULTIMEDIA],
        ".avif": [Category.MULTIMEDIA],
        ".jxl": [Category.MULTIMEDIA],
        # Multimedia — images (vector & design)
        ".svg": [Category.MULTIMEDIA],
        ".eps": [Category.MULTIMEDIA],
        ".ai": [Category.MULTIMEDIA],
        ".psd": [Category.MULTIMEDIA],
        ".xcf": [Category.MULTIMEDIA],
        ".sketch": [Category.MULTIMEDIA],
        ".fig": [Category.MULTIMEDIA],
        # Multimedia — images (scientific & ML)
        ".dcm": [Category.MULTIMEDIA, Category.DATA],
        ".dicom": [Category.MULTIMEDIA, Category.DATA],
        ".nii": [Category.MULTIMEDIA, Category.DATA],
        ".nii.gz": [Category.MULTIMEDIA, Category.DATA],
        ".exr": [Category.MULTIMEDIA],
        ".hdr": [Category.MULTIMEDIA],
        ".raw": [Category.MULTIMEDIA],
        # Multimedia — audio
        ".wav": [Category.MULTIMEDIA],
        ".mp3": [Category.MULTIMEDIA],
        ".flac": [Category.MULTIMEDIA],
        ".ogg": [Category.MULTIMEDIA],
        ".m4a": [Category.MULTIMEDIA],
        ".aac": [Category.MULTIMEDIA],
        ".opus": [Category.MULTIMEDIA],
        ".wma": [Category.MULTIMEDIA],
        ".aiff": [Category.MULTIMEDIA],
        ".aif": [Category.MULTIMEDIA],
        ".mid": [Category.MULTIMEDIA],
        ".midi": [Category.MULTIMEDIA],
        # Multimedia — video
        ".mp4": [Category.MULTIMEDIA],
        ".avi": [Category.MULTIMEDIA],
        ".mov": [Category.MULTIMEDIA],
        ".mkv": [Category.MULTIMEDIA],
        ".webm": [Category.MULTIMEDIA],
        ".wmv": [Category.MULTIMEDIA],
        ".flv": [Category.MULTIMEDIA],
        ".m4v": [Category.MULTIMEDIA],
        ".mpg": [Category.MULTIMEDIA],
        ".mpeg": [Category.MULTIMEDIA],
        ".mts": [Category.MULTIMEDIA],       # MPEG transport stream (not .ts — conflicts with TypeScript)
        # Multimedia — 3D & point cloud (ML/CV)
        ".obj": [Category.MULTIMEDIA],
        ".stl": [Category.MULTIMEDIA],
        ".ply": [Category.MULTIMEDIA, Category.DATA],
        ".pcd": [Category.MULTIMEDIA, Category.DATA],
        ".glb": [Category.MULTIMEDIA],
        ".gltf": [Category.MULTIMEDIA],
        ".fbx": [Category.MULTIMEDIA],
        ".usdz": [Category.MULTIMEDIA],
        # Multimedia — subtitles & captions (NLP/speech)
        ".srt": [Category.MULTIMEDIA, Category.DATA],
        ".vtt": [Category.MULTIMEDIA, Category.DATA],
        ".ass": [Category.MULTIMEDIA],
        ".sub": [Category.MULTIMEDIA],
        # Skills (deterministic)
        ".skill": [Category.SKILL],
        ".tool": [Category.SKILL],
        # Agent orchestration
        ".prompt": [Category.AGENT, Category.CONFIG],
        ".systemprompt": [Category.AGENT, Category.CONFIG],
        ".goal": [Category.AGENT, Category.CONFIG],
        ".persona": [Category.AGENT, Category.CONFIG],
        ".instruction": [Category.AGENT, Category.CONFIG],
        # Secret / credential files
        ".pem": [Category.SECRET],
        ".key": [Category.SECRET],
        ".p12": [Category.SECRET],
        ".pfx": [Category.SECRET],
        ".keystore": [Category.SECRET],
        ".jks": [Category.SECRET],
        ".credentials": [Category.SECRET],
        ".secret": [Category.SECRET],
        ".token": [Category.SECRET],
        ".htpasswd": [Category.SECRET],
        ".pgpass": [Category.SECRET],
        ".netrc": [Category.SECRET],
    }

    AMBIGUOUS_EXTENSIONS = {".yaml", ".yml", ".bin", ".db", ".sqlite", ".sqlite3", ".json"}

    AI_YAML_KEYWORDS = [
        "model", "training", "epochs", "batch_size", "learning_rate", "lr",
        "optimizer", "loss", "dataset", "tokenizer", "transformer",
        "attention", "embedding", "hidden_size", "num_layers", "num_heads",
        "dropout", "weight_decay", "warmup", "scheduler", "checkpoint",
        "pretrained", "fine_tune", "finetune", "backbone", "encoder", "decoder",
        "vocab_size", "max_length", "max_seq_length", "gradient",
        "wandb", "tensorboard", "huggingface", "torch", "tensorflow",
        "diffusion", "lora", "qlora", "peft", "rlhf", "sft",
        "inference", "quantization", "pruning", "distillation",
        "pipeline", "feature_extractor", "image_processor",
        "prompt", "completion", "chat_template", "system_prompt",
    ]

    SKILL_KEYWORDS = [
        "tool_use", "function_call", "actions", "plugin",
        "skill", "capability", "langchain", "autogen", "crewai",
        "openai_function", "agent", "tool_choice",
        "langgraph", "dspy", "pydantic_ai", "smolagents",
    ]

    SKILL_FILENAME_REGEX = re.compile(
        r"(agent|plugin|tool_def|skill)", re.IGNORECASE
    )

    CHECKPOINT_REGEX = re.compile(
        r"^(checkpoint|model|epoch|step|ckpt|run|experiment)"
        r"[-_](\d+|best|latest|final|last)$",
        re.IGNORECASE,
    )

    # Agent detection
    AGENT_KEYWORDS = [
        "system_prompt", "you are", "your role", "assistant",
        "persona", "goal", "objective", "instruction",
        "behave as", "act as", "respond as",
        "rules:", "guidelines:", "constraints:",
        # CrewAI / agentic framework keywords
        "backstory", "expected_output", "crew", "crewai",
        "langchain", "langgraph", "autogen", "dspy",
        "agent_executor", "tool_node", "state_graph",
    ]

    AGENT_FILENAME_REGEX = re.compile(
        r"(system.?prompt|agent.?config|goal|persona|instruction|rules|prompt.?template"
        r"|crew.?config|task.?config|agent.?def|workflow.?def|chain.?config)",
        re.IGNORECASE,
    )

    AGENT_SPECIAL_FILENAMES = {
        "system_prompt", "systemprompt", "system-prompt",
        "agent_config", "agentconfig", "agent-config",
        "goals", "persona", "instructions", "rules",
        # CrewAI / agentic frameworks
        "agents", "tasks", "crew",
    }

    AGENT_EXACT_FILENAMES = {
        ".cursorrules", ".claude", "AGENTS.md",
        "OAI_CONFIG_LIST",      # AutoGen config
        "MLmodel",              # MLflow model definition
    }

    # Secret detection
    SECRET_ENV_PATTERN = re.compile(r"^\.env(\..+)?$", re.IGNORECASE)

    SECRET_FILENAME_PATTERNS = re.compile(
        r"^(credentials\.(json|yaml|yml)|"
        r"secrets?\.(json|yaml|yml|toml)|"
        r"service[-_]account\.json|"
        r"id_(rsa|ed25519|ecdsa)(\.pub)?|"
        r"\.npmrc|\.pypirc|\.htpasswd|\.pgpass|\.netrc)$",
        re.IGNORECASE,
    )

    SECRET_CONTENT_PATTERNS = [
        re.compile(r"(?i)(api[_-]?key|apikey)\s*[:=]\s*['\"]?\w{16,}"),
        re.compile(r"(?i)(secret|password|passwd|pwd)\s*[:=]\s*['\"]?\S{8,}"),
        re.compile(r"(?i)(access[_-]?token|auth[_-]?token|bearer)\s*[:=]\s*['\"]?\S{16,}"),
        re.compile(r"AKIA[0-9A-Z]{16}"),
        re.compile(r"(ghp_|gho_|ghs_|ghr_)[a-zA-Z0-9]{36}"),
        re.compile(r"sk-[a-zA-Z0-9]{20,}"),
        re.compile(r"-----BEGIN\s+(RSA\s+)?PRIVATE\s+KEY-----"),
        re.compile(r"(?i)(mongodb\+srv|postgres|mysql)://[^@\s]+@"),
    ]

    BINARY_CATEGORIES = {Category.MODEL, Category.MULTIMEDIA, Category.CHECKPOINT, Category.VECTOR}

    # Risk configuration for extensions
    RISK_MAP: Dict[str, tuple] = {
        ".pkl": ("danger", "Pickle deserialization can execute arbitrary code"),
        ".pickle": ("danger", "Pickle deserialization can execute arbitrary code"),
        ".joblib": ("warning", "Joblib uses pickle internally — similar risk"),
        ".mar": ("warning", "Model archives may contain pickled components"),
        ".bin": ("warning", "Binary model file — may use pickle serialization"),
        ".ggml": ("warning", "Legacy format — consider converting to GGUF"),
        # Secret files
        ".pem": ("danger", "Secret/credential file — contains sensitive data"),
        ".key": ("danger", "Secret/credential file — contains sensitive data"),
        ".p12": ("danger", "Secret/credential file — contains sensitive data"),
        ".pfx": ("danger", "Secret/credential file — contains sensitive data"),
        ".keystore": ("danger", "Secret/credential file — contains sensitive data"),
        ".jks": ("danger", "Secret/credential file — contains sensitive data"),
        ".credentials": ("danger", "Secret/credential file — contains sensitive data"),
        ".secret": ("danger", "Secret/credential file — contains sensitive data"),
        ".token": ("danger", "Secret/credential file — contains sensitive data"),
        ".htpasswd": ("danger", "Secret/credential file — contains sensitive data"),
        ".pgpass": ("danger", "Secret/credential file — contains sensitive data"),
        ".netrc": ("danger", "Secret/credential file — contains sensitive data"),
    }

    # Confidence for ambiguous extensions
    AMBIGUOUS_CONFIDENCE = {
        ".yaml": "medium",
        ".yml": "medium",
        ".bin": "medium",
        ".db": "medium",
        ".sqlite": "medium",
        ".sqlite3": "medium",
        ".pkl": "medium",
        ".pickle": "medium",
        ".index": "medium",
    }

    # Default config file locations (checked in order)
    CONFIG_FILENAMES = ["aifiles.config.json", ".aifiles.json"]

    def __init__(self, config_path: Optional[str] = None) -> None:
        """Initialise classifier, optionally merging a user config file.

        Config is looked up in this order:
          1. Explicit *config_path* argument
          2. ``aifiles.config.json`` in the current working directory
          3. ``.aifiles.json`` in the current working directory
          4. ``~/.aifiles.json`` in the user's home directory

        If no config file is found the built-in defaults are used as-is.
        """
        # Start with class-level defaults (shallow copies so mutations are per-instance)
        self.EXTENSION_MAP = dict(self.__class__.EXTENSION_MAP)
        self.RISK_MAP = dict(self.__class__.RISK_MAP)
        self.AMBIGUOUS_EXTENSIONS = set(self.__class__.AMBIGUOUS_EXTENSIONS)
        self.AMBIGUOUS_CONFIDENCE = dict(self.__class__.AMBIGUOUS_CONFIDENCE)

        # Locate config
        cfg_path = self._find_config(config_path)
        if cfg_path:
            self._load_config(cfg_path)

    def _find_config(self, explicit: Optional[str] = None) -> Optional[str]:
        """Return the path to the config file, or None."""
        if explicit:
            if os.path.isfile(explicit):
                return explicit
            return None
        # Check CWD then HOME
        for name in self.CONFIG_FILENAMES:
            for directory in [os.getcwd(), os.path.expanduser("~")]:
                candidate = os.path.join(directory, name)
                if os.path.isfile(candidate):
                    return candidate
        return None

    def _load_config(self, path: str) -> None:
        """Merge a JSON config file into the classifier's maps."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            sys.stderr.write(f"  Warning: failed to load config {path}: {e}\n")
            return

        # --- extensions: map ext → [category, ...] ---
        for ext, cats in cfg.get("extensions", {}).items():
            ext = ext if ext.startswith(".") else f".{ext}"
            ext = ext.lower()
            try:
                cat_list = [Category(c.lower()) for c in (cats if isinstance(cats, list) else [cats])]
            except ValueError as e:
                sys.stderr.write(f"  Warning: unknown category in config for {ext}: {e}\n")
                continue
            self.EXTENSION_MAP[ext] = cat_list

        # --- risks: map ext → {level, reason} ---
        for ext, risk in cfg.get("risks", {}).items():
            ext = ext if ext.startswith(".") else f".{ext}"
            ext = ext.lower()
            level = risk.get("level", "warning") if isinstance(risk, dict) else risk
            reason = risk.get("reason", "Custom risk rule") if isinstance(risk, dict) else "Custom risk rule"
            if level not in ("danger", "warning"):
                sys.stderr.write(f"  Warning: invalid risk level '{level}' for {ext} — use 'danger' or 'warning'\n")
                continue
            self.RISK_MAP[ext] = (level, reason)

        # --- ambiguous: list of extensions that need heuristic classification ---
        for ext in cfg.get("ambiguous", []):
            ext = ext if ext.startswith(".") else f".{ext}"
            self.AMBIGUOUS_EXTENSIONS.add(ext.lower())

        # --- Remove extensions (if user wants to un-classify something) ---
        for ext in cfg.get("ignore_extensions", []):
            ext = ext if ext.startswith(".") else f".{ext}"
            ext = ext.lower()
            self.EXTENSION_MAP.pop(ext, None)
            self.RISK_MAP.pop(ext, None)
            self.AMBIGUOUS_EXTENSIONS.discard(ext)

    def classify_file(self, path: str, entry: os.DirEntry) -> Optional[FileResult]:
        name = entry.name
        ext = os.path.splitext(name)[1].lower()

        # Special filename matches (SKILL.md)
        if name == "SKILL.md":
            try:
                stat = entry.stat(follow_symlinks=False)
            except OSError:
                return None
            return FileResult(
                path=path, filename=name, extension=".md",
                categories=[Category.SKILL, Category.DOCUMENT],
                size_bytes=stat.st_size,
                modified_timestamp=stat.st_mtime,
                detection_method="filename", confidence="high",
            )

        # Agent special filename matches
        name_lower = name.lower()
        stem = os.path.splitext(name_lower)[0]
        if stem in self.AGENT_SPECIAL_FILENAMES or name in self.AGENT_EXACT_FILENAMES:
            # Ambiguous stems only classify as agent for specific extensions
            if stem in ("goals", "rules", "agents", "tasks", "crew"):
                if ext not in (".yaml", ".yml", ".json", ".md", ".txt"):
                    pass  # fall through to normal classification
                else:
                    return self._make_agent_result(path, entry, name, ext)
            else:
                return self._make_agent_result(path, entry, name, ext)

        # Secret filename matches (.env, .env.local, credentials.json, etc.)
        if self.SECRET_ENV_PATTERN.match(name) or self.SECRET_FILENAME_PATTERNS.match(name):
            try:
                stat = entry.stat(follow_symlinks=False)
            except OSError:
                return None
            return FileResult(
                path=path, filename=name, extension=ext or name,
                categories=[Category.SECRET],
                size_bytes=stat.st_size,
                modified_timestamp=stat.st_mtime,
                detection_method="filename", confidence="high",
                risk_level="danger",
                risk_reason="Secret/credential file — contains sensitive data",
            )

        # Deterministic match
        if ext in self.EXTENSION_MAP and ext not in self.AMBIGUOUS_EXTENSIONS:
            try:
                stat = entry.stat(follow_symlinks=False)
            except OSError:
                return None
            confidence = self.AMBIGUOUS_CONFIDENCE.get(ext, "high")
            result = FileResult(
                path=path,
                filename=name,
                extension=ext,
                categories=list(self.EXTENSION_MAP[ext]),
                size_bytes=stat.st_size,
                modified_timestamp=stat.st_mtime,
                confidence=confidence,
            )
            # Apply risk flags
            if ext in self.RISK_MAP:
                result.risk_level, result.risk_reason = self.RISK_MAP[ext]
            return result

        # Ambiguous — run heuristics
        if ext in self.AMBIGUOUS_EXTENSIONS:
            return self._classify_ambiguous(path, entry, ext)

        # Check .txt files for agent heuristic
        if ext == ".txt" and self._check_agent_heuristic(path, name):
            try:
                stat = entry.stat(follow_symlinks=False)
            except OSError:
                return None
            return FileResult(
                path=path, filename=name, extension=ext,
                categories=[Category.AGENT, Category.DOCUMENT],
                size_bytes=stat.st_size,
                modified_timestamp=stat.st_mtime,
                detection_method="heuristic", confidence="medium",
            )

        return None

    def _classify_ambiguous(
        self, path: str, entry: os.DirEntry, ext: str
    ) -> Optional[FileResult]:
        try:
            stat = entry.stat(follow_symlinks=False)
        except OSError:
            return None

        detected = False
        categories: List[Category] = []
        confidence = "medium"
        method = "heuristic"
        risk_level = "none"
        risk_reason = ""

        if ext in (".yaml", ".yml"):
            # Check agent heuristic first (most specific)
            if self._check_agent_heuristic(path, entry.name):
                detected = True
                categories = [Category.AGENT, Category.CONFIG]
            elif self._check_skill_heuristic(path, entry.name):
                detected = True
                categories = [Category.SKILL, Category.CONFIG]
            else:
                detected = self._check_yaml_heuristic(path)
                categories = [Category.CONFIG]
        elif ext == ".json":
            if self._check_agent_heuristic(path, entry.name):
                detected = True
                categories = [Category.AGENT, Category.CONFIG]
            elif self._check_skill_heuristic(path, entry.name):
                detected = True
                categories = [Category.SKILL, Category.CONFIG]
        elif ext == ".bin":
            detected = self._check_bin_heuristic(path, stat.st_size)
            categories = [Category.MODEL]
            if detected:
                risk_level = "warning"
                risk_reason = "Unknown binary format may contain unsafe serialized data"
        elif ext in (".db", ".sqlite", ".sqlite3"):
            db_type = self._check_db_heuristic(path)
            if db_type == "vector":
                detected = True
                categories = [Category.VECTOR]
                confidence = "medium"
            elif db_type == "data":
                detected = True
                categories = [Category.DATA]
                confidence = "medium"

        if not detected:
            return None

        return FileResult(
            path=path,
            filename=entry.name,
            extension=ext,
            categories=categories,
            size_bytes=stat.st_size,
            modified_timestamp=stat.st_mtime,
            detection_method=method,
            confidence=confidence,
            risk_level=risk_level,
            risk_reason=risk_reason,
        )

    def _check_skill_heuristic(self, path: str, name: str) -> bool:
        if not self.SKILL_FILENAME_REGEX.search(name):
            return False
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read(2048).lower()
            matches = sum(1 for kw in self.SKILL_KEYWORDS if kw in content)
            return matches >= 2
        except OSError:
            return False

    def _check_yaml_heuristic(self, path: str) -> bool:
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read(4096).lower()
            matches = sum(1 for kw in self.AI_YAML_KEYWORDS if kw in content)
            return matches >= 2
        except OSError:
            return False

    def _check_bin_heuristic(self, path: str, size: int) -> bool:
        try:
            with open(path, "rb") as f:
                header = f.read(4)
        except OSError:
            return False

        # GGUF magic
        if header == b"GGUF":
            return True
        # GGML magic
        if header == b"\x67\x67\x6d\x6c":
            return True
        # PyTorch ZIP format (PK header)
        if header[:2] == b"PK":
            return True

        # Large .bin next to HuggingFace config files
        if size > 100 * 1024 * 1024:
            parent = os.path.dirname(path)
            try:
                siblings = set(os.listdir(parent))
            except OSError:
                return False
            model_indicators = {
                "config.json", "tokenizer.json",
                "tokenizer_config.json", "generation_config.json",
            }
            if model_indicators & siblings:
                return True

        return False

    VECTOR_DB_INDICATORS = [
        b"embedding", b"vector", b"chroma", b"langchain",
        b"collection", b"faiss", b"annoy", b"lance",
    ]

    def _check_db_heuristic(self, path: str) -> Optional[str]:
        """Classify a .db/.sqlite/.sqlite3 file by inspecting header and content.

        Returns:
            "vector" — SQLite with vector DB indicators
            "data"   — SQLite without vector indicators (regular database)
            None     — not a SQLite file at all
        """
        try:
            with open(path, "rb") as f:
                header = f.read(16)
        except OSError:
            return None

        # SQLite magic: "SQLite format 3\000"
        if not header.startswith(b"SQLite format 3"):
            return None

        # Scan first 64KB for vector DB table indicators
        try:
            with open(path, "rb") as f:
                chunk = f.read(65536).lower()
            if any(ind in chunk for ind in self.VECTOR_DB_INDICATORS):
                return "vector"
            return "data"
        except OSError:
            return "data"

    def _make_agent_result(self, path: str, entry: os.DirEntry, name: str, ext: str) -> Optional[FileResult]:
        try:
            stat = entry.stat(follow_symlinks=False)
        except OSError:
            return None
        if ext in (".md", ".txt"):
            categories = [Category.AGENT, Category.DOCUMENT]
        else:
            categories = [Category.AGENT, Category.CONFIG]
        return FileResult(
            path=path, filename=name, extension=ext,
            categories=categories,
            size_bytes=stat.st_size,
            modified_timestamp=stat.st_mtime,
            detection_method="filename", confidence="high",
        )

    def _check_agent_heuristic(self, path: str, name: str) -> bool:
        if not self.AGENT_FILENAME_REGEX.search(name):
            return False
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read(2048).lower()
            matches = sum(1 for kw in self.AGENT_KEYWORDS if kw in content)
            return matches >= 2
        except OSError:
            return False

    def apply_secret_scan(self, result: FileResult) -> None:
        """Post-classification: check for embedded secrets in text-based files."""
        if result.risk_level == "danger":
            return
        if set(result.categories) & self.BINARY_CATEGORIES:
            return
        if self._check_secret_content(result.path):
            result.risk_level = "danger"
            result.risk_reason = "Possible embedded secret/credential detected"

    def _check_secret_content(self, path: str) -> bool:
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read(4096)
            return any(pat.search(content) for pat in self.SECRET_CONTENT_PATTERNS)
        except OSError:
            return False

    def classify_directory(self, path: str, entry: os.DirEntry) -> Optional[FileResult]:
        if not self.CHECKPOINT_REGEX.match(entry.name):
            return None

        dir_size = _get_dir_size(path)
        try:
            stat = entry.stat(follow_symlinks=False)
        except OSError:
            return None

        return FileResult(
            path=path,
            filename=entry.name,
            extension="",
            categories=[Category.CHECKPOINT],
            size_bytes=dir_size,
            modified_timestamp=stat.st_mtime,
            is_directory=True,
            detection_method="pattern",
            confidence="high",
        )


def _get_dir_size(path: str) -> int:
    total = 0
    try:
        for dirpath, _dirnames, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                try:
                    total += os.path.getsize(fp)
                except OSError:
                    pass
    except OSError:
        pass
    return total


# ---------------------------------------------------------------------------
# Progress Indicators
# ---------------------------------------------------------------------------

class ProgressIndicator:
    def update(self, path: str) -> None:
        pass

    def finish(self) -> None:
        pass


class SimpleProgress(ProgressIndicator):
    SPINNER = "|/-\\"

    def __init__(self) -> None:
        self._count = 0
        self._last_update = 0.0
        self._spin_idx = 0

    def update(self, path: str) -> None:
        self._count += 1
        now = time.monotonic()
        if now - self._last_update < 0.1:
            return
        self._last_update = now
        self._spin_idx = (self._spin_idx + 1) % 4
        display_path = path
        if len(display_path) > 60:
            display_path = "..." + display_path[-57:]
        sys.stderr.write(
            f"\r  {self.SPINNER[self._spin_idx]} Scanned {self._count} items | {display_path:<60s}"
        )
        sys.stderr.flush()

    def finish(self) -> None:
        sys.stderr.write("\r" + " " * 80 + "\r")
        sys.stderr.flush()


# ---------------------------------------------------------------------------
# Scanner
# ---------------------------------------------------------------------------

class AIFileScanner:
    SKIP_DIRS = {
        "__pycache__", ".git", ".svn", ".hg", "node_modules",
        ".tox", ".mypy_cache", ".pytest_cache", ".ruff_cache",
        "venv", ".venv", "env", ".env",
    }

    def __init__(self, config: ScanConfig, classifier: FileClassifier) -> None:
        self.config = config
        self.classifier = classifier
        self.results: List[FileResult] = []
        self.errors: List[str] = []

        if config.quiet or not sys.stderr.isatty():
            self.progress: ProgressIndicator = ProgressIndicator()
        else:
            self.progress = SimpleProgress()

    def scan(self) -> Tuple[List[FileResult], ScanSummary]:
        start = time.monotonic()
        self._scan_recursive(self.config.root_path, 0)
        self.progress.finish()
        duration = time.monotonic() - start

        summary = self._build_summary(duration)
        return self.results, summary

    def _scan_recursive(self, path: str, depth: int) -> None:
        if self.config.max_depth is not None and depth > self.config.max_depth:
            return

        try:
            entries = list(os.scandir(path))
        except PermissionError:
            self.errors.append(f"Permission denied: {path}")
            return
        except OSError as e:
            self.errors.append(f"OS error scanning {path}: {e}")
            return

        for entry in entries:
            full_path = entry.path
            self.progress.update(full_path)

            if not self.config.include_hidden and entry.name.startswith("."):
                continue

            try:
                if entry.is_dir(follow_symlinks=self.config.follow_symlinks):
                    result = self.classifier.classify_directory(full_path, entry)
                    if result:
                        self._apply_on_demand(result, full_path)
                        if self._passes_filters(result):
                            self.results.append(result)
                        continue  # don't recurse into checkpoint dirs

                    if entry.name not in self.SKIP_DIRS:
                        self._scan_recursive(full_path, depth + 1)

                elif entry.is_file(follow_symlinks=self.config.follow_symlinks):
                    result = self.classifier.classify_file(full_path, entry)
                    if result:
                        self.classifier.apply_secret_scan(result)
                        self._apply_on_demand(result, full_path)
                        if self._passes_filters(result):
                            self.results.append(result)

            except PermissionError:
                self.errors.append(f"Permission denied: {full_path}")
            except OSError as e:
                self.errors.append(f"Error accessing {full_path}: {e}")

    def _apply_on_demand(self, result: FileResult, full_path: str) -> None:
        """Apply on-demand computations (permissions, hashes) to a result."""
        if self.config.compute_permissions:
            result.owner, result.group, result.permissions = _get_permissions(full_path)
            result.permission_alert, result.permission_alert_reason = _check_permission_alert(
                result.permissions, result.categories,
            )
        if self.config.compute_hashes:
            if result.is_directory:
                result.hash = _compute_dir_hash(full_path)
            else:
                result.hash = _compute_file_hash(full_path)

    def _passes_filters(self, result: FileResult) -> bool:
        if result.size_bytes < self.config.min_size_bytes:
            return False
        if self.config.categories and not {c.value for c in result.categories}.intersection(self.config.categories):
            return False
        return True

    def _build_summary(self, duration: float) -> ScanSummary:
        summary = ScanSummary(
            scan_path=self.config.root_path,
            scan_time=datetime.now().isoformat(timespec="seconds"),
            total_files=len(self.results),
            total_size=sum(r.size_bytes for r in self.results),
            errors=self.errors,
            scan_duration=duration,
        )

        for r in self.results:
            for cat_enum in r.categories:
                cat = cat_enum.value
                if cat not in summary.by_category:
                    summary.by_category[cat] = {"count": 0, "size": 0}
                summary.by_category[cat]["count"] += 1
                summary.by_category[cat]["size"] += r.size_bytes

            ext = r.extension or "[dir]"
            if ext not in summary.by_extension:
                summary.by_extension[ext] = {"count": 0, "size": 0}
            summary.by_extension[ext]["count"] += 1
            summary.by_extension[ext]["size"] += r.size_bytes

        return summary


# ---------------------------------------------------------------------------
# Output Formatting
# ---------------------------------------------------------------------------

SORT_KEYS = {
    "size": lambda r: -r.size_bytes,
    "date": lambda r: -r.modified_timestamp,
    "name": lambda r: r.filename.lower(),
    "category": lambda r: (r.categories[0].value, -r.size_bytes),
}


class OutputFormatter:
    def __init__(self, use_color: bool = True) -> None:
        self.use_color = use_color and RICH_AVAILABLE

    def print_results(
        self,
        results: List[FileResult],
        summary: ScanSummary,
        sort_by: str = "size",
        top_n: Optional[int] = None,
    ) -> None:
        if self.use_color:
            self._print_rich(results, summary, sort_by, top_n)
        else:
            self._print_plain(results, summary, sort_by, top_n)

    # ---- Rich output ----

    def _print_rich(
        self,
        results: List[FileResult],
        summary: ScanSummary,
        sort_by: str,
        top_n: Optional[int],
    ) -> None:
        console = Console()

        # Summary panel
        summary_text = (
            f"[bold]Scanned:[/bold]  {summary.scan_path}\n"
            f"[bold]Files:[/bold]    {summary.total_files}\n"
            f"[bold]Size:[/bold]     {format_size(summary.total_size)}\n"
            f"[bold]Duration:[/bold] {summary.scan_duration:.1f}s"
        )
        console.print(Panel(summary_text, title="AI File Discovery Results", border_style="bright_blue"))

        if not results:
            console.print("\n[dim]No AI/ML files found.[/dim]")
            return

        # Category breakdown
        cat_table = Table(title="By Category")
        cat_table.add_column("Category", style="cyan")
        cat_table.add_column("Count", justify="right")
        cat_table.add_column("Total Size", justify="right", style="yellow")
        for cat in sorted(summary.by_category):
            info = summary.by_category[cat]
            cat_table.add_row(cat, str(info["count"]), format_size(info["size"]))
        console.print(cat_table)

        # File listing
        sorted_results = sorted(results, key=SORT_KEYS.get(sort_by, SORT_KEYS["size"]))
        if top_n:
            sorted_results = sorted_results[:top_n]

        has_permissions = any(r.owner for r in sorted_results)
        has_hashes = any(r.hash for r in sorted_results)

        file_table = Table(title="Files Found")
        file_table.add_column("Category", width=12)
        file_table.add_column("Extension", style="green", width=14)
        file_table.add_column("Size", justify="right", style="yellow", width=10)
        file_table.add_column("Modified", width=19)
        file_table.add_column("C", width=1)
        file_table.add_column("Risk", width=8)
        if has_permissions:
            file_table.add_column("Owner", width=12)
            file_table.add_column("Group", width=12)
            file_table.add_column("Perms", width=10)
        if has_hashes:
            file_table.add_column("Hash", width=14, style="dim")
        file_table.add_column("Path", style="dim")

        for r in sorted_results:
            cat_str = " ".join(
                f"[{CATEGORY_COLORS.get(c, 'white')}]{c.value}[/{CATEGORY_COLORS.get(c, 'white')}]"
                for c in r.categories
            )
            risk_display = ""
            if r.risk_level == "danger":
                risk_display = "[red bold]DANGER[/red bold]"
            elif r.risk_level == "warning":
                risk_display = "[yellow]WARN[/yellow]"
            row = [
                cat_str,
                r.extension or "[dir]",
                format_size(r.size_bytes),
                format_timestamp(r.modified_timestamp),
                r.confidence[0].upper(),
                risk_display,
            ]
            if has_permissions:
                row.extend([r.owner, r.group, r.permissions])
            if has_hashes:
                row.append(r.hash[:12] + "…" if r.hash else "")
            row.append(r.path)
            file_table.add_row(*row)
        console.print(file_table)

        # Errors
        if summary.errors:
            console.print(f"\n[red]{len(summary.errors)} error(s) during scan:[/red]")
            for err in summary.errors[:10]:
                console.print(f"  [dim]- {err}[/dim]")
            if len(summary.errors) > 10:
                console.print(f"  [dim]... and {len(summary.errors) - 10} more[/dim]")

    # ---- Plain text output ----

    def _print_plain(
        self,
        results: List[FileResult],
        summary: ScanSummary,
        sort_by: str,
        top_n: Optional[int],
    ) -> None:
        print("=" * 70)
        print("  AI File Discovery Results")
        print("=" * 70)
        print(f"  Scanned:  {summary.scan_path}")
        print(f"  Files:    {summary.total_files}")
        print(f"  Size:     {format_size(summary.total_size)}")
        print(f"  Duration: {summary.scan_duration:.1f}s")
        print()

        if not results:
            print("  No AI/ML files found.")
            return

        # Category breakdown
        print("  Category Breakdown:")
        print("  " + "-" * 45)
        for cat in sorted(summary.by_category):
            info = summary.by_category[cat]
            print(f"    {cat:<12} {info['count']:>5} files  {format_size(info['size']):>10}")
        print()

        # File listing
        sorted_results = sorted(results, key=SORT_KEYS.get(sort_by, SORT_KEYS["size"]))
        if top_n:
            sorted_results = sorted_results[:top_n]

        has_permissions = any(r.owner for r in sorted_results)
        has_hashes = any(r.hash for r in sorted_results)

        header = f"  {'Category':<20} {'Extension':<14} {'Size':>10}  {'Modified':<19}  C  Risk"
        if has_permissions:
            header += f"  {'Owner':<12} {'Group':<12} {'Perms':<10}"
        if has_hashes:
            header += f"  {'Hash':<14}"
        header += "  Path"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for r in sorted_results:
            ext = r.extension or "[dir]"
            risk = r.risk_level if r.risk_level != "none" else ""
            cats = ",".join(c.value for c in r.categories)
            line = (
                f"  {cats:<20} {ext:<14} {format_size(r.size_bytes):>10}"
                f"  {format_timestamp(r.modified_timestamp):<19}"
                f"  {r.confidence[0].upper()}"
                f"  {risk:<6}"
            )
            if has_permissions:
                line += f"  {r.owner:<12} {r.group:<12} {r.permissions:<10}"
            if has_hashes:
                h = r.hash[:12] + "…" if r.hash else ""
                line += f"  {h:<14}"
            line += f"{r.path}"
            print(line)
        print()

        # Errors
        if summary.errors:
            print(f"  {len(summary.errors)} error(s) during scan:")
            for err in summary.errors[:10]:
                print(f"    - {err}")
            if len(summary.errors) > 10:
                print(f"    ... and {len(summary.errors) - 10} more")

    # ---- Export ----

    def export_json(
        self, results: List[FileResult], summary: ScanSummary, path: str
    ) -> None:
        data = _build_json_payload(results, summary)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def export_csv(
        self, results: List[FileResult], summary: ScanSummary, path: str
    ) -> None:
        has_permissions = any(r.owner for r in results)
        has_hashes = any(r.hash for r in results)
        fieldnames = [
            "path", "filename", "extension", "category", "categories",
            "size_bytes", "size_human", "modified", "type",
            "detection_method", "confidence", "risk_level", "risk_reason",
        ]
        if has_permissions:
            fieldnames.extend(["owner", "group", "permissions"])
        if has_hashes:
            fieldnames.append("hash")
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                row = {
                    "path": r.path,
                    "filename": r.filename,
                    "extension": r.extension,
                    "category": r.categories[0].value,
                    "categories": "|".join(c.value for c in r.categories),
                    "size_bytes": r.size_bytes,
                    "size_human": format_size(r.size_bytes),
                    "modified": format_timestamp(r.modified_timestamp),
                    "type": "directory" if r.is_directory else "file",
                    "detection_method": r.detection_method,
                    "confidence": r.confidence,
                    "risk_level": r.risk_level,
                    "risk_reason": r.risk_reason,
                }
                if has_permissions:
                    row.update({"owner": r.owner, "group": r.group, "permissions": r.permissions})
                if has_hashes:
                    row["hash"] = r.hash
                writer.writerow(row)


# ---------------------------------------------------------------------------
# JSON payload builder (shared by CLI --json and export_json)
# ---------------------------------------------------------------------------

def _build_json_payload(results: List[FileResult], summary: ScanSummary) -> dict:
    """Build a Pandas-friendly JSON structure from scan results."""
    return {
        "scan_metadata": {
            "scan_path": summary.scan_path,
            "scan_time": summary.scan_time,
            "exported_at": datetime.now().isoformat(),
            "total_files": summary.total_files,
            "total_size_bytes": summary.total_size,
            "total_size_human": format_size(summary.total_size),
            "scan_duration_seconds": round(summary.scan_duration, 2),
            "by_category": {
                cat: {**info, "size_human": format_size(info["size"])}
                for cat, info in summary.by_category.items()
            },
            "by_extension": {
                ext: {**info, "size_human": format_size(info["size"])}
                for ext, info in summary.by_extension.items()
            },
            "errors": summary.errors,
        },
        "files": [
            {
                "path": r.path,
                "filename": r.filename,
                "extension": r.extension,
                "category": r.categories[0].value,
                "categories": [c.value for c in r.categories],
                "size_bytes": r.size_bytes,
                "size_human": format_size(r.size_bytes),
                "modified": format_timestamp(r.modified_timestamp),
                "modified_timestamp": r.modified_timestamp,
                "type": "directory" if r.is_directory else "file",
                "detection_method": r.detection_method,
                "confidence": r.confidence,
                "risk_level": r.risk_level or "",
                "risk_reason": r.risk_reason or "",
                **({"owner": r.owner, "group": r.group, "permissions": r.permissions,
                    "permission_alert": r.permission_alert, "permission_alert_reason": r.permission_alert_reason} if r.owner else {}),
                **({"hash": r.hash} if r.hash else {}),
            }
            for r in results
        ],
    }


# ---------------------------------------------------------------------------
# Report directory helpers
# ---------------------------------------------------------------------------

REPORT_DIR = "report"


def _ensure_report_dir(base_path: str = ".") -> str:
    """Ensure the report/ subfolder exists, return its absolute path."""
    report_path = os.path.join(base_path, REPORT_DIR)
    os.makedirs(report_path, exist_ok=True)
    return report_path


def _get_latest_report(report_dir: str) -> Optional[str]:
    """Find the most recent aifiles_*.json report in the report directory."""
    if not os.path.isdir(report_dir):
        return None
    reports = sorted(
        (f for f in os.listdir(report_dir)
         if f.startswith("aifiles_") and f.endswith(".json")),
        reverse=True,
    )
    return os.path.join(report_dir, reports[0]) if reports else None


# ---------------------------------------------------------------------------
# Integrity / Change detection
# ---------------------------------------------------------------------------

@dataclass
class IntegrityReport:
    """Result of comparing current scan against a previous report."""
    previous_report: str
    previous_time: str
    new_files: List[dict]
    changed_files: List[dict]
    deleted_files: List[dict]

    @property
    def has_changes(self) -> bool:
        return bool(self.new_files or self.changed_files or self.deleted_files)

    def to_dict(self) -> dict:
        return {
            "previous_report": self.previous_report,
            "previous_time": self.previous_time,
            "new_files": self.new_files,
            "changed_files": self.changed_files,
            "deleted_files": self.deleted_files,
            "summary": {
                "new": len(self.new_files),
                "changed": len(self.changed_files),
                "deleted": len(self.deleted_files),
            },
        }


def run_integrity_check(
    current_results: List[FileResult], report_dir: str
) -> Optional[IntegrityReport]:
    """Compare current scan results against the latest previous report.

    Returns None if no previous report exists.
    """
    prev_path = _get_latest_report(report_dir)
    if prev_path is None:
        return None

    try:
        with open(prev_path, "r", encoding="utf-8") as f:
            prev_data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None

    prev_files = prev_data.get("files", [])
    prev_time = prev_data.get("scan_metadata", {}).get("scan_time", "unknown")

    # Build lookup: path -> file record
    prev_by_path: Dict[str, dict] = {}
    for pf in prev_files:
        prev_by_path[pf["path"]] = pf

    cur_by_path: Dict[str, dict] = {}
    for r in current_results:
        cur_by_path[r.path] = {
            "path": r.path,
            "filename": r.filename,
            "hash": r.hash or "",
            "size_bytes": r.size_bytes,
            "categories": [c.value for c in r.categories],
        }

    new_files = []
    changed_files = []
    deleted_files = []

    # Detect new and changed files
    for path, cur in cur_by_path.items():
        if path not in prev_by_path:
            new_files.append({"path": path, "filename": cur["filename"],
                              "size_bytes": cur["size_bytes"],
                              "categories": cur["categories"]})
        else:
            prev = prev_by_path[path]
            prev_hash = prev.get("hash", "")
            cur_hash = cur["hash"]
            if cur_hash and prev_hash and cur_hash != prev_hash:
                changed_files.append({
                    "path": path, "filename": cur["filename"],
                    "size_bytes": cur["size_bytes"],
                    "previous_size": prev.get("size_bytes", 0),
                    "previous_hash": prev_hash,
                    "current_hash": cur_hash,
                    "categories": cur["categories"],
                })

    # Detect deleted files
    for path, prev in prev_by_path.items():
        if path not in cur_by_path:
            deleted_files.append({
                "path": path,
                "filename": prev.get("filename", os.path.basename(path)),
                "size_bytes": prev.get("size_bytes", 0),
                "categories": prev.get("categories", []),
            })

    return IntegrityReport(
        previous_report=os.path.basename(prev_path),
        previous_time=prev_time,
        new_files=new_files,
        changed_files=changed_files,
        deleted_files=deleted_files,
    )


INTEGRITY_LOG = "integrity_check.jsonl"


def append_integrity_log(
    report_dir: str,
    scan_path: str,
    integrity: Optional[IntegrityReport],
    report_file: str,
) -> str:
    """Append an integrity check entry to the JSONL log in report/.

    Format: one JSON object per line (JSONL) — easy for humans to read,
    easy to import into Pandas with ``pd.read_json(path, lines=True)``,
    and compatible with ``jq``, ELK, Splunk, and other log systems.

    Returns the path to the log file.
    """
    log_path = os.path.join(report_dir, INTEGRITY_LOG)
    now = datetime.now()

    entry: Dict[str, object] = {
        "timestamp": now.isoformat(timespec="seconds"),
        "scan_path": scan_path,
        "report_file": os.path.basename(report_file),
    }

    if integrity is None:
        entry["status"] = "baseline"
        entry["message"] = "No previous report — this scan is the new baseline"
        entry["new"] = 0
        entry["changed"] = 0
        entry["deleted"] = 0
        entry["files"] = []
    elif not integrity.has_changes:
        entry["status"] = "clean"
        entry["compared_to"] = integrity.previous_report
        entry["compared_time"] = integrity.previous_time
        entry["message"] = "No changes detected"
        entry["new"] = 0
        entry["changed"] = 0
        entry["deleted"] = 0
        entry["files"] = []
    else:
        entry["status"] = "changes_detected"
        entry["compared_to"] = integrity.previous_report
        entry["compared_time"] = integrity.previous_time
        entry["new"] = len(integrity.new_files)
        entry["changed"] = len(integrity.changed_files)
        entry["deleted"] = len(integrity.deleted_files)

        # Detailed file list with per-item timestamps and action
        files = []
        for f in integrity.new_files:
            files.append({
                "action": "added",
                "path": f["path"],
                "filename": f["filename"],
                "size_bytes": f["size_bytes"],
                "categories": f["categories"],
            })
        for f in integrity.changed_files:
            files.append({
                "action": "modified",
                "path": f["path"],
                "filename": f["filename"],
                "size_bytes": f["size_bytes"],
                "previous_size": f.get("previous_size", 0),
                "previous_hash": f.get("previous_hash", ""),
                "current_hash": f.get("current_hash", ""),
                "categories": f["categories"],
            })
        for f in integrity.deleted_files:
            files.append({
                "action": "deleted",
                "path": f["path"],
                "filename": f["filename"],
                "size_bytes": f["size_bytes"],
                "categories": f["categories"],
            })
        entry["files"] = files

    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

    return log_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aifiles",
        description="AI File Discovery — Find AI/ML files consuming disk space.",
    )
    parser.add_argument(
        "path", nargs="?", default=".",
        help="Directory to scan (default: current directory)",
    )
    parser.add_argument(
        "--max-depth", type=int, default=None, metavar="N",
        help="Maximum directory depth to recurse into",
    )
    parser.add_argument(
        "--min-size", default=None, metavar="SIZE",
        help="Minimum file size filter (e.g., 10MB, 1GB, 500KB)",
    )
    parser.add_argument(
        "--export", choices=["json", "csv"], default=None,
        help="Export results to file",
    )
    parser.add_argument(
        "--export-path", default=None, metavar="FILE",
        help="Path for export file (default: aifiles_YYYYMMDD_HHMMSS.{ext})",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output scan results as JSON to stdout (for piping to Pandas)",
    )
    parser.add_argument(
        "--category", action="append", default=None,
        choices=["data", "model", "config", "vector", "checkpoint",
                 "source", "document", "multimedia", "skill", "agent", "secret"],
        help="Filter by category (can specify multiple times)",
    )
    parser.add_argument(
        "--sort", choices=["size", "date", "name", "category"], default="size",
        help="Sort results (default: size)",
    )
    parser.add_argument(
        "--top", type=int, default=None, metavar="N",
        help="Show only the top N largest files",
    )
    parser.add_argument(
        "--no-color", action="store_true",
        help="Disable colored output",
    )
    parser.add_argument(
        "--follow-symlinks", action="store_true",
        help="Follow symbolic links",
    )
    parser.add_argument(
        "--include-hidden", action="store_true",
        help="Include hidden files and directories",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress progress output",
    )
    parser.add_argument(
        "--permissions", action="store_true",
        help="Compute file owner/group/permissions",
    )
    parser.add_argument(
        "--hashes", action="store_true",
        help="Compute SHA-256 hashes for files and folders",
    )
    parser.add_argument(
        "--integrity", action="store_true",
        help="Compare scan against previous report to detect new/changed/deleted files",
    )
    parser.add_argument(
        "--config", default=None, metavar="FILE",
        help="Path to config file (default: auto-detect aifiles.config.json)",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    scan_path = os.path.abspath(args.path)
    if not os.path.isdir(scan_path):
        parser.error(f"Not a valid directory: {scan_path}")

    min_size = 0
    if args.min_size:
        try:
            min_size = parse_size(args.min_size)
        except ValueError as e:
            parser.error(str(e))

    # JSON exports and integrity checks always need hashes + permissions
    needs_full = args.json or (args.export == "json") or args.integrity

    config = ScanConfig(
        root_path=scan_path,
        max_depth=args.max_depth,
        min_size_bytes=min_size,
        categories=args.category,
        follow_symlinks=args.follow_symlinks,
        include_hidden=args.include_hidden,
        quiet=args.quiet,
        compute_permissions=args.permissions or needs_full,
        compute_hashes=args.hashes or needs_full,
    )

    classifier = FileClassifier(config_path=args.config)
    scanner = AIFileScanner(config, classifier)
    results, summary = scanner.scan()

    # Report directory (next to scanned path)
    report_dir = _ensure_report_dir()

    formatter = OutputFormatter(use_color=False)

    # --json: output JSON to stdout (for piping to Pandas)
    if args.json:
        data = _build_json_payload(results, summary)
        # Also save to report/ folder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(report_dir, f"aifiles_{timestamp}.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        json.dump(data, sys.stdout, indent=2, ensure_ascii=False)
        sys.stdout.write("\n")
        print(f"\nReport saved to: {report_path}", file=sys.stderr)
        sys.exit(0 if results else 1)

    # Normal interactive output
    use_color = not args.no_color and sys.stdout.isatty()
    formatter = OutputFormatter(use_color=use_color)
    formatter.print_results(results, summary, sort_by=args.sort, top_n=args.top)

    if args.export:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if args.export_path:
            export_path = args.export_path
        else:
            export_path = os.path.join(report_dir, f"aifiles_{timestamp}.{args.export}")
        try:
            if args.export == "json":
                formatter.export_json(results, summary, export_path)
            elif args.export == "csv":
                formatter.export_csv(results, summary, export_path)
            print(f"\nExported to: {export_path}")
        except OSError as e:
            print(f"\nExport failed: {e}", file=sys.stderr)
            sys.exit(2)

    # Integrity check
    if args.integrity:
        integrity = run_integrity_check(results, report_dir)
        if integrity is None:
            print("\n  ℹ  No previous report found in report/ — this scan will serve as the baseline.")
        elif not integrity.has_changes:
            print(f"\n  ✅ No changes detected (compared to {integrity.previous_report})")
        else:
            print(f"\n  🔍 Integrity check vs {integrity.previous_report} ({integrity.previous_time}):")
            if integrity.new_files:
                print(f"     ➕ {len(integrity.new_files)} new file(s)")
                for nf in integrity.new_files[:10]:
                    print(f"        {nf['path']}")
                if len(integrity.new_files) > 10:
                    print(f"        ... and {len(integrity.new_files) - 10} more")
            if integrity.changed_files:
                print(f"     ✏️  {len(integrity.changed_files)} changed file(s)")
                for cf in integrity.changed_files[:10]:
                    print(f"        {cf['path']}")
                if len(integrity.changed_files) > 10:
                    print(f"        ... and {len(integrity.changed_files) - 10} more")
            if integrity.deleted_files:
                print(f"     🗑  {len(integrity.deleted_files)} deleted file(s)")
                for df in integrity.deleted_files[:10]:
                    print(f"        {df['path']}")
                if len(integrity.deleted_files) > 10:
                    print(f"        ... and {len(integrity.deleted_files) - 10} more")

        # Always save current report after integrity check for next comparison
        if not args.export:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = os.path.join(report_dir, f"aifiles_{timestamp}.json")
            formatter_plain = OutputFormatter(use_color=False)
            formatter_plain.export_json(results, summary, report_path)
            print(f"\n  Report saved to: {report_path}")

        # Append to integrity log
        log_path = append_integrity_log(report_dir, scan_path, integrity, report_path)
        print(f"  Integrity log: {log_path}")

    sys.exit(0 if results else 1)


if __name__ == "__main__":
    main()
