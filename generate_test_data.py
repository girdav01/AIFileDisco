#!/usr/bin/env python3
"""Generate a synthetic test data folder for AI File Discovery.

Creates a realistic directory tree with various AI/ML file types at
different sizes. All files contain random or structured dummy content —
no real models or datasets.

Usage:
    python3 generate_test_data.py          # creates ./test_data/
    python3 generate_test_data.py /path    # creates at custom path
"""

import json
import os
import random
import struct
import sys

TARGET = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "test_data")


def mkfile(relpath: str, content: bytes) -> None:
    path = os.path.join(TARGET, relpath)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(content)


def rand_bytes(n: int) -> bytes:
    return random.randbytes(n)


def csv_content(rows: int, cols: int) -> bytes:
    header = ",".join(f"col_{i}" for i in range(cols))
    lines = [header]
    for _ in range(rows):
        lines.append(",".join(f"{random.uniform(-10, 10):.4f}" for _ in range(cols)))
    return "\n".join(lines).encode()


def tsv_content(rows: int, cols: int) -> bytes:
    header = "\t".join(f"feature_{i}" for i in range(cols))
    lines = [header]
    for _ in range(rows):
        lines.append("\t".join(f"{random.randint(0, 1000)}" for _ in range(cols)))
    return "\n".join(lines).encode()


def jsonl_content(rows: int) -> bytes:
    lines = []
    for i in range(rows):
        record = {
            "id": i,
            "text": f"Sample text entry number {i} for training purposes.",
            "label": random.choice(["positive", "negative", "neutral"]),
            "score": round(random.random(), 4),
        }
        lines.append(json.dumps(record))
    return "\n".join(lines).encode()


def notebook_content(cells: int, ai_related: bool = True) -> bytes:
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            }
        },
        "cells": [],
    }
    if ai_related:
        code_snippets = [
            "import torch\nimport transformers\nfrom datasets import load_dataset",
            "model = AutoModelForSequenceClassification.from_pretrained('bert-base')\ntokenizer = AutoTokenizer.from_pretrained('bert-base')",
            "optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)\nscheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=1000)",
            "for epoch in range(3):\n    for batch in dataloader:\n        outputs = model(**batch)\n        loss = outputs.loss\n        loss.backward()\n        optimizer.step()",
            "from sklearn.metrics import accuracy_score, f1_score\npreds = model.predict(X_test)\nprint(f'Accuracy: {accuracy_score(y_test, preds):.4f}')",
        ]
    else:
        code_snippets = [
            "import pandas as pd\ndf = pd.read_csv('data.csv')",
            "print(df.describe())",
        ]

    for i in range(cells):
        nb["cells"].append({
            "cell_type": "code",
            "source": code_snippets[i % len(code_snippets)],
            "metadata": {},
            "outputs": [],
            "execution_count": i + 1,
        })
    return json.dumps(nb, indent=1).encode()


def yaml_ai_config(name: str) -> bytes:
    configs = {
        "training": (
            "# Training configuration\n"
            "model:\n"
            "  name: bert-base-uncased\n"
            "  hidden_size: 768\n"
            "  num_layers: 12\n"
            "  num_heads: 12\n"
            "  dropout: 0.1\n"
            "\n"
            "training:\n"
            "  learning_rate: 2e-5\n"
            "  batch_size: 32\n"
            "  epochs: 5\n"
            "  warmup_steps: 500\n"
            "  weight_decay: 0.01\n"
            "  optimizer: adamw\n"
            "  scheduler: linear\n"
            "  gradient_accumulation_steps: 4\n"
            "  max_length: 512\n"
            "\n"
            "dataset:\n"
            "  name: imdb\n"
            "  train_split: train\n"
            "  eval_split: test\n"
            "\n"
            "wandb:\n"
            "  project: sentiment-analysis\n"
            "  entity: ml-team\n"
        ),
        "lora": (
            "# LoRA fine-tuning config\n"
            "base_model: meta-llama/Llama-2-7b-hf\n"
            "lora:\n"
            "  r: 16\n"
            "  alpha: 32\n"
            "  dropout: 0.05\n"
            "  target_modules:\n"
            "    - q_proj\n"
            "    - v_proj\n"
            "\n"
            "training:\n"
            "  learning_rate: 3e-4\n"
            "  batch_size: 4\n"
            "  gradient_accumulation_steps: 8\n"
            "  epochs: 3\n"
            "  warmup_ratio: 0.03\n"
            "  quantization: 4bit\n"
            "\n"
            "inference:\n"
            "  max_new_tokens: 256\n"
            "  temperature: 0.7\n"
        ),
        "diffusion": (
            "# Stable Diffusion training config\n"
            "model:\n"
            "  pretrained: stabilityai/stable-diffusion-2-1\n"
            "  attention: xformers\n"
            "  embedding_dim: 768\n"
            "\n"
            "training:\n"
            "  learning_rate: 1e-5\n"
            "  batch_size: 1\n"
            "  gradient_checkpointing: true\n"
            "  mixed_precision: fp16\n"
            "  epochs: 100\n"
            "  scheduler: constant_with_warmup\n"
            "\n"
            "dataset:\n"
            "  resolution: 512\n"
            "  center_crop: true\n"
        ),
    }
    return configs.get(name, configs["training"]).encode()


def yaml_non_ai() -> bytes:
    return (
        "# Docker Compose configuration\n"
        "version: '3.8'\n"
        "services:\n"
        "  web:\n"
        "    image: nginx:latest\n"
        "    ports:\n"
        "      - '80:80'\n"
        "  redis:\n"
        "    image: redis:alpine\n"
        "    volumes:\n"
        "      - redis_data:/data\n"
        "volumes:\n"
        "  redis_data:\n"
    ).encode()


def fake_parquet_header() -> bytes:
    # Real Parquet magic number: PAR1
    return b"PAR1" + rand_bytes(196)


def fake_hdf5_header() -> bytes:
    # Real HDF5 signature
    return b"\x89HDF\r\n\x1a\n" + rand_bytes(192)


def fake_npy_header() -> bytes:
    # NumPy .npy magic + version + header
    return b"\x93NUMPY\x01\x00" + b"\x76\x00" + b"{'descr': '<f4', 'fortran_order': False, 'shape': (1000, 128), }" + b" " * 30 + b"\n" + rand_bytes(400)


def fake_onnx() -> bytes:
    # ONNX uses protobuf, starts with field tags
    return b"\x08\x07\x12\x0bonnx-model" + rand_bytes(500)


def fake_safetensors(size: int) -> bytes:
    header = json.dumps({
        "__metadata__": {"format": "pt"},
        "model.embed_tokens.weight": {
            "dtype": "F32",
            "shape": [32000, 4096],
            "data_offsets": [0, size - 100],
        },
    }).encode()
    header_len = struct.pack("<Q", len(header))
    return header_len + header + rand_bytes(size - len(header) - 8)


def fake_gguf(size: int) -> bytes:
    # GGUF magic + version
    return b"GGUF" + struct.pack("<I", 3) + rand_bytes(size - 8)


def fake_pytorch(size: int) -> bytes:
    # PyTorch models are ZIP files (PK header)
    return b"PK\x03\x04" + rand_bytes(size - 4)


def fake_tflite() -> bytes:
    # TFLite flatbuffer
    return rand_bytes(800)


def fake_sqlite_vector_db(size: int) -> bytes:
    # SQLite header + vector DB table names embedded
    header = b"SQLite format 3\x00"
    # Pad to page size with embedded vector-db indicators
    content = header + rand_bytes(100)
    content += b"\x00" + b"CREATE TABLE embedding_vectors" + b"\x00"
    content += b"CREATE TABLE collection_metadata" + b"\x00"
    content += rand_bytes(size - len(content))
    return content


def fake_faiss_index(size: int) -> bytes:
    return rand_bytes(size)


def fake_tfrecord(size: int) -> bytes:
    return rand_bytes(size)


def main() -> None:
    random.seed(42)
    print(f"Generating test data in: {TARGET}")

    # ---------------------------------------------------------------
    # DATA files — various formats and sizes
    # ---------------------------------------------------------------

    # Small CSVs
    mkfile("datasets/sentiment/train.csv", csv_content(500, 5))        # ~15 KB
    mkfile("datasets/sentiment/val.csv", csv_content(100, 5))          # ~3 KB
    mkfile("datasets/sentiment/test.csv", csv_content(200, 5))         # ~6 KB

    # Medium CSV
    mkfile("datasets/tabular/housing_prices.csv", csv_content(5000, 12))  # ~200 KB

    # TSV files
    mkfile("datasets/ner/conll_train.tsv", tsv_content(3000, 4))      # ~50 KB
    mkfile("datasets/ner/conll_test.tsv", tsv_content(800, 4))        # ~13 KB

    # JSONL files
    mkfile("datasets/instruction/alpaca_train.jsonl", jsonl_content(2000))  # ~200 KB
    mkfile("datasets/instruction/sharegpt_subset.jsonl", jsonl_content(500))  # ~50 KB
    mkfile("datasets/instruction/tiny_eval.jsonl", jsonl_content(50))       # ~5 KB

    # Parquet files (with real magic header)
    mkfile("datasets/embeddings/train_embeddings.parquet", fake_parquet_header() + rand_bytes(50_000))  # ~50 KB
    mkfile("datasets/embeddings/val_embeddings.parquet", fake_parquet_header() + rand_bytes(10_000))    # ~10 KB
    mkfile("datasets/large/wikipedia_chunks.parquet", fake_parquet_header() + rand_bytes(500_000))      # ~500 KB

    # Arrow file
    mkfile("datasets/cached/tokenized_dataset.arrow", rand_bytes(100_000))   # ~100 KB

    # HDF5 files
    mkfile("datasets/scientific/spectral_data.hdf5", fake_hdf5_header() + rand_bytes(80_000))  # ~80 KB
    mkfile("datasets/scientific/simulation.h5", fake_hdf5_header() + rand_bytes(200_000))      # ~200 KB

    # NumPy files
    mkfile("datasets/features/word_vectors.npy", fake_npy_header() + rand_bytes(150_000))  # ~150 KB
    mkfile("datasets/features/labels.npy", fake_npy_header() + rand_bytes(5_000))          # ~5 KB
    mkfile("datasets/features/cached_embeds.npz", rand_bytes(75_000))                       # ~75 KB

    # TFRecord files
    mkfile("datasets/tfrecords/train-00000-of-00004.tfrecord", fake_tfrecord(120_000))  # ~120 KB
    mkfile("datasets/tfrecords/train-00001-of-00004.tfrecord", fake_tfrecord(120_000))
    mkfile("datasets/tfrecords/train-00002-of-00004.tfrecord", fake_tfrecord(120_000))
    mkfile("datasets/tfrecords/train-00003-of-00004.tfrecord", fake_tfrecord(120_000))

    # Apache ecosystem data formats
    mkfile("datasets/streaming/events.avro", rand_bytes(85_000))                 # ~85 KB  Avro
    mkfile("datasets/warehouse/transactions.orc", rand_bytes(95_000))            # ~95 KB  ORC
    mkfile("datasets/cached/features.feather", rand_bytes(60_000))               # ~60 KB  Feather

    # Scientific data formats
    mkfile("datasets/scientific/climate_sim.nc", rand_bytes(70_000))             # ~70 KB  NetCDF
    mkfile("datasets/scientific/brain_scans.zarr", rand_bytes(55_000))           # ~55 KB  Zarr

    # ML-specific data formats
    mkfile("datasets/vision/train.beton", rand_bytes(110_000))                   # ~110 KB FFCV beton
    mkfile("datasets/mxnet/train.recordio", rand_bytes(80_000))                  # ~80 KB  MXNet RecordIO
    mkfile("datasets/mnist/t10k-images.idx", rand_bytes(40_000))                 # ~40 KB  IDX (MNIST)
    mkfile("datasets/webdataset/shard-00000.wds", rand_bytes(90_000))            # ~90 KB  WebDataset

    # ---------------------------------------------------------------
    # MODEL files — small to medium sizes
    # ---------------------------------------------------------------

    # SafeTensors
    mkfile("models/bert-sentiment/model.safetensors", fake_safetensors(300_000))   # ~300 KB
    mkfile("models/bert-sentiment/config.json", json.dumps({
        "model_type": "bert", "hidden_size": 768,
        "num_attention_heads": 12, "num_hidden_layers": 12,
    }, indent=2).encode())
    mkfile("models/bert-sentiment/tokenizer.json", json.dumps({
        "version": "1.0", "model": {"type": "WordPiece"}
    }).encode())
    mkfile("models/bert-sentiment/tokenizer_config.json", b'{"do_lower_case": true}')

    # GGUF model
    mkfile("models/llama-q4/llama-7b-q4_K_M.gguf", fake_gguf(800_000))  # ~800 KB

    # PyTorch models
    mkfile("models/resnet50/resnet50.pt", fake_pytorch(400_000))    # ~400 KB
    mkfile("models/vae/decoder.pth", fake_pytorch(200_000))         # ~200 KB

    # ONNX
    mkfile("models/exported/classifier.onnx", fake_onnx() + rand_bytes(150_000))  # ~150 KB

    # TFLite
    mkfile("models/mobile/detector.tflite", fake_tflite() + rand_bytes(100_000))  # ~100 KB

    # Pickle / Joblib (sklearn-style)
    mkfile("models/sklearn/random_forest.pkl", rand_bytes(50_000))      # ~50 KB
    mkfile("models/sklearn/scaler.pickle", rand_bytes(5_000))           # ~5 KB
    mkfile("models/sklearn/pipeline.joblib", rand_bytes(80_000))        # ~80 KB

    # .bin with GGUF magic (should be detected by heuristic)
    mkfile("models/mistral/pytorch_model.bin", fake_pytorch(500_000))   # ~500 KB
    mkfile("models/mistral/config.json", json.dumps({
        "model_type": "mistral", "hidden_size": 4096,
    }, indent=2).encode())
    mkfile("models/mistral/tokenizer_config.json", b'{"model_max_length": 32768}')
    mkfile("models/mistral/generation_config.json", b'{"max_new_tokens": 256}')

    # Protocol buffer model
    mkfile("models/tensorflow/saved_model.pb", rand_bytes(250_000))  # ~250 KB

    # MLModel (CoreML)
    mkfile("models/ios/image_classifier.mlmodel", rand_bytes(180_000))  # ~180 KB

    # Model archive
    mkfile("models/torchserve/sentiment.mar", rand_bytes(350_000))  # ~350 KB

    # Additional model formats
    mkfile("models/mobile/detector_edge.pte", rand_bytes(90_000))       # ~90 KB  ExecuTorch
    mkfile("models/tensorrt/classifier.engine", rand_bytes(200_000))    # ~200 KB TensorRT engine
    mkfile("models/tensorrt/detector.plan", rand_bytes(180_000))        # ~180 KB TensorRT plan
    mkfile("models/keras/text_classifier.keras", rand_bytes(120_000))   # ~120 KB Keras v3
    mkfile("models/paddle/lm_model.pdmodel", rand_bytes(150_000))      # ~150 KB PaddlePaddle
    mkfile("models/paddle/lm_params.pdparams", rand_bytes(250_000))    # ~250 KB PaddlePaddle params
    mkfile("models/mobile/face_detect.ncnn", rand_bytes(65_000))       # ~65 KB  NCNN mobile
    mkfile("models/mobile/object_detect.mnn", rand_bytes(70_000))      # ~70 KB  MNN mobile
    mkfile("models/coreml/vision.mlpackage", rand_bytes(160_000))      # ~160 KB CoreML package
    mkfile("models/onnxrt/optimized.ort", rand_bytes(130_000))         # ~130 KB ONNX Runtime
    mkfile("models/legacy/classifier.ggml", rand_bytes(400_000))       # ~400 KB Legacy GGML
    mkfile("models/bentoml/sentiment.bento", rand_bytes(100_000))      # ~100 KB BentoML

    # ---------------------------------------------------------------
    # CONFIG files — YAML (AI-related) and notebooks
    # ---------------------------------------------------------------

    mkfile("configs/training_bert.yaml", yaml_ai_config("training"))
    mkfile("configs/lora_finetune.yml", yaml_ai_config("lora"))
    mkfile("configs/diffusion_train.yaml", yaml_ai_config("diffusion"))

    # Non-AI YAML (should NOT be detected)
    mkfile("configs/docker-compose.yaml", yaml_non_ai())
    mkfile("infra/k8s-deploy.yml", yaml_non_ai())

    # Jupyter notebooks
    mkfile("notebooks/01_data_exploration.ipynb", notebook_content(6, ai_related=True))
    mkfile("notebooks/02_model_training.ipynb", notebook_content(8, ai_related=True))
    mkfile("notebooks/03_evaluation.ipynb", notebook_content(5, ai_related=True))
    mkfile("notebooks/utils_demo.ipynb", notebook_content(3, ai_related=False))

    # ---------------------------------------------------------------
    # VECTOR / EMBEDDING files
    # ---------------------------------------------------------------

    mkfile("vector_stores/chroma/chroma.db", fake_sqlite_vector_db(200_000))  # ~200 KB
    mkfile("vector_stores/faiss/docs.faiss", fake_faiss_index(300_000))       # ~300 KB
    mkfile("vector_stores/faiss/docs.index", rand_bytes(50_000))              # ~50 KB
    mkfile("vector_stores/annoy/embeddings.annoy", rand_bytes(150_000))       # ~150 KB
    mkfile("vector_stores/lance/vectors.lancedb", rand_bytes(100_000))        # ~100 KB

    # Additional vector store formats
    mkfile("vector_stores/hnsw/document_index.hnsw", rand_bytes(120_000))     # ~120 KB HNSW
    mkfile("vector_stores/hnsw/search_index.hnswlib", rand_bytes(80_000))     # ~80 KB  hnswlib
    mkfile("vector_stores/usearch/vectors.usearch", rand_bytes(90_000))       # ~90 KB  USearch
    mkfile("vector_stores/scann/product_embeddings.scann", rand_bytes(110_000))  # ~110 KB Google ScaNN
    mkfile("vector_stores/nmslib/nearest.nmslib", rand_bytes(75_000))         # ~75 KB  NMSLIB

    # ---------------------------------------------------------------
    # CHECKPOINT directories
    # ---------------------------------------------------------------

    # Training checkpoints
    for step in [500, 1000, 1500]:
        prefix = f"experiments/run-01/checkpoint-{step}"
        mkfile(f"{prefix}/optimizer.pt", fake_pytorch(100_000))
        mkfile(f"{prefix}/scheduler.pt", fake_pytorch(2_000))
        mkfile(f"{prefix}/model.safetensors", fake_safetensors(200_000))
        mkfile(f"{prefix}/training_args.json", json.dumps({
            "learning_rate": 2e-5, "per_device_train_batch_size": 8,
            "num_train_epochs": 3, "global_step": step,
        }, indent=2).encode())

    mkfile("experiments/run-01/model-best/model.safetensors", fake_safetensors(200_000))
    mkfile("experiments/run-01/model-best/config.json", b'{"model_type": "bert"}')

    mkfile("experiments/run-02/epoch-5/weights.pt", fake_pytorch(300_000))
    mkfile("experiments/run-02/epoch-5/optimizer_state.pt", fake_pytorch(50_000))

    # ---------------------------------------------------------------
    # SOURCE CODE files
    # ---------------------------------------------------------------

    mkfile("src/train.py", (
        b"import torch\nimport torch.nn as nn\nfrom transformers import AutoModel\n\n"
        b"class Trainer:\n    def __init__(self, model, lr=2e-5):\n"
        b"        self.model = model\n        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)\n\n"
        b"    def train_epoch(self, dataloader):\n        self.model.train()\n"
        b"        for batch in dataloader:\n            outputs = self.model(**batch)\n"
        b"            loss = outputs.loss\n            loss.backward()\n"
        b"            self.optimizer.step()\n            self.optimizer.zero_grad()\n"
    ))
    mkfile("src/model.py", (
        b"import torch.nn as nn\nfrom transformers import AutoModelForSequenceClassification\n\n"
        b"class SentimentModel(nn.Module):\n    def __init__(self, num_labels=3):\n"
        b"        super().__init__()\n        self.backbone = AutoModelForSequenceClassification.from_pretrained(\n"
        b"            'bert-base-uncased', num_labels=num_labels\n        )\n\n"
        b"    def forward(self, **kwargs):\n        return self.backbone(**kwargs)\n"
    ))
    mkfile("src/dataset.py", (
        b"import torch\nfrom torch.utils.data import Dataset\n\n"
        b"class TextDataset(Dataset):\n    def __init__(self, texts, labels, tokenizer):\n"
        b"        self.encodings = tokenizer(texts, truncation=True, padding=True)\n"
        b"        self.labels = labels\n\n"
        b"    def __len__(self):\n        return len(self.labels)\n\n"
        b"    def __getitem__(self, idx):\n        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}\n"
        b"        item['labels'] = torch.tensor(self.labels[idx])\n        return item\n"
    ))
    mkfile("src/preprocess.js", (
        b"const fs = require('fs');\nconst path = require('path');\n\n"
        b"function preprocessData(inputPath, outputPath) {\n"
        b"  const raw = fs.readFileSync(inputPath, 'utf-8');\n"
        b"  const lines = raw.split('\\n').filter(l => l.trim());\n"
        b"  const processed = lines.map(JSON.parse).map(item => ({\n"
        b"    text: item.text.toLowerCase().trim(),\n"
        b"    label: item.label\n  }));\n"
        b"  fs.writeFileSync(outputPath, processed.map(JSON.stringify).join('\\n'));\n}\n"
    ))
    mkfile("scripts/evaluate.sh", (
        b"#!/bin/bash\nset -euo pipefail\n\n"
        b"MODEL_DIR=${1:-models/bert-sentiment}\n"
        b"DATA_DIR=${2:-datasets/sentiment}\n\n"
        b"echo \"Evaluating model: $MODEL_DIR\"\n"
        b"python3 -m evaluate --model $MODEL_DIR --data $DATA_DIR/test.csv --metrics accuracy f1\n"
    ))
    mkfile("src/inference.rs", (
        b"use std::fs;\nuse serde::{Deserialize, Serialize};\n\n"
        b"#[derive(Serialize, Deserialize)]\nstruct Prediction {\n    label: String,\n    score: f32,\n}\n\n"
        b"fn main() {\n    let model_path = \"models/exported/classifier.onnx\";\n"
        b"    println!(\"Loading model from: {}\", model_path);\n}\n"
    ))
    mkfile("notebooks/helpers.R", (
        b"library(ggplot2)\nlibrary(dplyr)\n\n"
        b"load_results <- function(path) {\n  data <- read.csv(path)\n  return(data)\n}\n\n"
        b"plot_training_curve <- function(data) {\n"
        b"  ggplot(data, aes(x=epoch, y=loss)) + geom_line() + theme_minimal()\n}\n"
    ))
    mkfile("src/pipeline.jl", (
        b"using Flux\nusing DataFrames\nusing CSV\n\n"
        b"function build_model(input_dim, hidden_dim, output_dim)\n"
        b"    return Chain(\n        Dense(input_dim, hidden_dim, relu),\n"
        b"        Dense(hidden_dim, output_dim),\n        softmax\n    )\nend\n"
    ))

    # ---------------------------------------------------------------
    # DOCUMENT files
    # ---------------------------------------------------------------

    mkfile("docs/README.md", (
        b"# ML Project Documentation\n\n"
        b"This repository contains a sentiment analysis pipeline built with PyTorch and Transformers.\n\n"
        b"## Quick Start\n\n1. Install dependencies: `pip install -r requirements.txt`\n"
        b"2. Train model: `python src/train.py`\n3. Evaluate: `bash scripts/evaluate.sh`\n\n"
        b"## Project Structure\n\n- `src/` - Source code\n- `models/` - Trained models\n- `datasets/` - Training data\n"
    ))
    mkfile("docs/model_card.md", (
        b"# Model Card: BERT Sentiment Classifier\n\n"
        b"## Model Details\n- Architecture: BERT-base-uncased\n- Parameters: 110M\n- Fine-tuned on: IMDB dataset\n\n"
        b"## Intended Use\nSentiment classification of English text into positive, negative, or neutral.\n\n"
        b"## Limitations\n- English only\n- May not generalize to domain-specific text\n- Not suitable for hate speech detection\n\n"
        b"## Training Data\n- IMDB movie reviews (50K samples)\n- 80/10/10 train/val/test split\n"
    ))
    mkfile("docs/training_guide.md", (
        b"# Training Guide\n\n## Prerequisites\n- Python 3.9+\n- CUDA 11.8+ (for GPU training)\n\n"
        b"## Configuration\nEdit `configs/training_bert.yaml` to adjust hyperparameters.\n\n"
        b"## Distributed Training\nUse `torchrun --nproc_per_node=4 src/train.py` for multi-GPU training.\n\n"
        b"## Monitoring\nTraining metrics are logged to Weights & Biases.\n"
    ))
    mkfile("docs/paper_notes.rst", (
        b"===================\nPaper Notes\n===================\n\n"
        b"Attention Is All You Need (2017)\n--------------------------------\n\n"
        b"Key contributions:\n\n* Self-attention mechanism replaces recurrence\n"
        b"* Multi-head attention enables parallel processing\n* Positional encoding preserves sequence order\n\n"
        b"BERT: Pre-training of Deep Bidirectional Transformers (2018)\n------------------------------------------------------------\n\n"
        b"* Masked language modeling objective\n* Next sentence prediction\n"
    ))
    mkfile("docs/dataset_description.tex", (
        b"\\documentclass{article}\n\\begin{document}\n\n"
        b"\\title{Dataset Description}\n\\author{ML Team}\n\\maketitle\n\n"
        b"\\section{Overview}\nThe dataset consists of 50,000 movie reviews from IMDB.\n\n"
        b"\\section{Statistics}\n\\begin{itemize}\n  \\item Training: 40,000 samples\n"
        b"  \\item Validation: 5,000 samples\n  \\item Test: 5,000 samples\n\\end{itemize}\n\n"
        b"\\end{document}\n"
    ))

    # ---------------------------------------------------------------
    # MULTIMEDIA files (for multimodal AI)
    # ---------------------------------------------------------------

    mkfile("media/sample_image.png", rand_bytes(8_000))      # ~8 KB
    mkfile("media/diagram.jpg", rand_bytes(12_000))           # ~12 KB
    mkfile("media/logo.svg", (
        b'<?xml version="1.0" encoding="UTF-8"?>\n'
        b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">\n'
        b'  <circle cx="50" cy="50" r="40" fill="#4a90d9"/>\n'
        b'  <text x="50" y="55" text-anchor="middle" fill="white" font-size="20">AI</text>\n'
        b'</svg>\n'
    ))
    mkfile("media/audio_sample.wav", rand_bytes(20_000))      # ~20 KB
    mkfile("media/narration.mp3", rand_bytes(15_000))         # ~15 KB
    mkfile("media/demo_video.mp4", rand_bytes(50_000))        # ~50 KB

    # Additional image formats
    mkfile("media/photo.heic", rand_bytes(25_000))             # ~25 KB HEIC
    mkfile("media/next_gen.avif", rand_bytes(10_000))          # ~10 KB AVIF
    mkfile("media/icon.ico", rand_bytes(4_000))                # ~4 KB  ICO

    # Scientific / medical imaging
    mkfile("media/medical/brain_scan.dcm", rand_bytes(80_000))       # ~80 KB  DICOM
    mkfile("media/medical/mri_volume.nii", rand_bytes(60_000))       # ~60 KB  NIfTI

    # Design files
    mkfile("media/design/mockup.psd", rand_bytes(45_000))      # ~45 KB PSD
    mkfile("media/design/vector_art.eps", rand_bytes(18_000))  # ~18 KB EPS

    # Audio — additional formats
    mkfile("media/speech_corpus/sample_001.opus", rand_bytes(12_000))  # ~12 KB Opus
    mkfile("media/speech_corpus/sample_002.flac", rand_bytes(30_000))  # ~30 KB FLAC
    mkfile("media/midi/melody.mid", rand_bytes(3_000))                 # ~3 KB  MIDI

    # Video — additional formats
    mkfile("media/training_clip.webm", rand_bytes(35_000))    # ~35 KB WebM
    mkfile("media/presentation.mov", rand_bytes(40_000))      # ~40 KB MOV

    # 3D / Point cloud (computer vision / ML)
    mkfile("media/3d/scene.ply", rand_bytes(50_000))           # ~50 KB PLY point cloud
    mkfile("media/3d/lidar_scan.pcd", rand_bytes(65_000))      # ~65 KB PCD point cloud
    mkfile("media/3d/object.obj", rand_bytes(20_000))          # ~20 KB OBJ mesh
    mkfile("media/3d/model.glb", rand_bytes(55_000))           # ~55 KB GLB (glTF binary)

    # Subtitles / captions (speech/NLP tasks)
    mkfile("media/captions/lecture.srt", (
        b"1\n00:00:00,000 --> 00:00:05,000\n"
        b"Welcome to the machine learning lecture.\n\n"
        b"2\n00:00:05,000 --> 00:00:12,000\n"
        b"Today we will discuss transformer architectures.\n\n"
        b"3\n00:00:12,000 --> 00:00:20,000\n"
        b"Let's start with self-attention mechanisms.\n"
    ))
    mkfile("media/captions/tutorial.vtt", (
        b"WEBVTT\n\n"
        b"00:00:00.000 --> 00:00:04.000\n"
        b"This tutorial covers fine-tuning BERT.\n\n"
        b"00:00:04.000 --> 00:00:09.000\n"
        b"We will use the Hugging Face Transformers library.\n"
    ))

    # ---------------------------------------------------------------
    # DOCUMENT files — additional formats
    # ---------------------------------------------------------------

    # PDF document
    mkfile("docs/whitepaper.pdf", b"%PDF-1.4\n" + rand_bytes(35_000))  # ~35 KB fake PDF

    # Annotation / labeling formats (NLP/CV)
    mkfile("datasets/annotations/entities.ann", (
        b"T1\tPerson 0 5\tAlice\n"
        b"T2\tOrganization 25 35\tOpenAI Inc\n"
        b"R1\tWorks_For Arg1:T1 Arg2:T2\n"
    ))
    mkfile("datasets/annotations/ner_labels.conll", (
        b"Alice B-PER\n"
        b"works O\n"
        b"at O\n"
        b"OpenAI B-ORG\n"
        b"\n"
        b"Bob B-PER\n"
        b"studies O\n"
        b"at O\n"
        b"MIT B-ORG\n"
    ))
    mkfile("datasets/annotations/bio_tags.bio", (
        b"The O\n"
        b"model B-MODEL\n"
        b"was O\n"
        b"trained O\n"
        b"on O\n"
        b"ImageNet B-DATASET\n"
    ))

    # R Markdown and Quarto notebooks
    mkfile("notebooks/analysis.rmd", (
        b"---\ntitle: Model Evaluation\noutput: html_document\n---\n\n"
        b"```{r setup}\nlibrary(tidyverse)\nlibrary(caret)\n```\n\n"
        b"## Results\n\nThe model achieved 94.2% accuracy on the test set.\n\n"
        b"```{r plot}\nggplot(results, aes(x=epoch, y=loss)) + geom_line()\n```\n"
    ))
    mkfile("notebooks/report.qmd", (
        b"---\ntitle: Training Report\nformat: html\njupyter: python3\n---\n\n"
        b"## Model Performance\n\n```{python}\nimport pandas as pd\n"
        b"df = pd.read_csv('results.csv')\ndf.describe()\n```\n\n"
        b"The transformer model outperformed the baseline by 12%.\n"
    ))

    # HTML report
    mkfile("docs/eval_report.html", (
        b"<!DOCTYPE html>\n<html><head><title>Evaluation Report</title></head>\n"
        b"<body><h1>Model Evaluation</h1>\n"
        b"<p>Accuracy: 95.3% | F1: 0.94 | AUC: 0.98</p>\n"
        b"<table><tr><th>Metric</th><th>Value</th></tr>\n"
        b"<tr><td>Precision</td><td>0.95</td></tr></table>\n"
        b"</body></html>\n"
    ))

    # ---------------------------------------------------------------
    # SKILL / AGENT files
    # ---------------------------------------------------------------

    mkfile("agents/search_agent.yaml", (
        b"name: web_search\ndescription: Search the web for information\n"
        b"tool_use:\n  type: function_call\n  parameters:\n"
        b"    query:\n      type: string\n      description: Search query\n"
        b"    max_results:\n      type: integer\n      default: 5\n"
        b"actions:\n  - search\n  - summarize\n"
        b"langchain:\n  tool_class: WebSearchTool\n"
    ))
    mkfile("agents/scraper_plugin.json", json.dumps({
        "name": "web_scraper",
        "description": "Scrape and parse web pages",
        "plugin": True,
        "actions": [
            {"name": "fetch_url", "parameters": {"url": {"type": "string"}}},
            {"name": "extract_text", "parameters": {"selector": {"type": "string"}}},
        ],
        "tool_use": {"type": "function_call"},
        "autogen": {"agent_type": "AssistantAgent"},
    }, indent=2).encode())
    mkfile("agents/data_analyst.skill", (
        b"name: data_analyst\nversion: 1.0\n"
        b"capabilities:\n  - csv_analysis\n  - chart_generation\n  - statistical_summary\n"
        b"inputs:\n  data_path:\n    type: file\n    extensions: [csv, parquet, json]\n"
    ))
    mkfile("agents/SKILL.md", (
        b"# Data Pipeline Skill\n\n"
        b"This skill processes raw data files and prepares them for model training.\n\n"
        b"## Capabilities\n- Load CSV/Parquet files\n- Clean and normalize data\n- Feature engineering\n\n"
        b"## Usage\nInvoke via the agent orchestrator with `skill: data_pipeline`.\n"
    ))

    # ---------------------------------------------------------------
    # Agent files
    # ---------------------------------------------------------------

    mkfile("agents/system_prompt.md", (
        b"# System Prompt\n\n"
        b"You are a helpful assistant. Your role is to answer questions\n"
        b"about AI and machine learning. Follow these instructions carefully.\n"
    ))
    mkfile("agents/agent_config.yaml", (
        b"name: research_assistant\n"
        b"system_prompt: You are a research assistant\n"
        b"your role: analyze papers and summarize findings\n"
        b"persona: professional and thorough\n"
    ))
    mkfile("agents/goals.yaml", (
        b"objective: answer user questions accurately\n"
        b"goal: provide helpful and concise responses\n"
        b"instruction: always cite sources\n"
        b"constraints: stay on topic\n"
    ))
    mkfile("agents/persona.txt", (
        b"You are a friendly data science tutor.\n"
        b"Your role is to explain complex concepts simply.\n"
        b"Act as a patient teacher. Respond as an expert.\n"
    ))
    mkfile("configs/prompt_template.prompt", b"You are {{role}}. Your goal is {{objective}}.\n")

    # ---------------------------------------------------------------
    # Agentic framework files (CrewAI, LangGraph, AutoGen, DSPy)
    # ---------------------------------------------------------------

    # CrewAI project structure
    mkfile("agentic/crewai/config/agents.yaml", (
        b"researcher:\n"
        b"  role: Senior Research Analyst\n"
        b"  goal: Uncover cutting-edge developments in AI\n"
        b"  backstory: You are a seasoned researcher with a knack for finding the most relevant information.\n"
        b"  verbose: true\n\n"
        b"writer:\n"
        b"  role: Tech Content Writer\n"
        b"  goal: Write compelling articles about AI discoveries\n"
        b"  backstory: You are a skilled writer who translates complex topics into engaging content.\n"
    ))
    mkfile("agentic/crewai/config/tasks.yaml", (
        b"research_task:\n"
        b"  description: Research the latest AI trends and developments in {topic}\n"
        b"  expected_output: A detailed report with key findings and sources\n"
        b"  agent: researcher\n\n"
        b"writing_task:\n"
        b"  description: Write an article based on the research findings\n"
        b"  expected_output: A well-structured 500-word article\n"
        b"  agent: writer\n"
    ))
    mkfile("agentic/crewai/crew.py", (
        b"from crewai import Agent, Crew, Task\n"
        b"from crewai_tools import FirecrawlScrapeWebsiteTool\n\n"
        b"class ResearchCrew:\n"
        b"    agents_config = 'config/agents.yaml'\n"
        b"    tasks_config = 'config/tasks.yaml'\n\n"
        b"    def crew(self):\n"
        b"        return Crew(agents=self.agents, tasks=self.tasks, verbose=True)\n"
    ))

    # LangGraph / LangChain agent
    mkfile("agentic/langgraph/workflow.py", (
        b"from langgraph.graph import StateGraph, END\n"
        b"from langchain_core.messages import HumanMessage\n\n"
        b"def agent_node(state):\n"
        b"    messages = state['messages']\n"
        b"    response = model.invoke(messages)\n"
        b"    return {'messages': [response]}\n\n"
        b"workflow = StateGraph(AgentState)\n"
        b"workflow.add_node('agent', agent_node)\n"
        b"workflow.add_node('tool_node', tool_node)\n"
        b"workflow.set_entry_point('agent')\n"
    ))

    # AutoGen config
    mkfile("agentic/autogen/OAI_CONFIG_LIST", json.dumps([
        {"model": "gpt-4", "api_key": "REPLACE_WITH_YOUR_KEY"},
        {"model": "gpt-3.5-turbo", "api_key": "REPLACE_WITH_YOUR_KEY"},
    ], indent=2).encode())

    # DSPy module
    mkfile("agentic/dspy/retriever.py", (
        b"import dspy\n\n"
        b"class RAGModule(dspy.Module):\n"
        b"    def __init__(self, num_passages=3):\n"
        b"        super().__init__()\n"
        b"        self.retrieve = dspy.Retrieve(k=num_passages)\n"
        b"        self.generate = dspy.ChainOfThought('context, question -> answer')\n\n"
        b"    def forward(self, question):\n"
        b"        context = self.retrieve(question).passages\n"
        b"        return self.generate(context=context, question=question)\n"
    ))

    # MLflow model definition
    mkfile("agentic/mlflow/MLmodel", (
        b"artifact_path: model\n"
        b"flavors:\n"
        b"  python_function:\n"
        b"    env: conda.yaml\n"
        b"    loader_module: mlflow.sklearn\n"
        b"    model_path: model.pkl\n"
        b"    python_version: 3.10.12\n"
        b"  sklearn:\n"
        b"    code: null\n"
        b"    pickled_model: model.pkl\n"
        b"    sklearn_version: 1.3.0\n"
    ))

    # ---------------------------------------------------------------
    # Secret / credential files
    # ---------------------------------------------------------------

    mkfile("secrets/.env", (
        b"DB_PASSWORD=supersecret123\n"
        b"API_KEY=sk-fake1234567890abcdef1234567890ab\n"
        b"REDIS_URL=redis://localhost:6379\n"
    ))
    mkfile("secrets/credentials.json", json.dumps({
        "type": "service_account",
        "project_id": "test-project",
        "private_key_id": "fake-key-id",
        "client_email": "test@test.iam.gserviceaccount.com",
    }, indent=2).encode())
    mkfile("secrets/id_rsa", (
        b"-----BEGIN RSA PRIVATE KEY-----\n"
        b"MIIEpAIBAAKCAQEA0Z3VS5JJcds3xfn/yGw7FAKEFAKEFAKE\n"
        b"-----END RSA PRIVATE KEY-----\n"
    ))
    mkfile("secrets/api.token", b"ghp_abcdefghijklmnopqrstuvwxyz0123456789\n")

    # ---------------------------------------------------------------
    # NON-AI files (should be IGNORED by the scanner)
    # ---------------------------------------------------------------

    mkfile("src/requirements.txt", b"torch==2.1.0\ntransformers==4.36.0\n")
    mkfile("Makefile", b"train:\n\tpython train.py\n")
    mkfile(".gitignore", b"__pycache__/\n*.pyc\ntest_data/\n")
    mkfile("logs/train_2024-01-15.log", b"Epoch 1/3 - loss: 0.342 - acc: 0.891\n" * 100)
    mkfile("data/raw/notes.txt", b"Some random notes about the project.\n")

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------

    total_size = 0
    file_count = 0
    for dirpath, _dirs, files in os.walk(TARGET):
        for f in files:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
            file_count += 1

    size_mb = total_size / (1024 * 1024)
    print(f"Created {file_count} files ({size_mb:.1f} MB total)")
    print()
    print("Expected scanner results:")
    print("  DATA:       ~32 files (CSV, TSV, JSONL, Parquet, Arrow, Feather, HDF5, NPY, NPZ, TFRecord,")
    print("                         Avro, ORC, NetCDF, Zarr, Beton, RecordIO, IDX, WDS)")
    print("  MODEL:      ~26 files (SafeTensors, GGUF, GGML, PT, PTH, ONNX, TFLite, PKL, Pickle, Joblib,")
    print("                         PB, MLModel, MAR, BIN, PTE, Engine, Plan, Keras, PaddlePaddle,")
    print("                         NCNN, MNN, MLPackage, ORT, BentoML)")
    print("  CONFIG:     ~7 files  (3 YAML + 4 notebooks)")
    print("  VECTOR:     ~10 files (DB, FAISS, Index, Annoy, LanceDB, HNSW, hnswlib, USearch, ScaNN, NMSLIB)")
    print("  CHECKPOINT: ~5 dirs   (checkpoint-500/1000/1500, model-best, epoch-5)")
    print("  SOURCE:     ~12 files (PY, JS, SH, RS, R, JL + agentic framework source)")
    print("  DOCUMENT:   ~14 files (MD, RST, TEX, PDF, HTML, ANN, CoNLL, BIO, RMD, QMD)")
    print("  MULTIMEDIA: ~25 files (PNG, JPG, SVG, HEIC, AVIF, ICO, PSD, EPS, WAV, MP3, OPUS,")
    print("                         FLAC, MIDI, MP4, WebM, MOV, DCM, NII, PLY, PCD, OBJ, GLB, SRT, VTT)")
    print("  SKILL:      ~4 files  (YAML, JSON, SKILL, SKILL.MD)")
    print("  AGENT:      ~12 files (system prompts, agent configs, CrewAI agents/tasks,")
    print("                         AutoGen OAI_CONFIG_LIST, MLflow MLmodel, .prompt)")
    print("  SECRET:     ~4 files  (.env, credentials.json, id_rsa, .token) — requires --include-hidden for .env")
    print("  RISK:       ~5 files  (PKL=danger, PICKLE=danger, JOBLIB=warning, BIN=warning, GGML=warning)")
    print("  IGNORED:    ~5 files  (txt, Makefile, .gitignore, log, non-AI yaml)")


if __name__ == "__main__":
    main()
