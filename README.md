# AI File Discovery

**Find, classify, and monitor AI/ML files on disk.**

Scans directories to discover AI-related files — datasets, models, configs, vector stores, checkpoints, source code, documents, multimedia, agent prompts, skills, and secrets — with an interactive web dashboard and CLI.

**Zero dependencies.** Pure Python 3.8+ stdlib. No pip packages required.

---

## Quick Start

### Option 1: Run directly (no install)

```bash
git clone https://github.com/davidgirard/AIFileFinder.git
cd AIFileFinder

# CLI scan
python3 aifiles.py /path/to/scan

# Web dashboard
python3 server.py
# Open http://localhost:8505
```

### Option 2: pip install

```bash
pip install .

# Now available system-wide
aifiles /path/to/scan
aifiles-server
```

### Option 3: Docker

```bash
docker build -t aifilefinder .
docker run -p 8505:8505 -v /your/data:/data aifilefinder
# Open http://localhost:8505 and scan /data
```

### Option 4: One-line install

```bash
curl -fsSL https://raw.githubusercontent.com/davidgirard/AIFileFinder/main/install.sh | bash
```

---

## Features

| Feature | Description |
|---------|-------------|
| **11 categories** | data, model, config, vector, checkpoint, source, document, multimedia, skill, agent, secret |
| **Multi-category** | Files can belong to multiple categories (e.g., notebooks → config + source) |
| **Secret detection** | Finds credential files and embedded secrets (API keys, tokens, private keys) |
| **Agent detection** | Identifies system prompts, goal files, agent configs |
| **Web dashboard** | Interactive charts, filters, category chips, table/tree toggle |
| **Directory browser** | Navigate and select folders from the web UI |
| **Integrity checks** | Compare scans to detect new, changed, and deleted files (SHA-256) |
| **Permissions** | Show file owner, group, and permission flags |
| **JSON reports** | Pandas-compatible exports with hashes and permissions |
| **Integrity log** | Append-only JSONL audit trail of all integrity checks |
| **Zero dependencies** | Pure Python stdlib — nothing to install |

---

## CLI Usage

```bash
# Basic scan
aifiles /path/to/scan

# Filter by category
aifiles /path --category model config

# Export JSON report (saved to report/ folder with hashes + permissions)
aifiles /path --export json

# JSON to stdout (pipe to jq, Pandas, etc.)
aifiles /path --json

# Integrity check (detect changes since last scan)
aifiles /path --integrity

# Include hidden files (.env, .cursorrules, etc.)
aifiles /path --include-hidden

# Compute permissions and hashes
aifiles /path --permissions --hashes

# Filter by size, sort, limit
aifiles /path --min-size 10MB --sort size --top 20
```

### All CLI Options

```
positional arguments:
  path                  Directory to scan

options:
  --category            Filter by category (data, model, config, vector,
                        checkpoint, source, document, multimedia, skill,
                        agent, secret)
  --sort                Sort results (size, name, date, category)
  --top N               Show only top N results
  --max-depth N         Maximum directory depth
  --min-size SIZE       Minimum file size (e.g., 10MB, 1GB, 500KB)
  --export {json,csv}   Export results to report/ folder
  --export-path FILE    Custom export path
  --json                Output JSON to stdout
  --no-color            Disable colored output
  --follow-symlinks     Follow symbolic links
  --include-hidden      Include hidden files/directories
  --quiet               Suppress progress output
  --permissions         Compute file owner/group/permissions
  --hashes              Compute SHA-256 hashes
  --integrity           Compare against previous report for changes
```

---

## Web Dashboard

```bash
aifiles-server                        # default port 8505
aifiles-server --port 9000            # custom port
aifiles-server --scan-path /data      # pre-fill path
```

### Dashboard Features

- **Browse button** — navigate the filesystem to select a scan directory
- **Category chips** — filter by file type with color-coded badges
- **Table / Tree toggle** — switch between flat table and collapsible folder tree
- **On-demand options** — checkboxes for Permissions, Hashes, Integrity Check
- **Export JSON** — download Pandas-compatible report with datetime filename
- **Security banners** — warnings for deserialization risks, secrets, embedded credentials
- **Column tooltips** — hover over headers for explanations

---

## Reports & Integrity

All JSON reports are saved to the `report/` subfolder with timestamps:

```
report/
├── aifiles_20260314_120000.json      # Full scan snapshot
├── aifiles_20260314_130000.json      # Next scan snapshot
└── integrity_check.jsonl             # Append-only audit log
```

### Integrity Check

Compares current scan against the most recent previous report:

- **➕ New files** — files that didn't exist before
- **✏️ Changed files** — same path but different SHA-256 hash
- **🗑 Deleted files** — files in previous report no longer found

```bash
# First run creates the baseline
aifiles /path --integrity
# → "No previous report — this scan will serve as the baseline"

# Subsequent runs compare against baseline
aifiles /path --integrity
# → "✏️ 1 changed file(s): /path/to/modified_file.log"
```

### Integrity Log (JSONL)

Every integrity check appends to `report/integrity_check.jsonl`:

```json
{"timestamp": "2026-03-14T12:00:00", "status": "baseline", "new": 0, "changed": 0, "deleted": 0, "files": []}
{"timestamp": "2026-03-14T13:00:00", "status": "clean", "compared_to": "aifiles_20260314_120000.json", "new": 0, "changed": 0, "deleted": 0}
{"timestamp": "2026-03-14T14:00:00", "status": "changes_detected", "new": 1, "changed": 2, "deleted": 0, "files": [{"action": "modified", "path": "...", "previous_hash": "abc...", "current_hash": "def..."}]}
```

Import into Pandas:

```python
import pandas as pd
df = pd.read_json("report/integrity_check.jsonl", lines=True)
```

Compatible with: **jq**, **Pandas**, **ELK/Splunk**, **Datadog**, **grep**.

---

## Using Reports with Pandas

### Load a scan report

```python
import pandas as pd
import json

# Load the full scan report
with open("report/aifiles_20260314_120000.json") as f:
    data = json.load(f)

df = pd.DataFrame(data["files"])

# Quick overview
print(df.shape)                          # (81, 18)
print(df["category"].value_counts())     # Files per category

# Filter models over 100KB
big_models = df[(df["category"] == "model") & (df["size_bytes"] > 100_000)]
print(big_models[["filename", "size_human", "extension"]])

# Find all files with security risks
risky = df[df["risk_level"].isin(["danger", "warning"])]
print(risky[["filename", "risk_level", "risk_reason"]])

# Files with permission alerts
perm_issues = df[df["permission_alert"] != ""]
print(perm_issues[["filename", "permissions", "permission_alert", "permission_alert_reason"]])

# Total size by category
print(df.groupby("category")["size_bytes"].sum().sort_values(ascending=False))
```

### Load the integrity log

```python
import pandas as pd

# Load all integrity checks (JSONL → DataFrame in one line)
log = pd.read_json("report/integrity_check.jsonl", lines=True)

print(log[["timestamp", "status", "new", "changed", "deleted"]])
#             timestamp            status  new  changed  deleted
# 0  2026-03-14 12:00:00          baseline    0        0        0
# 1  2026-03-14 13:00:00             clean    0        0        0
# 2  2026-03-14 14:00:00  changes_detected    1        2        0

# Filter only checks that found changes
changes = log[log["status"] == "changes_detected"]

# Explode the files list to get one row per changed file
if not changes.empty:
    details = changes.explode("files").reset_index(drop=True)
    file_details = pd.json_normalize(details["files"])
    print(file_details[["action", "path", "filename"]])
```

### Pipe CLI output directly to Pandas

```bash
# Output JSON to stdout, pipe into a script
aifiles /path --json | python3 -c "
import sys, json, pandas as pd
df = pd.DataFrame(json.load(sys.stdin)['files'])
print(df.groupby('category')['size_bytes'].sum().sort_values(ascending=False))
"
```

---

## Detected Categories

| Category | Extensions / Patterns | Color |
|----------|----------------------|-------|
| **data** | .csv, .parquet, .jsonl, .tfrecord, .arrow, .feather, .h5, .hdf5, .npy, .npz, .avro, .orc, .petastorm, .nc, .zarr, .loom, .anndata, .beton, .mindrecord, .recordio, .idx, .shard, .wds | cyan |
| **model** | .pt, .pth, .onnx, .safetensors, .gguf, .ggml, .tflite, .mlmodel, .mar, .pte, .ckpt, .engine, .plan, .torchscript, .keras, .bento, .paddle, .pdparams, .pdmodel, .mnn, .ncnn, .xmodel, .om, .air, .mindir, .ort, .mlpackage, .pmml | red |
| **config** | .yaml, .yml, .json (with AI keywords), .ipynb, .pb | green |
| **vector** | .faiss, .annoy, .lancedb, .index, .hnsw, .usearch, .voy, .ivfpq, .scann, .hnswlib, .nmslib, .milvus, .db (with vector indicators) | purple |
| **checkpoint** | checkpoint dirs, .ckpt | yellow |
| **source** | .py, .js, .ts, .r, .jl, .rs, .sh, .go, .java, .cpp, .c, .rb, .scala, .kt, .swift, .lua | silver |
| **document** | .md, .rst, .tex, .log, .txt (via heuristic) | light blue |
| **multimedia** | .png, .jpg, .mp4, .wav, .mp3, .svg, .gif, .webp, .flac, .ogg, .avi, .mov, .mkv, .webm | pink |
| **skill** | .skill, .tool, + heuristic on yaml/json/md (LangChain, CrewAI, AutoGen, DSPy) | orange |
| **agent** | system prompts, agent/task/crew configs, .prompt, .cursorrules, AGENTS.md, OAI_CONFIG_LIST, MLmodel | bright green |
| **secret** | .pem, .key, .env, credentials.json, id_rsa, embedded API keys, AWS keys, GitHub tokens | red |

### Permission Alerts

When permissions are computed, AI File Discovery flags overly permissive files:

| Alert | Meaning |
|-------|---------|
| 🔴 **world_writable** | Anyone on the system can modify this file |
| 🟡 **world_readable** | Secrets or agent prompts readable by all users |
| 🟠 **group_writable** | Sensitive AI files writable by group members |
| ⚠️ **too_open** | Unusual execute permissions on non-source files |

These alerts appear in the dashboard table, tree view, and JSON reports.

---

## Custom Configuration

Create `aifiles.config.json` in your project root (or `~/.aifiles.json` for global config):

```json
{
  "extensions": {
    ".custom_model": ["model"],
    ".weights": ["model"],
    ".embeddings": ["vector", "data"],
    ".sql": ["source"],
    ".xml": ["config"]
  },
  "risks": {
    ".exe": { "level": "danger", "reason": "Executable — potential security risk" },
    ".wasm": { "level": "warning", "reason": "WebAssembly — review origin" }
  },
  "ambiguous": [".xml", ".toml"],
  "ignore_extensions": [".bak", ".tmp", ".swp"]
}
```

**Config lookup order:** explicit `--config` path → `aifiles.config.json` in CWD → `.aifiles.json` in CWD → `~/.aifiles.json` in HOME.

Valid categories: `data`, `model`, `config`, `vector`, `checkpoint`, `source`, `document`, `multimedia`, `skill`, `agent`, `secret`.

See [`aifiles.config.example.json`](aifiles.config.example.json) for a full annotated example.

---

## Test Data

Generate sample AI project structure for testing:

```bash
python3 generate_test_data.py
```

Run tests:

```bash
python3 -m pytest tests/ -v
```

---

## License

MIT
