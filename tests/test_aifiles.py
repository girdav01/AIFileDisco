"""Tests for AI File Storage Scanner."""

import json
import os
import tempfile
import unittest

# Allow running from repo root or tests/ directory
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from aifiles import (
    Category,
    FileClassifier,
    AIFileScanner,
    OutputFormatter,
    ScanConfig,
    format_size,
    parse_size,
    format_timestamp,
    _compute_file_hash,
    _get_permissions,
)


class TestFormatSize(unittest.TestCase):
    def test_bytes(self):
        self.assertEqual(format_size(0), "0 B")
        self.assertEqual(format_size(512), "512 B")
        self.assertEqual(format_size(1023), "1023 B")

    def test_kilobytes(self):
        self.assertEqual(format_size(1024), "1.0 KB")
        self.assertEqual(format_size(1536), "1.5 KB")

    def test_megabytes(self):
        self.assertEqual(format_size(1024 ** 2), "1.0 MB")

    def test_gigabytes(self):
        self.assertEqual(format_size(1024 ** 3), "1.0 GB")
        self.assertEqual(format_size(int(4.5 * 1024 ** 3)), "4.5 GB")

    def test_terabytes(self):
        self.assertEqual(format_size(1024 ** 4), "1.0 TB")


class TestParseSize(unittest.TestCase):
    def test_plain_bytes(self):
        self.assertEqual(parse_size("100"), 100)
        self.assertEqual(parse_size("100B"), 100)

    def test_kilobytes(self):
        self.assertEqual(parse_size("1KB"), 1024)
        self.assertEqual(parse_size("1K"), 1024)

    def test_megabytes(self):
        self.assertEqual(parse_size("10MB"), 10 * 1024 ** 2)
        self.assertEqual(parse_size("10M"), 10 * 1024 ** 2)

    def test_gigabytes(self):
        self.assertEqual(parse_size("1GB"), 1024 ** 3)

    def test_fractional(self):
        self.assertEqual(parse_size("1.5GB"), int(1.5 * 1024 ** 3))

    def test_case_insensitive(self):
        self.assertEqual(parse_size("10mb"), 10 * 1024 ** 2)
        self.assertEqual(parse_size("1gb"), 1024 ** 3)

    def test_whitespace(self):
        self.assertEqual(parse_size("  10 MB  "), 10 * 1024 ** 2)

    def test_invalid(self):
        with self.assertRaises(ValueError):
            parse_size("abc")
        with self.assertRaises(ValueError):
            parse_size("10XB")


class TestFormatTimestamp(unittest.TestCase):
    def test_format(self):
        # Just check it produces a string with expected shape
        result = format_timestamp(1700000000.0)
        self.assertRegex(result, r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}")


class TestFileClassifier(unittest.TestCase):
    def setUp(self):
        self.classifier = FileClassifier()
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _create_file(self, name: str, content: bytes = b"") -> str:
        path = os.path.join(self.tmpdir, name)
        with open(path, "wb") as f:
            f.write(content)
        return path

    def _scandir_entry(self, path: str):
        parent = os.path.dirname(path)
        for entry in os.scandir(parent):
            if entry.path == path:
                return entry
        raise FileNotFoundError(path)

    def test_deterministic_data_files(self):
        for ext in [".jsonl", ".csv", ".tsv", ".parquet", ".arrow", ".npy"]:
            path = self._create_file(f"test{ext}", b"data")
            entry = self._scandir_entry(path)
            result = self.classifier.classify_file(path, entry)
            self.assertIsNotNone(result, f"Should detect {ext}")
            self.assertIn(Category.DATA, result.categories, f"{ext} should be DATA")
            self.assertEqual(result.confidence, "high")
            os.unlink(path)

    def test_deterministic_model_files(self):
        for ext in [".onnx", ".pt", ".pth", ".safetensors", ".gguf", ".tflite"]:
            path = self._create_file(f"model{ext}", b"model data")
            entry = self._scandir_entry(path)
            result = self.classifier.classify_file(path, entry)
            self.assertIsNotNone(result, f"Should detect {ext}")
            self.assertIn(Category.MODEL, result.categories, f"{ext} should be MODEL")
            os.unlink(path)

    def test_ipynb_is_config(self):
        path = self._create_file("notebook.ipynb", b'{"cells": []}')
        entry = self._scandir_entry(path)
        result = self.classifier.classify_file(path, entry)
        self.assertIsNotNone(result)
        self.assertIn(Category.CONFIG, result.categories)

    def test_unknown_extension_ignored(self):
        path = self._create_file("readme.txt", b"hello")
        entry = self._scandir_entry(path)
        result = self.classifier.classify_file(path, entry)
        self.assertIsNone(result)

    def test_yaml_heuristic_positive(self):
        content = b"model:\n  learning_rate: 0.001\n  batch_size: 32\n  epochs: 10\n"
        path = self._create_file("config.yaml", content)
        entry = self._scandir_entry(path)
        result = self.classifier.classify_file(path, entry)
        self.assertIsNotNone(result)
        self.assertIn(Category.CONFIG, result.categories)
        self.assertEqual(result.detection_method, "heuristic")

    def test_yaml_heuristic_negative(self):
        content = b"services:\n  web:\n    image: nginx\n    ports:\n      - 80:80\n"
        path = self._create_file("docker-compose.yaml", content)
        entry = self._scandir_entry(path)
        result = self.classifier.classify_file(path, entry)
        self.assertIsNone(result)

    def test_bin_heuristic_gguf(self):
        path = self._create_file("model.bin", b"GGUF" + b"\x00" * 100)
        entry = self._scandir_entry(path)
        result = self.classifier.classify_file(path, entry)
        self.assertIsNotNone(result)
        self.assertIn(Category.MODEL, result.categories)

    def test_bin_heuristic_no_match(self):
        path = self._create_file("random.bin", b"\x00\x01\x02\x03" * 10)
        entry = self._scandir_entry(path)
        result = self.classifier.classify_file(path, entry)
        self.assertIsNone(result)

    def test_vector_extensions(self):
        for ext in [".faiss", ".annoy", ".lancedb"]:
            path = self._create_file(f"index{ext}", b"data")
            entry = self._scandir_entry(path)
            result = self.classifier.classify_file(path, entry)
            self.assertIsNotNone(result, f"Should detect {ext}")
            self.assertIn(Category.VECTOR, result.categories)
            os.unlink(path)

    def test_db_vector_sqlite(self):
        """SQLite .db with vector indicators → VECTOR."""
        # SQLite header + vector keyword in content
        content = b"SQLite format 3\x00" + b"\x00" * 100 + b"embedding vectors here"
        path = self._create_file("index.db", content)
        entry = self._scandir_entry(path)
        result = self.classifier.classify_file(path, entry)
        self.assertIsNotNone(result)
        self.assertIn(Category.VECTOR, result.categories)
        self.assertEqual(result.confidence, "medium")

    def test_db_regular_sqlite(self):
        """SQLite .db without vector indicators → DATA."""
        content = b"SQLite format 3\x00" + b"\x00" * 100 + b"CREATE TABLE users (id, name)"
        path = self._create_file("app.db", content)
        entry = self._scandir_entry(path)
        result = self.classifier.classify_file(path, entry)
        self.assertIsNotNone(result)
        self.assertIn(Category.DATA, result.categories)
        self.assertNotIn(Category.VECTOR, result.categories)
        self.assertEqual(result.confidence, "medium")

    def test_db_not_sqlite(self):
        """Non-SQLite .db file → not classified."""
        content = b"\x00\x01\x02\x03random binary data"
        path = self._create_file("random.db", content)
        entry = self._scandir_entry(path)
        result = self.classifier.classify_file(path, entry)
        self.assertIsNone(result)

    def test_sqlite_extension(self):
        """Files with .sqlite extension → classified based on content."""
        content = b"SQLite format 3\x00" + b"\x00" * 100 + b"CREATE TABLE logs (ts, msg)"
        path = self._create_file("logs.sqlite", content)
        entry = self._scandir_entry(path)
        result = self.classifier.classify_file(path, entry)
        self.assertIsNotNone(result)
        self.assertIn(Category.DATA, result.categories)

    def test_sqlite3_vector_extension(self):
        """.sqlite3 with vector content → VECTOR."""
        content = b"SQLite format 3\x00" + b"\x00" * 100 + b"chroma collection embeddings"
        path = self._create_file("vectors.sqlite3", content)
        entry = self._scandir_entry(path)
        result = self.classifier.classify_file(path, entry)
        self.assertIsNotNone(result)
        self.assertIn(Category.VECTOR, result.categories)

    def test_checkpoint_directory(self):
        ckpt_dir = os.path.join(self.tmpdir, "checkpoint-1500")
        os.makedirs(ckpt_dir)
        # Put a file inside
        with open(os.path.join(ckpt_dir, "model.bin"), "wb") as f:
            f.write(b"\x00" * 1024)

        for entry in os.scandir(self.tmpdir):
            if entry.name == "checkpoint-1500":
                result = self.classifier.classify_directory(entry.path, entry)
                self.assertIsNotNone(result)
                self.assertIn(Category.CHECKPOINT, result.categories)
                self.assertTrue(result.is_directory)
                self.assertEqual(result.size_bytes, 1024)
                break
        else:
            self.fail("checkpoint-1500 dir not found in scandir")

    def test_non_checkpoint_directory(self):
        normal_dir = os.path.join(self.tmpdir, "src")
        os.makedirs(normal_dir)
        for entry in os.scandir(self.tmpdir):
            if entry.name == "src":
                result = self.classifier.classify_directory(entry.path, entry)
                self.assertIsNone(result)
                break


class TestAIFileScanner(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _create_file(self, relpath: str, content: bytes = b"data") -> str:
        path = os.path.join(self.tmpdir, relpath)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(content)
        return path

    def test_basic_scan(self):
        self._create_file("data/train.parquet", b"x" * 1000)
        self._create_file("data/val.csv", b"a,b,c\n1,2,3\n")
        self._create_file("models/model.safetensors", b"tensors")
        self._create_file("src/main.py", b"print('hello')")

        config = ScanConfig(root_path=self.tmpdir, quiet=True)
        scanner = AIFileScanner(config, FileClassifier())
        results, summary = scanner.scan()

        self.assertEqual(summary.total_files, 4)
        categories = {c for r in results for c in r.categories}
        self.assertIn(Category.DATA, categories)
        self.assertIn(Category.MODEL, categories)
        self.assertIn(Category.SOURCE, categories)

    def test_min_size_filter(self):
        self._create_file("small.csv", b"a,b\n")
        self._create_file("big.parquet", b"x" * 10000)

        config = ScanConfig(root_path=self.tmpdir, min_size_bytes=1000, quiet=True)
        scanner = AIFileScanner(config, FileClassifier())
        results, summary = scanner.scan()

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].filename, "big.parquet")

    def test_category_filter(self):
        self._create_file("data.csv", b"data")
        self._create_file("model.pt", b"model")

        config = ScanConfig(root_path=self.tmpdir, categories=["model"], quiet=True)
        scanner = AIFileScanner(config, FileClassifier())
        results, _ = scanner.scan()

        self.assertEqual(len(results), 1)
        self.assertIn(Category.MODEL, results[0].categories)

    def test_max_depth(self):
        self._create_file("level0.csv", b"data")
        self._create_file("a/level1.csv", b"data")
        self._create_file("a/b/level2.csv", b"data")

        config = ScanConfig(root_path=self.tmpdir, max_depth=1, quiet=True)
        scanner = AIFileScanner(config, FileClassifier())
        results, _ = scanner.scan()

        paths = {r.filename for r in results}
        self.assertIn("level0.csv", paths)
        self.assertIn("level1.csv", paths)
        self.assertNotIn("level2.csv", paths)

    def test_skips_hidden_by_default(self):
        self._create_file(".hidden/secret.parquet", b"data")
        self._create_file("visible/data.parquet", b"data")

        config = ScanConfig(root_path=self.tmpdir, quiet=True)
        scanner = AIFileScanner(config, FileClassifier())
        results, _ = scanner.scan()

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].filename, "data.parquet")

    def test_include_hidden(self):
        self._create_file(".hidden/secret.parquet", b"data")
        self._create_file("visible/data.parquet", b"data")

        config = ScanConfig(root_path=self.tmpdir, include_hidden=True, quiet=True)
        scanner = AIFileScanner(config, FileClassifier())
        results, _ = scanner.scan()

        self.assertEqual(len(results), 2)

    def test_permission_error_handled(self):
        self._create_file("data.csv", b"data")
        restricted = os.path.join(self.tmpdir, "restricted")
        os.makedirs(restricted)
        os.chmod(restricted, 0o000)

        config = ScanConfig(root_path=self.tmpdir, quiet=True)
        scanner = AIFileScanner(config, FileClassifier())
        results, summary = scanner.scan()

        # Should find data.csv and have an error for restricted
        self.assertEqual(len(results), 1)
        self.assertTrue(len(summary.errors) > 0)

        # Cleanup
        os.chmod(restricted, 0o755)


class TestExport(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _create_file(self, relpath: str, content: bytes = b"data") -> str:
        path = os.path.join(self.tmpdir, relpath)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(content)
        return path

    def test_json_export(self):
        self._create_file("train.parquet", b"x" * 500)
        self._create_file("model.pt", b"y" * 1000)

        config = ScanConfig(root_path=self.tmpdir, quiet=True)
        scanner = AIFileScanner(config, FileClassifier())
        results, summary = scanner.scan()

        export_path = os.path.join(self.tmpdir, "report.json")
        formatter = OutputFormatter(use_color=False)
        formatter.export_json(results, summary, export_path)

        with open(export_path) as f:
            data = json.load(f)

        self.assertIn("scan_metadata", data)
        self.assertIn("files", data)
        self.assertEqual(data["scan_metadata"]["total_files"], 2)
        self.assertEqual(len(data["files"]), 2)

    def test_csv_export(self):
        self._create_file("data.jsonl", b'{"a": 1}\n')

        config = ScanConfig(root_path=self.tmpdir, quiet=True)
        scanner = AIFileScanner(config, FileClassifier())
        results, summary = scanner.scan()

        export_path = os.path.join(self.tmpdir, "report.csv")
        formatter = OutputFormatter(use_color=False)
        formatter.export_csv(results, summary, export_path)

        with open(export_path) as f:
            content = f.read()
        self.assertIn("path,filename,extension,category", content)
        self.assertIn(".jsonl", content)


class TestNewCategories(unittest.TestCase):
    """Tests for source, document, multimedia, and skill categories."""

    def setUp(self):
        self.classifier = FileClassifier()
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _create_file(self, name: str, content: bytes = b"") -> str:
        path = os.path.join(self.tmpdir, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(content)
        return path

    def _scandir_entry(self, path: str):
        parent = os.path.dirname(path)
        for entry in os.scandir(parent):
            if entry.path == path:
                return entry
        raise FileNotFoundError(path)

    # --- SOURCE ---
    def test_source_code_extensions(self):
        for ext in [".py", ".js", ".ts", ".java", ".cpp", ".go", ".rs", ".rb", ".r", ".jl"]:
            path = self._create_file(f"code{ext}", b"some code")
            entry = self._scandir_entry(path)
            result = self.classifier.classify_file(path, entry)
            self.assertIsNotNone(result, f"Should detect {ext}")
            self.assertIn(Category.SOURCE, result.categories, f"{ext} should be SOURCE")
            os.unlink(path)

    # --- DOCUMENT ---
    def test_document_extensions(self):
        for ext in [".md", ".markdown", ".rst", ".tex", ".adoc"]:
            path = self._create_file(f"doc{ext}", b"document content")
            entry = self._scandir_entry(path)
            result = self.classifier.classify_file(path, entry)
            self.assertIsNotNone(result, f"Should detect {ext}")
            self.assertIn(Category.DOCUMENT, result.categories, f"{ext} should be DOCUMENT")
            os.unlink(path)

    # --- MULTIMEDIA ---
    def test_multimedia_image_extensions(self):
        for ext in [".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"]:
            path = self._create_file(f"img{ext}", b"image data")
            entry = self._scandir_entry(path)
            result = self.classifier.classify_file(path, entry)
            self.assertIsNotNone(result, f"Should detect {ext}")
            self.assertIn(Category.MULTIMEDIA, result.categories, f"{ext} should be MULTIMEDIA")
            os.unlink(path)

    def test_multimedia_audio_extensions(self):
        for ext in [".wav", ".mp3", ".flac", ".ogg"]:
            path = self._create_file(f"audio{ext}", b"audio data")
            entry = self._scandir_entry(path)
            result = self.classifier.classify_file(path, entry)
            self.assertIsNotNone(result, f"Should detect {ext}")
            self.assertIn(Category.MULTIMEDIA, result.categories, f"{ext} should be MULTIMEDIA")
            os.unlink(path)

    def test_multimedia_video_extensions(self):
        for ext in [".mp4", ".avi", ".mov", ".mkv", ".webm"]:
            path = self._create_file(f"video{ext}", b"video data")
            entry = self._scandir_entry(path)
            result = self.classifier.classify_file(path, entry)
            self.assertIsNotNone(result, f"Should detect {ext}")
            self.assertIn(Category.MULTIMEDIA, result.categories, f"{ext} should be MULTIMEDIA")
            os.unlink(path)

    # --- SKILL ---
    def test_skill_deterministic_extension(self):
        path = self._create_file("agent.skill", b"skill definition")
        entry = self._scandir_entry(path)
        result = self.classifier.classify_file(path, entry)
        self.assertIsNotNone(result)
        self.assertIn(Category.SKILL, result.categories)

    def test_skill_md_filename(self):
        path = self._create_file("SKILL.md", b"# Skill definition\nSome skill content.")
        entry = self._scandir_entry(path)
        result = self.classifier.classify_file(path, entry)
        self.assertIsNotNone(result)
        self.assertIn(Category.SKILL, result.categories)

    def test_skill_heuristic_yaml(self):
        content = (
            b"name: search_agent\ntool_use:\n  type: function_call\n"
            b"actions:\n  - search\nlangchain:\n  tool_class: WebSearchTool\n"
        )
        path = self._create_file("search_agent.yaml", content)
        entry = self._scandir_entry(path)
        result = self.classifier.classify_file(path, entry)
        self.assertIsNotNone(result)
        self.assertIn(Category.SKILL, result.categories)

    def test_skill_heuristic_no_match_plain_yaml(self):
        """A YAML file with agent-like name but no skill keywords should not be SKILL."""
        content = b"name: my_agent\nversion: 1.0\nkey: value\n"
        path = self._create_file("my_agent.yaml", content)
        entry = self._scandir_entry(path)
        result = self.classifier.classify_file(path, entry)
        # Should either be None or CONFIG (via AI heuristic), but NOT SKILL
        if result is not None:
            self.assertNotIn(Category.SKILL, result.categories)


class TestRiskFlags(unittest.TestCase):
    """Tests for risk_level and risk_reason on FileResult."""

    def setUp(self):
        self.classifier = FileClassifier()
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _create_file(self, name: str, content: bytes = b"") -> str:
        path = os.path.join(self.tmpdir, name)
        with open(path, "wb") as f:
            f.write(content)
        return path

    def _scandir_entry(self, path: str):
        parent = os.path.dirname(path)
        for entry in os.scandir(parent):
            if entry.path == path:
                return entry
        raise FileNotFoundError(path)

    def test_pkl_danger(self):
        path = self._create_file("model.pkl", b"pickle data")
        entry = self._scandir_entry(path)
        result = self.classifier.classify_file(path, entry)
        self.assertIsNotNone(result)
        self.assertEqual(result.risk_level, "danger")
        self.assertIn("arbitrary code", result.risk_reason)

    def test_pickle_danger(self):
        path = self._create_file("model.pickle", b"pickle data")
        entry = self._scandir_entry(path)
        result = self.classifier.classify_file(path, entry)
        self.assertIsNotNone(result)
        self.assertEqual(result.risk_level, "danger")

    def test_joblib_warning(self):
        path = self._create_file("pipeline.joblib", b"joblib data")
        entry = self._scandir_entry(path)
        result = self.classifier.classify_file(path, entry)
        self.assertIsNotNone(result)
        self.assertEqual(result.risk_level, "warning")

    def test_safetensors_no_risk(self):
        path = self._create_file("model.safetensors", b"safe data")
        entry = self._scandir_entry(path)
        result = self.classifier.classify_file(path, entry)
        self.assertIsNotNone(result)
        self.assertEqual(result.risk_level, "none")

    def test_json_export_includes_risk(self):
        self._create_file("risky.pkl", b"pickle")
        config = ScanConfig(root_path=self.tmpdir, quiet=True)
        scanner = AIFileScanner(config, FileClassifier())
        results, summary = scanner.scan()

        export_path = os.path.join(self.tmpdir, "report.json")
        formatter = OutputFormatter(use_color=False)
        formatter.export_json(results, summary, export_path)

        with open(export_path) as f:
            data = json.load(f)

        pkl_file = data["files"][0]
        self.assertEqual(pkl_file["risk_level"], "danger")
        self.assertIn("arbitrary code", pkl_file["risk_reason"])


class TestMultiCategory(unittest.TestCase):
    """Tests for multi-category file classification."""

    def setUp(self):
        self.classifier = FileClassifier()
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _create_file(self, name: str, content: bytes = b"") -> str:
        path = os.path.join(self.tmpdir, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(content)
        return path

    def _scandir_entry(self, path: str):
        parent = os.path.dirname(path)
        for entry in os.scandir(parent):
            if entry.path == path:
                return entry
        raise FileNotFoundError(path)

    def test_ipynb_is_config_and_source(self):
        path = self._create_file("notebook.ipynb", b'{"cells": []}')
        entry = self._scandir_entry(path)
        result = self.classifier.classify_file(path, entry)
        self.assertIsNotNone(result)
        self.assertIn(Category.CONFIG, result.categories)
        self.assertIn(Category.SOURCE, result.categories)
        self.assertEqual(len(result.categories), 2)

    def test_pkl_is_model_and_data(self):
        path = self._create_file("model.pkl", b"pickle data")
        entry = self._scandir_entry(path)
        result = self.classifier.classify_file(path, entry)
        self.assertIsNotNone(result)
        self.assertIn(Category.MODEL, result.categories)
        self.assertIn(Category.DATA, result.categories)

    def test_skill_md_is_skill_and_document(self):
        path = self._create_file("SKILL.md", b"# Skill\nDefinition here.")
        entry = self._scandir_entry(path)
        result = self.classifier.classify_file(path, entry)
        self.assertIsNotNone(result)
        self.assertIn(Category.SKILL, result.categories)
        self.assertIn(Category.DOCUMENT, result.categories)

    def test_skill_yaml_is_skill_and_config(self):
        content = (
            b"name: search_agent\ntool_use:\n  type: function_call\n"
            b"actions:\n  - search\nlangchain:\n  tool_class: WebSearchTool\n"
        )
        path = self._create_file("search_agent.yaml", content)
        entry = self._scandir_entry(path)
        result = self.classifier.classify_file(path, entry)
        self.assertIsNotNone(result)
        self.assertIn(Category.SKILL, result.categories)
        self.assertIn(Category.CONFIG, result.categories)

    def test_single_category_file_has_list(self):
        path = self._create_file("data.csv", b"a,b\n1,2\n")
        entry = self._scandir_entry(path)
        result = self.classifier.classify_file(path, entry)
        self.assertIsNotNone(result)
        self.assertIsInstance(result.categories, list)
        self.assertEqual(len(result.categories), 1)
        self.assertEqual(result.categories[0], Category.DATA)

    def test_category_filter_any_match(self):
        """Multi-category files appear when filtering by ANY of their categories."""
        self._create_file("model.pkl", b"pickle")  # MODEL + DATA
        self._create_file("plain.csv", b"data")    # DATA only

        # Filter by "data" should include both
        config = ScanConfig(root_path=self.tmpdir, categories=["data"], quiet=True)
        scanner = AIFileScanner(config, FileClassifier())
        results, _ = scanner.scan()
        filenames = {r.filename for r in results}
        self.assertIn("model.pkl", filenames)
        self.assertIn("plain.csv", filenames)

        # Filter by "model" should include only pkl
        config2 = ScanConfig(root_path=self.tmpdir, categories=["model"], quiet=True)
        scanner2 = AIFileScanner(config2, FileClassifier())
        results2, _ = scanner2.scan()
        filenames2 = {r.filename for r in results2}
        self.assertIn("model.pkl", filenames2)
        self.assertNotIn("plain.csv", filenames2)

    def test_summary_counts_multi_category(self):
        """Multi-category files count in each of their categories."""
        self._create_file("model.pkl", b"x" * 100)  # MODEL + DATA

        config = ScanConfig(root_path=self.tmpdir, quiet=True)
        scanner = AIFileScanner(config, FileClassifier())
        results, summary = scanner.scan()

        self.assertEqual(summary.total_files, 1)
        self.assertIn("model", summary.by_category)
        self.assertIn("data", summary.by_category)
        self.assertEqual(summary.by_category["model"]["count"], 1)
        self.assertEqual(summary.by_category["data"]["count"], 1)

    def test_json_export_includes_categories(self):
        self._create_file("notebook.ipynb", b'{"cells": []}')
        config = ScanConfig(root_path=self.tmpdir, quiet=True)
        scanner = AIFileScanner(config, FileClassifier())
        results, summary = scanner.scan()

        export_path = os.path.join(self.tmpdir, "report.json")
        formatter = OutputFormatter(use_color=False)
        formatter.export_json(results, summary, export_path)

        with open(export_path) as f:
            data = json.load(f)

        file_entry = data["files"][0]
        self.assertIn("categories", file_entry)
        self.assertIn("config", file_entry["categories"])
        self.assertIn("source", file_entry["categories"])
        # backward compat — "category" is first category
        self.assertEqual(file_entry["category"], "config")


class TestAgentCategory(unittest.TestCase):
    """Tests for AGENT category detection."""

    def setUp(self):
        self.classifier = FileClassifier()
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _create_file(self, name: str, content: bytes = b"") -> str:
        path = os.path.join(self.tmpdir, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(content)
        return path

    def _scandir_entry(self, path: str):
        parent = os.path.dirname(path)
        for entry in os.scandir(parent):
            if entry.path == path:
                return entry
        raise FileNotFoundError(path)

    def test_agent_deterministic_extensions(self):
        """Test .prompt, .systemprompt, .goal, .persona, .instruction → AGENT+CONFIG."""
        for ext in [".prompt", ".systemprompt", ".goal", ".persona", ".instruction"]:
            path = self._create_file(f"test{ext}", b"agent content")
            entry = self._scandir_entry(path)
            result = self.classifier.classify_file(path, entry)
            self.assertIsNotNone(result, f"Should detect {ext}")
            self.assertIn(Category.AGENT, result.categories, f"{ext} should be AGENT")
            self.assertIn(Category.CONFIG, result.categories, f"{ext} should also be CONFIG")
            os.unlink(path)

    def test_agent_special_filename_md(self):
        """system_prompt.md → AGENT + DOCUMENT."""
        path = self._create_file("system_prompt.md", b"# System Prompt\nYou are a helpful assistant.")
        entry = self._scandir_entry(path)
        result = self.classifier.classify_file(path, entry)
        self.assertIsNotNone(result)
        self.assertIn(Category.AGENT, result.categories)
        self.assertIn(Category.DOCUMENT, result.categories)

    def test_agent_special_filename_yaml(self):
        """agent_config.yaml → AGENT + CONFIG."""
        path = self._create_file("agent_config.yaml", b"name: assistant\nrole: helper")
        entry = self._scandir_entry(path)
        result = self.classifier.classify_file(path, entry)
        self.assertIsNotNone(result)
        self.assertIn(Category.AGENT, result.categories)
        self.assertIn(Category.CONFIG, result.categories)

    def test_agent_heuristic_positive(self):
        """YAML with agent filename pattern + 2+ keywords → AGENT+CONFIG."""
        content = (
            b"system_prompt: You are a helpful assistant\n"
            b"your role is to answer questions\n"
            b"persona: friendly\n"
        )
        path = self._create_file("prompt_template.yaml", content)
        entry = self._scandir_entry(path)
        result = self.classifier.classify_file(path, entry)
        self.assertIsNotNone(result)
        self.assertIn(Category.AGENT, result.categories)
        self.assertIn(Category.CONFIG, result.categories)

    def test_agent_heuristic_negative(self):
        """File with agent-like name but no keywords → should NOT be AGENT."""
        content = b"name: config\nversion: 1.0\nkey: value\n"
        path = self._create_file("goal_tracker.yaml", content)
        entry = self._scandir_entry(path)
        result = self.classifier.classify_file(path, entry)
        # This will be picked up by yaml heuristic if it has AI keywords, or skipped
        if result is not None:
            self.assertNotIn(Category.AGENT, result.categories)

    def test_goals_py_not_agent(self):
        """goals.py should be SOURCE, not AGENT (only yaml/json/md/txt qualify)."""
        path = self._create_file("goals.py", b"def main():\n    pass\n")
        entry = self._scandir_entry(path)
        result = self.classifier.classify_file(path, entry)
        self.assertIsNotNone(result)
        self.assertIn(Category.SOURCE, result.categories)
        self.assertNotIn(Category.AGENT, result.categories)

    def test_agent_txt_heuristic(self):
        """A .txt file matching agent pattern + keywords → AGENT+DOCUMENT."""
        content = (
            b"You are a helpful assistant.\n"
            b"Your role is to answer questions about AI.\n"
            b"Instructions: always be polite\n"
        )
        path = self._create_file("instructions.txt", content)
        entry = self._scandir_entry(path)
        result = self.classifier.classify_file(path, entry)
        self.assertIsNotNone(result)
        self.assertIn(Category.AGENT, result.categories)
        self.assertIn(Category.DOCUMENT, result.categories)


class TestSecretDetection(unittest.TestCase):
    """Tests for SECRET category and embedded secret detection."""

    def setUp(self):
        self.classifier = FileClassifier()
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _create_file(self, name: str, content: bytes = b"") -> str:
        path = os.path.join(self.tmpdir, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(content)
        return path

    def _scandir_entry(self, path: str):
        parent = os.path.dirname(path)
        for entry in os.scandir(parent):
            if entry.path == path:
                return entry
        raise FileNotFoundError(path)

    def test_secret_extensions(self):
        """Test .pem, .key, .p12 etc. → SECRET + danger."""
        for ext in [".pem", ".key", ".p12", ".credentials", ".secret", ".token"]:
            path = self._create_file(f"test{ext}", b"secret data")
            entry = self._scandir_entry(path)
            result = self.classifier.classify_file(path, entry)
            self.assertIsNotNone(result, f"Should detect {ext}")
            self.assertIn(Category.SECRET, result.categories, f"{ext} should be SECRET")
            self.assertEqual(result.risk_level, "danger", f"{ext} should be danger")
            os.unlink(path)

    def test_secret_filename_credentials_json(self):
        """credentials.json → SECRET + danger."""
        path = self._create_file("credentials.json", b'{"key": "value"}')
        entry = self._scandir_entry(path)
        result = self.classifier.classify_file(path, entry)
        self.assertIsNotNone(result)
        self.assertIn(Category.SECRET, result.categories)
        self.assertEqual(result.risk_level, "danger")

    def test_secret_filename_id_rsa(self):
        """id_rsa → SECRET + danger."""
        path = self._create_file("id_rsa", b"-----BEGIN RSA PRIVATE KEY-----")
        entry = self._scandir_entry(path)
        result = self.classifier.classify_file(path, entry)
        self.assertIsNotNone(result)
        self.assertIn(Category.SECRET, result.categories)
        self.assertEqual(result.risk_level, "danger")

    def test_secret_content_heuristic(self):
        """Source file with embedded API key → risk flag upgrade."""
        content = b'API_KEY = "sk-1234567890abcdef1234567890abcdef"\nprint("hello")\n'
        path = self._create_file("config_loader.py", content)
        entry = self._scandir_entry(path)
        result = self.classifier.classify_file(path, entry)
        self.assertIsNotNone(result)
        self.assertIn(Category.SOURCE, result.categories)
        # Apply the secret scan
        self.classifier.apply_secret_scan(result)
        self.assertEqual(result.risk_level, "danger")
        self.assertIn("embedded secret", result.risk_reason)

    def test_secret_no_false_positive(self):
        """Normal source code should not trigger secret detection."""
        content = b'print("hello world")\nx = 42\ndef foo(): pass\n'
        path = self._create_file("normal.py", content)
        entry = self._scandir_entry(path)
        result = self.classifier.classify_file(path, entry)
        self.assertIsNotNone(result)
        self.classifier.apply_secret_scan(result)
        self.assertEqual(result.risk_level, "none")

    def test_binary_not_scanned_for_secrets(self):
        """Model files should not be scanned for secret content."""
        content = b'API_KEY = "sk-1234567890abcdef1234567890abcdef"'
        path = self._create_file("model.safetensors", content)
        entry = self._scandir_entry(path)
        result = self.classifier.classify_file(path, entry)
        self.assertIsNotNone(result)
        self.classifier.apply_secret_scan(result)
        self.assertEqual(result.risk_level, "none")  # MODEL is binary, skipped

    def test_aws_key_pattern(self):
        """AWS access key pattern detected."""
        content = b'aws_access_key_id = AKIAIOSFODNN7EXAMPLE\n'
        path = self._create_file("deploy.sh", content)
        entry = self._scandir_entry(path)
        result = self.classifier.classify_file(path, entry)
        self.assertIsNotNone(result)
        self.classifier.apply_secret_scan(result)
        self.assertEqual(result.risk_level, "danger")

    def test_github_token_pattern(self):
        """GitHub personal access token detected."""
        content = b'GITHUB_TOKEN=ghp_abcdefghijklmnopqrstuvwxyz0123456789\n'
        path = self._create_file("ci.sh", content)
        entry = self._scandir_entry(path)
        result = self.classifier.classify_file(path, entry)
        self.assertIsNotNone(result)
        self.classifier.apply_secret_scan(result)
        self.assertEqual(result.risk_level, "danger")

    def test_private_key_pattern(self):
        """Private key header detected."""
        content = b'-----BEGIN RSA PRIVATE KEY-----\nMIIE...\n'
        path = self._create_file("setup.py", content)
        entry = self._scandir_entry(path)
        result = self.classifier.classify_file(path, entry)
        self.assertIsNotNone(result)
        self.classifier.apply_secret_scan(result)
        self.assertEqual(result.risk_level, "danger")


class TestPermissions(unittest.TestCase):
    """Tests for on-demand file permission computation."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _create_file(self, relpath: str, content: bytes = b"data") -> str:
        path = os.path.join(self.tmpdir, relpath)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(content)
        return path

    def test_permissions_disabled_by_default(self):
        """Permissions fields are empty when compute_permissions=False."""
        self._create_file("data/train.csv", b"a,b\n1,2\n")
        config = ScanConfig(root_path=self.tmpdir, quiet=True)
        scanner = AIFileScanner(config, FileClassifier())
        results, _ = scanner.scan()
        self.assertTrue(len(results) > 0)
        for r in results:
            self.assertEqual(r.owner, "")
            self.assertEqual(r.group, "")
            self.assertEqual(r.permissions, "")

    def test_permissions_enabled(self):
        """Permissions are populated when compute_permissions=True."""
        self._create_file("data/train.csv", b"a,b\n1,2\n")
        config = ScanConfig(root_path=self.tmpdir, quiet=True, compute_permissions=True)
        scanner = AIFileScanner(config, FileClassifier())
        results, _ = scanner.scan()
        self.assertTrue(len(results) > 0)
        for r in results:
            self.assertNotEqual(r.owner, "")
            self.assertNotEqual(r.permissions, "")
            # Permissions string starts with - (file) or d (dir)
            self.assertIn(r.permissions[0], "-dlcbps")

    def test_permissions_helper(self):
        """_get_permissions returns valid tuple."""
        path = self._create_file("test.csv", b"data")
        owner, group, perms = _get_permissions(path)
        self.assertNotEqual(owner, "")
        self.assertNotEqual(perms, "")
        self.assertTrue(perms.startswith("-"))

    def test_permissions_nonexistent(self):
        """_get_permissions returns empty tuple for nonexistent path."""
        owner, group, perms = _get_permissions("/nonexistent/path/file.txt")
        self.assertEqual(owner, "")
        self.assertEqual(group, "")
        self.assertEqual(perms, "")


class TestHashing(unittest.TestCase):
    """Tests for on-demand file and directory hashing."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _create_file(self, relpath: str, content: bytes = b"data") -> str:
        path = os.path.join(self.tmpdir, relpath)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(content)
        return path

    def test_hashes_disabled_by_default(self):
        """Hash field is empty when compute_hashes=False."""
        self._create_file("data/train.csv", b"a,b\n1,2\n")
        config = ScanConfig(root_path=self.tmpdir, quiet=True)
        scanner = AIFileScanner(config, FileClassifier())
        results, _ = scanner.scan()
        self.assertTrue(len(results) > 0)
        for r in results:
            self.assertEqual(r.hash, "")

    def test_hashes_enabled(self):
        """Hash is populated when compute_hashes=True."""
        self._create_file("data/train.csv", b"a,b\n1,2\n")
        config = ScanConfig(root_path=self.tmpdir, quiet=True, compute_hashes=True)
        scanner = AIFileScanner(config, FileClassifier())
        results, _ = scanner.scan()
        self.assertTrue(len(results) > 0)
        for r in results:
            self.assertEqual(len(r.hash), 64)
            # Hex string
            int(r.hash, 16)

    def test_hash_correctness(self):
        """Hash matches manually computed SHA-256."""
        import hashlib
        content = b"known content for hashing test"
        path = self._create_file("test.csv", content)
        expected = hashlib.sha256(content).hexdigest()
        result = _compute_file_hash(path)
        self.assertEqual(result, expected)

    def test_hash_deterministic(self):
        """Same file produces same hash on repeated calls."""
        path = self._create_file("test.csv", b"deterministic content")
        h1 = _compute_file_hash(path)
        h2 = _compute_file_hash(path)
        self.assertEqual(h1, h2)

    def test_hash_nonexistent(self):
        """Hash of nonexistent file returns empty string."""
        result = _compute_file_hash("/nonexistent/path.txt")
        self.assertEqual(result, "")

    def test_directory_hash(self):
        """Checkpoint directory gets a hash when compute_hashes=True."""
        # Create a checkpoint directory structure
        self._create_file("checkpoint-1000/config.json", b'{"model":"test"}')
        self._create_file("checkpoint-1000/model.safetensors", b"weights")
        config = ScanConfig(root_path=self.tmpdir, quiet=True, compute_hashes=True)
        scanner = AIFileScanner(config, FileClassifier())
        results, _ = scanner.scan()
        dir_results = [r for r in results if r.is_directory]
        self.assertTrue(len(dir_results) > 0)
        for r in dir_results:
            self.assertEqual(len(r.hash), 64)

    def test_directory_hash_changes(self):
        """Directory hash changes when file content changes."""
        self._create_file("checkpoint-1000/config.json", b'{"version":1}')
        self._create_file("checkpoint-1000/model.safetensors", b"weights_v1")
        config = ScanConfig(root_path=self.tmpdir, quiet=True, compute_hashes=True)
        scanner = AIFileScanner(config, FileClassifier())
        results1, _ = scanner.scan()
        dir_hash1 = [r for r in results1 if r.is_directory][0].hash

        # Modify a file inside the checkpoint
        with open(os.path.join(self.tmpdir, "checkpoint-1000/model.safetensors"), "wb") as f:
            f.write(b"weights_v2_modified")

        scanner2 = AIFileScanner(config, FileClassifier())
        results2, _ = scanner2.scan()
        dir_hash2 = [r for r in results2 if r.is_directory][0].hash

        self.assertNotEqual(dir_hash1, dir_hash2)


if __name__ == "__main__":
    unittest.main()
