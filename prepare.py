#!/usr/bin/env python3
"""Prepare 3GPP or AMF log splits, or train MLX-LM on a prepared split.

Examples:
  python prepare.py                         # rebuild 3GPP train/valid/test splits
  python prepare.py --corpus logs            # prepare logs/ AMF log splits
  python prepare.py --corpus logs --train    # train one epoch on AMF logs
  python prepare.py --corpus logs --evaluate --adapter-path ./adapters/<saved-adapter>
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import random
import re
import subprocess
import sys
import threading
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CORPORA = {
    "3gpp": {
        "source_dir": ROOT / "3gpp_specs" / "23-series-text",
        "data_dir": ROOT / "data",
        "run_name": "23gpp",
        "document_label": "3GPP document",
    },
    "logs": {
        "source_dir": ROOT / "logs",
        "data_dir": ROOT / "data" / "logs",
        "run_name": "logs",
        "document_label": "AMF log",
    },
}
RUNS_DIR = ROOT / "runs"
CHUNK_SIZE = 1600
OVERLAP = 200
SPLIT_SEED = 42
DEFAULT_MODEL = "mlx-community/Qwen3-1.7B-4bit"
RESOURCE_SAMPLE_INTERVAL_SEC = 1.0

# These are measurement definitions, not claimed results. Quality target values
# must be agreed against the intended product/SLA before they can be scored.
PERFORMANCE_ROWS = [
    ("Correctness (%)", "", "Score each answer against an expert-approved answer key on a held-out corpus-specific QA set; report exact/semantic correctness.", "Set from product acceptance criteria.", "Measures whether users receive accurate answers.", "Incorrect answers can cause rework, defects, or operational risk.", "Pending QA benchmark"),
    ("Groundedness (%)", "", "For each answer, verify every factual claim against the supplied source passages; human-review a sample.", "Set from product acceptance criteria.", "Measures whether answers are supported by the source corpus.", "Grounded responses improve trust and auditability.", "Pending QA benchmark and retrieval/citation setup"),
    ("Hallucination rate (%)", "", "On the held-out QA set, count answers with unsupported or contradicted claims / all scored answers.", "Set a maximum acceptable rate before evaluation.", "Measures unsupported claims that can mislead standards users.", "Lower risk of incorrect engineering decisions and costly corrections.", "Pending QA benchmark and expert review"),
    ("Latency (p95 sec)", "", "Run the same representative prompts repeatedly against the final adapter; report the 95th percentile end-to-end response time.", "Set from product SLA.", "Measures responsiveness for normal and slower requests.", "Slow answers reduce adoption and interrupt workflows.", "Pending representative prompt benchmark"),
    ("Cost per 1,000 tasks", "", "Measure local energy/runtime or hosted inference charges for 1,000 defined tasks; state hardware and energy/price assumptions.", "Set an approved cost ceiling.", "Compares operating cost at expected usage.", "Supports deployment and capacity planning.", "Pending task definition and cost inputs"),
    ("Recall (if applicable)", "N/A", "For RAG only: Recall@k = relevant source chunks retrieved in top k / all relevant chunks, using labeled queries.", "Set from retrieval acceptance criteria.", "Measures whether the system finds the relevant standards passages.", "Missed clauses can lead to incomplete answers.", "Not measured by this fine-tuning run; requires RAG and labeled relevance set"),
    ("License & data residency", "", "Record the base-model license and corpus-use rights; confirm where data, logs, and inference run and whether any external service receives them.", "Must satisfy legal, security, and data-residency policy.", "Checks whether the model and source data can be used in the intended setting.", "Avoids licensing, privacy, and residency blockers.", "Review model card, corpus rights, and deployment path"),
    ("Weighted score", "", "Apply pre-agreed metric weights to normalized, measured results; document weights and scoring formula.", "Set weights before comparing models.", "Combines quality, speed, and cost into one decision aid.", "Makes model selection traceable to business priorities.", "Pending measured inputs and approved weights"),
    ("Recommendation", "", "Decide against the acceptance criteria after reviewing QA, latency, cost, license, and operational results.", "Pass all mandatory quality/compliance gates.", "Turns measured results into a deployment decision.", "Reduces deployment risk and focuses follow-up work.", "Pending evaluation"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--train", action="store_true", help="Train one epoch using the existing train split; evaluation is not run.")
    mode.add_argument("--evaluate", action="store_true", help="Evaluate a saved adapter on the full test split; optionally add --qa-eval for latency and answer review.")
    parser.add_argument("--corpus", choices=tuple(CORPORA), default="3gpp", help="Corpus to prepare, train, or evaluate (default: 3gpp).")
    parser.add_argument("--evaluate-split", choices=("valid", "test"), default="test", help="Split to score with --evaluate (default: test).")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"MLX model or Hugging Face model id (default: {DEFAULT_MODEL}).")
    parser.add_argument("--adapter-path", default=None, help="Output adapter path. Defaults to a new timestamped directory under adapters/.")
    parser.add_argument("--qa-train", default=None, help="Optional chat-format QA JSONL to train on instead of the document-chunk train split.")
    parser.add_argument("--resume-adapter-file", default=None, help="Continue fine-tuning from an existing MLX adapter checkpoint file.")
    parser.add_argument("--qa-test", default=None, help="Optional chat-format QA JSONL to score with --evaluate instead of the selected corpus split.")
    parser.add_argument("--qa-gold", default=None, help="Optional QA answer/evidence JSONL keyed by id; copied to predictions after generation and never shown in the prompt.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--qa-eval", default=None, help="Optional expert-authored QA JSONL for post-training response latency and answer review.")
    parser.add_argument("--qa-generation-tokens", type=int, default=256, help="Maximum answer tokens during the QA evaluation.")
    parser.add_argument("--score-qa", default=None, help="Aggregate human_review labels in a qa_predictions.jsonl file and update its run CSV.")
    parser.add_argument("--refresh-metrics", default=None, help="Re-parse train.log in an existing run directory and refresh run.json/performance_metrics.csv.")
    return parser.parse_args()


def iter_text_chunks(path: Path):
    """Yield overlapping character chunks without reading a whole log into memory."""
    buffer = ""
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        while block := handle.read(64 * 1024):
            buffer += block
            while len(buffer) > CHUNK_SIZE:
                yield buffer[:CHUNK_SIZE]
                buffer = buffer[CHUNK_SIZE - OVERLAP:]
    if buffer:
        yield buffer


def document_group(path: Path, corpus: str) -> str:
    """Keep related log replicas together to reduce train/holdout leakage."""
    if corpus == "logs":
        # Strip pod/node and replica suffixes, e.g. -n0-1 or -1-0.
        return re.sub(r"-(?:n\d+|\d+)-\d+$", "", path.stem, flags=re.IGNORECASE)
    return re.sub(r"v\d.*$", "", path.stem, flags=re.IGNORECASE)


def prepare_dataset(corpus: str = "3gpp") -> None:
    config = CORPORA[corpus]
    source_dir = config["source_dir"]
    output_dir = config["data_dir"]
    if not source_dir.is_dir():
        raise SystemExit(f"Text directory not found: {source_dir}")

    paths = sorted(
        path for path in source_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in ({".md", ".txt"} if corpus == "3gpp" else {".log", ".txt", ".md"})
    )
    if not paths:
        raise SystemExit(f"No supported text files found in {source_dir}")

    groups: dict[str, list[Path]] = {}
    seen_content: set[str] = set()
    duplicate_count = 0
    empty_count = 0
    for path in paths:
        if corpus == "3gpp":
            text = path.read_text(encoding="utf-8", errors="ignore").strip()
            if not text:
                empty_count += 1
                continue
            digest = hashlib.sha256(" ".join(text.split()).encode("utf-8")).hexdigest()
            if digest in seen_content:
                duplicate_count += 1
                continue
            seen_content.add(digest)
        groups.setdefault(document_group(path, corpus), []).append(path)

    group_keys = sorted(groups)
    random.Random(SPLIT_SEED).shuffle(group_keys)
    n = len(group_keys)
    if n < 3:
        raise SystemExit(f"Need at least three unique {corpus} document groups for train/valid/test splits; found {n}.")

    n_test = max(1, n // 10)
    n_valid = max(1, n // 10)
    if n_test + n_valid >= n:
        n_test = n_valid = 1
    test_groups = set(group_keys[:n_test])
    valid_groups = set(group_keys[n_test:n_test + n_valid])
    train_groups = set(group_keys[n_test + n_valid:])
    assignments = {"train": train_groups, "valid": valid_groups, "test": test_groups}

    output_dir.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    records_by_file: dict[str, int] = {}
    group_splits = {
        group: split
        for split, split_groups in assignments.items()
        for group in split_groups
    }
    handles = {
        name: (output_dir / f"{name}.jsonl").open("w", encoding="utf-8")
        for name in assignments
    }
    try:
        for group, docs in groups.items():
            split = group_splits[group]
            for path in docs:
                file_records = 0
                for raw_chunk in iter_text_chunks(path):
                    chunk = raw_chunk.strip()
                    if chunk:
                        record = {"text": f"{config['document_label']}: {path.name}\n\n{chunk}"}
                        handles[split].write(json.dumps(record, ensure_ascii=False) + "\n")
                        counts[split] += 1
                        file_records += 1
                records_by_file[path.relative_to(source_dir).as_posix()] = file_records
    finally:
        for handle in handles.values():
            handle.close()

    split_counts = {name: counts[name] for name in assignments}
    split_files = {
        name: output_dir / f"{name}.jsonl"
        for name in assignments
    }
    output_file_stats = {
        name: {
            "path": str(path),
            "records": split_counts[name],
            "size_bytes": path.stat().st_size,
        }
        for name, path in split_files.items()
    }
    source_file_stats = []
    for group, docs in groups.items():
        for path in docs:
            relative_path = path.relative_to(source_dir).as_posix()
            source_file_stats.append({
                "path": relative_path,
                "group": group,
                "split": group_splits[group],
                "size_bytes": path.stat().st_size,
                "records": records_by_file.get(relative_path, 0),
            })
    total_records = sum(split_counts.values())
    source_bytes = sum(path.stat().st_size for path in paths)
    output_bytes = sum(row["size_bytes"] for row in output_file_stats.values())
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "corpus": corpus,
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "source_files_scanned": len(paths),
        "source_bytes": source_bytes,
        "document_groups": len(groups),
        "group_to_split": group_splits,
        "records_by_split": split_counts,
        "total_records": total_records,
        "chunking": {
            "chunk_size_chars": CHUNK_SIZE,
            "overlap_chars": OVERLAP,
            "step_chars": CHUNK_SIZE - OVERLAP,
            "split_seed": SPLIT_SEED,
        },
        "duplicate_files_skipped": duplicate_count,
        "empty_3gpp_files_skipped": empty_count,
        "output_files": output_file_stats,
        "output_bytes": output_bytes,
        "source_file_stats": source_file_stats,
    }
    manifest_path = output_dir / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"Corpus: {corpus}")
    print(f"Scanned files: {len(paths)}")
    if corpus == "3gpp":
        print(f"Unique documents used: {len(seen_content)}")
        print(f"Exact duplicate files skipped: {duplicate_count}")
        print(f"Empty files skipped: {empty_count}")
    print(f"Document groups: {len(groups)}")
    print(f"Records written: train={split_counts['train']}, valid={split_counts['valid']}, test={split_counts['test']}, total={total_records}")
    print(f"Chunk settings: size={CHUNK_SIZE} chars, overlap={OVERLAP} chars, step={CHUNK_SIZE - OVERLAP} chars, seed={SPLIT_SEED}")
    print(f"Source size: {source_bytes / (1024 ** 3):.2f} GiB")
    print(f"Dataset size: {output_bytes / (1024 ** 3):.2f} GiB")
    print(f"Dataset files: {output_dir}")
    print(f"Dataset stats: {manifest_path}")


def inspect_jsonl(path: Path) -> tuple[int, str]:
    if not path.is_file():
        raise SystemExit(f"Required dataset file not found: {path}\nRun `python prepare.py` first.")
    digest = hashlib.sha256()
    count = 0
    with path.open("rb") as raw:
        for line_number, raw_line in enumerate(raw, 1):
            digest.update(raw_line)
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid JSON in {path.name}, line {line_number}: {exc}") from exc
            valid_text = isinstance(record, dict) and isinstance(record.get("text"), str)
            messages = record.get("messages") if isinstance(record, dict) else None
            valid_chat = (
                isinstance(messages, list)
                and len(messages) > 0
                and all(
                    isinstance(message, dict)
                    and isinstance(message.get("role"), str)
                    and isinstance(message.get("content"), str)
                    for message in messages
                )
            )
            if not (valid_text or valid_chat):
                raise SystemExit(
                    f"{path.name}, line {line_number} must contain either a string `text` field "
                    "or a chat `messages` list with string `role` and `content` fields."
                )
            count += 1
    if count == 0:
        raise SystemExit(f"Dataset file is empty: {path}")
    return count, digest.hexdigest()


def parse_training_log(log_text: str) -> dict:
    """Extract metrics from both MLX-LM's rich progress rows and verbose reports."""
    clean = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", log_text).replace("\r", "\n")
    rows = re.findall(
        r"(?m)^\s*(?P<iteration>[0-9][0-9,]*)\s+(?P<loss>[0-9]+\.[0-9]+)\s+[▲▼]\s+(?P<tps>[0-9]+(?:\.[0-9]+)?)\s+(?P<tokens>[0-9][0-9,.]*[kKmM]?)\s*$",
        clean,
    )
    result = {}
    if rows:
        iteration, loss, tps, trained_tokens = rows[-1]
        amount = float(re.sub(r"[kKmM]$", "", trained_tokens).replace(",", ""))
        suffix = trained_tokens[-1].lower() if trained_tokens[-1].isalpha() else ""
        multiplier = {"k": 1_000, "m": 1_000_000}.get(suffix, 1)
        result.update({
            "final_iteration": int(iteration.replace(",", "")),
            "final_train_loss": float(loss),
            "final_tokens_per_second": float(tps),
            "trained_tokens": int(amount * multiplier),
        })
    # Compatibility with MLX-LM output variants that print named metric labels.
    train_reports = re.findall(r"Train loss\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?).*?Tokens/sec\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?)", clean, re.IGNORECASE)
    if train_reports:
        result.setdefault("final_train_loss", float(train_reports[-1][0]))
        result.setdefault("final_tokens_per_second", float(train_reports[-1][1]))
    val_reports = re.findall(r"Val loss\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?)", clean, re.IGNORECASE)
    if val_reports:
        result["final_valid_loss"] = float(val_reports[-1])
    test_report = re.search(r"Test loss\s+([0-9]+(?:\.[0-9]+)?),\s*Test ppl\s+([0-9]+(?:\.[0-9]+)?)", clean)
    if test_report:
        result["test_loss"] = float(test_report.group(1))
        result["test_perplexity"] = float(test_report.group(2))
    peak_reports = re.findall(r"Peak mem\s+([0-9]+(?:\.[0-9]+)?)\s*GB", clean, re.IGNORECASE)
    if peak_reports:
        result["peak_mlx_memory_gb"] = max(map(float, peak_reports))
    return result


def parse_evaluation_log(log_text: str, example_count: int) -> dict:
    clean = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", log_text).replace("\r", "\n")
    # tqdm emits elapsed time and rolling examples/sec on the final progress row.
    progress = re.findall(r"\[(?:(\d+):)?(\d{2}):(\d{2})<[^]]*?(\d+(?:\.\d+)?)it/s", clean)
    result = {"evaluation_examples": example_count}
    if progress:
        hours, minutes, seconds, reported_rate = progress[-1]
        elapsed = int(hours or 0) * 3600 + int(minutes) * 60 + int(seconds)
        result["evaluation_duration_sec"] = elapsed
        result["evaluation_examples_per_sec"] = round(example_count / elapsed, 3) if elapsed else None
        result["evaluation_final_reported_examples_per_sec"] = float(reported_rate)
    return result


class ProcessResourceMonitor:
    """Sample CPU percentage and resident memory for one MLX-LM process."""

    def __init__(self, pid: int, interval_sec: float = RESOURCE_SAMPLE_INTERVAL_SEC):
        self.pid = pid
        self.interval_sec = interval_sec
        self.samples: list[tuple[float, float]] = []  # (cpu percent, RSS KiB)
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._sample_loop, daemon=True)

    def _sample(self) -> tuple[float, float] | None:
        try:
            result = subprocess.run(
                ["ps", "-p", str(self.pid), "-o", "%cpu=", "-o", "rss="],
                capture_output=True,
                text=True,
                check=False,
            )
            values = result.stdout.split()
            if result.returncode != 0 or len(values) < 2:
                return None
            return float(values[0]), float(values[1])
        except (OSError, ValueError):
            return None

    def _sample_loop(self) -> None:
        while not self.stop_event.is_set():
            sample = self._sample()
            if sample is not None:
                self.samples.append(sample)
            self.stop_event.wait(self.interval_sec)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> dict:
        self.stop_event.set()
        self.thread.join()
        if not self.samples:
            return {
                "resource_monitor": "ps process sampling",
                "resource_sample_interval_sec": self.interval_sec,
                "resource_samples": 0,
            }
        cpu = [sample[0] for sample in self.samples]
        rss_gb = [sample[1] / (1024 ** 2) for sample in self.samples]
        return {
            "resource_monitor": "ps process sampling",
            "resource_sample_interval_sec": self.interval_sec,
            "resource_samples": len(self.samples),
            "process_cpu_average_percent": round(sum(cpu) / len(cpu), 2),
            "process_cpu_peak_percent": round(max(cpu), 2),
            "process_rss_average_gb": round(sum(rss_gb) / len(rss_gb), 3),
            "process_rss_peak_gb": round(max(rss_gb), 3),
        }


def run_logged_command(command: list[str], log_path: Path) -> tuple[int, dict]:
    """Run an MLX command while recording its output and process resource use."""
    with log_path.open("w", encoding="utf-8") as log:
        log.write("Command: " + " ".join(command) + "\n\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        monitor = ProcessResourceMonitor(process.pid)
        monitor.start()
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
            log.flush()
        return_code = process.wait()
        resource_metrics = monitor.stop()
    return return_code, resource_metrics


def print_resource_summary(metrics: dict) -> None:
    if not metrics.get("resource_samples"):
        print("CPU/RAM usage was not captured (no process samples).")
        return
    print(
        "Run resources: "
        f"CPU {metrics['process_cpu_average_percent']}% average / {metrics['process_cpu_peak_percent']}% peak; "
        f"RAM {metrics['process_rss_average_gb']} GB average / {metrics['process_rss_peak_gb']} GB peak "
        f"({metrics['resource_samples']} samples)."
    )


def refresh_run_metrics(run_dir: Path) -> None:
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    manifest_path = run_dir / "run.json"
    train_log_path = run_dir / "train.log"
    eval_log_path = run_dir / "evaluation.log"
    if not manifest_path.is_file() or not (train_log_path.is_file() or eval_log_path.is_file()):
        raise SystemExit(f"Expected run.json and train.log or evaluation.log in {run_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    refreshed = []
    if train_log_path.is_file():
        manifest.update(parse_training_log(train_log_path.read_text(encoding="utf-8", errors="replace")))
        manifest["log_file"] = str(train_log_path)
        refreshed.append(train_log_path)
    if eval_log_path.is_file():
        eval_rows = int(manifest.get("evaluation_rows", 0))
        manifest.update(parse_evaluation_log(eval_log_path.read_text(encoding="utf-8", errors="replace"), eval_rows))
        manifest["evaluation_log"] = str(eval_log_path)
        refreshed.append(eval_log_path)
    manifest["performance_csv"] = str(run_dir / "performance_metrics.csv")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    write_performance_template(run_dir / "performance_metrics.csv", manifest)
    print("Refreshed metrics from " + ", ".join(str(path) for path in refreshed))
    print(f"Updated {run_dir / 'performance_metrics.csv'}")


def write_performance_template(path: Path, run_data: dict) -> None:
    fields = ["parameter", "actual", "target", "how_to_measure", "why_it_matters", "business_impact", "status"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        quality_rows = {
            "Correctness (%)": ("correctness_percent", "correct"),
            "Groundedness (%)": ("groundedness_percent", "grounded"),
            "Hallucination rate (%)": ("hallucination_percent", "hallucinated"),
        }
        for name, actual, how, target, why, impact, status in PERFORMANCE_ROWS:
            if name == "Latency (p95 sec)" and run_data.get("latency_p95_sec") is not None:
                actual = run_data["latency_p95_sec"]
                status = f"Measured over {run_data.get('qa_eval_count', 0)} QA prompts; see qa_predictions.jsonl"
            if name in quality_rows and run_data.get("qa_predictions_file"):
                status = "Awaiting expert labels in qa_predictions.jsonl"
            if name in quality_rows and run_data.get(quality_rows[name][0]) is not None:
                metric_key, label_key = quality_rows[name]
                actual = run_data[metric_key]
                reviewed = run_data.get("human_review_counts", {}).get(label_key, 0)
                status = f"Human reviewed ({reviewed} labeled answers)"
            writer.writerow({"parameter": name, "actual": actual, "target": target, "how_to_measure": how, "why_it_matters": why, "business_impact": impact, "status": status})
        writer.writerow({"parameter": "Training loss (final report)", "actual": run_data.get("final_train_loss", ""), "target": "Compare with initial/final validation and held-out test loss; lower alone does not prove QA quality.", "how_to_measure": "Read final MLX-LM training loss from train.log.", "why_it_matters": "Tracks next-token objective fit on training batches.", "business_impact": "Detects optimization progress; does not establish answer correctness.", "status": "Measured from training log" if run_data.get("final_train_loss") is not None else "See train.log"})
        valid_actual = run_data.get("final_valid_loss", "") if run_data.get("final_valid_loss") is not None else "Not run"
        valid_status = "Measured from training log" if run_data.get("final_valid_loss") is not None else ("Not run; evaluation deferred" if run_data.get("status") == "completed" else "Pending evaluation")
        if run_data.get("evaluation_split") == "valid" and run_data.get("evaluation_loss") is not None:
            valid_actual = run_data["evaluation_loss"]
            valid_status = "Measured on full valid split"
        writer.writerow({"parameter": "Validation loss (final report)", "actual": valid_actual, "target": "Should improve or remain stable without a widening train/validation gap.", "how_to_measure": "Run a separate validation evaluation after training.", "why_it_matters": "Checks generalization to held-out document groups.", "business_impact": "Flags overfitting before relying on the adapter.", "status": valid_status})
        eval_loss = run_data.get("evaluation_loss", run_data.get("test_loss"))
        eval_ppl = run_data.get("evaluation_perplexity", run_data.get("test_perplexity"))
        eval_actual = f"{eval_loss} / {eval_ppl}" if eval_loss is not None else "Not run"
        eval_split = run_data.get("evaluation_split", "test")
        eval_status = f"Measured on full {eval_split} split" if eval_loss is not None else ("Not run; evaluation deferred" if run_data.get("status") == "completed" else "Pending evaluation")
        writer.writerow({"parameter": "Evaluation loss / perplexity", "actual": eval_actual, "target": "Compare base and tuned model on the same held-out split.", "how_to_measure": "Run prepare.py --corpus <corpus> --evaluate --evaluate-split valid|test --adapter-path <adapter>; scores every row in the selected split.", "why_it_matters": "Measures held-out next-token prediction, not task-answer correctness.", "business_impact": "Helps detect regression or overfitting on unseen document groups.", "status": eval_status})
        throughput = f"{run_data['final_tokens_per_second']} tokens/sec" if run_data.get("final_tokens_per_second") is not None else "Not captured"
        if run_data.get("peak_mlx_memory_gb") is not None:
            peak_memory = f"{run_data['peak_mlx_memory_gb']} GB MLX peak"
        elif run_data.get("process_rss_peak_gb", run_data.get("peak_child_process_rss_gb")) is not None:
            peak_memory = f"{run_data.get('process_rss_peak_gb', run_data.get('peak_child_process_rss_gb'))} GB process RSS (OS metric)"
        else:
            peak_memory = "Not captured"
        writer.writerow({"parameter": "Training throughput / peak memory", "actual": f"{throughput}; {peak_memory}", "target": "Record device, software versions, sequence length, and batch size for fair comparisons.", "how_to_measure": "Tokens/sec from MLX progress output; process RSS sampled from the MLX-LM process. RSS is an OS metric, not MLX allocator peak.", "why_it_matters": "Quantifies local training speed and memory use.", "business_impact": "Supports hardware and training-time planning.", "status": "Throughput measured; process RSS captured" if run_data.get("final_tokens_per_second") is not None and run_data.get("process_rss_peak_gb", run_data.get("peak_child_process_rss_gb")) is not None else ("Throughput measured; memory unavailable" if run_data.get("final_tokens_per_second") is not None else "No training report parsed")})
        cpu_average = run_data.get("process_cpu_average_percent")
        cpu_peak = run_data.get("process_cpu_peak_percent")
        rss_average = run_data.get("process_rss_average_gb")
        rss_peak = run_data.get("process_rss_peak_gb")
        if run_data.get("resource_samples", 0):
            resource_actual = f"CPU {cpu_average}% average / {cpu_peak}% peak; RAM {rss_average} GB average / {rss_peak} GB peak; {run_data['resource_samples']} samples"
            resource_status = f"Sampled every {run_data.get('resource_sample_interval_sec', RESOURCE_SAMPLE_INTERVAL_SEC)} sec with ps"
        else:
            resource_actual = "Not captured"
            resource_status = "No process samples captured"
        writer.writerow({"parameter": "Run CPU / RAM usage", "actual": resource_actual, "target": "Compare runs on the same device and sampling interval.", "how_to_measure": "Sample MLX-LM process CPU percentage and resident set size (RSS) every second; RSS is resident system memory.", "why_it_matters": "Shows CPU load and memory footprint during this run.", "business_impact": "Supports hardware capacity and run-cost planning.", "status": resource_status})
        eval_summary = "Not run"
        if run_data.get("evaluation_rows") is not None:
            eval_summary = f"{run_data.get('evaluation_split', 'unknown')}: {run_data['evaluation_rows']} examples"
            if run_data.get("evaluation_duration_sec") is not None:
                eval_summary += f" / {run_data['evaluation_duration_sec']} sec / {run_data.get('evaluation_examples_per_sec', 'n/a')} examples/sec"
        writer.writerow({"parameter": "Evaluation split coverage / throughput", "actual": eval_summary, "target": "Score all rows in the selected held-out split.", "how_to_measure": "Dataset row count and complete evaluation progress log.", "why_it_matters": "Confirms which holdout was scored and whether it completed.", "business_impact": "Makes evaluation coverage and runtime auditable.", "status": f"Measured on {run_data.get('evaluation_split', 'selected split')}" if run_data.get("evaluation_rows") is not None and run_data.get("evaluation_duration_sec") is not None else ("Row count known; duration unavailable" if run_data.get("evaluation_rows") is not None else "No evaluation run")})
        writer.writerow({"parameter": "Training duration / final iteration / trained tokens", "actual": f"{run_data.get('elapsed_seconds', 'Not captured')} sec / {run_data.get('final_iteration', 'Not captured')} / {run_data.get('trained_tokens', 'Not captured')}", "target": "Compare runs on the same device and dataset.", "how_to_measure": "Wall-clock duration plus final MLX progress row.", "why_it_matters": "Shows total run cost and confirms training progress.", "business_impact": "Helps estimate future fine-tuning time.", "status": "Captured" if run_data.get("elapsed_seconds") is not None else "Not captured"})


def load_qa_eval(path: Path) -> list[dict]:
    if not path.is_file():
        raise SystemExit(f"QA evaluation file not found: {path}")
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid JSON in {path.name}, line {line_number}: {exc}") from exc
            if not isinstance(row, dict) or not isinstance(row.get("question"), str) or not row["question"].strip():
                raise SystemExit(f"{path.name}, line {line_number} needs a non-empty string `question` field.")
            if row["question"].startswith("REPLACE_"):
                raise SystemExit(f"Replace template text in {path.name}, line {line_number} with a real expert-authored question.")
            rows.append(row)
    if len(rows) < 20:
        raise SystemExit(f"Use at least 20 representative QA prompts for a useful p95 latency estimate; found {len(rows)}.")
    return rows


def load_qa_gold(path: Path) -> dict[str, dict]:
    if not path.is_file():
        raise SystemExit(f"QA gold file not found: {path}")
    gold = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid JSON in {path.name}, line {line_number}: {exc}") from exc
            if not isinstance(row, dict) or row.get("id") is None:
                raise SystemExit(f"{path.name}, line {line_number} needs an `id` field.")
            key = str(row["id"])
            if key in gold:
                raise SystemExit(f"Duplicate QA gold id {key!r} in {path.name}.")
            gold[key] = row
    return gold


def benchmark_qa(model_id: str, adapter_path: Path, qa_file: Path, output_path: Path, max_tokens: int, gold_file: Path | None = None, corpus: str = "3gpp") -> dict:
    """Measure response latency; attach optional gold answers after generation."""
    import mlx.core as mx
    from mlx_lm import load, stream_generate

    rows = load_qa_eval(qa_file)
    gold = load_qa_gold(gold_file) if gold_file else {}
    if gold:
        missing = [str(row.get("id")) for row in rows if str(row.get("id")) not in gold]
        if missing:
            raise SystemExit(f"QA gold is missing {len(missing)} prompt ids; first missing id: {missing[0]}")
    model, tokenizer = load(model_id, adapter_path=str(adapter_path))
    def make_prompt(row: dict) -> str:
        if corpus == "3gpp":
            instruction = "Answer the 3GPP standards question using the supplied reference excerpt when present. If the excerpt does not support an answer, say so. Include the specification and clause when known."
        else:
            instruction = "Answer the question using the supplied AMF log excerpt when present. If the excerpt does not support an answer, say so, and identify the relevant log source or event details when possible."
        prompt_parts = [instruction]
        if row.get("reference_context"):
            prompt_parts.append("Reference excerpt:\n" + str(row["reference_context"]))
        prompt_parts.append("Question:\n" + row["question"])
        user_content = "\n\n".join(prompt_parts)
        messages = [{"role": "user", "content": user_content}]
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except (AttributeError, TypeError, ValueError):
            return user_content + "\n\nAnswer:"

    prompts = [make_prompt(row) for row in rows]
    # Warm up generation so the p95 reflects steady-state requests, not first-use overhead.
    for _ in stream_generate(model, tokenizer, prompts[0], max_tokens=max_tokens):
        pass
    mx.synchronize()

    results = []
    latencies = []
    peak_memory_gb = None
    for row, prompt in zip(rows, prompts):
        start = time.perf_counter()
        response_text = []
        last_response = None
        for response in stream_generate(model, tokenizer, prompt, max_tokens=max_tokens):
            response_text.append(response.text)
            last_response = response
        mx.synchronize()
        elapsed = time.perf_counter() - start
        latencies.append(elapsed)
        peak = getattr(last_response, "peak_memory", None)
        if peak is not None:
            peak_memory_gb = max(peak_memory_gb or 0.0, float(peak))
        result_id = row.get("id", len(results) + 1)
        reference = gold.get(str(result_id), {})
        results.append({
            "id": result_id,
            "question": row["question"],
            "reference_answer": reference.get("reference_answer", row.get("reference_answer", "")),
            "reference_context": reference.get("reference_context", row.get("reference_context", "")),
            "source_ids": reference.get("source_ids", row.get("source_ids", [])),
            "model_answer": "".join(response_text),
            "latency_seconds": round(elapsed, 6),
            "prompt_tokens_per_second": getattr(last_response, "prompt_tps", None),
            "generation_tokens_per_second": getattr(last_response, "generation_tps", None),
            "peak_mlx_memory_gb": peak,
            "human_review": {"correct": None, "grounded": None, "hallucinated": None, "notes": ""},
        })
    output_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in results), encoding="utf-8")
    ordered = sorted(latencies)
    p95 = ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]
    return {
        "qa_eval_file": str(qa_file),
        "qa_predictions_file": str(output_path),
        "qa_eval_count": len(results),
        "latency_p50_sec": ordered[math.ceil(0.50 * len(ordered)) - 1],
        "latency_p95_sec": p95,
        "mean_latency_sec": sum(ordered) / len(ordered),
        "latency_warmup_requests": 1,
        "inference_peak_mlx_memory_gb": peak_memory_gb,
        "quality_metrics_note": "Answers exported for expert review. Correctness, groundedness, and hallucination rate are not automatically scored; fill human_review fields in qa_predictions.jsonl using the supplied answer/evidence.",
    }


def score_qa_annotations(predictions_path: Path) -> dict:
    if not predictions_path.is_file():
        raise SystemExit(f"QA predictions file not found: {predictions_path}")
    labels = {"correct": [], "grounded": [], "hallucinated": []}
    with predictions_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid JSON in {predictions_path.name}, line {line_number}: {exc}") from exc
            review = row.get("human_review", {})
            for key in labels:
                value = review.get(key)
                if value is not None:
                    if not isinstance(value, bool):
                        raise SystemExit(f"{predictions_path.name}, line {line_number}: human_review.{key} must be true, false, or null.")
                    labels[key].append(value)
    result = {"human_review_counts": {}, "human_review_source": str(predictions_path)}
    fields = {"correct": "correctness_percent", "grounded": "groundedness_percent", "hallucinated": "hallucination_percent"}
    for key, metric in fields.items():
        values = labels[key]
        result["human_review_counts"][key] = len(values)
        result[metric] = round(100.0 * sum(values) / len(values), 2) if values else None
    return result


def sync_evaluation_to_training_run(evaluation_manifest: dict) -> None:
    """Copy the latest evaluation metrics into the adapter's training run report."""
    training_dir_value = evaluation_manifest.get("training_run_dir")
    if not training_dir_value:
        return
    training_dir = Path(training_dir_value)
    training_manifest_path = training_dir / "run.json"
    if not training_manifest_path.is_file():
        return
    training_manifest = json.loads(training_manifest_path.read_text(encoding="utf-8"))
    fields = (
        "evaluation_only", "evaluation_split", "evaluation_rows", "evaluation_sha256",
        "evaluation_duration_sec", "evaluation_examples_per_sec", "evaluation_loss",
        "evaluation_perplexity", "evaluation_log", "qa_eval_file", "qa_gold_file",
        "qa_test_file", "qa_predictions_file", "qa_eval_count", "latency_p50_sec",
        "latency_p95_sec", "mean_latency_sec", "latency_warmup_requests",
        "inference_peak_mlx_memory_gb", "quality_metrics_note", "correctness_percent",
        "groundedness_percent", "hallucination_percent", "human_review_counts",
        "human_review_source",
    )
    for key in fields:
        if key in evaluation_manifest:
            training_manifest[key] = evaluation_manifest[key]
    training_manifest["latest_evaluation_run_id"] = evaluation_manifest.get("run_id")
    training_manifest["latest_evaluation_run_dir"] = str(evaluation_manifest.get("run_dir", ""))
    training_manifest["latest_evaluation_report"] = str(evaluation_manifest.get("performance_csv", ""))
    training_manifest_path.write_text(json.dumps(training_manifest, indent=2) + "\n", encoding="utf-8")
    write_performance_template(training_dir / "performance_metrics.csv", training_manifest)
    print(f"Updated training report: {training_dir / 'performance_metrics.csv'}")


def score_saved_qa(predictions_path: Path) -> None:
    metrics = score_qa_annotations(predictions_path)
    run_dir = predictions_path.parent
    manifest_path = run_dir / "run.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    manifest.update(metrics)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    write_performance_template(run_dir / "performance_metrics.csv", manifest)
    sync_evaluation_to_training_run(manifest)
    print(f"Updated human-scored percentages in {run_dir / 'performance_metrics.csv'}")
    print("Labels counted (correct / grounded / hallucinated): " + "/".join(str(metrics["human_review_counts"][key]) for key in ("correct", "grounded", "hallucinated")))


def write_qa_template(path: Path) -> None:
    if path.exists():
        return
    example = {
        "id": "q1",
        "question": "REPLACE_WITH_AN_EXPERT_AUTHORED_3GPP_QUESTION",
        "reference_answer": "REPLACE_WITH_AN_EXPERT_VERIFIED_ANSWER",
        "reference_context": "PASTE_THE_RELEVANT_3GPP_CLAUSE_OR_EXCERPT",
        "source_ids": ["TS 23.xxx clause x.y"],
    }
    path.write_text(json.dumps(example, ensure_ascii=False) + "\n", encoding="utf-8")


def train(args: argparse.Namespace) -> None:
    config = CORPORA[args.corpus]
    output_dir = config["data_dir"]
    run_name = config["run_name"]
    if args.batch_size < 1 or args.num_layers < 1 or args.max_seq_length < 1:
        raise SystemExit("batch-size, num-layers, and max-seq-length must be positive integers.")

    if args.qa_generation_tokens < 1:
        raise SystemExit("qa-generation-tokens must be positive.")
    if args.qa_eval:
        raise SystemExit("Training-only mode does not run QA evaluation. Run `python prepare.py --evaluate --adapter-path <adapter> --qa-eval data/qa_eval.jsonl` later.")

    if args.qa_test or args.qa_gold:
        raise SystemExit("--qa-test and --qa-gold are evaluation-only options.")
    resume_adapter_file = None
    if args.resume_adapter_file:
        resume_adapter_file = Path(args.resume_adapter_file)
        if not resume_adapter_file.is_absolute():
            resume_adapter_file = ROOT / resume_adapter_file
        if not resume_adapter_file.is_file():
            raise SystemExit(f"Resume adapter file not found: {resume_adapter_file}")
    counts = {}
    hashes = {}
    if args.qa_train:
        train_data_path = Path(args.qa_train)
        if not train_data_path.is_absolute():
            train_data_path = ROOT / train_data_path
        counts["train"], hashes["train"] = inspect_jsonl(train_data_path)
    else:
        train_data_path = output_dir / "train.jsonl"
        for split in ("train", "valid", "test"):
            counts[split], hashes[split] = inspect_jsonl(output_dir / f"{split}.jsonl")
    train_rows = counts["train"]
    if train_rows % args.batch_size:
        raise SystemExit(
            f"Train rows ({train_rows}) must be divisible by batch-size ({args.batch_size}) "
            "to guarantee one complete pass without dropping a partial batch."
        )
    iters = train_rows // args.batch_size
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    adapter_path = Path(args.adapter_path) if args.adapter_path else ROOT / "adapters" / f"qwen3-1.7b-{run_name}-full-{run_id}"
    if not adapter_path.is_absolute():
        adapter_path = ROOT / adapter_path
    if adapter_path.exists() and any(adapter_path.iterdir()):
        raise SystemExit(f"Adapter output already exists and is non-empty: {adapter_path}\nChoose a new --adapter-path to preserve the existing run.")
    adapter_path.mkdir(parents=True, exist_ok=True)

    run_dir = RUNS_DIR / f"qwen3-1.7b-{run_name}-{run_id}"
    run_dir.mkdir(parents=True, exist_ok=False)
    log_path = run_dir / "train.log"
    # Keep validation/test JSONL out of MLX-LM's data directory so --train stays training-only.
    training_data_dir = run_dir / "data"
    training_data_dir.mkdir()
    (training_data_dir / "train.jsonl").symlink_to(train_data_path)
    metrics_path = run_dir / "performance_metrics.csv"
    manifest_path = run_dir / "run.json"

    command = [
        "mlx_lm.lora", "--model", args.model, "--train",
        "--data", str(training_data_dir), "--adapter-path", str(adapter_path),
        "--batch-size", str(args.batch_size), "--num-layers", str(args.num_layers),
        "--max-seq-length", str(args.max_seq_length), "--learning-rate", str(args.learning_rate),
        "--grad-checkpoint", "--iters", str(iters), "--save-every", "5000",
        "--steps-per-report", "100", "--steps-per-eval", "1000",
        "--val-batches", "25", "--seed", str(args.seed),
    ]
    if resume_adapter_file:
        command.extend(["--resume-adapter-file", str(resume_adapter_file)])
    manifest = {
        "run_id": run_id,
        "corpus": args.corpus,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "model": args.model,
        "adapter_path": str(adapter_path),
        "resume_adapter_file": str(resume_adapter_file) if resume_adapter_file else None,
        "data_dir": str(train_data_path.parent),
        "training_dataset_type": "chat_qa" if args.qa_train else "document_chunks",
        "training_dataset_file": str(train_data_path),
        "dataset_rows": counts,
        "dataset_sha256": hashes,
        "epochs_requested": 1,
        "steps_requested": iters,
        "batch_size": args.batch_size,
        "max_seq_length_tokens": args.max_seq_length,
        "num_lora_layers": args.num_layers,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "python_version": sys.version,
        "platform": platform.platform(),
        "command": command,
        "notes": f"Each train.jsonl row is scheduled once in this epoch; examples are tokenized/truncated to max_seq_length_tokens. This training-only run uses a train-only data directory, so validation and test evaluation are deferred. Corpus: {args.corpus}.",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Training one epoch: {train_rows} rows, batch size {args.batch_size}, {iters} optimizer iterations.")
    print(f"Adapter: {adapter_path}")
    print(f"Run artifacts: {run_dir}")
    print("Note: rows longer than max-seq-length are truncated by MLX-LM for training.")

    start = time.perf_counter()
    return_code = 1
    resource_metrics = {}
    try:
        return_code, resource_metrics = run_logged_command(command, log_path)
    except FileNotFoundError as exc:
        print(f"Could not start MLX-LM: {exc}", file=sys.stderr)
        raise SystemExit("Activate the environment with mlx-lm installed, then rerun this command.") from exc
    finally:
        elapsed = time.perf_counter() - start
        log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
        manifest.update(parse_training_log(log_text))
        manifest.update(resource_metrics)
        manifest.update({
            "status": "completed" if return_code == 0 else "failed",
            "return_code": return_code,
            "elapsed_seconds": round(elapsed, 3),
            "log_file": str(log_path),
            "performance_csv": str(metrics_path),
        })
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        write_performance_template(metrics_path, manifest)
    if return_code != 0:
        raise SystemExit(f"MLX-LM exited with status {return_code}. See {log_path}")
    print(f"\nSaved run record: {manifest_path}")
    print(f"Saved performance table: {metrics_path}")
    print(f"Saved complete console log: {log_path}")
    print_resource_summary(resource_metrics)


def evaluate(args: argparse.Namespace) -> None:
    config = CORPORA[args.corpus]
    output_dir = config["data_dir"]
    run_name = config["run_name"]
    if not args.adapter_path:
        raise SystemExit("--evaluate requires --adapter-path pointing to the adapter produced by --train.")
    adapter_path = Path(args.adapter_path)
    if not adapter_path.is_absolute():
        adapter_path = ROOT / adapter_path
    if not adapter_path.is_dir():
        raise SystemExit(f"Adapter directory not found: {adapter_path}")

    if args.qa_train:
        raise SystemExit("--qa-train is training-only; do not pass it with --evaluate.")
    evaluation_split = args.evaluate_split
    if args.qa_test:
        evaluation_data_path = Path(args.qa_test)
        if not evaluation_data_path.is_absolute():
            evaluation_data_path = ROOT / evaluation_data_path
        evaluation_split = "qa_test"
    else:
        evaluation_data_path = output_dir / f"{evaluation_split}.jsonl"
    evaluation_rows, evaluation_hash = inspect_jsonl(evaluation_data_path)
    qa_file = None
    if args.qa_eval:
        qa_file = Path(args.qa_eval)
        if not qa_file.is_absolute():
            qa_file = ROOT / qa_file
        load_qa_eval(qa_file)
    qa_gold_file = None
    if args.qa_gold:
        if qa_file is None:
            raise SystemExit("--qa-gold requires --qa-eval.")
        qa_gold_file = Path(args.qa_gold)
        if not qa_gold_file.is_absolute():
            qa_gold_file = ROOT / qa_gold_file
        load_qa_gold(qa_gold_file)

    expected_adapter_path = adapter_path.resolve()
    matching_training_runs = []
    for candidate_manifest in RUNS_DIR.glob(f"qwen3-1.7b-{run_name}-*/run.json"):
        try:
            candidate = json.loads(candidate_manifest.read_text(encoding="utf-8"))
            candidate_adapter = candidate.get("adapter_path")
            if candidate_adapter and not candidate.get("evaluation_only") and Path(candidate_adapter).resolve() == expected_adapter_path:
                matching_training_runs.append((candidate_manifest.stat().st_mtime, candidate_manifest.parent))
        except (OSError, json.JSONDecodeError):
            continue
    training_run_dir = max(matching_training_runs, default=(0, None), key=lambda x: x[0])[1]

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = RUNS_DIR / f"qwen3-1.7b-{run_name}-evaluation-{run_id}"
    run_dir.mkdir(parents=True, exist_ok=False)
    log_path = run_dir / "evaluation.log"
    metrics_path = run_dir / "performance_metrics.csv"
    manifest_path = run_dir / "run.json"
    evaluation_data_dir = run_dir / "data"
    evaluation_data_dir.mkdir()
    (evaluation_data_dir / "test.jsonl").symlink_to(evaluation_data_path)
    command = ["mlx_lm.lora", "--model", args.model, "--adapter-path", str(adapter_path), "--data", str(evaluation_data_dir), "--test", "--test-batches", "-1", "--batch-size", str(args.batch_size), "--max-seq-length", str(args.max_seq_length)]
    manifest = {
        "run_id": run_id,
        "corpus": args.corpus,
        "status": "running",
        "evaluation_only": True,
        "model": args.model,
        "adapter_path": str(adapter_path),
        "training_run_dir": str(training_run_dir) if training_run_dir else None,
        "evaluation_split": evaluation_split,
        "evaluation_rows": evaluation_rows,
        "evaluation_sha256": evaluation_hash,
        "qa_eval_file": str(qa_file) if qa_file else None,
        "qa_gold_file": str(qa_gold_file) if qa_gold_file else None,
        "qa_test_file": str(evaluation_data_path) if args.qa_test else None,
        "command": command,
        "platform": platform.platform(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    evaluation_start = time.perf_counter()
    return_code, resource_metrics = run_logged_command(command, log_path)
    elapsed = time.perf_counter() - evaluation_start
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    test_match = re.search(r"Test loss\s+([0-9]+(?:\.[0-9]+)?),\s*Test ppl\s+([0-9]+(?:\.[0-9]+)?)", log_text)
    manifest.update(parse_evaluation_log(log_text, evaluation_rows))
    manifest.update({
        "status": "completed" if return_code == 0 else "failed",
        "return_code": return_code,
        "evaluation_duration_sec": round(elapsed, 3),
        "evaluation_examples_per_sec": round(evaluation_rows / elapsed, 3) if elapsed else None,
        "evaluation_loss": float(test_match.group(1)) if test_match else None,
        "evaluation_perplexity": float(test_match.group(2)) if test_match else None,
        "evaluation_log": str(log_path),
        "performance_csv": str(metrics_path),
        **resource_metrics,
    })
    if qa_file is not None and return_code == 0:
        predictions_path = run_dir / "qa_predictions.jsonl"
        manifest.update(benchmark_qa(args.model, adapter_path, qa_file, predictions_path, args.qa_generation_tokens, qa_gold_file, args.corpus))
    manifest["run_dir"] = str(run_dir)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    write_performance_template(metrics_path, manifest)
    if return_code == 0:
        sync_evaluation_to_training_run(manifest)
    if return_code != 0:
        raise SystemExit(f"MLX-LM evaluation exited with status {return_code}. See {log_path}")
    print(f"\nSaved evaluation report: {metrics_path}")
    print_resource_summary(resource_metrics)
    if qa_file is not None:
        print(f"Saved QA answers for expert scoring: {manifest['qa_predictions_file']}")


def main() -> None:
    args = parse_args()
    if args.refresh_metrics:
        refresh_run_metrics(Path(args.refresh_metrics))
    elif args.score_qa:
        score_path = Path(args.score_qa)
        if not score_path.is_absolute():
            score_path = ROOT / score_path
        score_saved_qa(score_path)
    elif args.train:
        train(args)
    elif args.evaluate:
        evaluate(args)
    elif args.qa_eval:
        raise SystemExit("--qa-eval is for later evaluation; use --evaluate --adapter-path <saved-adapter> --qa-eval <file>.")
    else:
        if args.adapter_path is not None:
            print("Note: --adapter-path is used with --train or --evaluate.")
        prepare_dataset(args.corpus)
        print(f"To train one complete pass over the generated train.jsonl, run: python prepare.py --corpus {args.corpus} --train")


if __name__ == "__main__":
    main()
