"""
Batch runner: process articles through the full pipeline with crash recovery.
Usage: py batch_runner.py --input data/train.csv --output results/results.csv --sample 100 --workers 3
"""
import argparse
import os
import threading
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from pipeline.core import analyze_article


def load_existing(output_path: str) -> set:
    """Return set of already-processed titles for resume support."""
    if not os.path.exists(output_path):
        return set()
    try:
        done = pd.read_csv(output_path)
        return set(done["title"].astype(str).tolist())
    except Exception:
        return set()


def sample_balanced(df: pd.DataFrame, n: int, label_col: str = "label") -> pd.DataFrame:
    """Sample n/2 fake + n/2 real, stratified."""
    half = n // 2
    fake = df[df[label_col] == 0].sample(min(half, (df[label_col] == 0).sum()), random_state=42)
    real = df[df[label_col] == 1].sample(min(half, (df[label_col] == 1).sum()), random_state=42)
    return pd.concat([fake, real]).sample(frac=1, random_state=42).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="Batch fake news detection runner")
    parser.add_argument("--input", default="data/train.csv", help="Input CSV path")
    parser.add_argument("--output", default="results/results.csv", help="Output CSV path")
    parser.add_argument("--sample", type=int, default=100, help="Number of articles to process")
    parser.add_argument("--workers", type=int, default=2, help="Number of articles to process in parallel")
    parser.add_argument("--model", default=None, help="Ollama model to use (e.g. qwen2.5:7b, qwen2.5:14b)")
    args = parser.parse_args()

    if args.model:
        import config
        config.OLLAMA_MODEL = args.model
        print(f"  Model override: {args.model}")

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)

    print(f"Loading dataset from {args.input}...")
    df = pd.read_csv(args.input)
    print(f"  Total articles: {len(df)}")

    if "label" in df.columns:
        sample_df = sample_balanced(df, args.sample)
    else:
        sample_df = df.sample(min(args.sample, len(df)), random_state=42).reset_index(drop=True)

    print(f"  Sampled: {len(sample_df)} articles")

    done_titles = load_existing(args.output)
    if done_titles:
        print(f"  Resuming — {len(done_titles)} already processed, skipping those...")

    write_header = not os.path.exists(args.output)

    processed = 0
    skipped = 0
    csv_lock = threading.Lock()

    todo_rows = [
        row for _, row in sample_df.iterrows()
        if str(row.get("title", "")) not in done_titles
    ]
    skipped = len(sample_df) - len(todo_rows)

    def process_row(row):
        title = str(row.get("title", ""))
        text = str(row.get("text", ""))
        true_label_int = row.get("label", None)
        true_label = {0: "FAKE", 1: "REAL"}.get(true_label_int, "") if true_label_int is not None else ""
        try:
            result = analyze_article(text, title)
        except Exception as e:
            print(f"  ERROR on '{title[:50]}': {e} — skipping")
            return None
        hf = result.get("hf_classifier", {})
        return {
            "title": title,
            "true_label": true_label,
            "predicted_label": result["final_label"],
            "confidence": result["final_confidence"],
            "hf_label": hf.get("hf_label", ""),
            "hf_confidence": hf.get("hf_confidence", ""),
            "bias_score": result["bias"].get("bias_score", None),
            "credibility_score": result["credibility"].get("credibility_score", None),
            "tone": result["bias"].get("tone", ""),
            "n_claims": len(result["claims"].get("claims", [])),
            "n_contradicted": sum(
                1 for r in result["fact_check"].get("results", [])
                if r.get("verdict") == "CONTRADICTED"
            ),
            "n_supported": sum(
                1 for r in result["fact_check"].get("results", [])
                if r.get("verdict") == "SUPPORTED"
            ),
            "devils_advocate_triggered": result["devils_advocate_triggered"],
            "da_recommendation": result["devils_advocate"].get("recommendation", "") if result["devils_advocate"] else "",
            "processing_time": result["total_processing_time"],
            "judge_summary": result["judge"].get("summary", ""),
            "subject": row.get("subject", ""),
            "date": row.get("date", ""),
        }

    counts = {"REAL": 0, "FAKE": 0}

    def print_progress():
        total = counts["REAL"] + counts["FAKE"]
        print(f"\r  Progress: {total}/{len(todo_rows)} done  |  REAL: {counts['REAL']}  FAKE: {counts['FAKE']}", end="", flush=True)

    print(f"  Running with {args.workers} parallel worker(s)...")
    print_progress()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_row, row): row for row in todo_rows}
        for future in as_completed(futures):
            row_data = future.result()
            if row_data is None:
                continue
            with csv_lock:
                pd.DataFrame([row_data]).to_csv(args.output, mode="a", header=write_header, index=False)
                write_header = False
            processed += 1
            counts[row_data["predicted_label"]] = counts.get(row_data["predicted_label"], 0) + 1
            print_progress()
    print()  # newline after progress line

    print(f"\n{'='*50}")
    print(f"Done. Processed {processed} articles (skipped {skipped} already done).")
    print(f"Results saved to {args.output}")

    if os.path.exists(args.output):
        final_df = pd.read_csv(args.output)
        if "true_label" in final_df.columns and "predicted_label" in final_df.columns:
            valid = final_df[final_df["true_label"].isin(["REAL", "FAKE"])]
            if len(valid) > 0:
                acc = (valid["true_label"] == valid["predicted_label"]).mean()
                print(f"Accuracy so far: {acc:.1%} on {len(valid)} labeled articles")


if __name__ == "__main__":
    main()
