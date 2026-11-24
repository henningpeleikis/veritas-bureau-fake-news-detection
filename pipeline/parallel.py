from concurrent.futures import ThreadPoolExecutor, as_completed
from agents.runners import run_fact_checker, run_bias_detector, run_source_credibility, run_hf_classifier


def run_parallel_agents(claims: list, article_text: str) -> tuple[dict, dict, dict, dict]:
    """Run Agents 2, 3, 5 and HF Classifier in parallel.
    Returns (fact_check, bias, credibility, hf_classifier)."""
    defaults = {
        "fact_check": {"results": [], "error": "Agent did not complete"},
        "bias": {"tone": "unknown", "bias_score": 0.5, "flags": [], "error": "Agent did not complete"},
        "credibility": {"credibility_score": 0.5, "flags": [], "error": "Agent did not complete"},
        "hf_classifier": {"hf_label": "UNKNOWN", "hf_confidence": 0.5, "error": "Agent did not complete"},
    }
    results = defaults.copy()

    task_map = {}

    with ThreadPoolExecutor(max_workers=4) as executor:
        task_map[executor.submit(run_fact_checker, claims, article_text)] = "fact_check"
        task_map[executor.submit(run_bias_detector, article_text)] = "bias"
        task_map[executor.submit(run_source_credibility, article_text)] = "credibility"
        task_map[executor.submit(run_hf_classifier, article_text)] = "hf_classifier"

        for future in as_completed(task_map):
            key = task_map[future]
            try:
                results[key] = future.result()
            except Exception as exc:
                results[key] = {**defaults[key], "error": str(exc)}

    return results["fact_check"], results["bias"], results["credibility"], results["hf_classifier"]
