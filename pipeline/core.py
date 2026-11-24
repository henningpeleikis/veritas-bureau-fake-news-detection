import time
from agents.runners import (
    run_claim_extractor,
    run_judge,
    run_devils_advocate,
    should_overturn_with_devils_advocate,
)
from pipeline.parallel import run_parallel_agents


DEFAULT_FACT_CHECK = {"results": []}
DEFAULT_BIAS = {"tone": "unknown", "bias_score": 0.5, "flags": []}
DEFAULT_CREDIBILITY = {"credibility_score": 0.5, "flags": []}
DEFAULT_HF = {"hf_label": "UNKNOWN", "hf_confidence": 0.5}
DEFAULT_JUDGE = {"label": "REAL", "confidence": 0.5, "summary": "Could not determine"}


def analyze_article(article_text: str, title: str = "") -> dict:
    """
    Main pipeline orchestration.
    1. Agent 1: Claim Extractor
    2. Agents 2+3+5 in parallel: Fact Checker, Bias Detector, Source Credibility
    3. Agent 4: Judge
    4. If confidence < 0.6: Agent 6 Devil's Advocate
    Returns complete results dict.
    """
    pipeline_start = time.time()
    full_text = f"TITLE: {title}\n\n{article_text}" if title else article_text

    print("[Pipeline] Running Agent 1: Claim Extractor...")
    try:
        claim_result = run_claim_extractor(full_text)
    except Exception as exc:
        claim_result = {"claims": [], "error": str(exc), "_processing_time": 0.0}

    claims = claim_result.get("claims", [])
    if not claims:
        claims = ["No specific claims could be extracted from this article."]

    print("[Pipeline] Running Agents 2+3+5+HF in parallel (Fact Check, Bias, Credibility, HF Classifier)...")
    try:
        fact_check, bias, credibility, hf_classifier = run_parallel_agents(claims, full_text)
    except Exception as exc:
        fact_check = {**DEFAULT_FACT_CHECK, "error": str(exc)}
        bias = {**DEFAULT_BIAS, "error": str(exc)}
        credibility = {**DEFAULT_CREDIBILITY, "error": str(exc)}
        hf_classifier = {**DEFAULT_HF, "error": str(exc)}

    print(f"[Pipeline] HF Classifier: {hf_classifier.get('hf_label')} ({hf_classifier.get('hf_confidence', 0):.0%})")
    print("[Pipeline] Running Agent 4: Judge...")
    try:
        judge = run_judge(fact_check, bias, credibility, full_text, hf_classifier=hf_classifier)
    except Exception as exc:
        judge = {**DEFAULT_JUDGE, "summary": "Judge failed", "error": str(exc), "_processing_time": 0.0}

    confidence = judge.get("confidence", 0.5)
    devils_advocate = None
    devils_advocate_triggered = False

    if confidence < 0.9:
        print(f"[Pipeline] Low confidence ({confidence:.2f}) -> triggering Agent 6: Devil's Advocate...")
        devils_advocate_triggered = True
        try:
            devils_advocate = run_devils_advocate(judge, fact_check, bias, credibility)
        except Exception as exc:
            devils_advocate = {
                "recommendation": "UPHOLD",
                "reasoning": "Devil's Advocate failed",
                "error": str(exc),
                "_processing_time": 0.0,
            }

        if should_overturn_with_devils_advocate(judge, devils_advocate):
            original_label = judge.get("label", "UNKNOWN")
            judge["label"] = "FAKE" if original_label == "REAL" else "REAL"
            judge["confidence"] = devils_advocate.get("revised_confidence", confidence)
            judge["summary"] += f" [OVERTURNED by Devil's Advocate: {devils_advocate.get('reasoning', '')}]"
            print(f"[Pipeline] Verdict OVERTURNED to {judge['label']}")
        else:
            judge["confidence"] = devils_advocate.get("revised_confidence", confidence)
            print(f"[Pipeline] Verdict UPHELD. Revised confidence: {judge['confidence']:.2f}")

    total_time = round(time.time() - pipeline_start, 2)
    print(f"[Pipeline] Complete in {total_time}s. Verdict: {judge.get('label')} ({judge.get('confidence'):.2f})")

    return {
        "title": title,
        "article_preview": article_text[:300],
        "claims": claim_result,
        "fact_check": fact_check,
        "bias": bias,
        "credibility": credibility,
        "hf_classifier": hf_classifier,
        "judge": judge,
        "devils_advocate": devils_advocate,
        "devils_advocate_triggered": devils_advocate_triggered,
        "final_label": judge.get("label", "UNKNOWN"),
        "final_confidence": judge.get("confidence", 0.5),
        "total_processing_time": total_time,
    }
