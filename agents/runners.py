from config import call_llm
from agents.prompts import (
    CLAIM_EXTRACTOR_PROMPT, FACT_CHECKER_PROMPT, BIAS_DETECTOR_PROMPT,
    SOURCE_CREDIBILITY_PROMPT, JUDGE_PROMPT, DEVILS_ADVOCATE_PROMPT
)
from pipeline.json_parser import extract_json
import time
import warnings

try:
    import torch
    _torch_available = True
except ImportError:
    _torch_available = False

_hf_pipeline = None  # module-level cache so the model loads only once
CLAIM_EXTRACTOR_CHAR_LIMIT = 6000


def _target_claim_count(article_text: str) -> int:
    words = len(article_text.split())
    if words < 400:
        return 3
    if words < 900:
        return 5
    return 7


def run_claim_extractor(article_text: str) -> dict:
    """Agent 1: Extract factual claims from the article."""
    start = time.time()
    try:
        target_claims = _target_claim_count(article_text)
        user_msg = (
            f"Extract up to {target_claims} key factual claims from this article. "
            f"If the article does not support that many strong claims, return fewer.\n\n"
            f"Article:\n\n{article_text[:CLAIM_EXTRACTOR_CHAR_LIMIT]}"
        )
        raw = call_llm(CLAIM_EXTRACTOR_PROMPT, user_msg)
        result = extract_json(raw, default={"claims": []})
    except Exception as e:
        print(f"[Claim Extractor] Error: {e}")
        result = {"claims": [], "error": str(e)}
    result["_target_claim_count"] = _target_claim_count(article_text)
    result["_processing_time"] = round(time.time() - start, 2)
    return result


def run_fact_checker(claims: list, article_text: str) -> dict:
    """Agent 2: Verify each claim with live web search grounding."""
    start = time.time()
    try:
        from pipeline.search import search_claims, format_search_evidence
        print(f"[Fact Checker] Searching web for {len(claims)} claim(s)...")
        search_results = search_claims(claims, max_results=3)
        snippets_found = sum(len(v) for v in search_results.values())
        print(f"[Fact Checker] Retrieved {snippets_found} web snippet(s).")

        evidence_block = format_search_evidence(claims, search_results)
        user_msg = (
            f"Verify the following claims using the web evidence provided.\n\n"
            f"{evidence_block}\n"
            f"Original article context:\n{article_text[:800]}"
        )
        raw = call_llm(FACT_CHECKER_PROMPT, user_msg)
        result = extract_json(raw, default={"results": []})
        result["web_search_used"] = True
        result["web_snippets_found"] = snippets_found
    except Exception as e:
        print(f"[Fact Checker] Error: {e}")
        result = {"results": [], "error": str(e), "web_search_used": False}
    result["_processing_time"] = round(time.time() - start, 2)
    return result


def run_bias_detector(article_text: str) -> dict:
    """Agent 3: Analyze bias and tone."""
    start = time.time()
    try:
        raw = call_llm(BIAS_DETECTOR_PROMPT, f"Analyze the bias in this article:\n\n{article_text[:3000]}")
        result = extract_json(raw, default={"tone": "unknown", "bias_score": 0.5, "flags": []})
    except Exception as e:
        print(f"[Bias Detector] Error: {e}")
        result = {"tone": "unknown", "bias_score": 0.5, "flags": [], "error": str(e)}
    result["_processing_time"] = round(time.time() - start, 2)
    return result


def run_source_credibility(article_text: str) -> dict:
    """Agent 5: Check source credibility signals."""
    start = time.time()
    try:
        raw = call_llm(SOURCE_CREDIBILITY_PROMPT, f"Assess the credibility of this article:\n\n{article_text[:3000]}")
        result = extract_json(raw, default={"credibility_score": 0.5, "flags": []})
    except Exception as e:
        print(f"[Source Credibility] Error: {e}")
        result = {"credibility_score": 0.5, "flags": [], "error": str(e)}
    result["_processing_time"] = round(time.time() - start, 2)
    return result


def run_hf_classifier(article_text: str) -> dict:
    """HF tool: classify article with a pretrained BERT fake-news detector.
    Model: mrm8488/bert-tiny-finetuned-fake-news-detection (trained on ISOT dataset).
    Loaded once and cached for the process lifetime."""
    global _hf_pipeline
    start = time.time()
    if not _torch_available:
        return {"hf_label": "UNKNOWN", "hf_confidence": 0.5, "_processing_time": 0.0}
    try:
        if _hf_pipeline is None:
            from transformers import pipeline as _hf_fn
            if _torch_available:
                try:
                    if torch.cuda.is_available():
                        device = 0
                    elif torch.backends.mps.is_available():
                        device = "mps"
                    else:
                        device = -1
                except Exception:
                    device = -1
            else:
                device = -1
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                _hf_pipeline = _hf_fn(
                    "text-classification",
                    model="jy46604790/Fake-News-Bert-Detect",
                    truncation=True,
                    max_length=512,
                    device=device,
                )
        out = _hf_pipeline(article_text[:2000], truncation=True)[0]
        raw_label = out["label"].upper()
        # LABEL_1 = REAL, LABEL_0 = FAKE for jy46604790/Fake-News-Bert-Detect
        if raw_label in ("LABEL_1", "REAL", "TRUE", "RELIABLE"):
            label = "REAL"
        else:
            label = "FAKE"
        result = {
            "hf_label": label,
            "hf_confidence": round(float(out["score"]), 4),
            "hf_raw_label": out["label"],
        }
    except Exception as e:
        print(f"[HF Classifier] Error: {e}")
        result = {"hf_label": "UNKNOWN", "hf_confidence": 0.5, "error": str(e)}
    result["_processing_time"] = round(time.time() - start, 2)
    return result


def run_judge(fact_check: dict, bias: dict, credibility: dict, article_text: str,
              hf_classifier: dict = None) -> dict:
    """Agent 4: Final verdict."""
    start = time.time()
    try:
        hf_section = ""
        if hf_classifier and hf_classifier.get("hf_label") != "UNKNOWN":
            hf_section = f"""
HF CLASSIFIER (pretrained BERT on ISOT dataset):
Label: {hf_classifier.get('hf_label')}  Confidence: {hf_classifier.get('hf_confidence', 0):.0%}
"""
        evidence = f"""Here is the evidence from the investigation:

FACT CHECK RESULTS:
{fact_check}

BIAS ANALYSIS:
{bias}

SOURCE CREDIBILITY:
{credibility}{hf_section}
ORIGINAL ARTICLE (first 500 chars):
{article_text[:500]}"""
        raw = call_llm(JUDGE_PROMPT, evidence)
        hf_fallback = (hf_classifier or {}).get("hf_label", "REAL")
        if hf_fallback not in ("REAL", "FAKE"):
            hf_fallback = "REAL"
        result = extract_json(raw, default={"label": hf_fallback, "confidence": 0.5, "summary": "Could not determine"})
    except Exception as e:
        print(f"[Judge] Error: {e}")
        hf_fallback = (hf_classifier or {}).get("hf_label", "REAL")
        if hf_fallback not in ("REAL", "FAKE"):
            hf_fallback = "REAL"
        result = {"label": hf_fallback, "confidence": 0.5, "summary": "Error", "error": str(e)}
    result["_processing_time"] = round(time.time() - start, 2)
    return result


def run_devils_advocate(judge_result: dict, fact_check: dict, bias: dict, credibility: dict) -> dict:
    """Agent 6: Challenge the verdict (only called when confidence < 0.6)."""
    start = time.time()
    try:
        evidence = f"""The Judge has issued this preliminary verdict:
{judge_result}

Based on this evidence:
Fact Check: {fact_check}
Bias: {bias}
Credibility: {credibility}

Argue the OPPOSITE case."""
        raw = call_llm(DEVILS_ADVOCATE_PROMPT, evidence)
        result = extract_json(raw, default={"recommendation": "UPHOLD", "reasoning": "Could not generate counter-argument"})
    except Exception as e:
        print(f"[Devil's Advocate] Error: {e}")
        result = {"recommendation": "UPHOLD", "reasoning": "Error", "error": str(e)}
    result["_processing_time"] = round(time.time() - start, 2)
    return result


def should_overturn_with_devils_advocate(judge_result: dict, da_result: dict) -> bool:
    """Apply hard guardrails before allowing the Devil's Advocate to flip the verdict."""
    if da_result.get("recommendation") != "OVERTURN":
        return False

    original_label = judge_result.get("label", "UNKNOWN")
    summary = judge_result.get("agent_inputs_summary", {}) or {}
    verdicts = summary.get("claim_verdicts", []) or []
    contradicted = sum(1 for v in verdicts if str(v).upper() == "CONTRADICTED")

    # Guard against false negatives: do not flip FAKE -> REAL while the judge
    # still reports unresolved contradicted claims.
    if original_label == "FAKE" and contradicted > 0:
        return False

    return True
