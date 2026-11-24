# PLAN.md — Veritas Bureau: Multi-Agent Fake News Detection System
# IE University — Gen AI & Agents Group Project
# Dr. Jaume Manero, April 2026

## PROJECT OVERVIEW

Build a multi-agent fake news detection system using plain Python + Ollama (NO CrewAI).
6 AI agents analyze news articles and produce a REAL/FAKE verdict.
Each agent is a Python function with a dedicated system prompt that calls Ollama directly.
This approach is ~7x faster than using CrewAI framework overhead.
The UI follows a dark "detective case file" aesthetic using Streamlit.

Work through each phase sequentially. Test each component before moving on.
Only ask the user a question if truly blocked. Otherwise make reasonable decisions
and document assumptions in DECISIONS.md.

---

## PHASE 1: PROJECT STRUCTURE

Create this folder structure:

```
fake-news-mas/
├── PLAN.md              (this file)
├── DECISIONS.md          (log assumptions here)
├── requirements.txt
├── .env.example
├── config.py             (LLM config, shared constants)
├── agents/
│   ├── __init__.py
│   ├── prompts.py        (all 6 system prompts as string constants)
│   └── runners.py        (one function per agent that calls Ollama)
├── pipeline/
│   ├── __init__.py
│   ├── core.py           (main pipeline logic — orchestration)
│   ├── parallel.py       (ThreadPoolExecutor for parallel agents)
│   └── json_parser.py    (robust JSON extraction from LLM output)
├── app.py                (Streamlit UI — case file theme)
├── batch_runner.py       (process 100 articles with crash recovery)
├── test_pipeline.py      (quick test on 2-3 articles)
├── notebooks/
│   ├── notebook_1_pipeline.ipynb
│   └── notebook_2_evaluation.ipynb
├── data/
│   └── (user places dataset here)
└── results/
    └── (outputs go here)
```

### requirements.txt
```
ollama
streamlit
pandas
scikit-learn
seaborn
matplotlib
python-dotenv
```

NOTE: No crewai or crewai-tools. We call Ollama directly via the `ollama` Python package.

### .env.example
```
LLM_BACKEND=ollama
OLLAMA_MODEL=qwen2.5:7b
OLLAMA_BASE_URL=http://localhost:11434
# Optional: for OpenAI
# LLM_BACKEND=openai
# OPENAI_API_KEY=sk-proj-xxxxx
# OPENAI_MODEL=gpt-4o-mini
```

## PHASE 2: CONFIG AND LLM SETUP

### config.py

Create a config module that:
- Loads .env using python-dotenv
- Supports switching between Ollama and OpenAI via LLM_BACKEND env var
- Default: Ollama with qwen2.5:7b at localhost:11434
- Exposes a `call_llm(system_prompt, user_message)` function that:
  - If Ollama: uses `ollama.chat()` directly
  - If OpenAI: uses `openai.chat.completions.create()`
  - Returns the raw text response as a string
- Temperature: 0.2 for all calls

```python
import ollama
from dotenv import load_dotenv
import os

load_dotenv()

BACKEND = os.getenv("LLM_BACKEND", "ollama")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

def call_llm(system_prompt: str, user_message: str, temperature: float = 0.2) -> str:
    """Call the LLM with a system prompt and user message. Returns raw text."""
    if BACKEND == "openai":
        import openai
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=temperature
        )
        return response.choices[0].message.content
    else:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            options={"temperature": temperature}
        )
        return response["message"]["content"]
```

---

## PHASE 3: AGENT DEFINITIONS

### agents/prompts.py

Store each agent's system prompt as a string constant. These are the "brains" of each agent.

#### CLAIM_EXTRACTOR_PROMPT
```
You are a veteran investigative journalist with 20 years of experience at Reuters and the Associated Press. You have an exceptional eye for distinguishing factual claims from opinion, rhetoric, and editorialization.

Your task: Read the provided news article and extract exactly 3-5 key factual claims that can be verified. Focus on statistics, named entities, events, dates, and specific assertions. Ignore opinions, rhetoric, and editorial language.

You MUST respond with ONLY valid JSON. No explanations, no markdown, no extra text.

Output format:
{
  "claims": [
    "Claim 1 text here",
    "Claim 2 text here",
    "Claim 3 text here"
  ]
}
```

#### FACT_CHECKER_PROMPT
```
You are a senior researcher at a fact-checking organization similar to Snopes or PolitiFact. You have deep knowledge of world events, politics, science, and economics.

Your task: For each claim provided, evaluate it using your knowledge. Determine if it is:
- SUPPORTED: likely true based on known facts
- CONTRADICTED: conflicts with known facts
- UNVERIFIABLE: cannot determine truth with available knowledge

You never fabricate evidence. If you don't know, say UNVERIFIABLE.

You MUST respond with ONLY valid JSON. No explanations, no markdown, no extra text.

Output format:
{
  "results": [
    {
      "claim": "The claim text",
      "verdict": "SUPPORTED|CONTRADICTED|UNVERIFIABLE",
      "confidence": 0.85,
      "evidence": "Brief explanation of why"
    }
  ]
}
```

#### BIAS_DETECTOR_PROMPT
```
You are a linguistics professor specializing in media studies and computational rhetoric at Columbia Journalism School. You have published extensively on propaganda detection, framing theory, and media manipulation.

Your task: Analyze the language, tone, and framing of the article independently of its factual content. Identify:
- Emotional manipulation or loaded language
- Sensationalist or clickbait headlines
- One-sided framing or missing counter-arguments
- Unattributed quotes or anonymous sources
- Fear-mongering or urgency language

Produce a bias score from 0.0 (completely neutral) to 1.0 (extremely biased).

You MUST respond with ONLY valid JSON. No explanations, no markdown, no extra text.

Output format:
{
  "tone": "neutral|informative|sensationalist|inflammatory|fearful",
  "bias_score": 0.72,
  "flags": [
    "Emotionally loaded headline",
    "No counter-arguments presented",
    "Unattributed quotes"
  ]
}
```

#### SOURCE_CREDIBILITY_PROMPT (CUSTOM ADDITION)
```
You are a digital forensics expert and media literacy educator. You have trained thousands of journalists and students in identifying unreliable sources.

Your task: Examine the structural and metadata signals of the article to assess its credibility. Check for:
- Named author with credentials
- Publication date
- Cited sources or references
- Balanced headline (not clickbait)
- Proper article structure (lead, body, conclusion)
- Editorial standards (attribution, sourcing)

Score credibility from 0.0 (no credibility markers) to 1.0 (fully credible structure).

You MUST respond with ONLY valid JSON. No explanations, no markdown, no extra text.

Output format:
{
  "credibility_score": 0.35,
  "flags": [
    "No author byline",
    "No publication date",
    "Zero cited sources"
  ],
  "has_author": false,
  "has_date": false,
  "has_sources": false,
  "headline_quality": "clickbait|sensational|neutral|professional"
}
```

#### JUDGE_PROMPT
```
You are the chief editor of an award-winning fact-checking platform with 25 years of editorial experience.

Your task: Synthesize all evidence from the Fact Checker, Bias Detector, and Source Credibility Analyzer to produce a final classification.

Weighting guidelines:
- Contradicted claims are the strongest signal of fake news
- High bias scores reinforce a FAKE verdict
- Low source credibility reinforces a FAKE verdict
- If most claims are supported AND bias is low AND credibility is high, lean REAL

Produce a confidence score from 0.0 to 1.0 and a human-readable summary.

You MUST respond with ONLY valid JSON. No explanations, no markdown, no extra text.

Output format:
{
  "label": "REAL|FAKE",
  "confidence": 0.88,
  "summary": "2 out of 3 claims were contradicted... The article also uses sensationalist language...",
  "agent_inputs_summary": {
    "claim_verdicts": ["CONTRADICTED", "UNVERIFIABLE", "SUPPORTED"],
    "bias_score": 0.72,
    "tone": "sensationalist",
    "credibility_score": 0.35
  }
}
```

#### DEVILS_ADVOCATE_PROMPT (CUSTOM ADDITION)
```
You are a former defense attorney turned media analyst. You believe every conclusion deserves rigorous challenge.

Your task: You receive a preliminary verdict (REAL or FAKE) with supporting evidence. Argue the OPPOSITE case as strongly as possible:
- If the verdict is FAKE, find every reason it could be REAL
- If the verdict is REAL, find every reason it could be FAKE

Be thorough and fair. If you cannot find compelling counter-evidence, say so honestly. Your purpose is to strengthen the final verdict by stress-testing it.

You MUST respond with ONLY valid JSON. No explanations, no markdown, no extra text.

Output format:
{
  "original_verdict": "FAKE",
  "counter_argument": "While the bias score is high, some claims are about real government programs...",
  "recommendation": "UPHOLD|OVERTURN",
  "revised_confidence": 0.91,
  "reasoning": "Counter-arguments were weak. No supporting evidence found for key claims."
}
```

### agents/runners.py

Each agent is a simple function that calls `call_llm()` from config.py and parses the response.

```python
from config import call_llm
from agents.prompts import (
    CLAIM_EXTRACTOR_PROMPT, FACT_CHECKER_PROMPT, BIAS_DETECTOR_PROMPT,
    SOURCE_CREDIBILITY_PROMPT, JUDGE_PROMPT, DEVILS_ADVOCATE_PROMPT
)
from pipeline.json_parser import extract_json
import time

def run_claim_extractor(article_text: str) -> dict:
    """Agent 1: Extract factual claims from the article."""
    start = time.time()
    raw = call_llm(CLAIM_EXTRACTOR_PROMPT, f"Analyze this article:\n\n{article_text}")
    result = extract_json(raw, default={"claims": []})
    result["_processing_time"] = round(time.time() - start, 2)
    return result

def run_fact_checker(claims: list, article_text: str) -> dict:
    """Agent 2: Verify each claim."""
    start = time.time()
    claims_text = "\n".join([f"- {c}" for c in claims])
    user_msg = f"Verify these claims:\n{claims_text}\n\nOriginal article:\n{article_text[:1000]}"
    raw = call_llm(FACT_CHECKER_PROMPT, user_msg)
    result = extract_json(raw, default={"results": []})
    result["_processing_time"] = round(time.time() - start, 2)
    return result

def run_bias_detector(article_text: str) -> dict:
    """Agent 3: Analyze bias and tone."""
    start = time.time()
    raw = call_llm(BIAS_DETECTOR_PROMPT, f"Analyze the bias in this article:\n\n{article_text}")
    result = extract_json(raw, default={"tone": "unknown", "bias_score": 0.5, "flags": []})
    result["_processing_time"] = round(time.time() - start, 2)
    return result

def run_source_credibility(article_text: str) -> dict:
    """Agent 5: Check source credibility signals."""
    start = time.time()
    raw = call_llm(SOURCE_CREDIBILITY_PROMPT, f"Assess the credibility of this article:\n\n{article_text}")
    result = extract_json(raw, default={"credibility_score": 0.5, "flags": []})
    result["_processing_time"] = round(time.time() - start, 2)
    return result

def run_judge(fact_check: dict, bias: dict, credibility: dict, article_text: str) -> dict:
    """Agent 4: Final verdict."""
    start = time.time()
    evidence = f"""Here is the evidence from the investigation:

FACT CHECK RESULTS:
{fact_check}

BIAS ANALYSIS:
{bias}

SOURCE CREDIBILITY:
{credibility}

ORIGINAL ARTICLE (first 500 chars):
{article_text[:500]}"""
    raw = call_llm(JUDGE_PROMPT, evidence)
    result = extract_json(raw, default={"label": "UNKNOWN", "confidence": 0.5, "summary": "Could not determine"})
    result["_processing_time"] = round(time.time() - start, 2)
    return result

def run_devils_advocate(judge_result: dict, fact_check: dict, bias: dict, credibility: dict) -> dict:
    """Agent 6: Challenge the verdict (only called when confidence < 0.6)."""
    start = time.time()
    evidence = f"""The Judge has issued this preliminary verdict:
{judge_result}

Based on this evidence:
Fact Check: {fact_check}
Bias: {bias}
Credibility: {credibility}

Argue the OPPOSITE case."""
    raw = call_llm(DEVILS_ADVOCATE_PROMPT, evidence)
    result = extract_json(raw, default={"recommendation": "UPHOLD", "reasoning": "Could not generate counter-argument"})
    result["_processing_time"] = round(time.time() - start, 2)
    return result
```

---

## PHASE 4: PIPELINE LOGIC

### pipeline/json_parser.py

Create a robust JSON parser that handles common LLM output issues:
- Strip markdown code fences
- Strip text before first { or after last }
- Handle single quotes vs double quotes
- Try json.loads first, then regex extraction, then return default
- Log warnings when fallback parsing is used

### pipeline/parallel.py

Run Agents 2, 3, and 5 in parallel using ThreadPoolExecutor with max_workers=3.

### pipeline/core.py

Main orchestration function `analyze_article(article_text, title)`:
1. Run Agent 1 (Claim Extractor)
2. Run Agents 2, 3, 5 in parallel
3. Run Agent 4 (Judge) with merged results
4. If confidence < 0.6 → run Agent 6 (Devil's Advocate) → handle UPHOLD/OVERTURN
5. Return complete results dict with all agent outputs and timing

Wrap each agent call in try/except. Never crash the whole pipeline because one agent failed.

---

## PHASE 5: TEST PIPELINE

### test_pipeline.py

Test with 2-3 inline articles (one fake, one real, one ambiguous).
Print all agent outputs formatted. Report processing times.
Fix all issues before moving on.

---

## PHASE 6: STREAMLIT UI

### app.py — "Veritas Bureau" Case File Theme

Dark detective aesthetic. Two tabs: Single Investigation + Batch Processing.

Colors: bg #0f0e0c, cards #171614, borders #2a2825, text #C8C3B8, muted #6B6660
Fonts: Courier Prime (monospace labels), IBM Plex Sans (body) from Google Fonts

Single mode: text input → agent progress feed → case file results with verdict stamp,
signal meters, claims list, bias flags, credibility flags, devil's advocate section.

Batch mode: CSV upload → progress bar → download results.csv → summary metrics.

---

## PHASE 7: BATCH RUNNER

### batch_runner.py

Standalone script: loads CSV, samples 100 articles (50 real, 50 fake), processes each
through analyze_article(), saves to results.csv after EACH article for crash recovery.
Supports resume if output file already exists. Command line args for input/output/sample size.

---

## PHASE 8: NOTEBOOK 1 — PIPELINE DEMONSTRATION

Show architecture, agent prompts, 3 walkthrough articles (fake, real, ambiguous).
Emphasize: no framework, plain Python, 7x faster than CrewAI.

---

## PHASE 9: NOTEBOOK 2 — EVALUATION AND ANALYSIS

Load results.csv. Compute accuracy/precision/recall/F1. Confusion matrix.
Confidence distribution. Agent signal analysis. Devil's Advocate impact.
Error analysis on 5-10 wrong predictions. Limitations discussion.

---

## FINAL CHECKLIST

- [x] `pip install -r requirements.txt` works (NO crewai dependency)
- [x] Ollama running with qwen2.5:7b pulled
- [x] `python test_pipeline.py` works end-to-end
- [x] `streamlit run app.py` launches, all four tabs functional
- [x] Batch processing saves incrementally
- [x] Notebook 1 runs with 3 examples
- [x] Notebook 2 runs with metrics and plots
- [x] results.csv complete (100 deduplicated articles, 87% accuracy)
- [x] DECISIONS.md documents assumptions
- [x] .env not included in the final submitted folder

---

## CUSTOM EXTENSIONS (beyond original plan)

These were added after the initial build and are documented here for completeness.

### HuggingFace BERT Classifier (Phase 3 addition)

Model: `mrm8488/bert-tiny-finetuned-fake-news-detection`

- Runs as `run_hf_classifier()` in `agents/runners.py`
- Executes in the same parallel stage as Agents 2, 3, 5 (max_workers bumped to 4)
- Returns `hf_label` and `hf_confidence`
- The Judge prompt treats it as a corroborating signal alongside the other agents
- Shown as "HF Tool" in the Streamlit pipeline diagram and Agent Conclusions tab
- Output captured in batch CSV as additional columns

### Web Search Grounding for Fact Checker (Phase 4 addition)

New module: `pipeline/search.py`

- `web_search(query, max_results)` — DuckDuckGo via `ddgs` package (no API key)
- `search_claims(claims)` — parallel search for all claims (max 3 workers)
- `format_search_evidence(claims, results)` — formats snippets as `[WEB]` evidence block
- `run_fact_checker()` now fetches snippets before calling the LLM and injects them into the user message
- The Fact Checker prompt instructs the model to prefer web evidence over training-data assumptions
- Falls back silently to LLM-only evaluation if search fails for any claim
