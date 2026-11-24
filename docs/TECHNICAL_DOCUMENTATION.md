# Veritas Bureau Technical Documentation

## Purpose

Veritas Bureau is a multi-agent fake news detection system that classifies an article as `REAL` or `FAKE` and returns structured supporting analysis. The project is implemented in plain Python orchestration rather than CrewAI, following the updated course instruction not to use CrewAI in the final submission.

The system is designed around two goals:

- decompose the classification task into specialized agent roles
- produce an interpretable final verdict rather than a single opaque label

## High-Level Architecture

The system has four main layers:

1. `config.py`
   Handles environment loading and LLM backend selection.
2. `agents/`
   Defines prompts and runner functions for each agent and tool.
3. `pipeline/`
   Orchestrates the end-to-end workflow, parallel execution, web search, and robust JSON parsing.
4. `app.py` and `batch_runner.py`
   Provide interactive and batch execution modes.

## Backend Configuration

The project supports two interchangeable LLM backends:

- `ollama` for local inference
- `openai` for remote API inference

Backend selection is controlled through environment variables in `.env`.

Default configuration:

```env
LLM_BACKEND=ollama
OLLAMA_MODEL=qwen2.5:14b
OLLAMA_BASE_URL=http://localhost:11434
```

Optional OpenAI configuration:

```env
LLM_BACKEND=openai
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-4o-mini
```

In `config.py`, both backends are exposed through a single function:

- `call_llm(system_prompt, user_message, temperature=0.2)`

This abstraction keeps the rest of the pipeline backend-agnostic.

## Agent Design

The system uses six LLM agents plus one HuggingFace classifier tool. Each LLM agent is implemented as:

- a system prompt in `agents/prompts.py`
- a runner function in `agents/runners.py`

### Agent 1: Claim Extractor

Function: `run_claim_extractor(article_text)`

Responsibility:
- extracts the article's main verifiable factual claims
- ignores rhetorical and opinion-only content

Implementation details:
- claim count is adaptive rather than fixed
- under 400 words: target 3 claims
- 400-899 words: target 5 claims
- 900+ words: target 7 claims
- input text is truncated to 3000 characters

### Agent 2: Fact Checker (with web search grounding)

Function: `run_fact_checker(claims, article_text)`

Responsibility:
- evaluates each extracted claim as `SUPPORTED`, `CONTRADICTED`, or `UNVERIFIABLE`
- returns a confidence score and evidence for each claim

Implementation details:
- before calling the LLM, searches DuckDuckGo for each claim in parallel using `pipeline/search.py`
- up to 3 web snippets per claim are formatted into a `[WEB]` evidence block and injected into the user message
- the LLM is instructed to prefer web evidence over training-data assumptions and to cite the relevant snippet
- if search fails for any claim, that claim falls back to LLM-only evaluation silently
- receives article context truncated to 800 characters alongside the web evidence block
- stores `web_search_used` and `web_snippets_found` in its output for debugging

### Agent 3: Bias Detector

Function: `run_bias_detector(article_text)`

Responsibility:
- analyzes tone, framing, emotional manipulation, and sensationalism
- returns a `bias_score` between `0.0` and `1.0`

Implementation details:
- input text is truncated to 3000 characters
- output includes tone and a list of bias flags

### Agent 5: Source Credibility Analyzer

Function: `run_source_credibility(article_text)`

Responsibility:
- evaluates visible structural credibility cues in the provided text
- checks sourcing signals, headline quality, and report-like structure

Important design choice:
- this agent only evaluates cues present in the supplied article text
- it does not assume access to a publisher database, URL metadata, or hidden article fields
- low visible metadata is treated as a secondary signal, not decisive proof of fakery

### HF Tool: BERT Classifier

Function: `run_hf_classifier(article_text)`

Responsibility:
- provides an independent classification signal from a pretrained BERT model
- returns `hf_label` (REAL or FAKE) and `hf_confidence`

Implementation details:
- model: `jy46604790/Fake-News-Bert-Detect`, fine-tuned on the ISOT dataset
- `LABEL_1` maps to REAL, `LABEL_0` maps to FAKE for this model
- loaded once on first call and cached at module level for the process lifetime
- if PyTorch is unavailable (e.g. Python 3.13), returns `hf_label=UNKNOWN` immediately without error
- input is truncated to 512 tokens (BERT limit); article text is pre-clipped to 2000 chars before tokenization
- label normalization handles multiple HuggingFace naming conventions (`LABEL_0`/`LABEL_1`, `FAKE`/`REAL`, etc.)
- runs in the same parallel stage as Agents 2, 3, 5
- its output is injected into the Judge evidence bundle as a minor tiebreaker signal only

### Agent 4: Judge

Function: `run_judge(fact_check, bias, credibility, article_text, hf_classifier=None)`

Responsibility:
- synthesizes the outputs of Agents 2, 3, 5, and the HF classifier
- produces the final classification, confidence score, and human-readable summary

Decision logic encoded in the prompt (in order of priority):
1. Contradicted claims — strongest signal toward `FAKE`
2. Bias + credibility combination — high bias (>0.6) and low credibility (<0.45) together are sufficient for `FAKE`; low bias (<0.4) is a strong signal for `REAL`
3. HF Classifier — minor tiebreaker only; does not override fact-check or bias signals
- Web search contradictions are treated skeptically for older articles — a single CONTRADICTED claim is downgraded to UNVERIFIABLE when credibility is high (>0.6), bias is low (<0.5), or HF says REAL
- The Judge must always output REAL or FAKE; UNKNOWN is not permitted
- Confidence must reflect signal consistency; the full range 0.0–1.0 is expected

### Agent 6: Devil's Advocate

Function: `run_devils_advocate(judge_result, fact_check, bias, credibility)`

Responsibility:
- challenges low-confidence verdicts by arguing the opposite case
- can uphold or overturn the Judge's preliminary decision

Trigger rule:
- runs when Judge confidence is below `0.90`

If recommendation is `OVERTURN`:
- the final label is flipped
- the revised confidence is used
- the Judge summary is annotated with the challenge result

## Agentic Flow

The canonical orchestration entry point is:

- `pipeline/core.py`
- function: `analyze_article(article_text, title="")`

Execution flow:

1. Build a combined text payload using title plus article body when a title is available.
2. Run Agent 1 to extract claims.
3. If no claims are extracted, insert a fallback placeholder claim so the downstream pipeline still executes.
4. Run Agents 2, 3, 5, and the HF classifier in parallel.
5. Merge their outputs and send them to Agent 4.
6. If Judge confidence is below `0.90`, run Agent 6.
7. Return a full structured result object with intermediate outputs and timing metadata.

## Parallelism

Parallel execution is implemented in `pipeline/parallel.py` with `ThreadPoolExecutor(max_workers=4)`.

The following agents and tools run concurrently:

- Fact Checker (itself runs inner parallel web searches before calling the LLM)
- Bias Detector
- Source Credibility Analyzer
- HF BERT Classifier

This reduces total latency because these components are independent once claims are available.

## Web Search Module

`pipeline/search.py` provides three functions:

- `web_search(query, max_results=3)` — queries DuckDuckGo via the `ddgs` package, returns `[{title, url, snippet}]`. No API key required.
- `search_claims(claims, max_results=3)` — runs `web_search` for every claim in parallel (max 3 workers), returns `{claim_text: [results]}`.
- `format_search_evidence(claims, results)` — formats the search output into a compact `[WEB] snippet...` block for injection into the LLM user message.

Fallback: if a search fails for any reason (network error, rate limit, empty result), the function returns an empty list for that claim and the fact checker falls back to LLM-only evaluation for that claim.

## Fault Tolerance and Parsing Robustness

Two reliability mechanisms are central to the project.

### 1. Agent-Level Error Handling

Each runner function in `agents/runners.py` is wrapped in `try/except`.

If an agent fails:
- the pipeline does not crash
- a safe default result is returned
- the error message is attached to the agent output

This allows the Judge or the final caller to proceed even with partial evidence.

### 2. JSON Recovery

All agent outputs are expected to be valid JSON, but LLM responses are not always perfectly formatted. To address this, `pipeline/json_parser.py` implements a recovery pipeline:

1. direct `json.loads`
2. remove Markdown code fences
3. extract the outermost JSON object
4. repair some malformed cases such as trailing commas
5. repair some unclosed array outputs
6. `_fix_truncated()` — rolls back to the last complete JSON value when the response is cut off mid-string (handles `num_predict` truncation)
7. fall back to a default value if all parsing attempts fail

This is especially important for smaller or local models that sometimes produce malformed structured output.

## Data Flow and Returned Structure

`analyze_article()` returns a dictionary containing:

- `title`
- `article_preview`
- `claims`
- `fact_check` (includes `web_search_used`, `web_snippets_found`)
- `bias`
- `credibility`
- `hf_classifier` (includes `hf_label`, `hf_confidence`, `hf_raw_label`)
- `judge`
- `devils_advocate`
- `devils_advocate_triggered`
- `final_label`
- `final_confidence`
- `total_processing_time`

This structure is used consistently by the UI, batch runner, and notebooks.

## Interactive Application

The Streamlit interface is implemented in `app.py`.

It provides four tabs:

1. `Single Investigation`
   Runs one article through the full pipeline and displays the result as a case file. Shows the HF classifier signal as a labelled banner below the signal meters.
2. `Agent Conclusions`
   Displays per-agent outputs including the HF classifier as a dedicated card.
3. `Pipeline`
   Explains the system architecture and agent roles, including the HF tool as a custom addition.
4. `Batch Processing`
   Allows smaller CSV-based runs inside the UI and exports a result file.

The UI also includes:

- a model selector dropdown in the header bar to switch Ollama models at runtime without restarting the app
- a dark/light theme toggle button in the header bar, aligned with the model selector
- a live pipeline diagram for single-article execution, with the HF classifier box appearing in the parallel stage alongside Agents 2, 3, 5
- verdict stamps and signal meters
- claim-level verdict rendering
- dedicated display for Devil's Advocate review when triggered (threshold label reflects the 0.90 cutoff)

## Batch Evaluation Mode

Large-scale evaluation is implemented in `batch_runner.py`.

Key behaviors:

- loads an input CSV, default `data/train.csv`
- samples a balanced subset when a `label` column is available
- maps dataset labels `0 -> FAKE` and `1 -> REAL`
- saves each processed article immediately to the output CSV
- adds the processed title to `done_titles` after each write so duplicate titles in the source CSV are also skipped within a single run
- supports resume behavior by skipping titles already written to the output file
- `--workers N` processes multiple articles concurrently via `ThreadPoolExecutor`; CSV writes are protected by a threading lock
- `--model <name>` overrides the default Ollama model for that run (e.g. `qwen2.5:7b`)
- live progress is printed in-place: `Progress: X/N done | REAL: X FAKE: X`
- prints accuracy on labeled articles at the end of the run
- output CSV includes `hf_label`, `hf_confidence`, `judge_summary`, `subject`, and `date` columns

## Testing

The quick functional test is `test_pipeline.py`.

It runs three representative examples:

- clearly fake
- clearly real
- ambiguous or mixed

The script prints:

- final verdict, confidence, and total time
- extracted claims
- fact-check results (web-grounded)
- HF classifier label and confidence
- bias and credibility outputs
- Devil's Advocate output when triggered

## Design Rationale

The project deliberately avoids a monolithic single-prompt classifier. Instead, it separates the problem into interpretable subtasks:

- claim extraction isolates verifiable content
- fact checking focuses on factual consistency, grounded in live web evidence
- bias detection captures rhetorical manipulation
- credibility analysis captures structural trust cues
- the HF BERT classifier provides an independent pretrained signal
- the Judge combines these signals into a final decision
- Devil's Advocate stress-tests borderline cases

This structure makes the system easier to inspect, debug, and explain in a report.

## Known Limitations

- model outputs can still vary depending on backend and prompt sensitivity
- the pipeline operates on truncated article context to control latency and token cost
- credibility analysis is limited to the text that is provided
- the final verdict depends heavily on the quality of claim extraction upstream
- DuckDuckGo search is subject to rate limiting during high-volume batch runs; individual failures fall back to LLM-only evaluation silently
- the HF classifier was trained on the same ISOT dataset used for evaluation, which may inflate its apparent contribution to accuracy

## Alignment With The Updated Assignment

The original assignment PDF includes CrewAI examples and backend guidance for both OpenAI and Ollama. The final implementation in this repository follows the updated instruction given during the course:

- do not use CrewAI in the final submission

Accordingly, this project:

- keeps the two backend options mentioned in the assignment: OpenAI and Ollama
- implements the agentic workflow directly in Python
- preserves the multi-agent design, role decomposition, parallelism, and evaluation workflow required by the project
- adds a HuggingFace classifier and web search grounding as custom extensions beyond the base requirements
