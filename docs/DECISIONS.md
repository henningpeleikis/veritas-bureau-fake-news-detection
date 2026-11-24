# Veritas Bureau Project Decisions

## Dataset Handling

- The merged dataset uses `text` as the article body and `title` as the title.
- The `subject` and `date` columns are preserved for evaluation and downstream analysis.
- Batch sampling is stratified to keep the real and fake classes balanced in sampled runs.
- Duplicate titles are skipped at write time in `batch_runner.py` (the `done_titles` set is updated after each write, not just at startup) to prevent duplicates even when the source CSV contains repeated entries.

## Agent Architecture

- The pipeline uses six agents with distinct responsibilities: extraction, verification, bias analysis, credibility analysis, judgment, and adversarial review.
- A HuggingFace BERT classifier runs alongside the LLM-based agents as an additional independent signal.
- Fact checking, bias detection, source credibility analysis, and the HF classifier all run in parallel with `ThreadPoolExecutor(max_workers=4)`.
- The Devil's Advocate agent runs when the Judge confidence is below `0.90`.
- If the Devil's Advocate recommends `OVERTURN`, the final label is flipped and the revised confidence is used.

## HuggingFace Classifier

- Model: `jy46604790/Fake-News-Bert-Detect`, fine-tuned on the ISOT dataset (the same dataset used for evaluation).
- `LABEL_1` = REAL, `LABEL_0` = FAKE for this model; label normalization handles multiple HuggingFace naming conventions.
- The model is lazy-loaded on first use and cached for the process lifetime to avoid repeated load overhead.
- If PyTorch is unavailable (e.g. Python 3.13 base conda env), the classifier returns `UNKNOWN` immediately without crashing the pipeline.
- Its output (`hf_label`, `hf_confidence`) is injected into the Judge evidence bundle as a **minor tiebreaker signal only** — it does not override fact-check or bias+credibility signals.
- The Judge prompt explicitly deprioritizes the HF signal to prevent it from dominating the verdict.

## Web Search Grounding

- The Fact Checker uses live DuckDuckGo web search (`ddgs` package) to retrieve snippets for each extracted claim before calling the LLM.
- All claim searches run in parallel within the Fact Checker using `ThreadPoolExecutor(max_workers=3)` to avoid hammering the search API.
- Up to 3 snippets per claim are retrieved and formatted into a `[WEB]` evidence block injected into the LLM user message.
- The Fact Checker prompt instructs the LLM to prefer web evidence over training-data assumptions, and to cite the relevant snippet in the `evidence` field.
- If search fails for a claim (network error, rate limit), the agent falls back silently to the LLM's own knowledge and marks that claim's evidence as absent.
- `web_search_used` and `web_snippets_found` are stored in the fact checker output for debugging.

## JSON Parsing

- LLM responses are parsed through a fault-tolerant extraction flow.
- The parser attempts direct JSON loading, fenced-code cleanup, regex extraction of the first JSON object, and finally a safe default.

## Article Truncation

- Claim extraction and bias analysis use up to 3000 characters of article text.
- Fact checking receives the evidence block (web snippets per claim) plus up to 800 characters of article context.
- The Judge receives summarized agent outputs plus up to 500 characters of article context.
- These limits keep latency and token use manageable while preserving the main signals needed by each stage.

## Interface

- The Streamlit app uses the "Veritas Bureau" case-file visual theme.
- Single-article analysis uses `session_state` to avoid unnecessary reruns during interaction.
- The pipeline diagram shows all parallel agents including the HF classifier as a live-animated box.
- A model selector dropdown in the main header bar lets users switch Ollama models at runtime without restarting the app.
- A dark/light theme toggle sits next to the model selector in the header bar at the same visual height.

## Batch Processing

- Batch results are written incrementally after each processed article for crash recovery.
- Resume behavior skips articles that have already been saved in the output file.
- `--workers N` enables parallel article processing via `ThreadPoolExecutor`; a threading lock protects concurrent CSV writes.
- `--model <name>` overrides the configured Ollama model at runtime, enabling side-by-side comparison runs without changing `.env`.
- Live progress is printed in-place showing total processed, REAL count, and FAKE count.
- Accuracy on labeled articles is printed at the end of each run.
- The output CSV includes `hf_label`, `hf_confidence`, `judge_summary`, `subject`, and `date` columns alongside the existing agent signals.

## Evaluation

- Evaluation notebooks analyze overall metrics and also compare performance by subject and date when those fields are available.
- Final evaluation uses 100 deduplicated articles (50 REAL / 50 FAKE) from the ISOT dataset.
