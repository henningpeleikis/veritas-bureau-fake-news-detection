"""Web search grounding for the fact checker agent.

Uses DuckDuckGo (no API key required). Each claim is searched in parallel
and the top snippets are injected into the fact checker's evidence bundle.
Falls back silently if search fails or returns no results.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed


def web_search(query: str, max_results: int = 3) -> list[dict]:
    """Search DuckDuckGo for query. Returns list of {title, url, snippet} dicts."""
    try:
        from ddgs import DDGS
        with DDGS() as ddgs:
            raw = list(ddgs.text(query, max_results=max_results))
        return [
            {
                "title": r.get("title", ""),
                "url": r.get("href", ""),
                "snippet": r.get("body", ""),
            }
            for r in raw
        ]
    except Exception as exc:
        print(f"[WebSearch] '{query[:60]}' failed: {exc}")
        return []


def search_claims(claims: list[str], max_results: int = 3) -> dict[str, list[dict]]:
    """Search all claims in parallel. Returns {claim_text: [result, ...]}."""
    output: dict[str, list[dict]] = {c: [] for c in claims}
    if not claims:
        return output

    # Cap workers to avoid hammering DuckDuckGo
    workers = min(len(claims), 3)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_claim = {
            executor.submit(web_search, claim, max_results): claim
            for claim in claims
        }
        for future in as_completed(future_to_claim):
            claim = future_to_claim[future]
            try:
                output[claim] = future.result()
            except Exception as exc:
                print(f"[WebSearch] claim search failed: {exc}")
    return output


def format_search_evidence(claims: list[str], search_results: dict[str, list[dict]]) -> str:
    """Format search results into a compact evidence block for the LLM."""
    lines = []
    for i, claim in enumerate(claims, 1):
        snippets = search_results.get(claim, [])
        lines.append(f"Claim {i}: {claim}")
        if snippets:
            for s in snippets:
                text = s.get("snippet", "").strip()
                if text:
                    lines.append(f"  [WEB] {text[:220]}")
        else:
            lines.append("  [WEB] No results found — rely on your own knowledge.")
        lines.append("")
    return "\n".join(lines)
