"""Quick pipeline test with 3 inline articles (fake, real, ambiguous)."""
import json
from pipeline.core import analyze_article

ARTICLES = [
    {
        "label": "FAKE",
        "title": "BREAKING: Hillary Clinton Arrested for Treason — Deep State Exposed",
        "text": (
            "Sources close to the White House confirmed Thursday that Hillary Clinton was secretly arrested "
            "by military tribunals for treason against the United States. The Deep State operation, which "
            "has been running for decades, was finally exposed by patriot insiders. Clinton was allegedly "
            "caught selling classified information to China and funding radical terrorist groups. President "
            "Trump reportedly signed the arrest warrant last night. Mainstream media refuses to cover this "
            "EXPLOSIVE story because they are complicit. Share this before it gets deleted! The arrests "
            "are just beginning — hundreds of globalist elites will face military justice. God bless America!"
        ),
    },
    {
        "label": "REAL",
        "title": "Federal Reserve raises interest rates by 0.25 percentage points",
        "text": (
            "The Federal Reserve on Wednesday raised its benchmark interest rate by a quarter of a percentage "
            "point, as policymakers continued their effort to bring inflation back down to the central bank's "
            "2 percent target. The decision was unanimous among the 12 voting members of the Federal Open "
            "Market Committee. Fed Chair Jerome Powell said in a press conference that while inflation has "
            "eased somewhat, it remains above the Fed's longer-run goal. The new target range for the "
            "federal funds rate is now 5.25 to 5.5 percent, the highest level in 22 years. Economists "
            "widely expected the move and markets showed little reaction following the announcement."
        ),
    },
    {
        "label": "AMBIGUOUS",
        "title": "Scientists discover ancient virus frozen in Siberian permafrost",
        "text": (
            "French scientists have revived a 48,500-year-old virus from Siberian permafrost, raising concerns "
            "about potential pandemic risks as Arctic ice continues to melt. The virus, called Pandoravirus yedoma, "
            "was extracted from a frozen lake sediment sample in Russia. While the researchers say the virus can "
            "only infect amoeba and poses no direct risk to humans, some biosafety experts warn that the melting "
            "permafrost could release unknown ancient pathogens that human immune systems have never encountered. "
            "The study was published in the journal Viruses. Climate scientists note that permafrost thawing "
            "is accelerating due to global warming."
        ),
    },
]


def print_separator(label=""):
    print("\n" + "=" * 60)
    if label:
        print(f"  {label}")
    print("=" * 60)


def main():
    print_separator("VERITAS BUREAU — PIPELINE TEST")
    print(f"Testing {len(ARTICLES)} articles...\n")

    for i, article in enumerate(ARTICLES, 1):
        print_separator(f"Article {i}/{len(ARTICLES)}: [{article['label']}] {article['title'][:50]}...")
        result = analyze_article(article["text"], article["title"])

        print(f"\n{'-'*40}")
        print(f"FINAL VERDICT:  {result['final_label']}")
        print(f"CONFIDENCE:     {result['final_confidence']:.2%}")
        print(f"TOTAL TIME:     {result['total_processing_time']}s")
        print(f"\nCLAIMS EXTRACTED ({len(result['claims'].get('claims', []))}):")
        for c in result["claims"].get("claims", []):
            print(f"  • {c}")

        fc = result["fact_check"]
        print(f"\nFACT CHECK ({len(fc.get('results', []))} claims checked):")
        for r in fc.get("results", []):
            print(f"  [{r.get('verdict','?')}] {r.get('claim','?')[:60]}...")

        bias = result["bias"]
        print(f"\nBIAS: tone={bias.get('tone')}, score={bias.get('bias_score')}")
        for f in bias.get("flags", []):
            print(f"  * {f}")

        cred = result["credibility"]
        print(f"\nCREDIBILITY: score={cred.get('credibility_score')}")
        for f in cred.get("flags", []):
            print(f"  * {f}")

        hf = result.get("hf_classifier", {})
        if hf.get("hf_label"):
            print(f"\nHF CLASSIFIER: label={hf.get('hf_label')}, confidence={hf.get('hf_confidence', 0):.0%}")

        print(f"\nJUDGE SUMMARY:\n  {result['judge'].get('summary', 'N/A')}")

        if result["devils_advocate_triggered"] and result["devils_advocate"]:
            da = result["devils_advocate"]
            print(f"\nDEVIL'S ADVOCATE:")
            print(f"  Recommendation: {da.get('recommendation')}")
            print(f"  Reasoning: {da.get('reasoning','')[:120]}")

        print(f"\n[Expected: {article['label']}  |  Got: {result['final_label']}]")

    print_separator("TEST COMPLETE")


if __name__ == "__main__":
    main()
