"""Veritas Bureau — Multi-Agent Fake News Detection UI."""

import io, time
from datetime import datetime
import pandas as pd
import streamlit as st
import config

# ── Pipeline imports ──
from pipeline.core import analyze_article
try:
    from agents.runners import (
        run_claim_extractor, run_fact_checker, run_bias_detector,
        run_source_credibility, run_judge, run_devils_advocate,
        should_overturn_with_devils_advocate,
    )
    from pipeline.parallel import run_parallel_agents
    STEP_BY_STEP = True
except ImportError:
    STEP_BY_STEP = False

st.set_page_config(page_title="Veritas Bureau", page_icon="🗂️", layout="wide", initial_sidebar_state="collapsed")
st.markdown('<link href="https://fonts.googleapis.com/css2?family=Courier+Prime:wght@400;700&family=IBM+Plex+Sans:wght@300;400;500;600&family=Playfair+Display:wght@400;700;900&family=Old+Standard+TT:wght@400;700&display=swap" rel="stylesheet">', unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════════
# THEME
# ════════════════════════════════════════════════════════════════════════════════

if "theme" not in st.session_state:
    st.session_state.theme = "dark"

THEMES = {
    "dark": {
        "bg": "#0f0e0c", "card": "#171614", "border": "#2a2825",
        "text": "#C8C3B8", "text2": "#9B9590", "muted": "#6B6660",
        "heading": "#E8E3D8", "dim": "#4a4640", "track": "#1a1916",
        "input_bg": "#171614", "input_border": "#2a2825",
    },
    "light": {
        "bg": "#F5F3EF", "card": "#FFFFFF", "border": "#E0DCD5",
        "text": "#3D3A36", "text2": "#5C5852", "muted": "#8A857D",
        "heading": "#1A1815", "dim": "#B5B0A8", "track": "#E8E4DD",
        "input_bg": "#FFFFFF", "input_border": "#D5D0C8",
    },
}

def T():
    return THEMES[st.session_state.theme]

def gen_css():
    t = T()
    return f"""
<style>
html, body, [data-testid="stAppViewContainer"], .stApp,
section.main > div, [data-testid="stAppViewBlockContainer"] {{
    background: {t['bg']} !important;
    color: {t['text']} !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
}}
[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"] {{
    display: none !important;
}}
.block-container {{
    padding: 1.5rem 2rem 4rem !important;
    max-width: 1100px !important;
}}
h1, h2, h3 {{
    font-family: 'IBM Plex Sans', sans-serif !important;
    color: {t['heading']} !important;
}}
p, span, div, label {{
    color: {t['text']} !important;
}}
[data-testid="stTabs"] [role="tablist"] {{
    border-bottom: 1px solid {t['border']} !important;
    gap: 0 !important;
}}
[data-testid="stTabs"] [role="tab"] {{
    font-family: 'Courier Prime', monospace !important;
    font-size: 11px !important; letter-spacing: 2px !important;
    text-transform: uppercase !important; color: {t['muted']} !important;
    padding: 10px 20px !important; background: transparent !important;
    border: none !important; border-radius: 0 !important;
}}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {{
    color: {t['text']} !important;
    border-bottom: 2px solid #C0392B !important;
}}
.stTextArea textarea, .stTextInput input {{
    background: {t['input_bg']} !important;
    border: 1px solid {t['input_border']} !important;
    border-radius: 8px !important;
    color: {t['text']} !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
}}
.stTextArea textarea:focus, .stTextInput input:focus {{
    border-color: {t['muted']} !important;
    box-shadow: none !important;
}}
.stButton > button {{
    background: #C0392B !important; color: #fff !important;
    border: none !important; border-radius: 6px !important;
    font-family: 'Courier Prime', monospace !important;
    font-size: 12px !important; letter-spacing: 2px !important;
    text-transform: uppercase !important; padding: 10px 24px !important;
}}
.stButton > button:hover {{ opacity: 0.85 !important; }}
.stButton > button:disabled {{
    background: {t['border']} !important; color: {t['dim']} !important;
}}
.stDownloadButton > button {{
    background: transparent !important; color: {t['muted']} !important;
    border: 1px solid {t['border']} !important; border-radius: 6px !important;
    font-family: 'Courier Prime', monospace !important;
    font-size: 11px !important; letter-spacing: 1px !important;
    text-transform: uppercase !important;
}}
[data-testid="stFileUploader"] section {{
    background: {t['card']} !important;
    border: 1px dashed {t['border']} !important;
    border-radius: 8px !important;
}}
[data-testid="stExpander"] {{
    background: {t['card']} !important;
    border: 1px solid {t['border']} !important;
    border-radius: 8px !important;
}}
[data-testid="stExpander"] summary {{
    color: {t['text2']} !important;
}}
[data-testid="stStatus"] {{
    background: {t['card']} !important;
    border: 1px solid {t['border']} !important;
    border-radius: 8px !important;
    color: {t['text']} !important;
}}
[data-testid="stMetric"] {{
    background: {t['card']} !important;
    border: 1px solid {t['border']} !important;
    border-radius: 8px !important;
    padding: 12px 16px !important;
}}
[data-testid="stMetricLabel"] {{ color: {t['muted']} !important; }}
[data-testid="stMetricValue"] {{ color: {t['heading']} !important; }}
[data-testid="stCaptionContainer"] p {{ color: {t['muted']} !important; }}
[data-testid="stSlider"] label {{ color: {t['text']} !important; }}
</style>"""

st.markdown(gen_css(), unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ════════════════════════════════════════════════════════════════════════════════

AGENT_INFO = [
    {"id": 1, "icon": "🔍", "name": "Agent 1", "role": "Claim Extractor",
     "color": "#7F77DD",
     "desc": "Reads the article and extracts an adaptive number of verifiable factual claims based on article length. Focuses on statistics, named entities, events, and dates while ignoring opinions and rhetoric."},
    {"id": 2, "icon": "📋", "name": "Agent 2", "role": "Fact Checker",
     "color": "#1D9E75",
     "desc": "Evaluates each claim using LLM knowledge. Assigns SUPPORTED, CONTRADICTED, or UNVERIFIABLE verdicts with evidence."},
    {"id": 3, "icon": "⚖️", "name": "Agent 3", "role": "Bias Detector",
     "color": "#D85A30",
     "desc": "Analyzes tone, emotional language, sensationalism, one-sided framing, and missing context. Produces a bias score 0-1."},
    {"id": 5, "icon": "🏷️", "name": "Agent 5", "role": "Source Credibility",
     "color": "#D4537E",
     "desc": "Checks for author byline, publication date, cited sources, headline quality, and article structure. Custom addition to the pipeline."},
    {"id": 4, "icon": "⚡", "name": "Agent 4", "role": "Judge",
     "color": "#378ADD",
     "desc": "Synthesizes all evidence from Agents 2, 3, and 5. Weighs contradicted claims, bias, and credibility to produce a final REAL/FAKE verdict."},
    {"id": 6, "icon": "😈", "name": "Agent 6", "role": "Devil's Advocate",
     "color": "#E24B4A",
     "desc": "Only activates when Judge confidence < 0.6. Argues the opposite case. If counter-arguments are strong, the verdict is overturned. Custom addition."},
    {"id": 7, "icon": "🤗", "name": "HF Tool", "role": "BERT Classifier",
     "color": "#7B61FF",
     "desc": "Pretrained BERT model (jy46604790/Fake-News-Bert-Detect) fine-tuned on fake news detection. Runs in parallel alongside Agents 2, 3, 5 and provides an independent classification signal to the Judge. Custom addition."},
]


# ════════════════════════════════════════════════════════════════════════════════
# HTML HELPERS
# ════════════════════════════════════════════════════════════════════════════════

def html_header():
    t = T()
    tc, tm, tl = st.columns([4, 2, 1])
    with tc:
        st.markdown(f"""
<div style="display:flex;align-items:center;gap:14px;padding:20px 0 16px">
  <div style="width:44px;height:44px;background:linear-gradient(135deg,#C0392B,#7F77DD);border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:22px;flex-shrink:0">🗂️</div>
  <div>
    <div style="font-size:24px;font-weight:600;color:{t['heading']};letter-spacing:-0.5px">Veritas Bureau</div>
    <div style="font-family:'Courier Prime',monospace;font-size:10px;letter-spacing:3px;text-transform:uppercase;color:{t['muted']};margin-top:2px">Multi-agent investigation unit</div>
  </div>
</div>
""", unsafe_allow_html=True)
    with tm:
        st.markdown("<div style='margin-top:28px'>", unsafe_allow_html=True)
        available_models = get_ollama_models()
        current_model = config.OLLAMA_MODEL if config.OLLAMA_MODEL in available_models else available_models[0]
        selected_model = st.selectbox("Model", available_models, index=available_models.index(current_model), label_visibility="collapsed", key="model_selector")
        if selected_model != config.OLLAMA_MODEL:
            config.OLLAMA_MODEL = selected_model
        st.markdown("</div>", unsafe_allow_html=True)
    with tl:
        st.markdown("<div style='margin-top:28px'>", unsafe_allow_html=True)
        icon = "☀️" if st.session_state.theme == "dark" else "🌙"
        if st.button(f"{icon} Theme", use_container_width=True, key="theme_btn"):
            st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(f'<div style="border-bottom:1px solid {t["border"]};margin-bottom:16px"></div>', unsafe_allow_html=True)


def divider(label):
    t = T()
    st.markdown(f'<div style="display:flex;align-items:center;gap:12px;margin:24px 0 16px"><div style="flex:1;height:1px;background:{t["border"]}"></div><span style="font-family:\'Courier Prime\',monospace;font-size:10px;letter-spacing:3px;text-transform:uppercase;color:{t["muted"]}">{label}</span><div style="flex:1;height:1px;background:{t["border"]}"></div></div>', unsafe_allow_html=True)


def meter(label, value, color):
    t = T()
    pct = int(max(0, min(1, value)) * 100)
    st.markdown(f"""
<div style="margin-bottom:12px">
  <div style="display:flex;justify-content:space-between;margin-bottom:6px">
    <span style="font-family:'Courier Prime',monospace;font-size:10px;letter-spacing:1px;text-transform:uppercase;color:{t['muted']}">{label}</span>
    <span style="font-family:'Courier Prime',monospace;font-size:12px;font-weight:500;color:{color}">{pct}%</span>
  </div>
  <div style="height:5px;background:{t['track']};border-radius:3px;overflow:hidden">
    <div style="height:100%;width:{pct}%;background:{color};border-radius:3px"></div>
  </div>
</div>""", unsafe_allow_html=True)


def stamp(label, confidence):
    colors = {"REAL": "#27AE60", "FAKE": "#C0392B"}.get(label.upper(), "#D4A017")
    st.markdown(f"""
<div style="display:inline-flex;flex-direction:column;align-items:center;transform:rotate(-5deg);user-select:none;margin-top:8px">
  <div style="font-family:'Courier Prime',monospace;font-size:32px;font-weight:700;letter-spacing:6px;padding:8px 24px;border:3px solid {colors};border-radius:6px;color:{colors};opacity:0.85;text-transform:uppercase">{label}</div>
  <div style="font-family:'Courier Prime',monospace;font-size:10px;letter-spacing:2px;color:{colors};opacity:0.6;margin-top:5px">CONFIDENCE: {int(confidence*100)}%</div>
</div>""", unsafe_allow_html=True)


def pills(flags, color):
    if not flags:
        return
    html = "".join([f'<span style="font-family:\'Courier Prime\',monospace;font-size:11px;color:{color};border:1px solid {color}44;background:{color}11;padding:4px 12px;border-radius:4px">{f}</span>' for f in flags])
    st.markdown(f'<div style="display:flex;flex-wrap:wrap;gap:8px">{html}</div>', unsafe_allow_html=True)


def claim_card(idx, text, verdict, evidence):
    vc = {"SUPPORTED": "#27AE60", "CONTRADICTED": "#C0392B"}.get(verdict.upper(), "#D4A017")
    t = T()
    ev = f'<div style="font-size:12px;color:{t["muted"]};line-height:1.5;margin-top:6px">{evidence}</div>' if evidence else ""
    st.markdown(f"""
<div style="background:{t['bg']};border:1px solid {t['track']};border-radius:8px;padding:14px;margin-bottom:8px">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:12px">
    <span style="font-size:13px;color:{t['text']};line-height:1.5;flex:1">
      <span style="font-family:'Courier Prime',monospace;color:{t['muted']};margin-right:8px">#{idx}</span>"{text}"
    </span>
    <span style="font-family:'Courier Prime',monospace;font-size:10px;letter-spacing:1px;text-transform:uppercase;color:{vc};border:1px solid {vc}44;padding:3px 10px;border-radius:3px;flex-shrink:0;white-space:nowrap">{verdict}</span>
  </div>
  {ev}
</div>""", unsafe_allow_html=True)


def agent_card(icon, name, role, color, content):
    t = T()
    st.markdown(f"""
<div style="background:{t['card']};border:1px solid {t['border']};border-left:3px solid {color};border-radius:10px;padding:20px;margin-bottom:12px">
  <div style="display:flex;align-items:center;gap:10px;margin-bottom:14px">
    <div style="width:32px;height:32px;border-radius:50%;background:{color}22;border:2px solid {color};display:flex;align-items:center;justify-content:center;font-size:14px">{icon}</div>
    <div>
      <div style="font-family:'Courier Prime',monospace;font-size:12px;letter-spacing:1px;color:{t['text']}">{name}</div>
      <div style="font-size:11px;color:{t['muted']}">{role}</div>
    </div>
  </div>
  <div style="font-size:13px;color:{t['text2']};line-height:1.7">{content}</div>
</div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
# PIPELINE DIAGRAM
# ════════════════════════════════════════════════════════════════════════════════

def pipeline_diagram(states=None):
    """Render pipeline diagram. states = dict of agent_id -> 'idle'|'active'|'done'"""
    if states is None:
        states = {a["id"]: "idle" for a in AGENT_INFO}
    t = T()

    def box(agent, width="130px"):
        s = states.get(agent["id"], "idle")
        if s == "active":
            bg = f"{agent['color']}33"
            border = agent["color"]
            txt_color = t["heading"]
            extra = f"box-shadow:0 0 12px {agent['color']}44;"
            indicator = f'<div style="font-size:9px;color:{agent["color"]};margin-top:4px;font-family:\'Courier Prime\',monospace;letter-spacing:1px">⟳ WORKING...</div>'
        elif s == "done":
            bg = f"{agent['color']}22"
            border = agent["color"]
            txt_color = t["heading"]
            extra = ""
            indicator = f'<div style="font-size:9px;color:{agent["color"]};margin-top:4px">✓ Complete</div>'
        else:
            bg = t["card"]
            border = t["border"]
            txt_color = t["muted"]
            extra = "opacity:0.5;"
            indicator = ""
        return f"""<div style="display:inline-block;background:{bg};border:2px solid {border};border-radius:10px;padding:12px;text-align:center;width:{width};{extra}">
  <div style="font-size:16px;margin-bottom:4px">{agent['icon']}</div>
  <div style="font-family:'Courier Prime',monospace;font-size:10px;letter-spacing:1px;color:{txt_color}">{agent['name']}</div>
  <div style="font-size:10px;color:{t['muted']};margin-top:2px">{agent['role']}</div>
  {indicator}
</div>"""

    arrow_color = t["muted"]
    arrow_v = f'<div style="text-align:center;color:{arrow_color};font-size:18px;margin:6px 0">↓</div>'
    arrow_h = f'<span style="color:{arrow_color};font-size:18px;margin:0 8px">→</span>'

    agents = {a["id"]: a for a in AGENT_INFO}

    # Input
    input_box = f'<div style="background:{t["card"]};border:1px solid {t["border"]};border-radius:8px;padding:10px 20px;text-align:center;display:inline-block"><span style="font-family:\'Courier Prime\',monospace;font-size:11px;letter-spacing:1px;color:{t["text"]}">📄 News Article</span></div>'

    # Parallel label
    par_s = states.get(2, "idle")
    par_color = "#1D9E75" if par_s in ("active", "done") else t["muted"]
    par_label = f'<div style="text-align:center;font-family:\'Courier Prime\',monospace;font-size:9px;letter-spacing:2px;color:{par_color};margin:8px 0">RUN IN PARALLEL</div>'

    # Merge
    merge_box = f'<div style="background:{t["card"]};border:1px solid {t["border"]};border-radius:8px;padding:8px 16px;text-align:center;display:inline-block"><span style="font-family:\'Courier Prime\',monospace;font-size:10px;letter-spacing:1px;color:{t["muted"]}">Merge results</span></div>'

    # Confidence check
    conf_s = states.get(4, "idle")
    conf_color = "#D4A017" if conf_s == "done" else t["muted"]
    conf_box = f'<div style="background:{t["card"]};border:1px solid {conf_color};border-radius:8px;padding:8px 16px;text-align:center;display:inline-block"><span style="font-family:\'Courier Prime\',monospace;font-size:10px;color:{conf_color}">Confidence check</span></div>'

    # Output
    out_s = states.get(4, "idle")
    out_color = "#27AE60" if out_s == "done" else t["muted"]
    out_box = f'<div style="background:{out_color}22;border:1px solid {out_color};border-radius:8px;padding:10px 20px;text-align:center;display:inline-block"><span style="font-family:\'Courier Prime\',monospace;font-size:11px;letter-spacing:1px;color:{out_color}">📋 Verdict</span></div>'

    st.markdown(f"""
<div style="background:{t['card']};border:1px solid {t['border']};border-radius:12px;padding:28px;text-align:center">
  <div>{input_box}</div>
  {arrow_v}
  <div>{box(agents[1])}</div>
  {arrow_v}
  {par_label}
  <div style="display:flex;justify-content:center;gap:16px;flex-wrap:wrap">
    {box(agents[2])}
    {box(agents[3])}
    {box(agents[5])}
    {box(agents[7])}
  </div>
  {arrow_v}
  <div>{merge_box}</div>
  {arrow_v}
  <div>{box(agents[4])}</div>
  {arrow_v}
  <div>{conf_box}</div>
  <div style="display:flex;justify-content:center;gap:32px;margin-top:12px">
    <div style="text-align:center">
      <div style="font-size:9px;color:{t['muted']};font-family:'Courier Prime',monospace;margin-bottom:6px">≥ 0.90</div>
      {out_box}
    </div>
    <div style="text-align:center">
      <div style="font-size:9px;color:{t['muted']};font-family:'Courier Prime',monospace;margin-bottom:6px">{'< 0.90'}</div>
      {box(agents[6], "140px")}
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
# RENDER: CASE FILE — published as a newspaper front page
# ════════════════════════════════════════════════════════════════════════════════

def render_case_file(result):
    """Render the investigation result as a newspaper front page."""

    # Newsprint palette (constant regardless of theme)
    paper       = "#EFE8D4"
    ink         = "#1A1815"
    ink_soft    = "#5C5852"
    ink_faded   = "#8A857D"
    ink_red     = "#8B0000"
    ink_green   = "#1A5E3F"
    ink_sepia   = "#6B4423"

    # Pull data
    judge   = result.get("judge", {}) or {}
    bias    = result.get("bias", {}) or {}
    cred    = result.get("credibility", {}) or {}
    fact    = result.get("fact_check", {}) or {}
    claims  = (result.get("claims", {}) or {}).get("claims", []) or []
    hf      = result.get("hf_classifier", {}) or {}
    da      = result.get("devils_advocate") or {}
    da_on   = result.get("devils_advocate_triggered", False)
    label   = (result.get("final_label") or "UNKNOWN").upper()
    conf    = float(result.get("final_confidence", 0.5) or 0.5)
    title   = (result.get("article_title") or "Untitled article").strip() or "Untitled article"
    case_n  = f"{datetime.now().strftime('%Y%m%d')}-{abs(hash(title)) % 9000 + 1000}"
    today   = datetime.now().strftime("%A, %d %B %Y").upper()

    # Verdict banner
    if label == "REAL":
        banner_text  = "ARTICLE VERIFIED AUTHENTIC"
        banner_deck  = "Bureau finds claims substantiated; analysis corroborates the reporting."
        bc = ink_green
    elif label == "FAKE":
        banner_text  = "ARTICLE RULED FABRICATED"
        banner_deck  = "Investigation exposes contradicted claims and prejudicial framing."
        bc = ink_red
    else:
        banner_text  = "VERDICT RETURNED INCONCLUSIVE"
        banner_deck  = "Bureau declines to certify; the evidence is insufficient for a determination."
        bc = ink_sepia

    # Counts
    sup = con = unv = 0
    for r in fact.get("results", []) or []:
        v = (r.get("verdict") or "").upper()
        if   v == "SUPPORTED":    sup += 1
        elif v == "CONTRADICTED": con += 1
        else:                     unv += 1

    # Lead paragraph
    summary = (judge.get("summary") or
               "The Bureau has completed its review. Findings are documented below.").strip()
    drop_cap = summary[0].upper() if summary else "T"
    rest     = summary[1:] if len(summary) > 1 else ""

    # Claims ledger
    fact_rows = fact.get("results", []) or []
    claims_items = []
    for i, cl in enumerate(claims):
        matched = next(
            (r for r in fact_rows
             if cl[:35].lower() in (r.get("claim") or "").lower()
             or (r.get("claim") or "")[:35].lower() in cl.lower()),
            None,
        )
        verdict = (matched.get("verdict", "UNVERIFIABLE").upper() if matched else "UNVERIFIABLE")
        v_color = {"SUPPORTED": ink_green, "CONTRADICTED": ink_red}.get(verdict, ink_sepia)
        v_label = {"SUPPORTED": "STANDS", "CONTRADICTED": "FALSE", "UNVERIFIABLE": "UNCLEAR"}.get(verdict, "UNCLEAR")
        cl_safe = cl.replace("<", "&lt;").replace(">", "&gt;")
        cl_trunc = cl_safe if len(cl_safe) <= 150 else cl_safe[:147] + "&hellip;"
        claims_items.append(
            f'<div style="break-inside:avoid;padding:9px 0;border-bottom:1px dotted {ink_soft}">'
            f'<div style="display:flex;justify-content:space-between;align-items:baseline;gap:8px;margin-bottom:3px">'
            f'<span class="vg-mono" style="font-size:9px !important;letter-spacing:2px;color:{ink_faded} !important">CLAIM {i+1:02d}</span>'
            f'<span class="vg-mono" style="font-size:9px !important;letter-spacing:1px;color:{v_color} !important;font-weight:700">&mdash; {v_label}</span>'
            f'</div>'
            f'<div style="font-family:\'Old Standard TT\',\'Playfair Display\',serif !important;font-size:12.5px !important;color:{ink} !important;line-height:1.5;font-style:italic">&ldquo;{cl_trunc}&rdquo;</div>'
            f'</div>'
        )
    claims_html = "".join(claims_items) if claims_items else (
        f'<div style="font-style:italic;color:{ink_faded} !important;text-align:center;padding:14px">No claims were entered into the record.</div>'
    )

    # HF classifier
    hf_html = ""
    if hf and hf.get("hf_label") not in (None, "UNKNOWN"):
        hf_label = hf.get("hf_label", "?")
        hf_conf  = float(hf.get("hf_confidence", 0) or 0)
        hf_color = {"REAL": ink_green, "FAKE": ink_red}.get(hf_label, ink_sepia)
        hf_html = (
            f'<div style="border:1px solid {ink};padding:10px 12px;margin-top:14px">'
            f'<div class="vg-mono" style="font-size:9px !important;letter-spacing:3px;color:{ink_faded} !important">&middot; SECOND OPINION &middot;</div>'
            f'<div style="font-family:\'Playfair Display\',serif !important;font-size:15px !important;font-weight:900;color:{hf_color} !important;letter-spacing:1px;margin-top:4px">{hf_label} &middot; {int(hf_conf*100)}%</div>'
            f'<div class="vg-mono" style="font-size:9px !important;color:{ink_faded} !important;margin-top:3px;font-style:italic">BERT classifier, ISOT corpus</div>'
            f'</div>'
        )

    # Bureau flags
    bias_flags = bias.get("flags", []) or []
    cred_flags = cred.get("flags", []) or []
    all_flags  = [*bias_flags[:3], *cred_flags[:2]]
    flags_html = ""
    if all_flags:
        flags_display = " &middot; ".join([f.upper() for f in all_flags])
        flags_html = (
            f'<div style="margin:22px 0 4px;padding:12px 16px;border-top:1px solid {ink};border-bottom:1px solid {ink};text-align:center">'
            f'<div class="vg-mono" style="font-size:9px !important;letter-spacing:3px;color:{ink_faded} !important;margin-bottom:5px">FLAGS RAISED BY THE BUREAU</div>'
            f'<div style="font-family:\'Playfair Display\',serif !important;font-size:13px !important;color:{ink} !important;font-style:italic;letter-spacing:0.5px;line-height:1.4">{flags_display}</div>'
            f'</div>'
        )

    # Stop Press (devil's advocate)
    stop_press_html = ""
    if da_on and da:
        rec    = da.get("recommendation", "UPHOLD")
        reason = (da.get("counter_argument") or da.get("reasoning") or "").strip()
        reason_safe = reason.replace("<", "&lt;").replace(">", "&gt;")
        reason_trunc = reason_safe if len(reason_safe) <= 380 else reason_safe[:377] + "&hellip;"
        rec_color = ink_red if rec == "OVERTURN" else ink_sepia
        stop_press_html = (
            f'<div style="margin-top:30px;padding:18px 22px;border:3px double {ink_red}">'
            f'<div style="text-align:center;font-family:\'Playfair Display\',serif !important;font-size:22px !important;font-weight:900;color:{ink_red} !important;letter-spacing:8px;padding-bottom:10px;border-bottom:1px solid {ink_red};margin-bottom:12px">'
            f'&middot; STOP PRESS &middot;</div>'
            f'<div class="vg-mono" style="font-size:9px !important;letter-spacing:2px;color:{ink_faded} !important;margin-bottom:6px;text-align:center">DISSENT FILED BY THE DEVIL&rsquo;S ADVOCATE</div>'
            f'<div style="font-family:\'Old Standard TT\',\'Playfair Display\',serif !important;font-size:14px !important;color:{ink} !important;line-height:1.7;text-align:justify">'
            f'<strong style="font-family:\'Playfair Display\',serif !important;font-weight:900;color:{rec_color} !important">Recommendation: {rec}.</strong> '
            f'<span style="font-style:italic">{reason_trunc}</span>'
            f'</div></div>'
        )

    title_safe = title.replace("<", "&lt;").replace(">", "&gt;")
    rest_safe  = rest.replace("<", "&lt;").replace(">", "&gt;")

    # ────────────────────────────────────────────────────────────────────────
    # Build newspaper HTML.
    #
    # IMPORTANT for Streamlit rendering:
    # 1. No HTML comments (break markdown parser)
    # 2. No lines with 4+ leading spaces (triggers code block)
    # 3. Scoped <style> with !important to override Streamlit global CSS
    # ────────────────────────────────────────────────────────────────────────

    bias_score = float(bias.get('bias_score', 0) or 0)
    cred_score = float(cred.get('credibility_score', 0) or 0)
    tone_text = (bias.get('tone') or 'unknown').upper()
    proc_time = result.get('total_processing_time', 0)

    # Scoped CSS reset — overrides Streamlit's global color/font !important
    scoped = (
        f'<style>'
        f'.vg,.vg div,.vg span,.vg p,.vg td,.vg tr,.vg table,.vg strong,.vg label'
        f'{{color:{ink} !important;font-family:"Playfair Display",serif !important}}'
        f'.vg .vg-mono{{font-family:"Courier Prime",monospace !important}}'
        f'</style>'
    )

    # Build the paper — every line starts at column 0 (no indentation)
    html = (
        scoped
        + f'<div class="vg" style="background:{paper};padding:40px 48px 32px;border:1px solid {ink};max-width:1000px;margin:0 auto;box-shadow:0 2px 24px rgba(0,0,0,0.35)">'
        + f'<div style="border-bottom:4px double {ink};padding-bottom:14px">'
        + f'<div style="text-align:center">'
        + f'<div class="vg-mono" style="font-size:10px !important;letter-spacing:5px;color:{ink} !important;margin-bottom:6px">&middot; THE DAILY EDITION &middot;</div>'
        + f'<div style="font-size:64px !important;font-weight:900;letter-spacing:8px;line-height:0.95;color:{ink} !important">VERITAS GAZETTE</div>'
        + f'<div class="vg-mono" style="font-size:10px !important;letter-spacing:6px;color:{ink} !important;margin-top:10px">BUREAU OF INVESTIGATIVE ANALYSIS &middot; EST. MMXXV</div>'
        + f'</div>'
        + f'<div style="display:flex;justify-content:space-between;align-items:center;margin-top:14px;padding-top:10px;border-top:1px solid {ink}">'
        + f'<span class="vg-mono" style="font-size:10px !important;letter-spacing:1px;color:{ink} !important">VOL. I &middot; NO. {case_n}</span>'
        + f'<span class="vg-mono" style="font-size:10px !important;letter-spacing:1px;color:{ink} !important">{today}</span>'
        + f'<span class="vg-mono" style="font-size:10px !important;letter-spacing:1px;color:{ink} !important">PRICE &mdash; ONE VERDICT</span>'
        + f'</div></div>'
        + f'<div style="text-align:center;padding:26px 0 18px;border-bottom:1px solid {ink}">'
        + f'<div style="font-size:46px !important;font-weight:900;letter-spacing:3px;line-height:1.05;color:{bc} !important;text-transform:uppercase">{banner_text}</div>'
        + f'<div style="font-size:16px !important;font-style:italic;color:{ink_soft} !important;margin-top:12px;line-height:1.4">{banner_deck}</div>'
        + f'<div class="vg-mono" style="font-size:10px !important;letter-spacing:3px;color:{ink_faded} !important;margin-top:14px">CONFIDENCE &middot; {int(conf*100)}% &nbsp;&middot;&nbsp; PROCESSING TIME &middot; {proc_time:.1f}s</div>'
        + f'</div>'
        + f'<div style="text-align:center;padding:16px 12px 14px;border-bottom:1px solid {ink_soft}">'
        + f'<div class="vg-mono" style="font-size:9px !important;letter-spacing:3px;color:{ink_faded} !important;margin-bottom:5px">SUBJECT OF INQUIRY</div>'
        + f'<div style="font-size:17px !important;color:{ink} !important;font-style:italic;line-height:1.4">&ldquo;{title_safe}&rdquo;</div>'
        + f'</div>'
        + f'<div style="display:grid;grid-template-columns:2fr 1fr;gap:30px;margin-top:22px">'
        + f'<div style="min-width:0">'
        + f'<div class="vg-mono" style="font-size:9px !important;letter-spacing:2px;color:{ink_faded} !important;margin-bottom:10px">BY THE MULTI-AGENT INVESTIGATION DESK</div>'
        + f'<div style="font-family:\'Old Standard TT\',\'Playfair Display\',serif !important;font-size:14px !important;color:{ink} !important;line-height:1.7;column-count:2;column-gap:24px;text-align:justify;hyphens:auto">'
        + f'<span style="font-family:\'Playfair Display\',serif !important;font-size:48px !important;font-weight:900;float:left;line-height:0.85;margin:4px 8px 0 0;color:{bc} !important">{drop_cap}</span>{rest_safe}'
        + f'</div>'
        + flags_html
        + f'</div>'
        + f'<div style="border-left:1px solid {ink_soft};padding-left:22px;min-width:0">'
        + f'<div style="border:2px solid {ink};padding:14px">'
        + f'<div style="font-size:15px !important;font-weight:900;text-align:center;letter-spacing:2px;color:{ink} !important;border-bottom:1px solid {ink};padding-bottom:8px;margin-bottom:10px">BY THE NUMBERS</div>'
        + f'<table style="width:100%;border-collapse:collapse">'
        + f'<tr><td class="vg-mono" style="padding:3px 0;font-size:10.5px !important;color:{ink} !important">Claims filed</td><td class="vg-mono" style="text-align:right;font-weight:700;font-size:10.5px !important;color:{ink} !important">{len(claims)}</td></tr>'
        + f'<tr><td class="vg-mono" style="padding:3px 0;font-size:10.5px !important;color:{ink} !important">Supported</td><td class="vg-mono" style="text-align:right;font-weight:700;font-size:10.5px !important;color:{ink_green} !important">{sup}</td></tr>'
        + f'<tr><td class="vg-mono" style="padding:3px 0;font-size:10.5px !important;color:{ink} !important">Contradicted</td><td class="vg-mono" style="text-align:right;font-weight:700;font-size:10.5px !important;color:{ink_red} !important">{con}</td></tr>'
        + f'<tr><td class="vg-mono" style="padding:3px 0;font-size:10.5px !important;color:{ink} !important">Unverified</td><td class="vg-mono" style="text-align:right;font-size:10.5px !important;color:{ink} !important">{unv}</td></tr>'
        + f'<tr><td colspan="2" style="padding:6px 0 4px;border-top:1px dotted {ink_soft}"></td></tr>'
        + f'<tr><td class="vg-mono" style="padding:3px 0;font-size:10.5px !important;color:{ink} !important">Bias score</td><td class="vg-mono" style="text-align:right;font-size:10.5px !important;color:{ink} !important">{bias_score:.2f}</td></tr>'
        + f'<tr><td class="vg-mono" style="padding:3px 0;font-size:10.5px !important;color:{ink} !important">Credibility</td><td class="vg-mono" style="text-align:right;font-size:10.5px !important;color:{ink} !important">{cred_score:.2f}</td></tr>'
        + f'<tr><td class="vg-mono" style="padding:3px 0;font-size:10.5px !important;color:{ink} !important">Tone</td><td class="vg-mono" style="text-align:right;font-style:italic;font-size:10.5px !important;color:{ink} !important">{tone_text}</td></tr>'
        + f'</table></div>'
        + hf_html
        + f'</div>'
        + f'</div>'
        + f'<div style="margin-top:32px;padding-top:18px;border-top:3px double {ink}">'
        + f'<div style="font-size:20px !important;font-weight:900;letter-spacing:2px;color:{ink} !important;margin-bottom:14px;text-align:center">EVIDENCE ENTERED INTO THE RECORD</div>'
        + f'<div style="column-count:2;column-gap:32px">{claims_html}</div>'
        + f'</div>'
        + stop_press_html
        + f'<div style="margin-top:32px;padding-top:14px;border-top:4px double {ink};text-align:center">'
        + f'<span class="vg-mono" style="font-size:10px !important;letter-spacing:4px;color:{ink_faded} !important">&mdash; END OF EDITION &middot; CASE {case_n} CLOSED &middot; VERITAS BUREAU &mdash;</span>'
        + f'</div>'
        + f'</div>'
    )

    st.markdown(html, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
# RENDER: AGENT CONCLUSIONS
# ════════════════════════════════════════════════════════════════════════════════

def render_conclusions(result):
    claims = result.get("claims", {})
    fact   = result.get("fact_check", {})
    bias   = result.get("bias", {})
    cred   = result.get("credibility", {})
    judge  = result.get("judge", {})
    da     = result.get("devils_advocate")
    da_on  = result.get("devils_advocate_triggered", False)

    # Agent 1
    cl = claims.get("claims", [])
    a1 = "<strong>Extracted claims:</strong><br>" + "<br>".join([f"{i+1}. {c}" for i, c in enumerate(cl)]) if cl else "No claims extracted."
    agent_card("🔍", "Agent 1", "Claim Extractor", "#7F77DD", a1)

    # Agent 2
    rows = fact.get("results", [])
    lines = []
    for r in rows:
        v = r.get("verdict", "?")
        c = {"SUPPORTED": "#27AE60", "CONTRADICTED": "#C0392B"}.get(v, "#D4A017")
        lines.append(f'<span style="color:{c};font-weight:500">{v}</span> — {r.get("claim","")}<br><span style="color:{T()["muted"]};font-size:12px">{r.get("evidence","")}</span>')
    agent_card("📋", "Agent 2", "Fact Checker", "#1D9E75", "<br>".join(lines) if lines else "No results.")

    # HF Classifier
    hf = result.get("hf_classifier", {})
    if hf and hf.get("hf_label") not in (None, "UNKNOWN"):
        hf_label = hf.get("hf_label", "?")
        hf_conf = hf.get("hf_confidence", 0)
        hf_col = {"REAL": "#27AE60", "FAKE": "#C0392B"}.get(hf_label, "#D4A017")
        agent_card("🤗", "HF Tool", "BERT Fake-News Classifier", "#7B61FF",
                   f'<strong>Label:</strong> <span style="color:{hf_col};font-weight:600">{hf_label}</span> ({hf_conf:.0%})<br>'
                   f'<span style="font-size:12px">Model: jy46604790/Fake-News-Bert-Detect</span>')

    # Agent 3
    bs = bias.get("bias_score", 0.5)
    tone = bias.get("tone", "unknown").upper()
    fl = ", ".join(bias.get("flags", [])) or "None"
    agent_card("⚖️", "Agent 3", "Bias Detector", "#D85A30", f"<strong>Tone:</strong> {tone}<br><strong>Bias score:</strong> {bs:.2f}<br><strong>Flags:</strong> {fl}")

    # Agent 5
    cs = cred.get("credibility_score", 0.5)
    hq = cred.get("headline_quality", "unknown").upper()
    cf = ", ".join(cred.get("flags", [])) or "None"
    agent_card("🏷️", "Agent 5", "Source Credibility", "#D4537E", f"<strong>Credibility:</strong> {cs:.2f}<br><strong>Headline:</strong> {hq}<br><strong>Flags:</strong> {cf}")

    # Agent 4
    jl = judge.get("label", "?")
    jc = judge.get("confidence", 0.5)
    js = judge.get("summary", "")
    jcolor = {"REAL": "#27AE60", "FAKE": "#C0392B"}.get(jl, "#D4A017")
    agent_card("⚡", "Agent 4", "Judge", "#378ADD", f'<strong>Verdict:</strong> <span style="color:{jcolor};font-weight:600">{jl}</span> ({jc:.0%})<br><br>{js}')

    # Agent 6
    if da_on and da:
        rec = da.get("recommendation", "UPHOLD")
        reason = da.get("reasoning") or da.get("counter_argument", "")
        rc = "#27AE60" if rec == "UPHOLD" else "#C0392B"
        agent_card("😈", "Agent 6", "Devil's Advocate", "#E24B4A", f'<strong>Recommendation:</strong> <span style="color:{rc};font-weight:600">{rec}</span><br><br>{reason}')


# ════════════════════════════════════════════════════════════════════════════════
# LIVE PIPELINE RUN
# ════════════════════════════════════════════════════════════════════════════════

def run_with_live_diagram(article_text, title):
    """Run pipeline step by step, updating the diagram between each stage."""
    diagram_slot = st.empty()
    status_slot = st.empty()
    states = {a["id"]: "idle" for a in AGENT_INFO}

    def update(active_ids, msg):
        for aid in active_ids:
            states[aid] = "active"
        with diagram_slot.container():
            pipeline_diagram(states)
        status_slot.caption(f"⏳ {msg}")

    def complete(done_ids):
        for aid in done_ids:
            states[aid] = "done"

    # Step 1: Claims
    update([1], "Agent 1: extracting claims...")
    if STEP_BY_STEP:
        claims_result = run_claim_extractor(article_text)
    else:
        result = analyze_article(article_text, title)
        with diagram_slot.container():
            states = {a["id"]: "done" for a in AGENT_INFO}
            pipeline_diagram(states)
        status_slot.success("Investigation complete.")
        return result
    complete([1])

    claims = claims_result.get("claims", [])

    # Step 2: Parallel
    update([2, 3, 5, 7], "Agents 2, 3, 5 + HF Classifier: running in parallel...")
    fact_check_result, bias_result, credibility_result, hf_result = run_parallel_agents(claims, article_text)
    complete([2, 3, 5, 7])

    # Step 3: Judge
    update([4], "Agent 4: judge deliberating...")
    judge_result = run_judge(
        fact_check_result, bias_result, credibility_result, article_text,
        hf_classifier=hf_result,
    )
    complete([4])

    # Step 4: Devil's advocate?
    da_triggered = False
    da_result = None
    conf = judge_result.get("confidence", 0.5)

    if conf < 0.9:
        da_triggered = True
        update([6], "Agent 6: devil's advocate challenging verdict...")
        da_result = run_devils_advocate(
            judge_result, fact_check_result, bias_result, credibility_result
        )
        complete([6])
        if should_overturn_with_devils_advocate(judge_result, da_result):
            orig = judge_result.get("label", "UNKNOWN")
            judge_result["label"] = "REAL" if orig == "FAKE" else "FAKE"
            judge_result["confidence"] = da_result.get("revised_confidence", conf)
            judge_result["summary"] = judge_result.get("summary", "") + f" [OVERTURNED: {da_result.get('reasoning','')}]"

    # Final diagram
    with diagram_slot.container():
        pipeline_diagram(states)
    status_slot.success("Investigation complete.")

    total_time = sum([
        claims_result.get("_processing_time", 0),
        fact_check_result.get("_processing_time", 0),
        bias_result.get("_processing_time", 0),
        credibility_result.get("_processing_time", 0),
        hf_result.get("_processing_time", 0),
        judge_result.get("_processing_time", 0),
        (da_result or {}).get("_processing_time", 0),
    ])

    return {
        "article_title": title,
        "article_text_preview": article_text[:200],
        "claims": claims_result,
        "fact_check": fact_check_result,
        "bias": bias_result,
        "credibility": credibility_result,
        "hf_classifier": hf_result,
        "judge": judge_result,
        "devils_advocate_triggered": da_triggered,
        "devils_advocate": da_result,
        "final_label": judge_result.get("label", "UNKNOWN"),
        "final_confidence": judge_result.get("confidence", 0.5),
        "total_processing_time": round(total_time, 2),
    }


# ════════════════════════════════════════════════════════════════════════════════
# PAGE
# ════════════════════════════════════════════════════════════════════════════════

def get_ollama_models() -> list[str]:
    try:
        import ollama
        client = ollama.Client(host=config.OLLAMA_BASE_URL)
        return sorted([m["model"] for m in client.list().get("models", [])])
    except Exception:
        return [config.OLLAMA_MODEL]

html_header()

tab_single, tab_agents, tab_pipeline, tab_batch = st.tabs([
    "Single Investigation", "Agent Conclusions", "Pipeline", "Batch Processing"
])

# ── Tab 1 ──
with tab_single:
    col_in, col_out = st.columns([1, 1], gap="large")
    with col_in:
        t = T()
        st.markdown(f'<div style="font-family:\'Courier Prime\',monospace;font-size:10px;letter-spacing:2px;text-transform:uppercase;color:{t["muted"]};margin-bottom:8px">Submit article for investigation</div>', unsafe_allow_html=True)
        title_input = st.text_input("Title", placeholder="Optional headline", label_visibility="collapsed")
        article_input = st.text_area("Body", placeholder="Paste the full article text here...", height=320, label_visibility="collapsed")
        st.caption("Typical runtime: 20–35 seconds with local Ollama.")
        open_case = st.button("Open Case", use_container_width=True)

    with col_out:
        if open_case:
            if not article_input.strip():
                st.warning("Paste article text before opening a case.")
            else:
                result = run_with_live_diagram(article_input.strip(), title_input.strip())
                st.session_state["last_result"] = result
        elif "last_result" in st.session_state:
            # Compact "case published below" pointer so col_out isn't empty
            t = T()
            st.markdown(f"""
<div style="background:{t['card']};border:1px solid {t['border']};border-radius:10px;padding:48px 24px;text-align:center">
  <div style="font-family:'Courier Prime',monospace;font-size:10px;letter-spacing:3px;text-transform:uppercase;color:{t['muted']};margin-bottom:12px">Case file published</div>
  <div style="font-size:32px;color:{t['dim']};margin:4px 0 10px;line-height:1">↓</div>
  <div style="font-family:'Courier Prime',monospace;font-size:10px;letter-spacing:1px;color:{t['dim']}">See the gazette below.<br>Submit new evidence to open another case.</div>
</div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
<div style="background:{T()['card']};border:1px solid {T()['border']};border-radius:10px;padding:48px 24px;text-align:center">
  <div style="font-family:'Courier Prime',monospace;font-size:12px;letter-spacing:1px;color:{T()['dim']}">No active case file.<br>Submit evidence on the left to begin.</div>
</div>""", unsafe_allow_html=True)

    # Render the newspaper full-width below the columns (needs the space for
    # the 2-column lead + sidebar layout — a half-column would crush it).
    if "last_result" in st.session_state:
        st.markdown("<div style='margin-top:32px'></div>", unsafe_allow_html=True)
        render_case_file(st.session_state["last_result"])

# ── Tab 2: Agent conclusions ──
with tab_agents:
    if "last_result" in st.session_state:
        st.markdown(f'<div style="font-family:\'Courier Prime\',monospace;font-size:10px;letter-spacing:2px;text-transform:uppercase;color:{T()["muted"]};margin-bottom:16px">Detailed findings from each agent</div>', unsafe_allow_html=True)
        render_conclusions(st.session_state["last_result"])
    else:
        st.markdown(f'<div style="background:{T()["card"]};border:1px solid {T()["border"]};border-radius:10px;padding:48px 24px;text-align:center"><div style="font-family:\'Courier Prime\',monospace;font-size:12px;letter-spacing:1px;color:{T()["dim"]}">Run a single investigation first to see agent conclusions.</div></div>', unsafe_allow_html=True)

# ── Tab 3: Pipeline ──
with tab_pipeline:
    t = T()
    st.markdown(f'<div style="font-family:\'Courier Prime\',monospace;font-size:10px;letter-spacing:2px;text-transform:uppercase;color:{t["muted"]};margin-bottom:16px">System architecture — enhanced 6-agent pipeline</div>', unsafe_allow_html=True)

    # Static diagram
    if "last_result" in st.session_state:
        all_done = {a["id"]: "done" for a in AGENT_INFO}
        if not st.session_state["last_result"].get("devils_advocate_triggered"):
            all_done[6] = "idle"
        pipeline_diagram(all_done)
    else:
        pipeline_diagram()

    st.write("")

    # Agent role cards
    divider("Agent roles and responsibilities")
    for a in AGENT_INFO:
        custom = " (custom addition)" if a["id"] in (5, 6, 7) else ""
        agent_card(a["icon"], a["name"], a["role"] + custom, a["color"], a["desc"])

# ── Tab 4: Batch ──
with tab_batch:
    t = T()
    st.markdown(f'<div style="font-family:\'Courier Prime\',monospace;font-size:10px;letter-spacing:2px;text-transform:uppercase;color:{t["muted"]};margin-bottom:8px">Batch article intake</div>', unsafe_allow_html=True)
    st.caption("For crash-safe incremental saves, use `batch_runner.py` from the terminal.")

    up_col, cfg_col = st.columns([1.2, 0.8], gap="large")
    with up_col:
        uploaded_file = st.file_uploader("Upload CSV (needs 'text' column)", type=["csv"])
    with cfg_col:
        max_articles = st.slider("Max articles", 1, 200, 20)

    run_batch = st.button("Run Batch Investigation", use_container_width=True, disabled=uploaded_file is None)

    if uploaded_file and run_batch:
        df = pd.read_csv(uploaded_file)
        if "text" not in df.columns:
            st.error("CSV must contain a `text` column.")
        else:
            df = df.head(max_articles).copy()
            records = []
            prog = st.progress(0.0)
            stat = st.empty()

            for idx, (_, row) in enumerate(df.iterrows(), start=1):
                stat.caption(f"[{idx}/{len(df)}] {str(row.get('title','Untitled'))[:90]}")
                try:
                    out = analyze_article(str(row.get("text", "")), str(row.get("title", "")))
                except Exception as exc:
                    st.warning(f"Skipped article {idx}: {exc}")
                    prog.progress(idx / len(df))
                    continue

                raw = row.get("label", "")
                gt = {0: "FAKE", 1: "REAL"}.get(raw, str(raw).upper() if raw else "")
                records.append({
                    "title": row.get("title", ""), "ground_truth": gt,
                    "predicted": out["final_label"], "confidence": out["final_confidence"],
                    "bias_score": out["bias"].get("bias_score"),
                    "credibility_score": out["credibility"].get("credibility_score"),
                    "tone": out["bias"].get("tone"),
                    "da_triggered": out["devils_advocate_triggered"],
                    "time_s": out.get("total_processing_time", 0),
                    "summary": out["judge"].get("summary", ""),
                })
                prog.progress(idx / len(df))

            stat.success(f"Processed {len(records)} of {len(df)} articles.")

            if records:
                res_df = pd.DataFrame(records)
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Processed", len(res_df))
                m2.metric("FAKE", int((res_df.predicted == "FAKE").sum()))
                m3.metric("REAL", int((res_df.predicted == "REAL").sum()))
                m4.metric("DA triggered", int(res_df.da_triggered.sum()))

                valid = res_df[res_df.ground_truth.isin(["REAL", "FAKE"])]
                if len(valid) > 0:
                    from sklearn.metrics import accuracy_score
                    acc = accuracy_score(valid.ground_truth, valid.predicted)
                    st.markdown(f'<div style="background:{t["card"]};border:1px solid {t["border"]};border-radius:10px;padding:20px;margin:16px 0;text-align:center"><div style="font-family:\'Courier Prime\',monospace;font-size:10px;letter-spacing:3px;text-transform:uppercase;color:{t["muted"]}">Overall accuracy</div><div style="font-size:2.5rem;font-weight:600;color:{t["heading"]};margin:8px 0">{acc:.1%}</div></div>', unsafe_allow_html=True)

                st.dataframe(res_df, use_container_width=True)
                buf = io.StringIO()
                res_df.to_csv(buf, index=False)
                st.download_button("Download results.csv", data=buf.getvalue(), file_name="results.csv", mime="text/csv", use_container_width=True)
