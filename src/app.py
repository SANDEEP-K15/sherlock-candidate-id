"""
Sherlock Candidate Identification — interactive demo.

Simulates a live meeting by revealing the transcript incrementally as you
move a slider (standing in for "time elapsed in the call"). Watch the
confidence bars update as more evidence arrives — this is the core thing
the challenge asks for: continuously updating confidence, not a one-shot
classification.

Run with: streamlit run src/app.py
"""

import json
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.fusion import score_meeting, pick_candidate

st.set_page_config(page_title="Sherlock — Candidate ID", page_icon="🕵️", layout="wide")

# ── Styling ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 2rem; max-width: 1100px; }

    .hero {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 2rem 2.2rem;
        border-radius: 16px;
        margin-bottom: 1.6rem;
        color: white;
    }
    .hero h1 { margin: 0 0 0.4rem 0; font-size: 1.9rem; }
    .hero p { margin: 0; opacity: 0.85; font-size: 0.95rem; line-height: 1.5; }
    .hero .badge {
        display: inline-block; background: rgba(255,255,255,0.15);
        padding: 0.15rem 0.7rem; border-radius: 20px; font-size: 0.75rem;
        margin-bottom: 0.8rem; letter-spacing: 0.03em;
    }

    .sidebar-card {
        background: #f6f5f1; border-radius: 12px; padding: 1rem 1.1rem;
        margin-bottom: 1rem; border: 1px solid #e5e3da;
    }
    .sidebar-card h4 { margin: 0 0 0.5rem 0; font-size: 0.85rem; color: #5f5e5a; text-transform: uppercase; letter-spacing: 0.04em; }
    .sidebar-card .row { font-size: 0.88rem; padding: 0.15rem 0; }

    .participant-chip {
        display: inline-flex; align-items: center; gap: 0.4rem;
        background: white; border: 1px solid #e5e3da; border-radius: 8px;
        padding: 0.35rem 0.6rem; margin: 0.2rem 0; font-size: 0.82rem; width: 100%;
    }
    .participant-chip .dot { width: 8px; height: 8px; border-radius: 50%; background: #b4b2a9; flex-shrink: 0; }
    .participant-chip .dot.on { background: #639922; }

    .conf-card {
        border-radius: 12px; padding: 1rem 1.2rem; margin-bottom: 0.7rem;
        border: 1px solid; transition: all 0.2s;
    }
    .conf-card.high { background: #eaf3de; border-color: #97c459; }
    .conf-card.mid { background: #faeeda; border-color: #ef9f27; }
    .conf-card.low { background: #f1efe8; border-color: #d3d1c7; }

    .conf-card .top-row { display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem; }
    .conf-card .name { font-weight: 600; font-size: 1rem; }
    .conf-card .pct { font-weight: 700; font-size: 1.3rem; }
    .conf-card.high .pct { color: #27500a; }
    .conf-card.mid .pct { color: #633806; }
    .conf-card.low .pct { color: #444441; }

    .verdict-tag {
        display: inline-block; font-size: 0.7rem; font-weight: 600; padding: 0.15rem 0.55rem;
        border-radius: 20px; text-transform: uppercase; letter-spacing: 0.03em;
    }
    .verdict-tag.high { background: #639922; color: white; }
    .verdict-tag.mid { background: #ef9f27; color: white; }
    .verdict-tag.low { background: #888780; color: white; }

    .bar-track { background: rgba(0,0,0,0.08); border-radius: 20px; height: 10px; overflow: hidden; }
    .bar-fill { height: 100%; border-radius: 20px; }
    .conf-card.high .bar-fill { background: #639922; }
    .conf-card.mid .bar-fill { background: #ef9f27; }
    .conf-card.low .bar-fill { background: #b4b2a9; }

    .decision-banner {
        border-radius: 14px; padding: 1.3rem 1.5rem; font-size: 1.05rem; font-weight: 500;
    }
    .decision-banner.success { background: #eaf3de; border: 1px solid #97c459; color: #27500a; }
    .decision-banner.pending { background: #faeeda; border: 1px solid #ef9f27; color: #633806; }
</style>
""", unsafe_allow_html=True)

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "mock_meeting.json"
meeting = json.loads(DATA_PATH.read_text())

# ── Hero header ──────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <span class="badge">🕵️ SHERLOCK — LIVE PROTOTYPE</span>
    <h1>Real-time candidate identification</h1>
    <p>Simulated meeting where the candidate joins as <b>"MacBook Pro"</b> — no name match possible —
    and no one ever explicitly labels a "candidate" field. The system has to converge on the right
    person purely from weak, individually-fallible signals.</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🕵️ Sherlock")
    st.caption("Candidate identification demo")
    st.divider()

    invite = meeting["calendar_invite"]
    st.markdown(f"""
    <div class="sidebar-card">
        <h4>📅 Calendar invite</h4>
        <div class="row">👤 <b>{invite['candidate_name']}</b></div>
        <div class="row">✉️ {invite['candidate_email']}</div>
        <div class="row">🎙️ Interviewers: {', '.join(invite['interviewers'])}</div>
    </div>
    """, unsafe_allow_html=True)

    chips = ""
    for p in meeting["participants"]:
        dot_class = "on" if p.get("webcam_on") else ""
        chips += f"""<div class="participant-chip"><span class="dot {dot_class}"></span>
        <b>{p['display_name']}</b>&nbsp;· joined {p['join_time'][11:16]}</div>"""
    st.markdown(f"""
    <div class="sidebar-card">
        <h4>👥 Participants (raw)</h4>
        {chips}
    </div>
    """, unsafe_allow_html=True)

    st.caption("🟢 = webcam on · ⚪ = webcam off")

# ── Timeline slider ──────────────────────────────────────────────────────
max_turns = len(meeting["transcript"])
turn = st.slider(
    "⏱️ Meeting progress — drag to simulate time passing",
    min_value=0, max_value=max_turns, value=0,
)
transcript_so_far = meeting["transcript"][:turn]

col_left, col_right = st.columns([1, 1], gap="large")

# ── Transcript (chat-style) ───────────────────────────────────────────────
with col_left:
    st.markdown("#### 💬 Transcript so far")
    if not transcript_so_far:
        st.info("No one has spoken yet — identification currently relies only on join metadata.", icon="🤫")
    else:
        for t in transcript_so_far:
            speaker = next(p["display_name"] for p in meeting["participants"] if p["id"] == t["speaker_id"])
            is_interviewer = speaker in invite["interviewers"]
            avatar = "🎙️" if is_interviewer else "🧑‍💻"
            with st.chat_message("assistant" if is_interviewer else "user", avatar=avatar):
                st.markdown(f"**{speaker}**")
                st.write(t["text"])

# ── Confidence cards ───────────────────────────────────────────────────────
with col_right:
    st.markdown("#### 📊 Confidence per participant")
    scores = score_meeting(meeting, transcript_so_far)

    for s in scores:
        tier = "high" if s.confidence >= 0.6 else ("mid" if s.confidence >= 0.3 else "low")
        verdict_label = {"likely_candidate": "Candidate", "likely_interviewer_or_other": "Not candidate", "uncertain": "Uncertain"}[s.verdict]

        st.markdown(f"""
        <div class="conf-card {tier}">
            <div class="top-row">
                <span class="name">{s.display_name}</span>
                <span class="pct">{s.confidence:.0%}</span>
            </div>
            <div class="bar-track"><div class="bar-fill" style="width:{s.confidence*100:.0f}%"></div></div>
            <div style="margin-top:0.5rem;"><span class="verdict-tag {tier}">{verdict_label}</span></div>
        </div>
        """, unsafe_allow_html=True)

        with st.expander(f"Why? — {s.display_name}"):
            st.text(s.explanation())

# ── Decision banner ─────────────────────────────────────────────────────
st.markdown("#### 🎯 System decision")
decision = pick_candidate(scores)
if decision:
    st.markdown(f"""
    <div class="decision-banner success">
        ✅ Identified candidate: <b>{decision.display_name}</b> — confidence {decision.confidence:.0%}
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="decision-banner pending">
        ⏳ Not confident enough yet to commit to a candidate — waiting for more signal rather than guessing.
    </div>
    """, unsafe_allow_html=True)
