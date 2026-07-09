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

st.set_page_config(page_title="Sherlock — Candidate ID", layout="wide")

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "mock_meeting.json"
meeting = json.loads(DATA_PATH.read_text())

st.title("Sherlock: live candidate identification")
st.caption(
    "Simulated meeting where the candidate joins as 'MacBook Pro' (no name match) "
    "and the interviewer never explicitly labels a 'candidate' field — the system "
    "has to figure it out from converging weak signals."
)

with st.sidebar:
    st.header("Calendar invite (known in advance)")
    st.json(meeting["calendar_invite"])
    st.header("Participants (raw)")
    for p in meeting["participants"]:
        st.write(f"**{p['display_name']}** — joined {p['join_time'][11:16]}")

max_turns = len(meeting["transcript"])
turn = st.slider(
    "Meeting progress (transcript turns revealed) — simulates real time",
    min_value=0, max_value=max_turns, value=0,
)

transcript_so_far = meeting["transcript"][:turn]

st.subheader("Transcript so far")
if not transcript_so_far:
    st.info("No one has spoken yet — identification currently relies only on join metadata.")
else:
    for t in transcript_so_far:
        speaker = next(p["display_name"] for p in meeting["participants"] if p["id"] == t["speaker_id"])
        st.write(f"**{speaker}**: {t['text']}")

st.subheader("Confidence per participant")
scores = score_meeting(meeting, transcript_so_far)

for s in scores:
    col1, col2 = st.columns([3, 1])
    with col1:
        st.progress(s.confidence, text=f"{s.display_name} — {s.confidence:.0%} ({s.verdict})")
    with col2:
        with st.expander("why?"):
            st.text(s.explanation())

decision = pick_candidate(scores)
st.subheader("System decision")
if decision:
    st.success(f"Identified candidate: **{decision.display_name}** (confidence {decision.confidence:.0%})")
else:
    st.warning("Not confident enough yet to commit to a candidate — waiting for more signal rather than guessing.")
