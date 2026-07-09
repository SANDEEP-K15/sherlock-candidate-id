# Sherlock — Real-time interview candidate identification

A prototype that identifies which meeting participant is the interview
candidate, using multiple weak signals fused into a confidence score that
updates continuously as the meeting progresses — rather than one rule
("match the display name") that breaks the moment someone joins as
"MacBook Pro".

## Quick start

```bash
pip install -r requirements.txt
streamlit run src/app.py
```

Open the local URL Streamlit prints. Use the slider to simulate the meeting
progressing turn-by-turn and watch confidence update live.

To run the scoring logic headlessly (e.g. for tests / CI):

```bash
python3 -c "
import json
from src.fusion import score_meeting, pick_candidate
meeting = json.load(open('data/mock_meeting.json'))
print(score_meeting(meeting))
"
```

## Problem framing

The invite tells us the *name* of the candidate, not their *identity in the
meeting*. Display names lie (device names, nicknames), the interviewer can
misname them, and there can be multiple interviewers or silent observers.
So the system treats candidate identification as an **inference problem
under uncertainty**, not a lookup.

## Architecture

```
Meeting event stream (participants, transcript, calendar invite)
        │
        ▼
five independent weak-signal extractors  (src/signals.py)
  • name_match                — fuzzy match vs invite name
  • email_match                — participant email vs candidate email on file
  • interviewer_exclusion      — rules OUT known interviewers (often more
                                  reliable than a positive candidate match)
  • join_timing                — join time vs scheduled start
  • transcript_self_identification — did they say "I'm <name>"?
  • transcript_addressed_as    — did someone else call them by that name,
                                  and did they respond right after?
  • screen_share_context       — shared screen right after being asked a
                                  technical question
        │
        ▼
fusion engine (src/fusion.py)
  weighted sum of signal scores → sigmoid → confidence in [0,1]
  recomputed from scratch on every new transcript chunk, so confidence
  can rise OR fall as new evidence arrives
        │
        ▼
decision layer
  only commits to a candidate if confidence ≥ 0.6 AND there's a clear
  margin over the second-best participant. Otherwise reports "uncertain"
  rather than guessing.
        │
        ▼
Streamlit dashboard — confidence bars + expandable per-signal explanation
per participant, updating live as the (simulated) meeting proceeds
```

Each `SignalResult` carries a `reason` string, so any decision can be
explained by listing which signals fired, their scores, and their weights
— this is what satisfies the "explain why it selected a participant"
requirement, and it's inspectable in the sidebar "why?" expander per
participant in the demo.

## Why fusion, not a single classifier

A single ML classifier trained end-to-end would be a black box and would
need a large labeled dataset of real interviews we don't have. A weighted
rule-based fusion of interpretable signals:
- works with zero training data (bootstrap problem — a startup doesn't have
  thousands of labeled interviews on day one)
- is auditable — every score traces back to a specific reason
- degrades gracefully — if one signal is missing (no email visible), it
  just contributes 0 rather than crashing the pipeline
- is trivially extensible — new signals (e.g. a CV face-match against a
  resume photo, or a voice-print match) slot in as one more `SignalResult`
  with a weight, no retraining needed

The natural "what's next" is to learn signal *weights* from labeled outcome
data once enough real interviews have been logged (i.e. replace hand-tuned
weights with logistic regression over the same signal vector) — same
signals, same explainability, better-calibrated confidence. This is the
"continues learning as more interview data becomes available" bonus point.

## Assumptions

- Speaker-attributed transcript is available in near-real-time (per the
  challenge's "Available Information" list).
- Calendar invite reliably provides candidate name/email and interviewer
  names in advance — this is the one piece of ground truth we anchor to.
- A meeting has exactly one candidate. Multiple candidates in one call is
  out of scope for this prototype.
- Real Zoom/Teams/Meet integration (webhooks, media streams) is replaced
  here with a JSON fixture (`data/mock_meeting.json`) that has the same
  shape the platforms would actually provide, so swapping in a real
  connector is a data-source change, not an architecture change.
- No CV/voice models are wired in for this prototype (see Limitations) —
  the signal framework is designed so they'd plug in as additional
  `SignalResult` producers without touching the fusion logic.

## Evaluation

**How I tested it**: `data/mock_meeting.json` intentionally encodes the
exact edge cases named in the challenge doc:
- candidate joins as a device name ("MacBook Pro") → name_match alone fails,
  transcript self-identification recovers it once they speak
- interviewer joins early with a real name → interviewer_exclusion
  correctly suppresses them despite otherwise-plausible timing
- a silent observer with no meaningful signals → stays "uncertain"
  indefinitely rather than being wrongly promoted, since it never gets
  positive signal from any source
- interviewer explicitly addresses the candidate by first name before a
  technical question → transcript_addressed_as fires even before the
  candidate has explicitly self-introduced

I ran the fusion engine at multiple points in the simulated timeline
(0, 1, 2, 4, 7 transcript turns) and confirmed confidence starts flat/
ambiguous, sharpens correctly once real evidence appears, and never
misfires onto an interviewer.

**Accuracy**: on this single scripted scenario, the system converges on
the correct participant within two transcript turns and stays correct.
This is not a statistically meaningful accuracy number — see limitations.

**Limitations**:
- Only one synthetic scenario is tested; no real interview recordings.
  Real accuracy needs a labeled dataset of actual calls.
- No audio/video-based signals (voice consistency, face presence) —
  currently text/metadata only. These would meaningfully help the
  "candidate changes display name mid-call" and "impersonation" cases.
- Weights are hand-tuned, not learned. They're defensible but not optimal.
- `interviewer_exclusion`'s domain-hint logic assumes interviewer emails
  share a company domain — won't work if interviewers use personal email.
- The "uncertain until clear margin" decision rule trades off latency for
  safety — in a live product you'd want to tune the 0.6/0.2 thresholds
  against how costly a wrong guess is vs. a slow guess.

## What I'd improve next

1. Add a lightweight voice/face consistency signal (does this person's
   voice/face stay the same across the call — catches mid-call swaps).
2. Replace hand-tuned weights with logistic regression once real labeled
   interviews are available.
3. Stream processing instead of full-recompute (currently O(participants ×
   transcript length) on every update — fine for a 30-min call, not for a
   6-hour marathon interview loop).
4. A "confidence decayed" alert if a previously high-confidence match goes
   silent for a long time (possible mid-call handoff / impersonation).
5. Real Zoom/Meet/Teams webhook adapters feeding the same `meeting` dict
   shape used here, so the fusion/signal code doesn't change at all.

## Repo structure

```
data/mock_meeting.json   simulated meeting fixture with edge cases baked in
src/signals.py           individual weak-signal extractors
src/fusion.py            combines signals into confidence + decision
src/app.py                Streamlit live demo
requirements.txt
```
