"""
Fusion engine: combines weak per-signal scores into one confidence per
participant, and produces a human-readable explanation.

Approach: weighted sum of signal scores, normalized through a sigmoid so the
final confidence is bounded in [0, 1] and doesn't blow up as more signals
get added. This is deliberately simple and auditable rather than a black-box
model — every score can be traced back to which signals fired and why,
which matters more than raw accuracy for a v1 (see README "Trade-offs").

As more transcript arrives (simulating a live meeting), call `score_participant`
again — signals are stateless and recomputed from scratch each time, so
confidence naturally rises as stronger evidence (like a self-introduction)
appears, and *falls* if new evidence contradicts an earlier guess. This also
sidesteps a whole class of bugs where stale incremental state gets out of sync.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field

from src import signals as sig


@dataclass
class ParticipantScore:
    participant_id: str
    display_name: str
    confidence: float
    signal_breakdown: list[sig.SignalResult] = field(default_factory=list)
    verdict: str = "unknown"  # "likely_candidate" | "likely_interviewer" | "uncertain"

    def explanation(self) -> str:
        lines = [f"Confidence this participant ({self.display_name}) is the candidate: {self.confidence:.0%}"]
        for s in sorted(self.signal_breakdown, key=lambda x: -abs(x.score * x.weight)):
            sign = "+" if s.score >= 0 else ""
            lines.append(f"  [{sign}{s.score:.1f} x{s.weight}] {s.name}: {s.reason}")
        return "\n".join(lines)


def _sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def score_participant(participant: dict, invite: dict, transcript: list[dict],
                       interviewer_domain_hint: str | None = None) -> ParticipantScore:
    results: list[sig.SignalResult] = []

    results.append(sig.name_match(participant["display_name"], invite["candidate_name"]))
    results.append(sig.email_match(participant.get("email"), invite.get("candidate_email")))
    results.append(sig.interviewer_exclusion(
        participant["display_name"], invite["interviewers"],
        participant.get("email"), interviewer_domain_hint,
    ))
    results.append(sig.join_timing(participant["join_time"], invite["scheduled_start"]))
    results.append(sig.transcript_self_identification(participant["id"], transcript, invite["candidate_name"]))
    results.append(sig.transcript_addressed_as(participant["id"], transcript, invite["candidate_name"]))

    follows_technical_prompt = any(
        t["speaker_id"] != participant["id"] and "share your screen" in t["text"].lower()
        for t in transcript
    )
    results.append(sig.screen_share_context(participant.get("screen_share", False), follows_technical_prompt))

    weighted_sum = sum(r.score * r.weight for r in results)
    confidence = _sigmoid(weighted_sum)

    if confidence >= 0.7:
        verdict = "likely_candidate"
    elif confidence <= 0.3:
        verdict = "likely_interviewer_or_other"
    else:
        verdict = "uncertain"

    return ParticipantScore(
        participant_id=participant["id"],
        display_name=participant["display_name"],
        confidence=confidence,
        signal_breakdown=results,
        verdict=verdict,
    )


def score_meeting(meeting: dict, transcript_so_far: list[dict] | None = None) -> list[ParticipantScore]:
    """Score every participant given the meeting data and however much
    transcript has arrived so far (this is what makes it 'continuously
    updating' in a live setting — call this on every new transcript chunk).
    """
    invite = meeting["calendar_invite"]
    transcript = transcript_so_far if transcript_so_far is not None else meeting["transcript"]

    interviewer_domains = set()
    for p in meeting["participants"]:
        if p.get("email") and any(
            sig.fuzz.token_sort_ratio(p["display_name"].lower(), name.lower()) > 75
            for name in invite["interviewers"]
        ):
            interviewer_domains.add(p["email"].split("@")[-1])
    domain_hint = next(iter(interviewer_domains), None)

    scores = [
        score_participant(p, invite, transcript, interviewer_domain_hint=domain_hint)
        for p in meeting["participants"]
    ]
    return sorted(scores, key=lambda s: -s.confidence)


def pick_candidate(scores: list[ParticipantScore]) -> ParticipantScore | None:
    """Return the top-confidence participant, but only if there's a clear
    winner — otherwise return None to signal 'ambiguous, need more data'.
    This is the 'gracefully handle uncertainty' requirement: better to say
    'not sure yet' than to lock onto the wrong person.
    """
    if not scores:
        return None
    if len(scores) == 1:
        return scores[0]
    top, second = scores[0], scores[1]
    if top.confidence >= 0.6 and (top.confidence - second.confidence) >= 0.2:
        return top
    return None
