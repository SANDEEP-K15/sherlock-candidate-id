"""
Weak signal extractors for candidate identification.

Design principle: no single signal is trusted. Each signal returns a score
in [-1, 1] (negative = evidence AGAINST being the candidate, positive = evidence
FOR) plus a short human-readable reason string. The fusion engine combines
these into a single confidence per participant.

Every function is pure and stateless — given the same inputs, same output.
This makes them trivial to unit test and to re-run as new events arrive.
"""

from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from rapidfuzz import fuzz


@dataclass
class SignalResult:
    name: str
    score: float          # in [-1, 1]
    reason: str
    weight: float = 1.0    # relative importance of this signal


def name_match(display_name: str, candidate_name: str) -> SignalResult:
    """Fuzzy match participant display name against the calendar invite name.
    Weak by design — deliberately fails on device names like 'MacBook Pro',
    which is exactly why we don't rely on it alone.
    """
    if not display_name:
        return SignalResult("name_match", 0.0, "No display name available", weight=0.8)

    ratio = fuzz.token_sort_ratio(display_name.lower(), candidate_name.lower()) / 100.0
    if ratio > 0.8:
        return SignalResult("name_match", 0.9, f"Display name '{display_name}' closely matches invite name", weight=0.8)
    if ratio > 0.4:
        return SignalResult("name_match", 0.3, f"Display name '{display_name}' partially matches invite name", weight=0.8)
    return SignalResult("name_match", -0.1, f"Display name '{display_name}' does not resemble invite name (e.g. device name)", weight=0.8)


def email_match(participant_email: str | None, candidate_email: str | None) -> SignalResult:
    """Exact/domain match against the known candidate email, if visible at all."""
    if not participant_email or not candidate_email:
        return SignalResult("email_match", 0.0, "Participant email not visible", weight=1.2)

    if participant_email.lower() == candidate_email.lower():
        return SignalResult("email_match", 1.0, "Email exactly matches candidate email on file", weight=1.2)

    p_domain = participant_email.split("@")[-1].lower()
    c_domain = candidate_email.split("@")[-1].lower()
    if p_domain == c_domain:
        return SignalResult("email_match", 0.4, "Email domain matches candidate's domain", weight=1.2)

    return SignalResult("email_match", -0.6, "Email does not match candidate on file", weight=1.2)


def interviewer_exclusion(display_name: str, interviewer_names: list[str], participant_email: str | None = None,
                            interviewer_domain_hint: str | None = None) -> SignalResult:
    """Strong negative signal if this participant closely matches a KNOWN interviewer.
    This is often more reliable than positively identifying the candidate, since
    interviewer identities are known in advance from the calendar invite.
    """
    for interviewer in interviewer_names:
        ratio = fuzz.token_sort_ratio(display_name.lower(), interviewer.lower()) / 100.0
        if ratio > 0.75:
            return SignalResult("interviewer_exclusion", -1.0, f"Display name matches known interviewer '{interviewer}'", weight=1.5)

    if interviewer_domain_hint and participant_email and participant_email.lower().endswith(interviewer_domain_hint.lower()):
        return SignalResult("interviewer_exclusion", -0.7, "Company email domain suggests interviewer/staff, not candidate", weight=1.5)

    return SignalResult("interviewer_exclusion", 0.1, "Does not match any known interviewer", weight=1.5)


def join_timing(join_time: str, scheduled_start: str) -> SignalResult:
    """Weak timing signal. Interviewers often join a couple minutes early to set
    up; candidates often join right at or slightly after the scheduled time.
    Deliberately low weight since this varies a lot in practice.
    """
    j = datetime.fromisoformat(join_time)
    s = datetime.fromisoformat(scheduled_start)
    delta_minutes = (j - s).total_seconds() / 60.0

    if -1 <= delta_minutes <= 5:
        return SignalResult("join_timing", 0.3, "Joined at/slightly after scheduled start (typical candidate pattern)", weight=0.4)
    if delta_minutes < -1:
        return SignalResult("join_timing", -0.2, "Joined notably early (more typical of interviewer/host)", weight=0.4)
    return SignalResult("join_timing", 0.0, "Joined significantly late — inconclusive", weight=0.4)


def transcript_self_identification(participant_id: str, transcript: list[dict], candidate_name: str) -> SignalResult:
    """Strong signal: does the participant's own speech contain a self-introduction
    with the candidate's first name? This is what catches the 'MacBook Pro' case —
    the display name lies, but the person's own words don't.
    """
    first_name = candidate_name.split()[0].lower()
    own_lines = [t["text"] for t in transcript if t["speaker_id"] == participant_id]
    for line in own_lines:
        low = line.lower()
        if first_name in low and ("i'm" in low or "i am" in low or "my name" in low):
            return SignalResult("transcript_self_id", 1.0, f"Participant introduced themselves as '{candidate_name.split()[0]}' in transcript", weight=1.6)
    return SignalResult("transcript_self_id", 0.0, "No self-introduction detected yet", weight=1.6)


def transcript_addressed_as(participant_id: str, transcript: list[dict], candidate_name: str) -> SignalResult:
    """Do OTHER speakers address this participant by the candidate's first name?
    e.g. an interviewer saying 'Ananya, can you share your screen?' right before
    this participant speaks is strong circumstantial evidence.
    """
    first_name = candidate_name.split()[0].lower()
    for i, turn in enumerate(transcript):
        if turn["speaker_id"] == participant_id:
            continue
        if first_name in turn["text"].lower():
            # check if this participant speaks shortly after being addressed
            for later in transcript[i + 1:i + 3]:
                if later["speaker_id"] == participant_id:
                    return SignalResult("transcript_addressed_as", 0.8,
                                         f"Addressed as '{candidate_name.split()[0]}' by another speaker, then responded", weight=1.3)
    return SignalResult("transcript_addressed_as", 0.0, "Not yet addressed by candidate name in transcript", weight=1.3)


def screen_share_context(screen_share: bool, follows_technical_prompt: bool) -> SignalResult:
    """Weak signal: candidates in technical interviews are usually the ones asked
    to share their screen for coding rounds. Interviewers sometimes share too
    (to present a problem), so this alone is not decisive.
    """
    if screen_share and follows_technical_prompt:
        return SignalResult("screen_share", 0.3, "Shared screen shortly after being asked a technical question", weight=0.5)
    return SignalResult("screen_share", 0.0, "No decisive screen-share signal", weight=0.5)
