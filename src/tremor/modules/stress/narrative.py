"""The risk memo's executive summary: prose by a local language model, every figure by the engine.

A chief risk officer reads a memo's first paragraph first. A language model writes that paragraph better
than a template, but it also invents and miscalculates figures: asked to copy "$264m" exactly, llama3.1
wrote "$267 million", and it called a CET1 ratio of 10.66% "below" a 9.5% requirement. A risk memo cannot
carry either. So the model never writes a number:

1. **Placeholders, not figures.** The model sees the facts in words - which way the scenario moves markets,
   where the losses come from, which borrowers are hit hardest - with placeholders where the figures go
   ("the book would lose [LOSS]"), and must use them; the engine fills them in with the exact values. The
   model never sees a digit, so a digit in its draft can only be one it made up. It does not see the news
   stories either: given them, it wrote that the bank itself "harbors dirty money" (the story was about
   another bank) - the memo prints the stories right below the summary instead.
2. **Checks, and one or two repairs.** A draft is rejected if it writes a digit of its own, uses a
   placeholder it was not given, leaves out the loss or the capital ratio, contradicts the capital position
   (says "below the requirement" when the ratio stays above it, or the reverse), says the CET1 ratio rises
   when it falls, tells the scenario as something that has happened ("has incurred", "has been hit"),
   talks about its own draft, or stops mid-sentence. A rejected draft goes back to the model with the
   reasons, at most twice.
3. **Deterministic fallback.** No server, too slow, or every draft rejected: the memo uses a template
   summary built from the same facts. Nothing on the default run path depends on a language model.

**Switched off by default** (``llm.enabled`` in configs/settings.yaml). Measured on the 33 stress tests of
the two crisis replays (``python scripts/check_llm_summaries.py``, docs/results/llm_summaries.json): 22
drafts pass the checks, none with a wrong figure or capital statement; read by hand, about half tie the
largest loss channel to the hardest-hit borrowers more tightly than the numbers do. Earlier prompts did
worse (a made-up headline with made-up sources), and small prompt changes moved acceptance between 0 and
31 of 33. A risk memo cannot carry prose that needs a second read, so the memo uses the deterministic
summary unless an analyst switches the model on.

The model runs locally (Ollama, llama3.1 by default): no API key, and no portfolio data leaves the machine.
"""

from __future__ import annotations

import queue
import re
import threading
import time
from collections.abc import Callable
from datetime import datetime

import requests

from tremor.config import LlmSettings
from tremor.modules.stress.engine import StressResult
from tremor.modules.stress.memo import names_to_review

_PLACEHOLDER = re.compile(r"\[([A-Z][A-Z0-9_]*)\]")
REQUIRED = ("LOSS", "CET1_AFTER")  # a summary that leaves these out is not a summary of a stress test
REPAIRS = 2  # a rejected draft goes back to the model with the reasons at most this many times
# Digits that are not figures: names ("CET1", "Common Equity Tier 1", "IFRS 9", "S&P 500") and the impact
# scale itself ("[IMPACT] out of 10"); "8 out of 10" still fails on the 8.
_TERMS = re.compile(r"\bCET\s?1\b|\bTier\s?[12]\b|\bIFRS\s?9\b|\bS&P\s?500\b|\bG7\b|\bG20\b"
                    r"|\bout of (?:10|ten)\b|\b(?:1|one)[\s-]*(?:-|to)[\s-]*(?:10|ten)\b", re.IGNORECASE)
# Figures spelled out are figures too ("the average of five earlier downturns" was made up).
_NUMBER_WORDS = re.compile(r"\b(?:two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|dozens?|twenty|thirty|forty|fifty|"
                           r"sixty|seventy|eighty|ninety|hundreds?|thousands?|millions?|billions?)\b", re.IGNORECASE)
_PROMPT = """You are drafting the opening paragraph of a risk memo for a bank's chief risk officer.
Reply with the paragraph only: 3 or 4 plain sentences, under 110 words, with no introduction (never start with
"Here is"), no headings, no bullet points and no notes.
Use ONLY the facts below. The facts contain placeholders in square brackets such as [LOSS]: whenever you
refer to one of those values, write the placeholder exactly as it appears, brackets included. Never write
a digit or a number yourself, and do not invent anything that is not in the facts.
The event is news about other companies or countries, not about the bank: never write that the bank took
part in it or has already been hit by it. The stress test is hypothetical: write that the book WOULD lose
money under the scenario, never that a loss has happened. Say whether the CET1 ratio stays above the
requirement or falls below it, exactly as the facts say.
Cover: the event and its impact; what the scenario assumes; the loss, where it comes from and what it does to
the CET1 ratio against the requirement; the watchlist names to review first.

FACTS
{facts}
{feedback}
Executive summary:"""
_FEEDBACK = """
Your previous draft was rejected because {problems}. The rejected draft was:
{draft}
Reply with the corrected paragraph only.
"""
# Which way the scenario moves each market, in words: (name, factors averaged, word if down, word if up, smallest
# move worth saying, in the factor's units).
_MOVES = (("equities", ("EQ_US", "EQ_EU", "EQ_EM"), "fall", "rise", 1.0),
          ("credit spreads", ("CS_IG", "CS_HY", "CS_EM"), "tighten", "widen", 5.0),
          ("government bond yields", ("IR_USD_10Y",), "fall", "rise", 5.0),
          ("oil", ("CMD_OIL",), "falls", "rises", 2.0),
          ("gold", ("CMD_GOLD",), "falls", "rises", 1.0),
          ("market volatility", ("VOL_VIX",), "falls", "rises", 1.0))
_LENDING = ("loan", "revolver")  # borrowers; a bond basket is not one
# The claims a CRO acts on: is capital above or below the requirement, and is the loss told as a scenario?
_IN_SENTENCE = r"(?:[^.]|\.(?=\d)){0,40}"  # up to 40 characters without leaving the sentence ("9.5%" is not a full stop)
_SAYS_BREACH = re.compile(rf"\b(below|under(?!\s+(?:the|this|that)\s+(?:scenario|stress|test))|short of|beneath)\b"
                          rf"{_IN_SENTENCE}\b(requirement|required|minimum)\b|\bbreach", re.IGNORECASE)  # not "under the scenario"
_SAYS_ABOVE = re.compile(rf"\b(above|over|clear of|within)\b{_IN_SENTENCE}\b(requirement|required|minimum)\b|\bheadroom\b",
                         re.IGNORECASE)
_SAYS_ABOVE_ITSELF = re.compile(r"\b(above|over|beyond)\s+(?:the\s+|its\s+)?\[CET1_(?:BEFORE|AFTER)\]", re.IGNORECASE)
_SAYS_CET1_UP = re.compile(rf"\bCET\s?1\b{_IN_SENTENCE}\b(increas\w*|ris(?:e|es|ing)|rose|improv\w*|grow\w*|climb\w*)",
                           re.IGNORECASE)
_SAYS_REALISED = re.compile(r"\b(has|have|had)\s+(incurred|suffered|lost|recorded|booked)\b|\bincurred\b", re.IGNORECASE)
_SAYS_ALREADY_HIT = re.compile(r"\b(has|have|had)\s+been\s+(impacted|affected|hit|harmed|exposed|downgraded|defaulted)\b",
                               re.IGNORECASE)
# The model is never shown what the reports say, only how many there are: a claim put in their mouth is made up
# ("according to reports from 25 publishers, this event could lead to significant market volatility").
_SAYS_REPORTS_CLAIM = re.compile(r"\baccording to\b|\breports?\s+(?:\w+\s+){0,4}?(?:say|says|said|suggest|suggests|indicate|"
                                 r"indicates|show|shows|warn|warns|claim|claims|point to|points to)\b", re.IGNORECASE)
_SAYS_META = re.compile(r"\bhere(?:'s| is| are)\b|\b(?:executive summary|opening paragraph|draft|revised|rewritten|placeholder)s?\b",
                        re.IGNORECASE)
_PREAMBLE = re.compile(r"^(?:here(?:'s| is| are)|below is|sure\b)[^:]{0,100}:\s*", re.IGNORECASE)  # "Here is the revised paragraph:"
_NOTE = re.compile(r"\s*\(?\b(?:note|notes|changes made)\s*:.*$", re.IGNORECASE | re.DOTALL)  # "Note: I corrected ..."
_ENDS = re.compile(r"[.!?][\"'”’)\]]*$")
_SLOT = re.compile(r"(?:\b([Aa]n?) )?(\$?)\[([A-Z][A-Z0-9_]*)\](%?)")  # a placeholder, with what the model typed around it
# "an 8.8", "an operational", "a 7.9" - and "a European", "a Ukraine", "a U.S.", "a one-off", "a unit"
_VOWEL_SOUND = re.compile(r"(?!(?:[Ee]u|[Uu](?:k|ni|s[eu]|\.)|[Oo]ne))[aeiouAEIOU]|8|1[18]\b")
_AVAILABLE: dict[str, tuple[float, bool]] = {}  # server url -> (checked at, reachable): probe at most once a minute


def _money(value: float) -> str:
    a = abs(value)
    return f"${a / 1e9:.2f}bn" if a >= 1e9 else f"${a / 1e6:,.0f}m"


def slots(run: StressResult, watch: list[dict]) -> dict[str, str]:
    """Placeholder -> the exact text the engine puts there."""
    t, tot, cap, credit, sc = run.trigger or {}, run.totals, run.capital, run.credit, run.scenario
    snap = t.get("snapshot") or {}
    gap = f"{abs(cap['cet1_ratio_after'] - cap['requirement_pct']) * 100:.0f} bp"
    label = (t.get("event_type_label") or "").lower()
    review = names_to_review(run, watch)[:4]
    values = {
        "HEADLINE": f"\"{t.get('headline', sc.title)}\"",
        "EVENT_TYPE": "analyst-defined scenario" if not label else label if label.endswith("event") else f"{label} event",
        "IMPACT": f"{t.get('impact_score', 0):.1f}", "DETECTED": (datetime.fromisoformat(t["detected_at"]).strftime("%d %b %Y %H:%M UTC")
                                                               if t.get("detected_at") else "on demand"),
        "REPORTS": f"{snap.get('n_docs', 0):,}", "PUBLISHERS": f"{snap.get('n_publishers', 0):,}",
        "ANALOGS": ", ".join(a["title"] for a in sc.analogs) or "the expert template for the event type",
        "PRIOR_CRISES": f"{sc.prior_episodes}", "VALUE_BEFORE": _money(tot["value_before"]), "VALUE_AFTER": _money(tot["value_after"]),
        "LOSS": _money(abs(tot["pnl"])), "LOSS_PCT": f"{abs(tot['pnl_pct']):.2f}%",
        "CET1_BEFORE": f"{cap['cet1_ratio_before']:.2f}%", "CET1_AFTER": f"{cap['cet1_ratio_after']:.2f}%",
        "REQUIREMENT": f"{cap['requirement_pct']:.1f}%", "HEADROOM": gap, "SHORTFALL": gap,
        "ECL_BEFORE": _money(credit["ecl_before"]), "ECL_AFTER": _money(credit["ecl_after"]),
        "DOWNGRADES": f"{credit['downgraded_obligors']}", "DEFAULTS": f"{credit['defaults']}",
        "NAMES": ", ".join(f"{w['name']} ({_money(w['exposure_musd'] * 1e6)})" for w in review) or "none flagged",
    }
    return values


def _words(text: str) -> bool:
    return not re.search(r"\d", text)


def scenario_moves(shocks: dict[str, float]) -> str:
    """'equities fall, credit spreads widen, oil rises': the scenario's direction, without its figures."""
    said = []
    for name, ids, down, up, floor in _MOVES:
        values = [shocks[f] for f in ids if f in shocks]
        mean = sum(values) / len(values) if values else 0.0
        if abs(mean) >= floor:
            said.append(f"{name} {up if mean > 0 else down}")
    return ", ".join(said)


def slot_facts(run: StressResult) -> list[str]:
    """The fact sheet as the model sees it: words, with placeholders where the figures go - never a digit."""
    breach, gain = run.capital["breach"], run.totals["pnl"] > 0
    moves = scenario_moves(run.scenario.shocks)
    worst = min(run.by_channel, key=lambda c: c["pnl"], default={"name": "", "pnl": 0.0})
    via = worst["name"] if worst["name"][:2].isupper() else worst["name"][:1].lower() + worst["name"][1:]  # "FX" stays "FX"
    hit = list(dict.fromkeys(x["obligor"] for x in run.top_losses if x["asset_class"] in _LENDING and _words(x["obligor"])))[:3]
    source = (f"; the largest losses come through {via}" if worst["pnl"] < 0 else "") + (
        f", the hardest-hit borrowers being {', '.join(hit)}" if hit else "")
    return ["Event (news about others, not about the bank): [HEADLINE], a [EVENT_TYPE] with impact [IMPACT] on a one-to-ten "
            "scale; the stress test ran on [DETECTED]",
            "Evidence: [REPORTS] reports from [PUBLISHERS] publishers",
            "Scenario: market moves of the closest historical crises ([ANALOGS]), blended with the average of [PRIOR_CRISES] "
            "earlier crises and scaled up for the event's impact" + (f"; in it, {moves}" if moves else ""),
            f"Result: under the scenario the book would {'gain' if gain else 'lose'} [LOSS] ([LOSS_PCT] of its value), "
            "from [VALUE_BEFORE] to [VALUE_AFTER]" + source,
            "Capital: the CET1 ratio would go from [CET1_BEFORE] to [CET1_AFTER], which "
            + ("falls BELOW the [REQUIREMENT] requirement, a shortfall of [SHORTFALL]" if breach
               else "stays ABOVE the [REQUIREMENT] requirement, with headroom of [HEADROOM]"),
            "Credit: expected credit losses from [ECL_BEFORE] to [ECL_AFTER]; [DOWNGRADES] borrowers downgraded, [DEFAULTS] in default",
            "Watchlist names to review first (early-warning Watch Negative, with the book's exposure): [NAMES]"]


def fill(text: str, values: dict[str, str]) -> str:
    """Put the engine's values in, and tidy what the model typed around a placeholder: no "$$" from "$[LOSS]",
    no "%%" from "[LOSS_PCT]%", "a"/"an" to match the value, a capital letter when a value starts a sentence."""
    def put(m: re.Match) -> str:
        article, dollar, name, pct = m.groups()
        value = values.get(name)
        if value is None:
            return m.group(0)
        before = m.string[:m.start()].rstrip()
        if article:
            article = article[0] + ("n " if _VOWEL_SOUND.match(value.lstrip('"')) else " ")
        elif not before or before[-1] in ".!?":
            value = value[:1].upper() + value[1:]
        return (article or "") + ("" if value.startswith("$") else dollar) + value + ("" if value.endswith("%") else pct)

    return re.sub(r"[\"“”]{2,}", '"', _SLOT.sub(put, text))


def fact_sheet(run: StressResult, watch: list[dict]) -> list[str]:
    """The fact sheet with the figures in: what the template summary says, and what the model may say."""
    values = slots(run, watch)
    return [fill(line, values) for line in slot_facts(run)]


def tidy(draft: str) -> str:
    """Drop what a chat model wraps around the paragraph: "Here is the revised paragraph:", "Note: I changed ..."."""
    draft = _NOTE.sub("", _PREAMBLE.sub("", " ".join(draft.split()))).strip()
    return draft[1:-1].strip() if len(draft) > 1 and draft[0] == draft[-1] == '"' else draft


def check_draft(draft: str, breach: bool, allowed: frozenset[str] | None = None, cet1_falls: bool = True) -> list[str]:
    """Why a placeholder draft cannot be shown (empty = accepted). ``allowed``: the placeholders the model was given."""
    problems = []
    unknown = sorted({p for p in _PLACEHOLDER.findall(draft) if p not in (allowed or _SLOT_NAMES)})
    if unknown:
        problems.append("it used placeholders that are not in the facts: " + ", ".join(unknown))
    plain = _TERMS.sub(" ", _PLACEHOLDER.sub(" ", draft))
    numbers = re.findall(r"\d[\d,.]*", plain) + _NUMBER_WORDS.findall(plain)
    if numbers:
        problems.append("it wrote numbers of its own (" + ", ".join(numbers[:3]) + ")")
    missing = [p for p in REQUIRED if f"[{p}]" not in draft]
    if missing:
        problems.append("it left out " + " and ".join(missing))
    if not breach and _SAYS_BREACH.search(draft):
        problems.append("it says capital falls below the requirement, but the CET1 ratio stays above it")
    if breach and _SAYS_ABOVE.search(draft):
        problems.append("it says capital stays above the requirement, but the CET1 ratio falls below it")
    if cet1_falls and _SAYS_CET1_UP.search(draft):
        problems.append("it says the CET1 ratio rises, but it falls")
    if _SAYS_ABOVE_ITSELF.search(draft):  # "would remain above 13.50% to 10.96%" (seen in a draft)
        problems.append("it compares the CET1 ratio with its own level; compare it with the requirement")
    if _SAYS_REALISED.search(draft):
        problems.append("it presents the scenario's loss as one that has happened")
    if _SAYS_ALREADY_HIT.search(draft):
        problems.append("it says the bank or its borrowers have already been hit; the stress test only says what would happen")
    if _SAYS_REPORTS_CLAIM.search(draft):
        problems.append("it says what the reports claim, but the facts only count the reports")
    if _SAYS_META.search(draft):
        problems.append("it talks about the draft instead of the risk")
    if draft and not _ENDS.search(draft):
        problems.append("it stops mid-sentence")
    return problems


_SLOT_NAMES = frozenset(("HEADLINE", "EVENT_TYPE", "IMPACT", "DETECTED", "REPORTS", "PUBLISHERS", "ANALOGS", "PRIOR_CRISES",
                         "VALUE_BEFORE", "VALUE_AFTER", "LOSS", "LOSS_PCT", "CET1_BEFORE", "CET1_AFTER", "REQUIREMENT",
                         "HEADROOM", "SHORTFALL", "ECL_BEFORE", "ECL_AFTER", "DOWNGRADES", "DEFAULTS", "NAMES"))


def template_summary(run: StressResult, watch: list[dict]) -> str:
    """The deterministic summary, from the same facts."""
    v, breach = slots(run, watch), run.capital["breach"]
    verb = "gain" if run.totals["pnl"] > 0 else "lose"
    text = (f"{'An' if _VOWEL_SOUND.match(v['EVENT_TYPE']) else 'A'} {v['EVENT_TYPE']}, {v['HEADLINE']}, reached impact {v['IMPACT']}. "
            f"The scenario blends the closest historical crises ({v['ANALOGS']}) with the average of {v['PRIOR_CRISES']} earlier ones. "
            f"Under it the book would {verb} {v['LOSS']} ({v['LOSS_PCT']} of its value) and the CET1 ratio would go from "
            f"{v['CET1_BEFORE']} to {v['CET1_AFTER']}, {'below' if breach else 'above'} the {v['REQUIREMENT']} requirement.")
    return text + (f" Review first: {v['NAMES']}." if v["NAMES"] != "none flagged" else "")


def _ollama(settings: LlmSettings) -> Callable[[str], str] | None:
    """A generate(prompt) function for the local server, or None when it does not answer."""
    checked, reachable = _AVAILABLE.get(settings.url, (0.0, False))
    if time.monotonic() - checked > 60:
        try:
            tags = requests.get(f"{settings.url}/api/tags", timeout=1.5).json()
            reachable = any(m.get("name", "").split(":")[0] == settings.model.split(":")[0] for m in tags.get("models", []))
        except (requests.RequestException, ValueError):
            reachable = False
        _AVAILABLE[settings.url] = (time.monotonic(), reachable)
    if not reachable:
        return None

    def generate(prompt: str) -> str:
        resp = requests.post(f"{settings.url}/api/generate", timeout=settings.timeout_seconds, json={
            "model": settings.model, "prompt": prompt, "stream": False,
            "options": {"temperature": 0, "seed": 7, "num_predict": 320}})
        resp.raise_for_status()
        return resp.json().get("response", "").strip()

    return generate


def executive_summary(run: StressResult, watch: list[dict], settings: LlmSettings | None = None,
                      generate: Callable[[str], str] | None = None) -> dict:
    """{"text", "source": "llm" | "template", "model", "note"} for the memo's opening paragraph."""
    fallback = {"text": template_summary(run, watch), "source": "template", "model": None}
    if generate is None and settings is not None and settings.enabled:
        generate = _ollama(settings)
    if generate is None:
        return {**fallback, "note": "template summary (no local language model running)"}
    model = settings.model if settings is not None else "language model"
    facts, feedback = "\n".join(f"- {line}" for line in slot_facts(run)), ""
    given = frozenset(_PLACEHOLDER.findall(facts))
    cet1_falls = run.capital["cet1_ratio_after"] < run.capital["cet1_ratio_before"]
    for attempt in range(1, REPAIRS + 2):
        try:
            draft = tidy(generate(_PROMPT.format(facts=facts, feedback=feedback)))
        except (requests.RequestException, ValueError) as exc:
            return {**fallback, "note": f"template summary (the language model failed: {type(exc).__name__})"}
        problems = check_draft(draft, bool(run.capital["breach"]), given, cet1_falls) if draft else ["it was empty"]
        if not problems:
            tries = f", accepted on attempt {attempt} after the checks sent it back" if attempt > 1 else ""
            return {"text": fill(draft, slots(run, watch)), "draft": draft, "source": "llm", "model": model, "attempts": attempt,
                    "note": f"prose drafted by a local language model ({model}{tries}); every figure inserted by the engine, "
                            "and the capital statement checked against the run"}
        feedback = _FEEDBACK.format(problems="; ".join(problems), draft=draft)
    return {**fallback, "rejected_draft": draft, "attempts": REPAIRS + 1,
            "note": f"template summary (all {REPAIRS + 1} of the language model's drafts were rejected; the last because "
                    + "; ".join(problems) + ")"}


class SummaryDrafts:
    """Drafts executive summaries in one background thread, so a memo never waits for the model.

    The first request for a run's memo queues a draft and the memo shows the template summary with a note;
    once the draft is ready (and has passed the checks) the memo shows it. The draft is kept with its
    placeholders and filled at every request, so the names it lists are the ones on the watchlist printed
    below it, not the ones of the moment it was drafted. A daemon thread: a slow model can never hold up
    shutting the server down."""

    def __init__(self, settings: LlmSettings):
        self._settings = settings
        self._results: dict[str, dict] = {}
        self._queued: set[str] = set()
        self._jobs: queue.Queue = queue.Queue()
        self._worker: threading.Thread | None = None

    def get(self, run: StressResult, watch: list[dict]) -> dict:
        done = self._results.get(run.run_id)
        if done is not None:
            text = fill(done["draft"], slots(run, watch)) if done.get("draft") else template_summary(run, watch)
            return {**done, "text": text}
        template = executive_summary(run, watch)  # no generator: the deterministic summary
        if not self._settings.enabled:  # the default: the memo is deterministic and says nothing about a model
            return {k: v for k, v in template.items() if k != "note"}
        if _ollama(self._settings) is None:
            return template
        if run.run_id not in self._queued:
            self._queued.add(run.run_id)
            self._jobs.put((run, list(watch)))
            if self._worker is None or not self._worker.is_alive():
                self._worker = threading.Thread(target=self._work, name="memo-summaries", daemon=True)
                self._worker.start()
        return {**template, "note": f"template summary for now: the local language model ({self._settings.model}) is "
                                    "drafting one - refresh in about half a minute"}

    def _work(self) -> None:
        while True:
            run, watch = self._jobs.get()
            try:
                self._results[run.run_id] = executive_summary(run, watch, self._settings)
            except Exception as exc:  # whatever goes wrong, the memo must not say "drafting" for good
                self._results[run.run_id] = {**executive_summary(run, watch),
                                             "note": f"template summary (the language model failed: {type(exc).__name__})"}
