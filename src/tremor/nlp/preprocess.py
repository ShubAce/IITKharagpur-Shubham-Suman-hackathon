"""Text normalisation shared by every source."""

from __future__ import annotations

import hashlib
import html
import re

_URL = re.compile(r"https?://\S+|www\.\S+")
_RETWEET = re.compile(r"^\s*RT @\w+:\s*", re.IGNORECASE)
_WHITESPACE = re.compile(r"\s+")
_NON_ALNUM = re.compile(r"[^a-z0-9$]+")
# Publisher suffix appended by news aggregators: "Headline text - Reuters"
_PUBLISHER_SUFFIX = re.compile(r"\s+[-|–—]\s+[A-Z][\w&.' ]{1,30}$")
# Words that cannot end a complete headline: a title ending in one was cut off upstream (GDELT stores
# some page titles truncated: "Russian troops try to seize Chernobyl nuclear plant amid").
_DANGLING = frozenset(
    "a an the of to amid with by from into for and or but as at its their his her than".split()
)
# Some pages report a file name as their title ("Charles Schwab.pdf"): a document, not a headline.
_FILE_NAME = re.compile(r"^[\w\s().,&'-]{1,80}\.(pdf|docx?|xlsx?|pptx?|csv|txt)$", re.IGNORECASE)


def is_file_name(text: str) -> bool:
    return bool(_FILE_NAME.match(text.strip()))


def clean_text(text: str, strip_publisher_suffix: bool = False) -> str:
    """Unescape HTML, drop URLs and the retweet prefix, collapse whitespace."""
    text = html.unescape(text or "")
    text = _URL.sub(" ", text)
    text = _RETWEET.sub("", text)
    text = _WHITESPACE.sub(" ", text).strip()
    if strip_publisher_suffix:
        text = _PUBLISHER_SUFFIX.sub("", text)
    return text


def tidy_headline(text: str) -> str:
    """Display-only cleanup of web page titles: drop site names glued on with " | "
    ("NATO chief: peace shattered | iNFOnews | Thompson-Okanagan's News Source") and mark titles
    that were truncated upstream with an ellipsis. The engine itself always analyses the text exactly
    as received."""
    if " | " in text:
        parts = [p.strip() for p in text.split(" | ") if p.strip()]
        best = max(parts, key=len)
        if len(best) >= 25:
            text = best
    words = text.rstrip().split(" ")
    if len(words) > 3 and words[-1].lower() in _DANGLING:
        text = text.rstrip() + "…"
    return text


def fingerprint(text: str) -> str:
    """Hash of the text with case, punctuation and spacing removed.

    Retweets and syndicated copies of one article collapse to the same fingerprint, so they are
    counted as corroboration of an existing story instead of being scored as fresh information.
    """
    canonical = _NON_ALNUM.sub(" ", clean_text(text).lower()).strip()
    return hashlib.md5(canonical.encode("utf-8")).hexdigest()
