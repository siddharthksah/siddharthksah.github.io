#!/usr/bin/env python3
"""Count AI-writing tells in a Markdown post. Usage: prose-tells.py FILE [FILE...]"""
import re, statistics, sys

PATTERNS = {
    "em/en dash": r"[—–]|\s--\s",
    "honest*": r"\bhonest(ly)?\b",
    "actual(ly)": r"\bactual(ly)?\b",
    "the whole": r"\bthe whole\b",
    "which is/means": r",\s+which (is|means|makes)\b",
    "X, not Y": r",\s+(not|never)\s+\w+",
    "not just / not merely": r"\b(not just|not merely|isn'?t just|doesn'?t just|isn'?t about)\b",
    "rather than/instead of": r"\b(rather than|instead of)\b",
    "earn": r"\bearn(s|ed)?\b",
    "deliberate/on purpose": r"\b(deliberate(ly)?|on purpose)\b",
    "semicolon": r";",
    "colon reveal": r"[a-z]:\s+[a-z]",
    "That is (caption)": r"(^|\n)That is\b",
    "the real": r"\bthe real\b",
    "load-bearing": r"load-bearing",
    "at its core": r"at its core",
    "boring": r"\bboring\b",
    "tuition/diploma/liturgy/religion/syllabus/ledger": r"\b(tuition|diploma|liturgy|religion|syllabus|ledger|confession)\b",
    "figurative (ladder/fence/tripwire/...)": r"\b(ladder|rungs?|fences?|tripwires?|war stor(y|ies)|battle|referee|negotiat\w*|invoice|sandwich|knife|poison|blast radius|keys on day one|changed everything|demands respect|worst possible|vanity metric|fashionable|off day|feel good about|the way \w+ (do|does)|settles every|eats? \w+|gone dark|goes dark)\b",
    "one-word sentence": r"(?:^|[.!?] )[A-Z][a-z]+\.(?: |$)",
    "X does A. Y does B. parallel": r"\b(\w+) (costs|does|is) [^.]{2,30}\. (\w+) (costs|does|is) [^.]{2,30}\.",
}

def body(text):
    parts = text.split("---", 2)
    t = parts[2] if len(parts) > 2 else text
    t = re.sub(r"```.*?```", "", t, flags=re.S)
    t = re.sub(r"!\[[^\]]*\]\([^)]*\)(\{[^}]*\})?", "", t)
    t = re.sub(r"<video[^>]*>.*?</video>", "", t, flags=re.S)
    t = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", t)
    t = re.sub(r"^\|.*$", "", t, flags=re.M)
    t = re.sub(r"^#+ .*$", "", t, flags=re.M)
    return t

for path in sys.argv[1:]:
    text = open(path).read()
    b = body(text)
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", b) if len(s.strip()) > 1]
    lens = [len(s.split()) for s in sents]
    words = sum(lens)
    print(f"== {path}")
    print(f"   words={words} sentences={len(sents)} avg={statistics.mean(lens):.1f} sd={statistics.pstdev(lens):.1f} max={max(lens)} over30={sum(1 for l in lens if l > 30)}")
    hits = []
    for name, pat in PATTERNS.items():
        n = len(re.findall(pat, b, flags=re.I | re.M))
        if n:
            hits.append(f"{name}={n}")
    print("   " + (", ".join(hits) if hits else "no keyword tells"))
