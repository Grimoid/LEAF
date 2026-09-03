from __future__ import annotations

import re
from typing import List

_WS_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    text = text.strip()
    return _WS_RE.sub(" ", text)


def whitespace_words(text: str) -> List[str]:
    text = normalize_text(text)
    return text.split(" ") if text else []
