"""Answer-format rules applied around the legacy verifier.

The ground-truth convention: multiple-choice answers are sorted letters
separated by ASCII commas (``A,C``).
"""

from __future__ import annotations

import re
import unicodedata

_CHOICE_GT_RE = re.compile(r"[A-Z](?:,[A-Z])*")
_LATEX_WRAPPER_RE = re.compile(r"\\(?:text|textbf|mathrm|mathbf)\s*\{([^{}]*)\}")
# After NFKC, full-width commas, semicolons and parentheses become ASCII.
_CHOICE_PRED_RE = re.compile(r"[A-Z(),、;\s]+")


def choice_letters(gt: str) -> str | None:
    """Return the letters of a multiple-choice ground truth such as ``A,C``, else None."""
    if not _CHOICE_GT_RE.fullmatch(gt):
        return None
    letters = gt.split(",")
    if letters != sorted(set(letters)):
        return None
    return "".join(letters)


def parse_choice(pred: str) -> str | None:
    """Parse predicted choice labels into sorted letters, or None if not a pure label list.

    Accepts e.g. ``AC``, ``A、C``, ``(A)(C)``, ``\\text{A, C}`` and full-width letters.
    Rejects any other text (``A或C``) and repeated letters (``II``).
    """
    value = unicodedata.normalize("NFKC", pred).replace("$", "")
    while True:
        unwrapped = _LATEX_WRAPPER_RE.sub(r"\1", value)
        if unwrapped == value:
            break
        value = unwrapped
    if not _CHOICE_PRED_RE.fullmatch(value):
        return None
    letters = re.findall(r"[A-Z]", value)
    if not letters or len(set(letters)) != len(letters):
        return None
    return "".join(sorted(letters))


def split_top_level(text: str, separators: str) -> list[str] | None:
    """Split on separators outside brackets; None if brackets are unbalanced or a part is empty.

    Bracket kinds are counted together so half-open intervals such as ``[1,2)``
    stay one part. A separator escaped as ``\\,`` (LaTeX thin space) is not split.
    """
    parts, depth, start = [], 0, 0
    for i, char in enumerate(text):
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
            if depth < 0:
                return None
        elif char in separators and depth == 0 and not (i > 0 and text[i - 1] == "\\"):
            parts.append(text[start:i].strip())
            start = i + 1
    if depth != 0:
        return None
    parts.append(text[start:].strip())
    return parts if all(parts) else None


# A label is a simple identifier before the only "=": x, x_1, x_{1}, \alpha, f(x).
_LABEL_RE = re.compile(
    r"\s*((?:[A-Za-z]|\\[A-Za-z]+)(?:_(?:\{[A-Za-z0-9]+\}|[A-Za-z0-9]))?(?:\([A-Za-z]\))?)\s*=(.*)",
    re.S,
)


def split_label(field: str) -> tuple[str | None, str]:
    """Split ``x_{1}=3`` into a normalized label ``x_1`` and value ``3``; unlabeled fields give (None, field)."""
    if field.count("=") != 1:
        return None, field
    match = _LABEL_RE.fullmatch(field)
    if not match or not match.group(2).strip():
        return None, field
    return re.sub(r"[\s{}]", "", match.group(1)), match.group(2).strip()


_SPACE = r"(?:\s|~|\\[,;:! ])*"
_POWER = r"(?:\^\{?[23]\}?|[²³])?"
# Units allowed inside \text{} / \mathrm{} / \mbox{}; single letters only when wrapped.
_WRAPPED_UNITS = {"%", "cm", "mm", "km", "dm", "kg", "mg", "ml", "mL", "m", "g", "s", "h", "L", "min"}
_WRAPPED_UNIT_RE = re.compile(rf"\\(?:text|mathrm|mbox)\s*\{{{_SPACE}(?P<unit>[^{{}}]*?){_SPACE}\}}{_POWER}$")
_SYMBOL_UNIT_RE = re.compile(r"(?:\^\{\\circ\}|\^\\circ|\\circ|°|\\%|%)$")
_BARE_UNIT_RE = re.compile(rf"(?<![A-Za-z])(?:cm|mm|km|dm|kg|mg|ml|mL){_POWER}$")
_TRAILING_SPACE_RE = re.compile(rf"{_SPACE}$")


def strip_unit(field: str) -> str:
    """Remove one trailing unit (%, degree sign, metric unit) from a numeric field.

    No conversion is done: ``25%`` becomes ``25``, never ``0.25``. The unit is
    kept when nothing remains or the rest ends with a letter (``xcm``, ``2m``).
    """
    value = field.strip()
    match = _WRAPPED_UNIT_RE.search(value)
    if match and match.group("unit") not in _WRAPPED_UNITS:
        match = None
    match = match or _SYMBOL_UNIT_RE.search(value) or _BARE_UNIT_RE.search(value)
    if not match:
        return value
    rest = _TRAILING_SPACE_RE.sub("", value[: match.start()])
    if not rest or re.search(r"[A-Za-z]$", rest):
        return value
    return rest
