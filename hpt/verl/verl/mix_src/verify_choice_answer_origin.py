#!/usr/bin/env python3
"""Conservative automatic answer verification. See VERIFIER.md."""
from __future__ import annotations
import argparse
import re
import unicodedata
from collections.abc import Sequence
from decimal import Decimal
from functools import lru_cache

import sympy as sp
from sympy.core.relational import Relational
from math_verify import parse as math_parse, verify as math_verify
from math_verify.parser import LatexExtractionConfig, NormalizationConfig

DEFAULT_CHOICES = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _group(text, start):
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{": depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0: return text[start + 1:i], i + 1
    raise ValueError("Unbalanced braces")


def strip_latex_wrappers(text: str) -> str:
    value = str(text).strip()
    while True:
        old = value
        if value.startswith("$") and value.endswith("$"):
            value = value.strip("$").strip()
        match = re.match(r"\\(?:boxed|text|mathrm)\s*\{", value)
        if match:
            body, end = _group(value, match.end() - 1)
            if not value[end:].strip(): value = body.strip()
        if old == value: return value


def _answers(text):
    if text is None: raise ValueError("Missing answer")
    value = str(text).strip()
    if "</think>" in value: value = value.rsplit("</think>", 1)[1].strip()
    elif "<think>" in value: raise ValueError("Unclosed reasoning")
    boxes, end = [], 0
    for match in re.finditer(r"\\boxed\s*\{", value):
        if match.start() < end: raise ValueError("Nested boxes")
        body, end = _group(value, match.end() - 1)
        boxes.append(body)
    if boxes: return boxes
    lines = re.findall(r"(?im)^\s*(?:最终答案|正确答案|答案|故选|因此选|(?:final\s+|correct\s+)?answer|choice|option)\s*(?:为|是|is\b|[:：])?\s*[:：]?\s*(.*?)\s*$", value)
    return [v.rstrip("。.").strip() for v in lines] if lines else [value]


def _choice(value, valid):
    value = unicodedata.normalize("NFKC", strip_latex_wrappers(value)).upper()
    if not re.fullmatch(r"[A-Z\s,、;()\[\]{}]+", value): raise ValueError("Not choice labels")
    labels = re.sub("[^A-Z]", "", value)
    if not labels or any(c not in valid for c in labels): raise ValueError("Invalid labels")
    return "".join(c for c in dict.fromkeys(valid) if c in labels)


def extract_choice_answer(text: str, valid_choices: str = DEFAULT_CHOICES) -> str:
    try:
        values = {_choice(v, valid_choices.upper()) for v in _answers(text)}
        return values.pop() if len(values) == 1 else ""
    except ValueError: return ""


def verify_choice_answer(prediction: str, ground_truth: str, valid_choices: str = DEFAULT_CHOICES) -> bool:
    pred = extract_choice_answer(prediction, valid_choices)
    gold = extract_choice_answer(ground_truth, valid_choices)
    return bool(gold) and pred == gold


UNITS = {}
for dimension, entries in {
    "length": [("m 米", "1"), ("cm 厘米", "1/100"), ("mm 毫米", "1/1000"), ("km 千米 公里", "1000")],
    "mass": [("kg 千克 公斤", "1"), ("g 克", "1/1000")],
    "time": [("s 秒", "1"), ("min 分钟", "60"), ("h 小时", "3600")],
    "angle": [("° 度", "1")],
    "temperature": [("°C ℃ 摄氏度", "1")],
    "scalar": [("% \\%", "1/100")],
}.items():
    for aliases, scale in entries:
        for alias in aliases.split(): UNITS[alias] = (dimension, sp.Rational(scale))
for alias in ("m", "米", "cm", "厘米", "mm", "毫米", "km", "千米"):
    for power in (2, 3):
        for suffix in (str(power), "^" + str(power), "^{" + str(power) + "}", "²" if power == 2 else "³"):
            UNITS[alias + suffix] = ("area" if power == 2 else "volume", UNITS[alias][1] ** power)


def _split(value, separators=",，"):
    stack, parts, start = [], [], 0
    for i, c in enumerate(value):
        if c in "{([": stack.append(c)
        elif c in "})]":
            if not stack or stack.pop() != {"}": "{", ")": "(", "]": "["}[c]: raise ValueError("Unbalanced answer")
        elif c in separators and not stack and (i == 0 or value[i - 1] != "\\"):
            parts.append(value[start:i].strip()); start = i + 1
    parts.append(value[start:].strip())
    if stack or not all(parts): raise ValueError("Invalid answers")
    return parts


_NUMBERED_FIELD_RE = re.compile(
    r"(?<!\w)(?:\((?P<ascii>\d+)\)|（(?P<wide>\d+)）|(?P<plain>\d+)(?:\.(?!\d)|[、:：]))\s*"
)


def _strip_ordered_markers(parts):
    """Strip (1)/(2), （1）/（2）, or 1./2、 only as a complete sequence."""
    named = [re.fullmatch(r"([a-z])\s*=\s*(.+)", part) for part in parts]
    if len(parts) > 1 and all(named):
        if [match.group(1) for match in named] != [chr(ord("a") + i) for i in range(len(parts))]:
            raise ValueError("Invalid answer labels")
        return [match.group(2).strip() for match in named]
    stripped, numbers = [], []
    for part in parts:
        match = _NUMBERED_FIELD_RE.match(part)
        if not match:
            if numbers:
                raise ValueError("Incomplete numbered answer")
            return parts
        numbers.append(int(next(v for v in match.groupdict().values() if v)))
        stripped.append(part[match.end():].strip())
    if numbers != list(range(1, len(parts) + 1)) or not all(stripped):
        raise ValueError("Invalid answer numbering")
    return stripped


def _split_prediction_fields(value, expected_count):
    """Parse tolerant multi-blank presentation into an ordered field list."""
    value = strip_latex_wrappers(value).strip()
    value = re.sub(
        r"[①-⑳]",
        lambda match: f"({ord(match.group()) - ord('①') + 1})",
        value,
    )

    # A list of selected item numbers can itself be the answer, not a set of
    # empty numbered blanks: (1)(3) means the two values 1 and 3.
    selected = re.fullmatch(r"\s*(?:\(\d+\)\s*)+", value)
    if selected:
        fields = re.findall(r"\((\d+)\)", value)
        if len(fields) == expected_count:
            return fields

    # Accept adjacent numbered fields even if the model omitted separators.
    markers = list(_NUMBERED_FIELD_RE.finditer(value))
    if len(markers) == expected_count and expected_count > 1:
        numbers = [
            int(next(v for v in marker.groupdict().values() if v))
            for marker in markers
        ]
        if numbers == list(range(1, expected_count + 1)) and not value[:markers[0].start()].strip():
            fields = [
                value[marker.end() : markers[i + 1].start()].strip(" \t,，;；")
                if i + 1 < len(markers)
                else value[marker.end():].strip(" \t,，;；")
                for i, marker in enumerate(markers)
            ]
            if all(fields):
                return fields

    parts = _split(value, separators=",，;；")
    if len(parts) == expected_count:
        return _strip_ordered_markers(parts)

    # Chinese enumeration marks can be separators within groups separated by
    # commas. Only use them when they recover exactly the reference field count.
    enumerated = _split(value, separators=",，;；、")
    if len(enumerated) == expected_count:
        return _strip_ordered_markers(enumerated)

    alternatives = re.split(r"\s*(?:\\text\s*\{\s*或\s*\}|或)\s*", value)
    if len(alternatives) == expected_count and all(alternatives):
        return alternatives
    return _strip_ordered_markers(parts)


def _literal(value):
    value = unicodedata.normalize("NFKC", strip_latex_wrappers(value)).strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        value = value[1:-1].strip()
    return re.sub(r"\s+", " ", value)


def _latex_surface(value):
    """Discard presentation-only LaTeX differences without changing content."""
    value = re.sub(r"\\(?:left|right)(?![A-Za-z])", "", value)
    value = re.sub(r"\\(?:dfrac|tfrac)(?![A-Za-z])", r"\\frac", value)
    return re.sub(r"\s*,\s*", ",", value)


_LATEX_CONFIG = [LatexExtractionConfig(
    normalization_config=NormalizationConfig(
        basic_latex=True, units=True, malformed_operators=False,
        nits=False, boxed="all", equations=False,
    ),
)]


class InvalidMathValue(ValueError):
    """A parsed mathematical value that must never be rewarded."""


@lru_cache(maxsize=4096)
def _math_value(value):
    """Parse one complete field with math-verify, without string fallback."""
    value = strip_latex_wrappers(value).strip()
    if not value or len(value) > 512:
        raise ValueError("Invalid mathematical answer length")
    if value.count("=") > 1:
        raise ValueError("Ambiguous equation chain")
    # Math-Verify reads bare "pi" as p*i; in math fields it denotes \pi.
    value = re.sub(r"(?<![A-Za-z\\])pi(?![A-Za-z])", lambda _: r"\pi", value)
    parsed = math_parse(
        f"${value}$", extraction_config=_LATEX_CONFIG,
        fallback_mode="no_fallback", extraction_mode="first_match",
    )
    if len(parsed) != 1 or not isinstance(parsed[0], sp.Basic):
        raise ValueError("Unparsed mathematical answer")
    result = parsed[0]
    if isinstance(result, sp.Expr) and result.has(sp.zoo, sp.nan, sp.oo, -sp.oo):
        raise InvalidMathValue("Nonfinite mathematical answer")
    return result


def _math_equal(gold, prediction):
    if isinstance(gold, Relational) != isinstance(prediction, Relational):
        return False
    return bool(math_verify([gold], [prediction]))


def _looks_math(value):
    # Math-Verify interprets plain words such as abc as products of symbols.
    return bool(value == "pi" or re.search(r"[0-9\\+*/^=<>≤≥√π]", value) or re.fullmatch(r"[A-Za-z]", value))


def _quantity(value, expected_unit):
    value = strip_latex_wrappers(value)
    value = re.sub(r"\\(?:mathrm|text)\s*\{([^{}]*)\}", r"\1", value)
    value = value.replace(r"\,", " ").replace(r"\;", " ").replace("~", " ").strip()
    value = re.sub(r"(?:\^\{?\\circ\}?|\\circ)", "°", value)
    if value in UNITS:
        raise ValueError("Unit without magnitude")
    for unit in sorted(UNITS, key=len, reverse=True):
        if value.endswith(unit) and value[:-len(unit)].strip():
            number = value[:-len(unit)].strip()
            # Unseparated symbolic names such as 'xcm' are ambiguous.
            if re.search(r"[A-Za-z]$", number): raise ValueError("Ambiguous unit")
            dimension, scale = UNITS[unit]
            parsed = _math_value(number)
            if not isinstance(parsed, sp.Expr): raise ValueError("Non-numeric quantity")
            return parsed * scale, dimension
    if expected_unit:
        dimension, scale = UNITS[expected_unit]
        parsed = _math_value(value)
        if not isinstance(parsed, sp.Expr): raise ValueError("Non-numeric quantity")
        return parsed * scale, dimension
    return _math_value(value), "scalar"


def _boolean(value):
    value = _literal(value).lower()
    if value in ("true", "正确", "对", "√", "✓"): return True
    if value in ("false", "错误", "错", "×", "✗"): return False
    raise ValueError("Invalid boolean")


def _field_equal(prediction, ground_truth, expected_unit=None, *, allow_percent_display=True):
    """Compare one blank, preferring exact text before mathematical equivalence."""
    pred_text = _literal(prediction)
    gold_text = _literal(ground_truth)
    if pred_text == gold_text:
        if _looks_math(gold_text):
            try:
                _math_value(gold_text)
            except InvalidMathValue:
                return False
            except ValueError:
                pass
        return bool(gold_text)

    if "\\" in pred_text and "\\" in gold_text and _latex_surface(pred_text) == _latex_surface(gold_text):
        return True

    # A parenthesized pair is one structured answer, not two blank slots.
    if all(text.startswith("(") and text.endswith(")") for text in (pred_text, gold_text)):
        try:
            pred_items = _split(pred_text[1:-1])
            gold_items = _split(gold_text[1:-1])
            if len(gold_items) > 1 and len(pred_items) == len(gold_items):
                return all(_field_equal(pred, gold, expected_unit, allow_percent_display=allow_percent_display) for pred, gold in zip(pred_items, gold_items))
        except ValueError:
            pass

    try:
        return _boolean(pred_text) == _boolean(gold_text)
    except ValueError:
        pass

    if pred_text in UNITS and gold_text in UNITS:
        return UNITS[pred_text] == UNITS[gold_text]

    if expected_unit is not None or _looks_math(gold_text) or gold_text in UNITS:
        try:
            pred = _quantity(pred_text, expected_unit)
            gold = _quantity(gold_text, expected_unit)
            if pred[1] == gold[1] and _math_equal(gold[0], pred[0]):
                return True
        except Exception:
            pass
    return expected_unit is None and allow_percent_display and _percent_display_equal(pred_text, gold_text)


_BARE_NUMBER_RE = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)")


def _percent_display_equal(prediction, ground_truth):
    """Match a percent sign used as display notation for the same number."""
    def matches(percent_text, bare_text):
        percent_text = _literal(percent_text).replace(r"\%", "%")
        bare_text = _literal(bare_text)
        if not percent_text.endswith("%") or not _BARE_NUMBER_RE.fullmatch(bare_text):
            return False
        number = percent_text[:-1].strip()
        return bool(_BARE_NUMBER_RE.fullmatch(number)) and Decimal(number) == Decimal(bare_text)

    return matches(prediction, ground_truth) or matches(ground_truth, prediction)


def _auto_fields(text, expected_count, *, reference=False):
    answers = _answers(text)
    if len(answers) == 1:
        if reference:
            # The canonical reference schema uses only top-level ASCII commas.
            return _split(strip_latex_wrappers(answers[0]), separators=",")
        return _split_prediction_fields(answers[0], expected_count)

    # Multiple boxes can naturally mean one box per blank. Multiple unboxed
    # answer declarations remain ambiguous and are rejected.
    final = str(text).rsplit("</think>", 1)[-1]
    box_count = len(re.findall(r"\\boxed\s*\{", final))
    if not reference and expected_count == 1 and box_count == len(answers):
        fields = [strip_latex_wrappers(value).strip() for value in answers]
        if fields and all(field == fields[0] for field in fields) and len(_split_prediction_fields(fields[0], 1)) == 1:
            return fields[:1]
    if not reference and box_count == expected_count == len(answers):
        fields = [strip_latex_wrappers(value).strip() for value in answers]
        if all(fields) and all(len(_split_prediction_fields(value, 1)) == 1 for value in fields):
            return fields
    if not reference and box_count == len(answers) and box_count > 1:
        fields = [
            field
            for answer in answers
            for field in _split_prediction_fields(answer, expected_count)
        ]
        if len(fields) == expected_count:
            return fields
    raise ValueError("Ambiguous answer declarations")


def verify_auto_answer(prediction, ground_truth, *, valid_choices=DEFAULT_CHOICES, expected_unit=None, expected_units: Sequence[str | None] | None = None):
    """Infer the answer schema from the canonical ground-truth serialization.

    A top-level ASCII comma in the reference means ordered blanks. Multi-choice
    references must therefore use compact letters such as ``AC``, as required by
    the training prompt.
    """
    try:
        gold_fields = _auto_fields(ground_truth, 1, reference=True)
        if not gold_fields:
            return False
        if expected_units is not None:
            if len(expected_units) != len(gold_fields) or any(unit is not None and unit not in UNITS for unit in expected_units):
                raise ValueError("Invalid per-field units")
            units = expected_units
        else:
            units = [expected_unit] * len(gold_fields)

        if len(gold_fields) == 1:
            gold = _literal(gold_fields[0])
            valid = "".join(dict.fromkeys(valid_choices.upper()))
            if re.fullmatch(r"[A-Z]+", gold) and all(char in valid for char in gold):
                return verify_choice_answer(prediction, ground_truth, valid_choices)

        pred_fields = _auto_fields(prediction, len(gold_fields))
        return len(pred_fields) == len(gold_fields) and all(
            _field_equal(pred, gold, unit, allow_percent_display=expected_units is None)
            for pred, gold, unit in zip(pred_fields, gold_fields, units)
        )
    except (ValueError, SyntaxError, TypeError, ZeroDivisionError, RecursionError):
        return False


def verify_answer(prediction: str, ground_truth: str, answer_type: str = "auto", *, valid_choices: str = DEFAULT_CHOICES, expected_unit: str | None = None, expected_units: Sequence[str | None] | None = None) -> bool:
    """Multiple blanks are ordered; conflicting final declarations fail closed.

    expected_unit defines the unit of bare numbers, on both sides. Use 'unit'
    when the question asks for a unit itself. Sympy is needed for short/unit.
    """
    if answer_type not in ("auto", "choice", "boolean", "short", "unit"): raise ValueError("Unknown answer type")
    if expected_unit is not None and expected_unit not in UNITS: raise ValueError("Unknown expected unit")
    if expected_units is not None and (answer_type != "auto" or expected_unit is not None): raise ValueError("Per-field units require auto mode without a global unit")
    if answer_type == "auto": return verify_auto_answer(prediction, ground_truth, valid_choices=valid_choices, expected_unit=expected_unit, expected_units=expected_units)
    if answer_type == "choice": return verify_choice_answer(prediction, ground_truth, valid_choices)
    try:
        def normalize(text):
            if answer_type == "boolean":
                return [_boolean(text)]
            if answer_type == "unit":
                unit = strip_latex_wrappers(text)
                if unit not in UNITS: raise ValueError("Unknown unit")
                dimension, scale = UNITS[unit]
                return [(scale, dimension)]
            return [_quantity(v, expected_unit) for v in _split(strip_latex_wrappers(text))]
        pred_answers = _answers(prediction)
        gold_answers = _answers(ground_truth)
        predictions = [normalize(v) for v in pred_answers]
        references = [normalize(v) for v in gold_answers]
        gold = references[0]
        gold_raw = _split(strip_latex_wrappers(gold_answers[0])) if answer_type == "short" else []
        def equal(items, answer):
            if len(items) != len(gold): return False
            if answer_type == "boolean": return items == gold
            raw = _split(strip_latex_wrappers(answer)) if answer_type == "short" else []
            return all(
                (a[1] == b[1] and _math_equal(b[0], a[0]))
                or (answer_type == "short" and expected_unit is None and _percent_display_equal(raw[i], gold_raw[i]))
                for i, (a, b) in enumerate(zip(items, gold))
            )
        return all(equal(items, answer) for items, answer in zip(predictions, pred_answers)) and all(
            equal(items, answer) for items, answer in zip(references, gold_answers)
        )
    except (ValueError, SyntaxError, TypeError, ZeroDivisionError, RecursionError): return False


def compute_score(
    data_source,
    solution_str: str,
    ground_truth: str,
    extra_info=None,
    **kwargs,
) -> float:
    """VERL custom-reward entry point using automatic answer verification."""
    del data_source, extra_info, kwargs
    return float(verify_answer(solution_str, ground_truth, answer_type="auto"))


def format_diagnostics(text: str) -> dict[str, bool]:
    final = str(text).rsplit("</think>", 1)[-1]
    boxed = False
    try:
        matches = list(re.finditer(r"\\boxed\s*\{", final))
        for match in matches: _group(final, match.end() - 1)
        boxed = len(matches) == 1
    except ValueError: pass
    return {"has_end_think": "</think>" in str(text), "single_boxed_final": boxed}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prediction"); parser.add_argument("ground_truth")
    parser.add_argument("--choices", default=DEFAULT_CHOICES)
    parser.add_argument("--type", choices=("auto", "choice", "boolean", "short", "unit"), default="auto")
    parser.add_argument("--expected-unit", help="Unit of bare numbers, e.g. cm, %% or °")
    parser.add_argument("--expected-units", help="Comma-separated units per blank; use - for unspecified, e.g. -,-,%%")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    units = None if args.expected_units is None else [None if value == "-" else value for value in args.expected_units.split(",")]
    correct = verify_answer(args.prediction, args.ground_truth, args.type, valid_choices=args.choices, expected_unit=args.expected_unit, expected_units=units)
    if not args.quiet:
        if args.type == "choice":
            print(f"prediction: {extract_choice_answer(args.prediction, args.choices) or '<empty>'}")
            print(f"ground_truth: {extract_choice_answer(args.ground_truth, args.choices) or '<empty>'}")
        print(f"correct: {correct}")
        print(f"format: {format_diagnostics(args.prediction)}")
    return 0 if correct else 1


if __name__ == "__main__": raise SystemExit(main())
