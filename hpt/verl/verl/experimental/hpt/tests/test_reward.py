import pytest

OK = {"response_length": 3000}


def score(reward, solution, gt, extra_info=OK, **kwargs):
    return reward.compute_score("test", solution, gt, extra_info, **kwargs)


@pytest.mark.parametrize(
    "solution, gt, expected",
    [
        ("think</think>\n答案为 \\boxed{3}", "3", 1.0),
        ("think \\boxed{3}", "3", 0.0),  # no </think>
        ("think \\boxed{3}</think> 答案是3", "3", 0.0),  # box only in thinking
        ("t</think>\\boxed{3}" + "验算" * 100, "3", 1.0),
        ("t</think>\\boxed{5} 改正：\\boxed{3}", "3", 1.0),
        ("t</think>\\boxed{3} 改正：\\boxed{5}", "3", 0.0),
        ("t</think>\\boxed{3", "3", 0.0),  # unbalanced box
        ("t</think>\\boxed{3}", ["5", "3"], 1.0),  # list ground truth
        ("t</think>\\boxed{3}", "\\boxed{3}", 1.0),  # boxed ground truth
        ("t</think>\\boxed{3}", 3, 1.0),  # numeric ground truth
    ],
)
def test_extraction(reward, solution, gt, expected):
    assert score(reward, solution, gt) == expected


def test_length_metadata_is_backward_compatible_but_not_required(reward):
    solution = "t</think>\\boxed{3}"
    assert score(reward, solution, "3", {"response_length": 8191}) == 1.0
    assert score(reward, solution, "3", {"response_length": 8192}) == 1.0
    assert score(reward, solution, "3", {"response_length": 4096}, max_response_length=4096) == 1.0
    assert score(reward, solution, "3", None) == 1.0


def test_origin_and_current_reward_interfaces(origin_reward, reward):
    plain = "reasoning \\boxed{3}"
    tagged = "reasoning</think>\\boxed{3}"
    kwargs = {"extra_info": {"response_length": 10}, "max_response_length": 10}

    # Origin grades the last box anywhere; current only grades the final answer.
    assert origin_reward.compute_score("test", plain, "3", **kwargs) == 1.0
    assert reward.compute_score("test", plain, "3", **kwargs) == 0.0
    assert origin_reward.compute_score("test", tagged, "3", **kwargs) == 1.0
    assert reward.compute_score("test", tagged, "3", **kwargs) == 1.0

    # Both process-pool entry points accept the same manager call contract.
    assert origin_reward.compute_score_in_process("test", tagged, "3", **kwargs) == 1.0
    assert reward.compute_score_in_process("test", tagged, "3", **kwargs) == 1.0


# Single-field answers without units or labels must keep the legacy verdict.
LEGACY_EQUIVALENT = [
    ("\\frac{1}{2}", "0.5"),
    ("x^2+2x+1", "(x+1)^2"),
    ("\\sqrt{53}", "\\sqrt{53}"),
    ("(160,560,960)", "(160, 560, 960)"),
    ("[1,2)", "[1,2)"),
    ("\\text{Evelyn}", "\\text{Evelyn}"),
    ("7", "8"),
    # Not stripped by strip_unit, but legacy's own unit handling accepts them.
    ("2m", "2"),
    ("5厘米", "5"),
    ("30度", "30"),
]


@pytest.mark.parametrize("pred, gt", LEGACY_EQUIVALENT)
def test_single_field_matches_legacy(reward, legacy, pred, gt):
    final = "\n\\boxed{" + pred + "}"
    assert score(reward, "t</think>" + final, gt) == float(bool(legacy.compute_score(final, gt)))


@pytest.mark.parametrize(
    "pred, gt, expected",
    [
        ("A,C", "A,C", 1.0),
        ("AC", "A,C", 1.0),
        ("A、C", "A,C", 1.0),
        ("A，C", "A,C", 1.0),
        ("A C", "A,C", 1.0),
        ("(A)(C)", "A,C", 1.0),
        ("\\text{A, C}", "A,C", 1.0),
        ("ＡＣ", "A,C", 1.0),  # full-width
        ("CA", "A,C", 1.0),  # order-insensitive
        ("\\textbf{(B)}", "B", 1.0),
        ("A", "A,C", 0.0),
        ("ACD", "A,C", 0.0),
        ("A或C", "A,C", 0.0),
        ("选A", "A", 0.0),
        ("II", "I", 0.0),  # repeated letters
        ("AAC", "A,C", 0.0),
        ("B", "A", 0.0),
    ],
)
def test_choice(reward, pred, gt, expected):
    assert score(reward, "t</think>\\boxed{" + pred + "}", gt) == expected


@pytest.mark.parametrize("gt", ["C,A", "A,A", "AB", "a", "A, C"])
def test_not_choice_ground_truth(reward, gt):
    from verl090_verifier.answer_rules import choice_letters

    assert choice_letters(gt) is None


@pytest.mark.parametrize(
    "pred, gt, expected",
    [
        ("2,3", "2,3", 1.0),
        ("2，3", "2,3", 1.0),
        ("2;3", "2,3", 1.0),
        ("2；3", "2,3", 1.0),
        ("\\frac{1}{2}, 3", "0.5,3", 1.0),
        ("3,2", "2,3", 0.0),  # order matters
        ("2,3,4", "2,3", 0.0),
        ("2", "2,3", 0.0),
        ("(2,3)", "2,3", 0.0),  # one tuple, not two blanks
        ("(1,2], 3", "(1,2],3", 1.0),  # interval stays one blank
        ("x=1,y=2", "x=1,y=2", 1.0),
        ("x_{1}=1,x_2=2", "x_1=1,x_{2}=2", 1.0),
        ("y=1,x=2", "x=1,y=2", 0.0),  # labels differ
        ("1,2", "x=1,y=2", 1.0),  # one-sided labels compare values
        ("x=1,y=2", "1,2", 1.0),
    ],
)
def test_multi_blank(reward, pred, gt, expected):
    assert score(reward, "t</think>\\boxed{" + pred + "}", gt) == expected


@pytest.mark.parametrize(
    "pred, gt, expected",
    [
        ("x=3", "x=3", 1.0),
        ("y=3", "x=3", 0.0),  # legacy alone accepts this
        ("3", "x=3", 1.0),
        ("y=2x+1", "y=2x+1", 1.0),
        ("p^{2}-2q=1", "p^{2}-2q=1", 1.0),  # equation, not a label
    ],
)
def test_single_field_labels(reward, pred, gt, expected):
    assert score(reward, "t</think>\\boxed{" + pred + "}", gt) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("1,2", ["1", "2"]),
        ("(1,2),3", ["(1,2)", "3"]),
        ("[1,2),\\{3,4\\}", ["[1,2)", "\\{3,4\\}"]),
        ("1\\,000", ["1\\,000"]),
        ("(1,2", None),
        ("1,", None),
    ],
)
def test_split_top_level(text, expected):
    from verl090_verifier.answer_rules import split_top_level

    assert split_top_level(text, ",") == expected


def test_thousands_separator_in_ground_truth_is_two_blanks(reward):
    # By convention a comma in the ground truth separates blanks; "1,000" must not appear.
    assert score(reward, "t</think>\\boxed{1000}", "1,000") == 0.0


@pytest.mark.parametrize(
    "pred, gt, expected",
    [
        ("30°", "30", 1.0),
        ("30^\\circ", "30", 1.0),
        ("30^{\\circ}", "30", 1.0),
        ("25\\%", "25", 1.0),
        ("25%", "25", 1.0),
        ("5\\text{ cm}", "5", 1.0),
        ("5\\mathrm{~cm}", "5", 1.0),
        ("5\\,\\text{m}", "5", 1.0),
        ("5\\text{m}^2", "5", 1.0),
        ("5cm", "5", 1.0),
        ("5 km", "5", 1.0),
        ("\\frac{1}{2}\\%", "\\frac{1}{2}", 1.0),
        ("25\\%", "0.25", 0.0),  # no percent/decimal conversion
        ("0.25", "25", 0.0),
        ("50\\%", "0.5", 0.0),
        ("50cm", "0.5", 0.0),  # no unit conversion
        ("30°,40\\%", "30,40", 1.0),  # units per blank
        ("x=30^\\circ", "x=30", 1.0),
    ],
)
def test_units(reward, pred, gt, expected):
    assert score(reward, "t</think>\\boxed{" + pred + "}", gt) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("30°", "30"),
        ("xcm", "xcm"),
        ("%", "%"),
        ("\\text{cm}", "\\text{cm}"),
        ("5\\text{ apples}", "5\\text{ apples}"),
        ("5", "5"),
    ],
)
def test_strip_unit(text, expected):
    from verl090_verifier.answer_rules import strip_unit

    assert strip_unit(text) == expected
