# ============================================================
# FLUENTPATH
# TOPIC 8 — STEP 3
# SCORE SCALE GUARDRAIL V1
#
# Purpose:
# - Detect suspicious / ambiguous AI score scales
# - Protect existing Interview AI score
# - Never blindly convert 1-10 scores into 0-100
# - Validate explicit 0-100 scoring contracts
# - Provide safe fallback when score scale is uncertain
#
# No Streamlit
# No database
# No API calls
# No main-app changes
# ============================================================

from __future__ import annotations

import re
from typing import Any, Dict, Optional


# ============================================================
# CONSTANTS
# ============================================================

MIN_SCORE = 0.0
MAX_SCORE = 100.0

DEFAULT_SAFE_SCORE = None


# ============================================================
# BASIC NUMBER EXTRACTION
# ============================================================

def extract_number(
    value: Any,
) -> Optional[float]:

    if value is None:
        return None

    if isinstance(value, bool):
        return None

    if isinstance(
        value,
        (int, float),
    ):
        return float(value)

    text = str(value).strip()

    if not text:
        return None

    match = re.search(
        r"-?\d+(?:\.\d+)?",
        text,
    )

    if not match:
        return None

    try:
        return float(
            match.group(0)
        )

    except (
        TypeError,
        ValueError,
        OverflowError,
    ):
        return None


# ============================================================
# SCORE SCALE HINT DETECTION
# ============================================================

def detect_explicit_scale(
    value: Any,
) -> Optional[str]:

    """
    Detect explicit scale information from the raw score.

    Examples:
        "8/10"   -> "0-10"
        "82/100" -> "0-100"
        "82%"    -> "0-100"

    Plain values such as:
        8
        4
        82

    do NOT contain an explicit scale.
    """

    if not isinstance(
        value,
        str,
    ):
        return None

    text = value.strip().lower()

    if not text:
        return None

    if re.search(
        r"/\s*10(?:\D|$)",
        text,
    ):
        return "0-10"

    if re.search(
        r"/\s*100(?:\D|$)",
        text,
    ):
        return "0-100"

    if "%" in text:
        return "0-100"

    return None


# ============================================================
# SAFE 0-100 CLAMP
# ============================================================

def clamp_100(
    score: Any,
) -> Optional[float]:

    number = extract_number(
        score
    )

    if number is None:
        return None

    number = max(
        MIN_SCORE,
        min(
            MAX_SCORE,
            number,
        ),
    )

    return round(
        number,
        1,
    )


# ============================================================
# SCORE SCALE ANALYSIS
# ============================================================

def analyze_score_scale(
    raw_score: Any,
    expected_scale: str = "0-100",
) -> Dict[str, Any]:

    number = extract_number(
        raw_score
    )

    explicit_scale = (
        detect_explicit_scale(
            raw_score
        )
    )

    result = {
        "raw_score":
            raw_score,

        "numeric_score":
            number,

        "expected_scale":
            expected_scale,

        "explicit_scale":
            explicit_scale,

        "status":
            "UNKNOWN",

        "trusted":
            False,

        "normalized_score":
            None,

        "reason":
            "",
    }

    if number is None:

        result["status"] = (
            "INVALID"
        )

        result["reason"] = (
            "No numeric score was found."
        )

        return result


    # --------------------------------------------------------
    # Explicit 0-100 score
    # --------------------------------------------------------

    if explicit_scale == "0-100":

        if (
            MIN_SCORE
            <= number
            <= MAX_SCORE
        ):

            result["status"] = (
                "VALID_0_100"
            )

            result["trusted"] = True

            result[
                "normalized_score"
            ] = round(
                number,
                1,
            )

            result["reason"] = (
                "The response explicitly uses "
                "a 0-100 or percentage scale."
            )

            return result


    # --------------------------------------------------------
    # Explicit 0-10 score
    # --------------------------------------------------------

    if explicit_scale == "0-10":

        if 0 <= number <= 10:

            result["status"] = (
                "EXPLICIT_0_10"
            )

            result["trusted"] = False

            result["reason"] = (
                "The AI explicitly returned a "
                "0-10 score while FluentPath "
                "expects 0-100."
            )

            return result


    # --------------------------------------------------------
    # Outside allowed 0-100 range
    # --------------------------------------------------------

    if (
        number < MIN_SCORE
        or number > MAX_SCORE
    ):

        result["status"] = (
            "OUT_OF_RANGE"
        )

        result["trusted"] = False

        result["reason"] = (
            "The score is outside the "
            "expected 0-100 range."
        )

        return result


    # --------------------------------------------------------
    # Plain number 0-10
    #
    # IMPORTANT:
    # A plain score such as 8 technically fits 0-100,
    # but may also represent 8/10.
    #
    # We do NOT automatically convert it to 80.
    # --------------------------------------------------------

    if 0 <= number <= 10:

        result["status"] = (
            "AMBIGUOUS_LOW_SCORE"
        )

        result["trusted"] = False

        result["reason"] = (
            "The score is between 0 and 10 "
            "without an explicit scale. "
            "It may represent either a very "
            "low 0-100 score or a 0-10 score."
        )

        return result


    # --------------------------------------------------------
    # Plain score >10 and <=100
    #
    # Under an explicit application contract expecting
    # 0-100, this is safe to accept.
    # --------------------------------------------------------

    result["status"] = (
        "VALID_0_100"
    )

    result["trusted"] = True

    result[
        "normalized_score"
    ] = round(
        number,
        1,
    )

    result["reason"] = (
        "The score is compatible with "
        "the expected 0-100 contract."
    )

    return result


# ============================================================
# PRIMARY SCORE PROTECTION
# ============================================================

def resolve_primary_score(
    ai_score: Any,
    existing_score: Any = None,
    expected_scale: str = "0-100",
) -> Dict[str, Any]:

    """
    Priority:

    1. Existing application score
       - Used when supplied and valid.

    2. New AI score
       - Used only when its scale is trusted.

    3. Safe fallback
       - No invented score.
    """

    # --------------------------------------------------------
    # Existing score protection
    # --------------------------------------------------------

    if existing_score is not None:

        existing_number = extract_number(
            existing_score
        )

        if (
            existing_number is not None
            and MIN_SCORE
            <= existing_number
            <= MAX_SCORE
        ):

            return {
                "score":
                    round(
                        existing_number,
                        1,
                    ),

                "source":
                    "existing_score",

                "trusted":
                    True,

                "status":
                    "EXISTING_SCORE_PROTECTED",

                "ai_score_analysis":
                    analyze_score_scale(
                        ai_score,
                        expected_scale,
                    ),

                "reason":
                    "Existing application score "
                    "remains authoritative.",
            }


    # --------------------------------------------------------
    # No existing score
    # --------------------------------------------------------

    analysis = analyze_score_scale(
        ai_score,
        expected_scale,
    )

    if analysis["trusted"]:

        return {
            "score":
                analysis[
                    "normalized_score"
                ],

            "source":
                "ai_score",

            "trusted":
                True,

            "status":
                analysis["status"],

            "ai_score_analysis":
                analysis,

            "reason":
                analysis["reason"],
        }


    # --------------------------------------------------------
    # Ambiguous / invalid score
    # --------------------------------------------------------

    return {
        "score":
            DEFAULT_SAFE_SCORE,

        "source":
            "fallback",

        "trusted":
            False,

        "status":
            analysis["status"],

        "ai_score_analysis":
            analysis,

        "reason":
            analysis["reason"],
    }


# ============================================================
# STRUCTURED SCORE CONTRACT
# ============================================================

def validate_score_contract(
    score: Any,
    existing_score: Any = None,
) -> Dict[str, Any]:

    resolved = resolve_primary_score(
        ai_score=score,
        existing_score=existing_score,
        expected_scale="0-100",
    )

    return {
        "valid":
            resolved["trusted"],

        "score":
            resolved["score"],

        "source":
            resolved["source"],

        "status":
            resolved["status"],

        "reason":
            resolved["reason"],

        "analysis":
            resolved[
                "ai_score_analysis"
            ],
    }


# ============================================================
# SELF TEST
# ============================================================

def run_self_tests():

    print()
    print("=" * 72)
    print(
        "FLUENTPATH SCORE SCALE GUARDRAIL V1"
    )
    print("=" * 72)


    # --------------------------------------------------------
    # TEST 1
    # Normal 0-100 score
    # --------------------------------------------------------

    result = validate_score_contract(
        82
    )

    assert result["valid"] is True
    assert result["score"] == 82.0
    assert result["source"] == "ai_score"

    print(
        "TEST 1 - Normal 0-100 score: PASS"
    )


    # --------------------------------------------------------
    # TEST 2
    # Percentage
    # --------------------------------------------------------

    result = validate_score_contract(
        "76%"
    )

    assert result["valid"] is True
    assert result["score"] == 76.0

    print(
        "TEST 2 - Explicit percentage: PASS"
    )


    # --------------------------------------------------------
    # TEST 3
    # Explicit /100
    # --------------------------------------------------------

    result = validate_score_contract(
        "91/100"
    )

    assert result["valid"] is True
    assert result["score"] == 91.0

    print(
        "TEST 3 - Explicit 0-100 scale: PASS"
    )


    # --------------------------------------------------------
    # TEST 4
    # Explicit /10
    # --------------------------------------------------------

    result = validate_score_contract(
        "8/10"
    )

    assert result["valid"] is False
    assert result["score"] is None
    assert (
        result["status"]
        == "EXPLICIT_0_10"
    )

    print(
        "TEST 4 - Explicit 0-10 detection: PASS"
    )


    # --------------------------------------------------------
    # TEST 5
    # Ambiguous score 8
    # --------------------------------------------------------

    result = validate_score_contract(
        8
    )

    assert result["valid"] is False
    assert result["score"] is None
    assert (
        result["status"]
        == "AMBIGUOUS_LOW_SCORE"
    )

    print(
        "TEST 5 - Ambiguous score 8 detection: PASS"
    )


    # --------------------------------------------------------
    # TEST 6
    # Ambiguous score 4
    # --------------------------------------------------------

    result = validate_score_contract(
        4
    )

    assert result["valid"] is False
    assert result["score"] is None

    print(
        "TEST 6 - Ambiguous score 4 detection: PASS"
    )


    # --------------------------------------------------------
    # TEST 7
    # Do not blindly convert 8 to 80
    # --------------------------------------------------------

    result = validate_score_contract(
        8
    )

    assert result["score"] != 80

    print(
        "TEST 7 - No blind 8-to-80 conversion: PASS"
    )


    # --------------------------------------------------------
    # TEST 8
    # Existing Interview score protection
    # --------------------------------------------------------

    result = validate_score_contract(
        score=8,
        existing_score=30,
    )

    assert result["valid"] is True
    assert result["score"] == 30.0
    assert (
        result["source"]
        == "existing_score"
    )

    print(
        "TEST 8 - Existing score protection: PASS"
    )


    # --------------------------------------------------------
    # TEST 9
    # Existing score protected even if new AI gives 95
    # --------------------------------------------------------

    result = validate_score_contract(
        score=95,
        existing_score=30,
    )

    assert result["score"] == 30.0
    assert (
        result["source"]
        == "existing_score"
    )

    print(
        "TEST 9 - Existing score remains authoritative: PASS"
    )


    # --------------------------------------------------------
    # TEST 10
    # Out of range
    # --------------------------------------------------------

    result = validate_score_contract(
        145
    )

    assert result["valid"] is False
    assert result["score"] is None
    assert (
        result["status"]
        == "OUT_OF_RANGE"
    )

    print(
        "TEST 10 - Out-of-range detection: PASS"
    )


    # --------------------------------------------------------
    # TEST 11
    # Negative score
    # --------------------------------------------------------

    result = validate_score_contract(
        -5
    )

    assert result["valid"] is False
    assert result["score"] is None

    print(
        "TEST 11 - Negative score rejection: PASS"
    )


    # --------------------------------------------------------
    # TEST 12
    # Invalid text
    # --------------------------------------------------------

    result = validate_score_contract(
        "good answer"
    )

    assert result["valid"] is False
    assert result["score"] is None
    assert (
        result["status"]
        == "INVALID"
    )

    print(
        "TEST 12 - Invalid score rejection: PASS"
    )


    # --------------------------------------------------------
    # TEST 13
    # Decimal 0-100 score
    # --------------------------------------------------------

    result = validate_score_contract(
        84.5
    )

    assert result["valid"] is True
    assert result["score"] == 84.5

    print(
        "TEST 13 - Decimal 0-100 score: PASS"
    )


    # --------------------------------------------------------
    # TEST 14
    # Boundary 100
    # --------------------------------------------------------

    result = validate_score_contract(
        100
    )

    assert result["valid"] is True
    assert result["score"] == 100.0

    print(
        "TEST 14 - Maximum boundary: PASS"
    )


    # --------------------------------------------------------
    # TEST 15
    # Existing score 0 must still be protected
    # --------------------------------------------------------

    result = validate_score_contract(
        score=90,
        existing_score=0,
    )

    assert result["valid"] is True
    assert result["score"] == 0.0
    assert (
        result["source"]
        == "existing_score"
    )

    print(
        "TEST 15 - Existing zero score protection: PASS"
    )


    # --------------------------------------------------------
    # FINAL
    # --------------------------------------------------------

    print()
    print("=" * 72)

    print(
        "SCORE SCALE GUARDRAIL: PASS"
    )

    print("=" * 72)

    print()
    print(
        "Verified:"
    )

    print(
        "- 0-100 score validation"
    )

    print(
        "- Percentage scale recognition"
    )

    print(
        "- Explicit /100 recognition"
    )

    print(
        "- Explicit /10 detection"
    )

    print(
        "- Ambiguous low-score detection"
    )

    print(
        "- No blind 1-10 to 0-100 conversion"
    )

    print(
        "- Existing Interview score protection"
    )

    print(
        "- Out-of-range rejection"
    )

    print(
        "- Invalid score rejection"
    )

    print(
        "- No invented fallback score"
    )


# ============================================================
# DIRECT RUN
# ============================================================

if __name__ == "__main__":

    run_self_tests()