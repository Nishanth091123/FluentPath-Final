"""
FluentPath - Interview Intelligence V2
=======================================

Purpose:
- Analyze interview answers safely and deterministically
- Detect answer strength / weakness
- Evaluate answer structure
- Recommend next difficulty
- Generate safe follow-up questions
- Track session-level performance
- Provide actionable coaching

Important:
- No separate LLM call is used in this V2 engine.
- Existing FluentPath AI interview scoring can remain unchanged.
- This module acts as an additional intelligence layer.
- It does not evaluate emotion, personality, confidence,
  eye contact, or mental state.
"""

import re
from typing import Dict, List, Optional, Any


# ============================================================
# CONFIGURATION
# ============================================================

VALID_LEVELS = {
    "Beginner",
    "Intermediate",
    "Advanced",
}

VALID_MODES = {
    "HR",
    "Technical",
    "Technical HR",
    "Project",
    "Mock",
    "Random",
}


# ============================================================
# BASIC HELPERS
# ============================================================

def clean_text(value: Any) -> str:

    if value is None:
        return ""

    return str(value).strip()


def clamp_score(value: Any) -> int:

    try:
        value = float(value)

    except Exception:
        value = 0

    return int(
        max(
            0,
            min(100, round(value))
        )
    )


def normalize_level(value: str) -> str:

    value = clean_text(value).lower()

    if value == "advanced":
        return "Advanced"

    if value == "intermediate":
        return "Intermediate"

    return "Beginner"


def normalize_mode(value: str) -> str:

    value = clean_text(value)

    for mode in VALID_MODES:

        if value.lower() == mode.lower():
            return mode

    return "Mock"


def get_words(text: str) -> List[str]:

    return re.findall(
        r"\b[\w'-]+\b",
        clean_text(text).lower(),
        flags=re.UNICODE
    )


def word_count(text: str) -> int:

    return len(
        get_words(text)
    )


# ============================================================
# QUESTION TYPE DETECTION
# ============================================================

def detect_question_type(
    mode: str,
    question: str
) -> str:

    mode = normalize_mode(mode)
    q = clean_text(question).lower()

    behavioural_markers = [
        "tell me about a time",
        "describe a time",
        "challenging problem",
        "challenge you faced",
        "conflict",
        "failure",
        "achievement",
        "difficult situation",
        "teamwork",
        "deadline",
        "pressure",
        "leadership",
        "problem you solved",
    ]

    project_markers = [
        "your project",
        "project",
        "implemented",
        "developed",
        "built",
        "architecture",
        "deployment",
        "real-time",
        "real time",
    ]

    technical_markers = [
        "what is",
        "difference between",
        "explain",
        "how does",
        "how do",
        "why do",
        "why is",
        "algorithm",
        "model",
        "python",
        "machine learning",
        "data science",
        "sql",
        "rag",
        "llm",
        "classification",
        "regression",
    ]

    if any(
        marker in q
        for marker in behavioural_markers
    ):
        return "Behavioural"

    if (
        mode == "Project"
        or any(
            marker in q
            for marker in project_markers
        )
    ):
        return "Project"

    if (
        mode in {
            "Technical",
            "Technical HR"
        }
        or any(
            marker in q
            for marker in technical_markers
        )
    ):
        return "Technical"

    return "General"


# ============================================================
# ANSWER SIGNALS
# ============================================================

def detect_answer_signals(
    answer: str
) -> Dict:

    text = clean_text(answer).lower()

    result_markers = [
        "result",
        "finally",
        "completed",
        "achieved",
        "improved",
        "reduced",
        "increased",
        "solved",
        "success",
        "outcome",
    ]

    action_markers = [
        "i checked",
        "i analyzed",
        "i analysed",
        "i discussed",
        "i decided",
        "i implemented",
        "i developed",
        "i created",
        "i used",
        "i handled",
        "i solved",
        "i planned",
        "i organized",
        "i organised",
        "i worked",
        "we implemented",
        "we developed",
        "we created",
        "we used",
    ]

    example_markers = [
        "for example",
        "for instance",
        "in one project",
        "in my project",
        "in my experience",
        "one time",
        "once",
    ]

    reason_markers = [
        "because",
        "so that",
        "therefore",
        "due to",
        "the reason",
    ]

    sequence_markers = [
        "first",
        "then",
        "next",
        "after that",
        "finally",
    ]

    has_result = any(
        marker in text
        for marker in result_markers
    )

    has_action = any(
        marker in text
        for marker in action_markers
    )

    has_example = any(
        marker in text
        for marker in example_markers
    )

    has_reason = any(
        marker in text
        for marker in reason_markers
    )

    has_sequence = any(
        marker in text
        for marker in sequence_markers
    )

    return {
        "has_result": has_result,
        "has_action": has_action,
        "has_example": has_example,
        "has_reason": has_reason,
        "has_sequence": has_sequence,
    }


# ============================================================
# RELEVANCE SCORE
# ============================================================

def calculate_relevance_score(
    question: str,
    answer: str
) -> int:

    question_words = set(
        get_words(question)
    )

    answer_words = set(
        get_words(answer)
    )

    stop_words = {
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "to",
        "of",
        "in",
        "on",
        "at",
        "for",
        "and",
        "or",
        "but",
        "with",
        "about",
        "me",
        "you",
        "your",
        "my",
        "i",
        "we",
        "they",
        "it",
        "this",
        "that",
        "what",
        "why",
        "how",
        "when",
        "where",
        "which",
        "do",
        "does",
        "did",
        "can",
        "could",
        "would",
        "should",
        "tell",
        "describe",
        "explain",
    }

    important_question_words = {
        word
        for word in question_words
        if (
            word not in stop_words
            and len(word) > 2
        )
    }

    if not important_question_words:

        return 70

    overlap = (
        important_question_words
        & answer_words
    )

    ratio = (
        len(overlap)
        / len(important_question_words)
    )

    score = 50 + (ratio * 45)

    if word_count(answer) < 8:
        score -= 15

    return clamp_score(score)


# ============================================================
# COMPLETENESS SCORE
# ============================================================

def calculate_completeness_score(
    answer: str,
    question_type: str,
    signals: Dict
) -> int:

    count = word_count(answer)

    if count < 8:
        score = 25

    elif count < 20:
        score = 45

    elif count < 40:
        score = 62

    elif count < 80:
        score = 78

    elif count <= 180:
        score = 88

    else:
        score = 82

    if question_type == "Behavioural":

        if signals["has_action"]:
            score += 5

        if signals["has_result"]:
            score += 5

        if signals["has_example"]:
            score += 3

    elif question_type == "Project":

        if signals["has_action"]:
            score += 5

        if signals["has_result"]:
            score += 5

    elif question_type == "Technical":

        if signals["has_reason"]:
            score += 4

        if signals["has_example"]:
            score += 3

    return clamp_score(score)


# ============================================================
# STRUCTURE SCORE
# ============================================================

def calculate_structure_score(
    answer: str,
    question_type: str,
    signals: Dict
) -> int:

    count = word_count(answer)

    if count < 8:
        score = 30

    elif count < 20:
        score = 48

    else:
        score = 62

    if signals["has_sequence"]:
        score += 10

    if signals["has_action"]:
        score += 8

    if signals["has_result"]:
        score += 8

    if signals["has_reason"]:
        score += 5

    if question_type == "Behavioural":

        if (
            signals["has_action"]
            and signals["has_result"]
        ):
            score += 5

    return clamp_score(score)


# ============================================================
# CLARITY SCORE
# ============================================================

def calculate_clarity_score(
    answer: str
) -> int:

    text = clean_text(answer)

    count = word_count(text)

    if count < 5:
        return 30

    sentences = [
        item.strip()
        for item in re.split(
            r"[.!?]+",
            text
        )
        if item.strip()
    ]

    if not sentences:
        sentences = [text]

    average_sentence_length = (
        count
        / len(sentences)
    )

    score = 75

    if count < 15:
        score -= 12

    if average_sentence_length > 35:
        score -= 10

    if average_sentence_length > 50:
        score -= 10

    repeated_fillers = [
        "actually actually",
        "so so",
        "like like",
        "basically basically",
    ]

    lower = text.lower()

    if any(
        filler in lower
        for filler in repeated_fillers
    ):
        score -= 8

    return clamp_score(score)


# ============================================================
# STRENGTH / WEAKNESS ANALYSIS
# ============================================================

def build_feedback(
    question_type: str,
    answer: str,
    signals: Dict,
    relevance: int,
    completeness: int,
    structure: int,
    clarity: int
) -> Dict:

    strengths = []
    weaknesses = []
    missing_points = []

    if relevance >= 70:
        strengths.append(
            "The answer stays relevant to the question."
        )

    if clarity >= 70:
        strengths.append(
            "The answer is reasonably clear and understandable."
        )

    if signals["has_action"]:
        strengths.append(
            "The answer explains an action or approach."
        )

    if signals["has_result"]:
        strengths.append(
            "The answer includes an outcome or result."
        )

    if relevance < 60:
        weaknesses.append(
            "Make the answer more directly connected to the question."
        )

    if completeness < 60:
        weaknesses.append(
            "Add enough supporting detail to make the answer complete."
        )

    if structure < 60:
        weaknesses.append(
            "Organize the answer in a clearer sequence."
        )

    if clarity < 60:
        weaknesses.append(
            "Use shorter and clearer sentences."
        )

    if question_type == "Behavioural":

        if not signals["has_action"]:
            missing_points.append(
                "Explain what action you personally took."
            )

        if not signals["has_result"]:
            missing_points.append(
                "Finish with the result or outcome."
            )

    elif question_type == "Project":

        if not signals["has_action"]:
            missing_points.append(
                "Explain your approach or implementation."
            )

        if not signals["has_result"]:
            missing_points.append(
                "Mention the project result or impact."
            )

    elif question_type == "Technical":

        if not signals["has_reason"]:
            missing_points.append(
                "Add a short reason or explanation where relevant."
            )

        if (
            word_count(answer) < 25
            and not signals["has_example"]
        ):
            missing_points.append(
                "Add one small example if it helps explain the concept."
            )

    if not strengths:

        strengths.append(
            "The answer provides a starting point that can be improved."
        )

    if not weaknesses:

        weaknesses.append(
            "Make the answer more specific and interview-focused."
        )

    if not missing_points:

        missing_points.append(
            "No major structural point is missing; improve precision."
        )

    return {
        "strengths": strengths,
        "weaknesses": weaknesses,
        "missing_points": missing_points,
    }


# ============================================================
# OVERALL SCORE
# ============================================================

def calculate_overall_score(
    relevance: int,
    completeness: int,
    structure: int,
    clarity: int
) -> int:

    score = (
        relevance * 0.30
        + completeness * 0.25
        + structure * 0.25
        + clarity * 0.20
    )

    return clamp_score(score)


# ============================================================
# SAFE FOLLOW-UP ENGINE
# ============================================================

def generate_safe_follow_up(
    question_type: str,
    signals: Dict,
    overall_score: int
) -> Dict:

    if overall_score >= 90:

        return {
            "follow_up_needed": False,
            "follow_up_question": None,
            "follow_up_reason":
                "The answer is already sufficiently complete."
        }

    if question_type == "Behavioural":

        if not signals["has_action"]:

            return {
                "follow_up_needed": True,
                "follow_up_question":
                    "What specific action did you personally take?",
                "follow_up_reason":
                    "The action taken is not clear enough."
            }

        if not signals["has_result"]:

            return {
                "follow_up_needed": True,
                "follow_up_question":
                    "What was the final result of your action?",
                "follow_up_reason":
                    "The outcome needs more detail."
            }

        return {
            "follow_up_needed": True,
            "follow_up_question":
                "What did you learn from that experience?",
            "follow_up_reason":
                "The follow-up checks reflection and learning."
        }

    if question_type == "Project":

        if not signals["has_action"]:

            return {
                "follow_up_needed": True,
                "follow_up_question":
                    "What approach did you use to solve this problem?",
                "follow_up_reason":
                    "The implementation approach needs clarification."
            }

        if not signals["has_result"]:

            return {
                "follow_up_needed": True,
                "follow_up_question":
                    "What result or improvement did the project achieve?",
                "follow_up_reason":
                    "The project impact is not clear."
            }

        return {
            "follow_up_needed": True,
            "follow_up_question":
                "What was the most important challenge in this project?",
            "follow_up_reason":
                "The follow-up checks deeper project understanding."
        }

    if question_type == "Technical":

        if not signals["has_reason"]:

            return {
                "follow_up_needed": True,
                "follow_up_question":
                    "Why is this important in a practical application?",
                "follow_up_reason":
                    "The explanation can be tested at a deeper level."
            }

        return {
            "follow_up_needed": True,
            "follow_up_question":
                "Can you explain this with one simple practical example?",
            "follow_up_reason":
                "The follow-up checks practical understanding."
        }

    return {
        "follow_up_needed": True,
        "follow_up_question":
            "Can you explain that with one specific example?",
        "follow_up_reason":
            "A specific example can make the answer stronger."
    }


# ============================================================
# ADAPTIVE DIFFICULTY
# ============================================================

def recommend_next_difficulty(
    current_level: str,
    current_score: int,
    recent_scores: Optional[List[float]] = None
) -> Dict:

    current_level = normalize_level(
        current_level
    )

    current_score = clamp_score(
        current_score
    )

    recent_scores = recent_scores or []

    clean_scores = [
        clamp_score(score)
        for score in recent_scores
    ]

    if clean_scores:

        history_average = (
            sum(clean_scores)
            / len(clean_scores)
        )

        decision_score = (
            current_score * 0.60
            + history_average * 0.40
        )

    else:

        decision_score = current_score

    if current_level == "Beginner":

        if decision_score >= 80:

            next_level = "Intermediate"

            reason = (
                "Performance is strong enough to move "
                "to an intermediate question."
            )

        else:

            next_level = "Beginner"

            reason = (
                "Continue at beginner level and build "
                "a stronger answer structure."
            )

    elif current_level == "Intermediate":

        if decision_score >= 85:

            next_level = "Advanced"

            reason = (
                "Performance supports moving to an "
                "advanced interview question."
            )

        elif decision_score < 55:

            next_level = "Beginner"

            reason = (
                "A beginner-level question can help "
                "strengthen the core concept."
            )

        else:

            next_level = "Intermediate"

            reason = (
                "Continue at intermediate level to "
                "build consistency."
            )

    else:

        if decision_score < 60:

            next_level = "Intermediate"

            reason = (
                "Intermediate practice is recommended "
                "before continuing advanced questions."
            )

        else:

            next_level = "Advanced"

            reason = (
                "Continue with advanced interview practice."
            )

    return {
        "current_level": current_level,
        "next_level": next_level,
        "decision_score": round(
            decision_score,
            1
        ),
        "reason": reason,
    }


# ============================================================
# SESSION INTELLIGENCE
# ============================================================

def build_session_intelligence(
    scores: List[float]
) -> Dict:

    clean_scores = [
        clamp_score(score)
        for score in scores
    ]

    if not clean_scores:

        return {
            "attempts": 0,
            "average_score": 0,
            "trend": "Not Enough Data",
            "session_status": "New Session",
        }

    average_score = round(
        sum(clean_scores)
        / len(clean_scores),
        1
    )

    if len(clean_scores) < 2:

        trend = "Not Enough Data"

    else:

        midpoint = max(
            1,
            len(clean_scores) // 2
        )

        first_half = clean_scores[:midpoint]
        second_half = clean_scores[midpoint:]

        if not second_half:

            trend = "Not Enough Data"

        else:

            first_average = (
                sum(first_half)
                / len(first_half)
            )

            second_average = (
                sum(second_half)
                / len(second_half)
            )

            difference = (
                second_average
                - first_average
            )

            if difference >= 5:
                trend = "Improving"

            elif difference <= -5:
                trend = "Declining"

            else:
                trend = "Stable"

    if average_score >= 85:
        status = "Strong"

    elif average_score >= 70:
        status = "Developing Well"

    elif average_score >= 50:
        status = "Needs Practice"

    else:
        status = "Priority Improvement"

    return {
        "attempts": len(clean_scores),
        "average_score": average_score,
        "trend": trend,
        "session_status": status,
    }


# ============================================================
# COACH TIP
# ============================================================

def build_coach_tip(
    question_type: str,
    overall_score: int,
    signals: Dict
) -> str:

    if overall_score >= 85:

        return (
            "Keep the answer concise and support your "
            "main point with precise detail."
        )

    if question_type == "Behavioural":

        if not signals["has_action"]:
            return (
                "Clearly explain what you personally did."
            )

        if not signals["has_result"]:
            return (
                "Finish your answer with the result."
            )

        return (
            "Use a simple Situation → Action → Result flow."
        )

    if question_type == "Project":

        if not signals["has_result"]:
            return (
                "Connect your approach to a clear project result."
            )

        return (
            "Explain the problem, your approach, and the result."
        )

    if question_type == "Technical":

        return (
            "Start with the definition, explain why it matters, "
            "then add one simple example."
        )

    return (
        "Answer directly, give one supporting point, "
        "and finish clearly."
    )


# ============================================================
# NEXT ACTION
# ============================================================

def build_next_action(
    overall_score: int,
    next_level: str
) -> str:

    if overall_score >= 85:

        return (
            f"Move to the next {next_level.lower()} "
            f"interview question."
        )

    if overall_score >= 70:

        return (
            "Improve the missing point once, then continue "
            f"with {next_level.lower()} practice."
        )

    if overall_score >= 50:

        return (
            "Retry this answer once using the coaching tip "
            "before moving to the next question."
        )

    return (
        "Review the core idea and answer the same question "
        "again in a short structured format."
    )


# ============================================================
# MAIN V2 ENGINE
# ============================================================

def run_interview_intelligence_v2(
    mode: str,
    level: str,
    question: str,
    answer: str,
    recent_scores: Optional[List[float]] = None,
    existing_score: Optional[float] = None
) -> Dict:

    mode = normalize_mode(mode)
    level = normalize_level(level)

    question = clean_text(question)
    answer = clean_text(answer)

    if not question:
        raise ValueError(
            "Interview question is required."
        )

    if not answer:
        raise ValueError(
            "Interview answer is required."
        )

    # --------------------------------------------------------
    # 1. Detect question type
    # --------------------------------------------------------

    question_type = detect_question_type(
        mode,
        question
    )

    # --------------------------------------------------------
    # 2. Detect answer signals
    # --------------------------------------------------------

    signals = detect_answer_signals(
        answer
    )

    # --------------------------------------------------------
    # 3. Score answer
    # --------------------------------------------------------

    relevance = calculate_relevance_score(
        question,
        answer
    )

    completeness = calculate_completeness_score(
        answer,
        question_type,
        signals
    )

    structure = calculate_structure_score(
        answer,
        question_type,
        signals
    )

    clarity = calculate_clarity_score(
        answer
    )

    heuristic_overall = calculate_overall_score(
        relevance,
        completeness,
        structure,
        clarity
    )

    # --------------------------------------------------------
    # Existing FluentPath Interview AI score
    # --------------------------------------------------------
    #
    # If the main Interview module already produced a valid
    # AI score, use that as the primary overall interview
    # score.
    #
    # The deterministic V2 metrics remain available for:
    # - relevance
    # - completeness
    # - structure
    # - clarity
    # - weakness detection
    # - follow-up generation
    #
    # This avoids creating two competing overall scores.
    # --------------------------------------------------------

    if existing_score is not None:

        overall = clamp_score(
            existing_score
        )

        overall_score_source = (
            "Existing Interview AI Score"
        )

    else:

        overall = heuristic_overall

        overall_score_source = (
            "V2 Structural Heuristic"
        )

    # --------------------------------------------------------
    # 4. Feedback
    # --------------------------------------------------------

    feedback = build_feedback(
        question_type=question_type,
        answer=answer,
        signals=signals,
        relevance=relevance,
        completeness=completeness,
        structure=structure,
        clarity=clarity,
    )

    # --------------------------------------------------------
    # 5. Safe deterministic follow-up
    # --------------------------------------------------------

    follow_up = generate_safe_follow_up(
        question_type=question_type,
        signals=signals,
        overall_score=overall,
    )

    # --------------------------------------------------------
    # 6. Adaptive difficulty
    # --------------------------------------------------------

    previous_scores = recent_scores or []

    difficulty = recommend_next_difficulty(
        current_level=level,
        current_score=overall,
        recent_scores=previous_scores,
    )

    # --------------------------------------------------------
    # 7. Session intelligence
    # --------------------------------------------------------

    session_scores = (
        list(previous_scores)
        + [overall]
    )

    session = build_session_intelligence(
        session_scores
    )

    # --------------------------------------------------------
    # 8. Coach tip
    # --------------------------------------------------------

    coach_tip = build_coach_tip(
        question_type,
        overall,
        signals
    )

    # --------------------------------------------------------
    # 9. Next action
    # --------------------------------------------------------

    next_action = build_next_action(
        overall,
        difficulty["next_level"]
    )

    # --------------------------------------------------------
    # 10. Final structured result
    # --------------------------------------------------------

    return {
        "engine_version":
            "Interview Intelligence V2",

        "mode":
            mode,

        "question_type":
            question_type,

        "current_level":
            level,

        "relevance_score":
            relevance,

        "completeness_score":
            completeness,

        "structure_score":
            structure,

        "clarity_score":
            clarity,

        "overall_score":
            overall,

        "overall_score_source":
            overall_score_source,

        "heuristic_overall_score":
            heuristic_overall,

        "existing_ai_score":
            (
                clamp_score(existing_score)
                if existing_score is not None
                else None
            ),

        "strengths":
            feedback["strengths"],

        "weaknesses":
            feedback["weaknesses"],

        "missing_points":
            feedback["missing_points"],

        "coach_tip":
            coach_tip,

        "follow_up_needed":
            follow_up["follow_up_needed"],

        "follow_up_question":
            follow_up["follow_up_question"],

        "follow_up_reason":
            follow_up["follow_up_reason"],

        "next_difficulty":
            difficulty["next_level"],

        "difficulty_decision_score":
            difficulty["decision_score"],

        "difficulty_reason":
            difficulty["reason"],

        "session_attempts":
            session["attempts"],

        "session_average":
            session["average_score"],

        "session_trend":
            session["trend"],

        "session_status":
            session["session_status"],

        "next_action":
            next_action,

        "signals":
            signals,
    }


# ============================================================
# COMPATIBILITY WRAPPER
# ============================================================

def run_interview_intelligence(
    mode: str,
    level: str,
    question: str,
    answer: str,
    recent_scores: Optional[List[float]] = None,
    existing_score: Optional[float] = None
) -> Dict:

    """
    Compatibility wrapper.

    Main app can later call:
    run_interview_intelligence(...)

    without depending on the V2 function name.
    """

    return run_interview_intelligence_v2(
        mode=mode,
        level=level,
        question=question,
        answer=answer,
        recent_scores=recent_scores,
        existing_score=existing_score,
    )


# ============================================================
# SELF TEST
# ============================================================

if __name__ == "__main__":

    print(
        "INTERVIEW INTELLIGENCE V2 TEST"
    )

    print("=" * 70)

    sample_question = (
        "Tell me about a challenging problem "
        "you solved."
    )

    sample_answer = (
        "In one project we had a delay because "
        "a required activity was not progressing "
        "as planned. I first checked the reason "
        "for the delay, discussed the issue with "
        "the team, and reorganized the immediate "
        "work priorities. We completed the "
        "critical activity and reduced further "
        "delay."
    )

    try:

        result = run_interview_intelligence_v2(
            mode="HR",
            level="Beginner",
            question=sample_question,
            answer=sample_answer,
            recent_scores=[68, 74],
            existing_score=82,
        )

        print(
            "INTERVIEW INTELLIGENCE V2: PASS"
        )

        print()

        print(
            "Engine:",
            result["engine_version"]
        )

        print(
            "Mode:",
            result["mode"]
        )

        print(
            "Question Type:",
            result["question_type"]
        )

        print(
            "Current Level:",
            result["current_level"]
        )

        print()

        print(
            "Relevance:",
            result["relevance_score"]
        )

        print(
            "Completeness:",
            result["completeness_score"]
        )

        print(
            "Structure:",
            result["structure_score"]
        )

        print(
            "Clarity:",
            result["clarity_score"]
        )

        print(
            "Overall Score:",
            result["overall_score"]
        )

        print(
            "Overall Score Source:",
            result["overall_score_source"]
        )

        print(
            "Structural Heuristic Score:",
            result["heuristic_overall_score"]
        )

        print(
            "Existing AI Score:",
            result["existing_ai_score"]
        )

        print()

        print(
            "Strengths:",
            result["strengths"]
        )

        print(
            "Weaknesses:",
            result["weaknesses"]
        )

        print(
            "Missing Points:",
            result["missing_points"]
        )

        print()

        print(
            "Coach Tip:",
            result["coach_tip"]
        )

        print()

        print(
            "Follow-up Needed:",
            result["follow_up_needed"]
        )

        print(
            "Follow-up Question:",
            result["follow_up_question"]
        )

        print(
            "Follow-up Reason:",
            result["follow_up_reason"]
        )

        print()

        print(
            "Next Difficulty:",
            result["next_difficulty"]
        )

        print(
            "Difficulty Decision Score:",
            result[
                "difficulty_decision_score"
            ]
        )

        print()

        print(
            "Session Attempts:",
            result["session_attempts"]
        )

        print(
            "Session Average:",
            result["session_average"]
        )

        print(
            "Session Trend:",
            result["session_trend"]
        )

        print(
            "Session Status:",
            result["session_status"]
        )

        print()

        print(
            "NEXT ACTION:"
        )

        print(
            result["next_action"]
        )

        print()

        # Critical safety test
        follow_up_text = clean_text(
            result["follow_up_question"]
        ).lower()

        if (
            result["follow_up_needed"]
            and not follow_up_text
        ):
            raise RuntimeError(
                "Follow-up required but question is empty."
            )

        if "user safety" in follow_up_text:
            raise RuntimeError(
                "Invalid safety text entered follow-up output."
            )

        print(
            "FOLLOW-UP VALIDATION: PASS"
        )

        # Existing Interview AI score must remain
        # the primary overall score when supplied.
        if result["overall_score"] != 82:
            raise RuntimeError(
                "Existing Interview AI score "
                "was not preserved."
            )

        if (
            result["overall_score_source"]
            != "Existing Interview AI Score"
        ):
            raise RuntimeError(
                "Overall score source is incorrect."
            )

        if result["existing_ai_score"] != 82:
            raise RuntimeError(
                "Existing AI score metadata "
                "is incorrect."
            )

        print(
            "EXISTING AI SCORE VALIDATION: PASS"
        )

    except Exception as e:

        print(
            "INTERVIEW INTELLIGENCE V2: FAILED"
        )

        print(
            type(e).__name__
        )

        print(
            str(e)
        )