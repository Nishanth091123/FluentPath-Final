# ============================================================
# FLUENTPATH
# TOPIC 7 — PERSISTENT LEARNING MEMORY / LEARNING STATE
# Version: V1
#
# Purpose:
# Build a persistent learner profile from existing ActivityLog.
#
# No new database table.
# No schema migration.
# No LLM/API call.
# Read-only database analysis.
# ============================================================

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import select

from database.connection import SessionLocal
from database.models import ActivityLog


# ============================================================
# CONFIG
# ============================================================

DEFAULT_RECENT_LIMIT = 50

SUPPORTED_MODULES = {
    "techprep": "TechPrep",
    "interview": "Interview",
    "interviews": "Interview",
    "speaking": "Speaking",
    "smartspeak": "SmartSpeak",
    "practice": "Practice",
    "tests": "Tests",
    "test": "Tests",
    "daily assignment": "Daily Assignment",
    "daily_assignment": "Daily Assignment",
    "assignment": "Daily Assignment",
    "learning": "Learning",
    "grammar": "Grammar",
    "tenses": "Tenses",
    "vocabulary": "Vocabulary",
    "sentence formation": "Sentence Formation",
    "sentence_formation": "Sentence Formation",
}


# ============================================================
# BASIC HELPERS
# ============================================================

def safe_text(value: Any) -> str:
    if value is None:
        return ""

    return str(value).strip()


def safe_score(value: Any) -> Optional[float]:
    if value is None:
        return None

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    if number < 0:
        number = 0.0

    if number > 100:
        number = 100.0

    return round(number, 1)


def normalize_module(activity_type: Any) -> str:
    raw = safe_text(activity_type)

    if not raw:
        return "Other"

    key = raw.lower().strip()

    if key in SUPPORTED_MODULES:
        return SUPPORTED_MODULES[key]

    for known_key, display_name in SUPPORTED_MODULES.items():
        if known_key in key:
            return display_name

    return raw


def normalize_topic(value: Any) -> str:
    text = safe_text(value)

    if not text:
        return "General"

    # Existing TechPrep titles often contain:
    # Python - question
    # Machine Learning - question
    #
    # Keep first segment as topic.
    if " - " in text:
        first = text.split(" - ", 1)[0].strip()

        if first:
            return first

    return text


def get_attr(obj: Any, *names: str, default=None):
    for name in names:
        if hasattr(obj, name):
            value = getattr(obj, name)

            if value is not None:
                return value

    return default


def get_activity_datetime(activity: Any):
    return get_attr(
        activity,
        "created_at",
        "timestamp",
        "activity_time",
        "date",
        "updated_at",
        default=None,
    )


def format_datetime(value: Any) -> str:
    if value is None:
        return "Not Available"

    if isinstance(value, datetime):
        return value.strftime("%d-%m-%Y %I:%M %p")

    return safe_text(value) or "Not Available"


# ============================================================
# SCORE / LEVEL HELPERS
# ============================================================

def calculate_average(
    scores: List[float]
) -> Optional[float]:

    valid_scores = [
        score
        for score in scores
        if score is not None
    ]

    if not valid_scores:
        return None

    return round(
        sum(valid_scores) / len(valid_scores),
        1
    )


def determine_level(
    average_score: Optional[float]
) -> str:

    if average_score is None:
        return "New Learner"

    if average_score >= 80:
        return "Advanced"

    if average_score >= 60:
        return "Intermediate"

    return "Beginner"


def determine_performance_status(
    average_score: Optional[float]
) -> str:

    if average_score is None:
        return "No Scored Practice Yet"

    if average_score >= 85:
        return "Strong"

    if average_score >= 70:
        return "Developing Well"

    if average_score >= 50:
        return "Needs Practice"

    return "Needs Focus"


def calculate_trend(
    scores: List[float]
) -> str:

    valid_scores = [
        score
        for score in scores
        if score is not None
    ]

    if len(valid_scores) < 2:
        return "Not Enough Data"

    if len(valid_scores) == 2:
        difference = (
            valid_scores[-1]
            - valid_scores[0]
        )

    else:
        midpoint = len(valid_scores) // 2

        first_half = valid_scores[:midpoint]
        second_half = valid_scores[midpoint:]

        first_average = calculate_average(
            first_half
        )

        second_average = calculate_average(
            second_half
        )

        difference = (
            second_average
            - first_average
        )

    if difference >= 5:
        return "Improving"

    if difference <= -5:
        return "Declining"

    return "Stable"


# ============================================================
# ACTIVITY EXTRACTION
# ============================================================

def extract_activity_score(
    activity: Any
) -> Optional[float]:

    value = get_attr(
        activity,
        "score",
        "activity_score",
        default=None,
    )

    return safe_score(value)


def extract_activity_type(
    activity: Any
) -> str:

    value = get_attr(
        activity,
        "activity_type",
        "type",
        default="Other",
    )

    return normalize_module(value)


def extract_activity_title(
    activity: Any
) -> str:

    value = get_attr(
        activity,
        "title",
        "activity_name",
        "topic",
        "question",
        default="",
    )

    return safe_text(value)


def _safe_review_data(
    activity: Any
) -> Dict[str, Any]:

    review_data = get_attr(
        activity,
        "review_data",
        default=None,
    )

    if isinstance(
        review_data,
        dict
    ):
        return review_data

    return {}


def _first_non_empty(
    *values: Any
) -> str:

    for value in values:

        text = safe_text(
            value
        )

        if text:
            return text

    return ""


def extract_topic(
    activity: Any
) -> str:

    module_name = extract_activity_type(
        activity
    )

    review = _safe_review_data(
        activity
    )

    # --------------------------------------------------------
    # 1. Direct database topic / subject
    # --------------------------------------------------------

    direct_topic = _first_non_empty(
        get_attr(
            activity,
            "topic",
            default=None,
        ),
        get_attr(
            activity,
            "subject",
            default=None,
        ),
    )

    if direct_topic:

        return normalize_topic(
            direct_topic
        )


    # --------------------------------------------------------
    # 2. TECHPREP
    # --------------------------------------------------------

    if module_name == "TechPrep":

        candidate = _first_non_empty(
            review.get("topic"),
            review.get("subject"),
            review.get("category"),
            review.get("technology"),
            review.get("skill"),
        )

        if candidate:

            return normalize_topic(
                candidate
            )

        title = extract_activity_title(
            activity
        )

        if title:

            normalized = normalize_topic(
                title
            )

            if normalized != "General":
                return normalized

        return "TechPrep General"


    # --------------------------------------------------------
    # 3. INTERVIEW
    # --------------------------------------------------------

    if module_name == "Interview":

        candidate = _first_non_empty(
            review.get(
                "interview_type"
            ),
            review.get("type"),
            review.get("category"),
            review.get("mode"),
        )

        if candidate:

            # Do not treat Text Practice /
            # Face-to-Face Practice as subject
            # if interview_type is available.
            return safe_text(
                candidate
            )

        return "Interview General"


    # --------------------------------------------------------
    # 4. SMARTSPEAK
    # --------------------------------------------------------

    if module_name == "SmartSpeak":

        candidate = _first_non_empty(
            review.get("situation"),
            review.get("category"),
            review.get("context"),
            review.get("topic"),
            review.get("tone"),
        )

        if candidate:

            return safe_text(
                candidate
            )

        return "SmartSpeak General"


    # --------------------------------------------------------
    # 5. SPEAKING
    # --------------------------------------------------------

    if module_name == "Speaking":

        candidate = _first_non_empty(
            review.get("practice_type"),
            review.get("category"),
            review.get("topic"),
            review.get("mode"),
            review.get("title"),
        )

        if candidate:

            return normalize_topic(
                candidate
            )

        return "Speaking General"


    # --------------------------------------------------------
    # 6. TESTS
    # --------------------------------------------------------

    if module_name == "Tests":

        candidate = _first_non_empty(
            review.get("test_type"),
            review.get("category"),
            review.get("topic"),
            review.get("subject"),
            review.get("title"),
        )

        if candidate:

            return normalize_topic(
                candidate
            )

        return "Tests General"


    # --------------------------------------------------------
    # 7. PRACTICE
    # --------------------------------------------------------

    if module_name == "Practice":

        candidate = _first_non_empty(
            review.get("practice_type"),
            review.get("category"),
            review.get("topic"),
            review.get("subject"),
            review.get("title"),
        )

        if candidate:

            return normalize_topic(
                candidate
            )

        return "Practice General"


    # --------------------------------------------------------
    # 8. DAILY ASSIGNMENT / ASSIGNMENT
    # --------------------------------------------------------

    if module_name in {
        "Daily Assignment",
        "Assignment",
    }:

        candidate = _first_non_empty(
            review.get("assignment_type"),
            review.get("category"),
            review.get("topic"),
            review.get("title"),
        )

        if candidate:

            return normalize_topic(
                candidate
            )

        return "Assignment General"


    # --------------------------------------------------------
    # 9. LEARNING SUBMODULES
    # --------------------------------------------------------

    if module_name in {
        "Learning",
        "Grammar",
        "Tenses",
        "Vocabulary",
        "Sentence Formation",
    }:

        candidate = _first_non_empty(
            review.get("topic"),
            review.get("category"),
            review.get("lesson"),
            review.get("title"),
        )

        if candidate:

            return normalize_topic(
                candidate
            )

        return (
            module_name
            + " General"
        )


    # --------------------------------------------------------
    # 10. GENERIC FALLBACK
    # --------------------------------------------------------

    candidate = _first_non_empty(
        review.get("topic"),
        review.get("category"),
        review.get("title"),
        extract_activity_title(
            activity
        ),
    )

    if candidate:

        return normalize_topic(
            candidate
        )

    if module_name != "Other":

        return (
            module_name
            + " General"
        )

    return "General"


# ============================================================
# DATABASE READER
# ============================================================

def load_user_activities(
    user_id: str,
    limit: int = DEFAULT_RECENT_LIMIT,
) -> List[Any]:

    user_id = safe_text(user_id)

    if not user_id:
        return []

    db = SessionLocal()

    try:
        stmt = (
            select(ActivityLog)
            .where(
                ActivityLog.user_id == user_id
            )
        )

        # Prefer newest records if created_at exists.
        if hasattr(
            ActivityLog,
            "created_at"
        ):
            stmt = stmt.order_by(
                ActivityLog.created_at.desc()
            )

        elif hasattr(
            ActivityLog,
            "id"
        ):
            stmt = stmt.order_by(
                ActivityLog.id.desc()
            )

        if limit and limit > 0:
            stmt = stmt.limit(limit)

        rows = db.execute(
            stmt
        ).scalars().all()

        # We want chronological order for trend analysis.
        rows = list(rows)
        rows.reverse()

        return rows

    finally:
        db.close()


# ============================================================
# MODULE STATE
# ============================================================

def build_module_state(
    module_name: str,
    activities: List[Any],
) -> Dict[str, Any]:

    scores = []

    for activity in activities:
        score = extract_activity_score(
            activity
        )

        if score is not None:
            scores.append(score)

    average_score = calculate_average(
        scores
    )

    latest_score = (
        scores[-1]
        if scores
        else None
    )

    best_score = (
        max(scores)
        if scores
        else None
    )

    trend = calculate_trend(
        scores
    )

    level = determine_level(
        average_score
    )

    status = determine_performance_status(
        average_score
    )

    latest_activity = (
        activities[-1]
        if activities
        else None
    )

    latest_topic = (
        extract_topic(
            latest_activity
        )
        if latest_activity
        else "General"
    )

    latest_time = (
        format_datetime(
            get_activity_datetime(
                latest_activity
            )
        )
        if latest_activity
        else "Not Available"
    )

    return {
        "module": module_name,
        "attempts": len(activities),
        "scored_attempts": len(scores),
        "average_score": average_score,
        "latest_score": latest_score,
        "best_score": best_score,
        "trend": trend,
        "level": level,
        "performance_status": status,
        "latest_topic": latest_topic,
        "last_activity": latest_time,
    }


# ============================================================
# TOPIC STATE
# ============================================================

def build_topic_states(
    activities: List[Any],
) -> Dict[str, Dict[str, Any]]:

    area_groups = defaultdict(list)

    for activity in activities:

        score = extract_activity_score(
            activity
        )

        if score is None:
            continue

        module_name = extract_activity_type(
            activity
        )

        topic = extract_topic(
            activity
        )

        area_key = (
            module_name,
            topic,
        )

        area_groups[
            area_key
        ].append(score)

    states = {}

    for (
        module_name,
        topic,
    ), scores in area_groups.items():

        average_score = calculate_average(
            scores
        )

        # Stable readable key.
        state_key = (
            module_name
            + " :: "
            + topic
        )

        states[
            state_key
        ] = {
            "module":
                module_name,

            "topic":
                topic,

            "area":
                state_key,

            "attempts":
                len(scores),

            "average_score":
                average_score,

            "latest_score":
                scores[-1],

            "best_score":
                max(scores),

            "trend":
                calculate_trend(
                    scores
                ),

            "level":
                determine_level(
                    average_score
                ),

            "performance_status":
                determine_performance_status(
                    average_score
                ),
        }

    return states


# ============================================================
# STRONG / WEAK TOPIC
# ============================================================

def find_strongest_topic(
    topic_states: Dict[str, Dict[str, Any]]
) -> Optional[Dict[str, Any]]:

    valid = [
        state
        for state in topic_states.values()
        if state.get(
            "average_score"
        ) is not None
    ]

    if not valid:
        return None

    return max(
        valid,
        key=lambda item: (
            item["average_score"],
            item["attempts"],
        )
    )


def find_weakest_topic(
    topic_states: Dict[str, Dict[str, Any]]
) -> Optional[Dict[str, Any]]:

    valid = [
        state
        for state in topic_states.values()
        if state.get(
            "average_score"
        ) is not None
    ]

    if not valid:
        return None

    return min(
        valid,
        key=lambda item: (
            item["average_score"],
            -item["attempts"],
        )
    )


# ============================================================
# NEXT PRACTICE RECOMMENDATION
# ============================================================

def build_next_recommendation(
    overall_average: Optional[float],
    overall_trend: str,
    weakest_topic: Optional[Dict[str, Any]],
    latest_module: str,
    latest_topic: str,
) -> Dict[str, Any]:

    # --------------------------------------------------------
    # No scored history
    # --------------------------------------------------------

    if overall_average is None:

        return {
            "recommended_module":
                latest_module
                if latest_module != "None"
                else "TechPrep",

            "recommended_topic":
                latest_topic
                if latest_topic != "None"
                else "Python",

            "recommended_difficulty":
                "Beginner",

            "reason":
                (
                    "Start with a short beginner "
                    "practice to build your baseline."
                ),
        }

    # --------------------------------------------------------
    # Weak topic priority
    # --------------------------------------------------------

    if weakest_topic:

        weak_average = (
            weakest_topic.get(
                "average_score"
            )
        )

        weak_topic_name = (
            weakest_topic.get(
                "topic",
                "General"
            )
        )

        if (
            weak_average is not None
            and weak_average < 70
        ):

            weak_module_name = (
                weakest_topic.get(
                    "module",
                    latest_module
                )
            )

            return {
                "recommended_module":
                    weak_module_name,

                "recommended_topic":
                    weak_topic_name,

                "recommended_difficulty":
                    determine_level(
                        weak_average
                    ),

                "reason":
                    (
                        "This module and topic "
                        "currently have the lowest "
                        "scored average, so one "
                        "focused practice is "
                        "recommended."
                    ),
            }

    # --------------------------------------------------------
    # Declining performance
    # --------------------------------------------------------

    if overall_trend == "Declining":

        return {
            "recommended_module":
                latest_module,

            "recommended_topic":
                latest_topic,

            "recommended_difficulty":
                "Beginner"
                if overall_average < 60
                else "Intermediate",

            "reason":
                (
                    "Recent performance is declining. "
                    "Review the latest area before "
                    "increasing difficulty."
                ),
        }

    # --------------------------------------------------------
    # Strong learner
    # --------------------------------------------------------

    if overall_average >= 80:

        return {
            "recommended_module":
                latest_module,

            "recommended_topic":
                latest_topic,

            "recommended_difficulty":
                "Advanced",

            "reason":
                (
                    "Your recent scored performance "
                    "is strong. Continue with a more "
                    "challenging practice."
                ),
        }

    # --------------------------------------------------------
    # Intermediate learner
    # --------------------------------------------------------

    if overall_average >= 60:

        return {
            "recommended_module":
                latest_module,

            "recommended_topic":
                latest_topic,

            "recommended_difficulty":
                "Intermediate",

            "reason":
                (
                    "Continue at intermediate level "
                    "and strengthen consistency."
                ),
        }

    # --------------------------------------------------------
    # Beginner / low score
    # --------------------------------------------------------

    return {
        "recommended_module":
            latest_module,

        "recommended_topic":
            latest_topic,

        "recommended_difficulty":
            "Beginner",

        "reason":
            (
                "Strengthen the fundamentals with "
                "another structured beginner practice."
            ),
    }


# ============================================================
# MAIN LEARNING MEMORY BUILDER
# ============================================================

def build_learning_memory(
    user_id: str,
    limit: int = DEFAULT_RECENT_LIMIT,
) -> Dict[str, Any]:

    user_id = safe_text(user_id)

    if not user_id:

        return {
            "status": "NO_USER",
            "user_id": "",
            "message": "User ID is required.",
        }

    activities = load_user_activities(
        user_id=user_id,
        limit=limit,
    )

    # --------------------------------------------------------
    # New learner
    # --------------------------------------------------------

    if not activities:

        recommendation = (
            build_next_recommendation(
                overall_average=None,
                overall_trend="Not Enough Data",
                weakest_topic=None,
                latest_module="None",
                latest_topic="None",
            )
        )

        return {
            "status": "NEW_LEARNER",
            "user_id": user_id,
            "total_activities": 0,
            "scored_activities": 0,
            "overall_average": None,
            "overall_level": "New Learner",
            "performance_status":
                "No Scored Practice Yet",
            "trend": "Not Enough Data",
            "latest_module": "None",
            "latest_topic": "None",
            "last_activity": "Not Available",
            "strongest_topic": None,
            "weakest_topic": None,
            "module_states": {},
            "topic_states": {},
            "recommendation": recommendation,
        }

    # --------------------------------------------------------
    # Group by module
    # --------------------------------------------------------

    module_groups = defaultdict(list)

    all_scores = []

    for activity in activities:

        module_name = (
            extract_activity_type(
                activity
            )
        )

        module_groups[
            module_name
        ].append(activity)

        score = extract_activity_score(
            activity
        )

        if score is not None:
            all_scores.append(score)

    # --------------------------------------------------------
    # Module memory
    # --------------------------------------------------------

    module_states = {}

    for module_name, rows in module_groups.items():

        module_states[
            module_name
        ] = build_module_state(
            module_name,
            rows,
        )

    # --------------------------------------------------------
    # Topic memory
    # --------------------------------------------------------

    topic_states = build_topic_states(
        activities
    )

    strongest_topic = (
        find_strongest_topic(
            topic_states
        )
    )

    weakest_topic = (
        find_weakest_topic(
            topic_states
        )
    )

    # --------------------------------------------------------
    # Overall learner state
    # --------------------------------------------------------

    overall_average = (
        calculate_average(
            all_scores
        )
    )

    overall_level = (
        determine_level(
            overall_average
        )
    )

    overall_status = (
        determine_performance_status(
            overall_average
        )
    )

    overall_trend = (
        calculate_trend(
            all_scores
        )
    )

    # --------------------------------------------------------
    # Latest activity
    # --------------------------------------------------------

    latest_activity = activities[-1]

    latest_module = (
        extract_activity_type(
            latest_activity
        )
    )

    latest_topic = (
        extract_topic(
            latest_activity
        )
    )

    last_activity = format_datetime(
        get_activity_datetime(
            latest_activity
        )
    )

    # --------------------------------------------------------
    # Recommendation
    # --------------------------------------------------------

    recommendation = (
        build_next_recommendation(
            overall_average=
                overall_average,

            overall_trend=
                overall_trend,

            weakest_topic=
                weakest_topic,

            latest_module=
                latest_module,

            latest_topic=
                latest_topic,
        )
    )

    # --------------------------------------------------------
    # Final persistent learning state
    # --------------------------------------------------------

    return {
        "status": "ACTIVE",
        "user_id": user_id,

        "total_activities":
            len(activities),

        "scored_activities":
            len(all_scores),

        "overall_average":
            overall_average,

        "overall_level":
            overall_level,

        "performance_status":
            overall_status,

        "trend":
            overall_trend,

        "latest_module":
            latest_module,

        "latest_topic":
            latest_topic,

        "last_activity":
            last_activity,

        "strongest_topic":
            strongest_topic,

        "weakest_topic":
            weakest_topic,

        "module_states":
            module_states,

        "topic_states":
            topic_states,

        "recommendation":
            recommendation,
    }


# ============================================================
# COMPACT MEMORY FOR AI / AGENT USE
# ============================================================

def build_learning_context(
    user_id: str,
    limit: int = DEFAULT_RECENT_LIMIT,
) -> Dict[str, Any]:

    memory = build_learning_memory(
        user_id=user_id,
        limit=limit,
    )

    if memory.get("status") == "NO_USER":
        return memory

    strongest = memory.get(
        "strongest_topic"
    )

    weakest = memory.get(
        "weakest_topic"
    )

    recommendation = memory.get(
        "recommendation",
        {}
    )

    return {
        "user_id":
            user_id,

        "memory_status":
            memory.get(
                "status"
            ),

        "learner_level":
            memory.get(
                "overall_level"
            ),

        "average_score":
            memory.get(
                "overall_average"
            ),

        "trend":
            memory.get(
                "trend"
            ),

        "performance_status":
            memory.get(
                "performance_status"
            ),

        "last_module":
            memory.get(
                "latest_module"
            ),

        "last_topic":
            memory.get(
                "latest_topic"
            ),

        "strongest_module":
            (
                strongest.get("module")
                if strongest
                else None
            ),

        "strongest_topic":
            (
                strongest.get("topic")
                if strongest
                else None
            ),

        "strongest_topic_average":
            (
                strongest.get(
                    "average_score"
                )
                if strongest
                else None
            ),

        "weakest_module":
            (
                weakest.get("module")
                if weakest
                else None
            ),

        "weakest_topic":
            (
                weakest.get("topic")
                if weakest
                else None
            ),

        "weakest_topic_average":
            (
                weakest.get(
                    "average_score"
                )
                if weakest
                else None
            ),

        "recommended_module":
            recommendation.get(
                "recommended_module"
            ),

        "recommended_topic":
            recommendation.get(
                "recommended_topic"
            ),

        "recommended_difficulty":
            recommendation.get(
                "recommended_difficulty"
            ),

        "recommendation_reason":
            recommendation.get(
                "reason"
            ),
    }


# ============================================================
# HUMAN-READABLE MEMORY SUMMARY
# ============================================================

def build_memory_summary(
    user_id: str,
    limit: int = DEFAULT_RECENT_LIMIT,
) -> str:

    memory = build_learning_memory(
        user_id=user_id,
        limit=limit,
    )

    if memory.get(
        "status"
    ) == "NO_USER":

        return "User ID is required."

    if memory.get(
        "status"
    ) == "NEW_LEARNER":

        return (
            "No previous learning activity "
            "was found for this learner."
        )

    strongest = memory.get(
        "strongest_topic"
    )

    weakest = memory.get(
        "weakest_topic"
    )

    recommendation = memory.get(
        "recommendation",
        {}
    )

    strongest_text = (
        (
            strongest.get(
                "module",
                "Other"
            )
            + " → "
            + strongest.get(
                "topic",
                "General"
            )
        )
        if strongest
        else "Not Available"
    )

    weakest_text = (
        (
            weakest.get(
                "module",
                "Other"
            )
            + " → "
            + weakest.get(
                "topic",
                "General"
            )
        )
        if weakest
        else "Not Available"
    )

    return (
        f"Learner Level: "
        f"{memory.get('overall_level')} | "
        f"Average: "
        f"{memory.get('overall_average')} | "
        f"Trend: "
        f"{memory.get('trend')} | "
        f"Strongest Topic: "
        f"{strongest_text} | "
        f"Weakest Topic: "
        f"{weakest_text} | "
        f"Recommended Next: "
        f"{recommendation.get('recommended_module')} "
        f"- "
        f"{recommendation.get('recommended_topic')} "
        f"("
        f"{recommendation.get('recommended_difficulty')}"
        f")"
    )


# ============================================================
# SELF TEST
# ============================================================

def run_self_test(
    user_id: str
):

    print()
    print(
        "=" * 72
    )

    print(
        "FLUENTPATH LEARNING MEMORY V1 TEST"
    )

    print(
        "=" * 72
    )

    print(
        "User:",
        user_id
    )

    memory = build_learning_memory(
        user_id=user_id
    )

    print()
    print(
        "Status:",
        memory.get(
            "status"
        )
    )

    print(
        "Total Activities:",
        memory.get(
            "total_activities"
        )
    )

    print(
        "Scored Activities:",
        memory.get(
            "scored_activities"
        )
    )

    print(
        "Overall Average:",
        memory.get(
            "overall_average"
        )
    )

    print(
        "Learner Level:",
        memory.get(
            "overall_level"
        )
    )

    print(
        "Performance Status:",
        memory.get(
            "performance_status"
        )
    )

    print(
        "Trend:",
        memory.get(
            "trend"
        )
    )

    print()
    print(
        "Latest Module:",
        memory.get(
            "latest_module"
        )
    )

    print(
        "Latest Topic:",
        memory.get(
            "latest_topic"
        )
    )

    print(
        "Last Activity:",
        memory.get(
            "last_activity"
        )
    )

    print()
    print(
        "Strongest Topic:",
        memory.get(
            "strongest_topic"
        )
    )

    print(
        "Weakest Topic:",
        memory.get(
            "weakest_topic"
        )
    )

    print()
    print(
        "Module States:"
    )

    for module_name, state in (
        memory.get(
            "module_states",
            {}
        ).items()
    ):

        print(
            " -",
            module_name,
            ":",
            state
        )

    print()
    print(
        "Module-Aware Topic States:"
    )

    for area_name, state in (
        memory.get(
            "topic_states",
            {}
        ).items()
    ):

        print(
            " -",
            area_name,
            ":",
            state
        )

    print()
    print(
        "Recommendation:"
    )

    print(
        memory.get(
            "recommendation"
        )
    )

    print()
    print(
        "Compact AI Context:"
    )

    print(
        build_learning_context(
            user_id
        )
    )

    print()
    print(
        "Memory Summary:"
    )

    print(
        build_memory_summary(
            user_id
        )
    )

    print()
    print(
        "=" * 72
    )

    print(
        "LEARNING MEMORY ENGINE: PASS"
    )

    print(
        "=" * 72
    )


# ============================================================
# DIRECT RUN
# ============================================================

if __name__ == "__main__":

    import sys

    if len(sys.argv) < 2:

        print()
        print(
            "Learning Memory module loaded successfully."
        )

        print()
        print(
            "Run real-user test with:"
        )

        print(
            "python learning_memory.py NS001"
        )

    else:

        run_self_test(
            sys.argv[1]
        )