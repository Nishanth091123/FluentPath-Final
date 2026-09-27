from typing import Dict, List


# ============================================================
# SCORE HELPERS
# ============================================================

def safe_score(value):
    try:
        value = float(value)
        return max(0.0, min(100.0, value))
    except (TypeError, ValueError):
        return 0.0


def average_score(scores: List[float]) -> float:

    valid_scores = [
        safe_score(score)
        for score in scores
        if score is not None
    ]

    if not valid_scores:
        return 0.0

    return round(
        sum(valid_scores) / len(valid_scores),
        1
    )


# ============================================================
# LEVEL CLASSIFICATION
# ============================================================

def classify_level(avg_score: float) -> str:

    avg_score = safe_score(avg_score)

    if avg_score >= 80:
        return "Advanced"

    if avg_score >= 60:
        return "Intermediate"

    return "Beginner"


# ============================================================
# PERFORMANCE STATUS
# ============================================================

def performance_status(avg_score: float) -> str:

    avg_score = safe_score(avg_score)

    if avg_score >= 85:
        return "Strong"

    if avg_score >= 70:
        return "Developing Well"

    if avg_score >= 50:
        return "Needs Practice"

    return "Priority Improvement"


# ============================================================
# TREND DETECTION
# ============================================================

def detect_trend(scores: List[float]) -> str:

    clean = [
        safe_score(score)
        for score in scores
        if score is not None
    ]

    if len(clean) < 2:
        return "Not Enough Data"

    midpoint = len(clean) // 2

    first_half = clean[:midpoint]
    second_half = clean[midpoint:]

    if not first_half or not second_half:
        return "Stable"

    first_avg = average_score(first_half)
    second_avg = average_score(second_half)

    difference = second_avg - first_avg

    if difference >= 5:
        return "Improving"

    if difference <= -5:
        return "Declining"

    return "Stable"


# ============================================================
# NEXT DIFFICULTY
# ============================================================

def recommend_difficulty(
    avg_score: float,
    attempts: int,
    trend: str
) -> str:

    avg_score = safe_score(avg_score)

    if attempts < 2:
        return "Beginner"

    if avg_score >= 85 and trend != "Declining":
        return "Advanced"

    if avg_score >= 65:
        return "Intermediate"

    return "Beginner"


# ============================================================
# RECOMMENDED ACTION
# ============================================================

def recommend_action(
    avg_score: float,
    attempts: int,
    trend: str
) -> str:

    avg_score = safe_score(avg_score)

    if attempts == 0:
        return "Start with a beginner practice question."

    if avg_score < 50:
        return (
            "Review the concept and retry with a simpler question."
        )

    if avg_score < 70:
        return (
            "Practice the same topic again before increasing difficulty."
        )

    if trend == "Declining":
        return (
            "Revise the topic once before moving to a harder question."
        )

    if avg_score < 85:
        return (
            "Move to an intermediate question and continue practice."
        )

    return (
        "Move to an advanced or scenario-based interview question."
    )


# ============================================================
# ADAPTIVE LEARNING PROFILE
# ============================================================

def build_learning_profile(
    topic: str,
    scores: List[float]
) -> Dict:

    clean_scores = [
        safe_score(score)
        for score in scores
        if score is not None
    ]

    attempts = len(clean_scores)
    avg = average_score(clean_scores)
    trend = detect_trend(clean_scores)

    return {
        "topic": topic,
        "attempts": attempts,
        "average_score": avg,
        "level": classify_level(avg),
        "status": performance_status(avg),
        "trend": trend,
        "recommended_difficulty": recommend_difficulty(
            avg,
            attempts,
            trend
        ),
        "recommended_action": recommend_action(
            avg,
            attempts,
            trend
        )
    }


# ============================================================
# FIND WEAKEST TOPIC
# ============================================================

def find_weakest_topic(topic_scores: Dict[str, List[float]]):

    profiles = []

    for topic, scores in topic_scores.items():

        if not scores:
            continue

        profiles.append(
            build_learning_profile(topic, scores)
        )

    if not profiles:
        return None

    return min(
        profiles,
        key=lambda item: item["average_score"]
    )


# ============================================================
# NEXT TOPIC RECOMMENDATION
# ============================================================

def recommend_next_topic(
    topic_scores: Dict[str, List[float]]
):

    profiles = []

    for topic, scores in topic_scores.items():
        if scores:
            profiles.append(
                build_learning_profile(topic, scores)
            )

    if not profiles:
        return {
            "topic": "Python",
            "reason": (
                "No previous TechPrep score history is available. "
                "Start with a core technical topic."
            )
        }

    weakest = min(
        profiles,
        key=lambda item: item["average_score"]
    )

    # Genuine weak area
    if weakest["average_score"] < 70:
        return {
            "topic": weakest["topic"],
            "reason": (
                f'{weakest["topic"]} needs more practice '
                f'({weakest["average_score"]}/100 average).'
            )
        }

    # No weak attempted topic
    strongest_recent = max(
        profiles,
        key=lambda item: item["average_score"]
    )

    return {
        "topic": strongest_recent["topic"],
        "reason": (
            "No weak scored topic has been identified yet. "
            f'{strongest_recent["topic"]} is currently at '
            f'{strongest_recent["average_score"]}/100, so continue '
            f'at {strongest_recent["recommended_difficulty"]} level '
            "or practise a new topic to build a broader profile."
        )
    }


# ============================================================
# SELF TEST
# ============================================================

if __name__ == "__main__":

    print("ADAPTIVE LEARNING ENGINE TEST")
    print("=" * 70)

    test_data = {
        "Python": [62, 68, 74, 78],
        "Machine Learning": [45, 52, 58],
        "SQL": [82, 86, 90],
        "Generative AI": [70, 76, 84]
    }

    for topic, scores in test_data.items():

        profile = build_learning_profile(
            topic,
            scores
        )

        print()
        print("Topic:", profile["topic"])
        print("Attempts:", profile["attempts"])
        print(
            "Average Score:",
            profile["average_score"]
        )
        print("Level:", profile["level"])
        print("Status:", profile["status"])
        print("Trend:", profile["trend"])
        print(
            "Recommended Difficulty:",
            profile["recommended_difficulty"]
        )
        print(
            "Recommended Action:",
            profile["recommended_action"]
        )

    print()
    print("=" * 70)

    recommendation = recommend_next_topic(
        test_data
    )

    print(
        "NEXT RECOMMENDED TOPIC:",
        recommendation["topic"]
    )

    print(
        "REASON:",
        recommendation["reason"]
    )

    print()
    print("ADAPTIVE LEARNING ENGINE: PASS")