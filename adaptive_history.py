from collections import defaultdict

from database.connection import SessionLocal
from database.models import ActivityLog

from adaptive_learning import (
    build_learning_profile,
    recommend_next_topic,
)


# ============================================================
# TECHPREP HISTORY
# ============================================================

def get_user_techprep_scores(user_id):

    db = SessionLocal()

    try:
        rows = (
            db.query(ActivityLog)
            .filter(
                ActivityLog.user_id == user_id,
                ActivityLog.activity_type == "TechPrep",
                ActivityLog.score.isnot(None),
            )
            .order_by(ActivityLog.created_at.asc())
            .all()
        )

        topic_scores = defaultdict(list)

        for row in rows:

            activity = (row.activity or "").strip()

            if " - " not in activity:
                continue

            topic = activity.split(" - ", 1)[0].strip()

            if not topic:
                continue

            try:
                score = float(row.score)
            except (TypeError, ValueError):
                continue

            topic_scores[topic].append(score)

        return dict(topic_scores)

    finally:
        db.close()


# ============================================================
# USER ADAPTIVE PROFILE
# ============================================================

def build_user_adaptive_profile(user_id):

    topic_scores = get_user_techprep_scores(user_id)

    profiles = {}

    for topic, scores in topic_scores.items():

        profiles[topic] = build_learning_profile(
            topic,
            scores
        )

    next_recommendation = recommend_next_topic(
        topic_scores
    )

    return {
        "user_id": user_id,
        "topic_scores": topic_scores,
        "profiles": profiles,
        "next_recommendation": next_recommendation,
    }


# ============================================================
# SELF TEST
# ============================================================

if __name__ == "__main__":

    test_user = "NS001"

    print("REAL USER ADAPTIVE LEARNING TEST")
    print("=" * 70)
    print("User:", test_user)

    result = build_user_adaptive_profile(
        test_user
    )

    print()

    if not result["profiles"]:

        print("No TechPrep score history found.")
        print()
        print("ADAPTIVE HISTORY CONNECTION: PASS")

    else:

        for topic, profile in result["profiles"].items():

            print("Topic:", topic)
            print(
                "Scores:",
                result["topic_scores"][topic]
            )
            print(
                "Attempts:",
                profile["attempts"]
            )
            print(
                "Average:",
                profile["average_score"]
            )
            print(
                "Level:",
                profile["level"]
            )
            print(
                "Status:",
                profile["status"]
            )
            print(
                "Trend:",
                profile["trend"]
            )
            print(
                "Recommended Difficulty:",
                profile["recommended_difficulty"]
            )
            print(
                "Recommended Action:",
                profile["recommended_action"]
            )
            print("-" * 70)

        recommendation = result[
            "next_recommendation"
        ]

        print(
            "NEXT RECOMMENDED TOPIC:",
            recommendation["topic"]
        )

        print(
            "REASON:",
            recommendation["reason"]
        )

        print()
        print("ADAPTIVE HISTORY CONNECTION: PASS")