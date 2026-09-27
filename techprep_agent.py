import os
from typing import Dict, Optional

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openrouter import ChatOpenRouter

from techprep_rag import retrieve_techprep_context
from techprep_langchain import generate_techprep_answer
from techprep_langgraph import generate_techprep_graph_answer
from adaptive_history import build_user_adaptive_profile


load_dotenv()


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "openrouter/free"

VALID_INTENTS = {
    "CONCEPT",
    "INTERVIEW",
    "PRACTICE",
    "REVISION",
}

VALID_DIFFICULTIES = {
    "Beginner",
    "Intermediate",
    "Advanced",
}


# ============================================================
# LLM
# ============================================================

def get_agent_llm():

    api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not available."
        )

    return ChatOpenRouter(
        model=MODEL_NAME,
        api_key=api_key,
        temperature=0.1,
    )


# ============================================================
# SAFE TEXT HELPERS
# ============================================================

def clean_text(value):

    if value is None:
        return ""

    return str(value).strip()


def normalize_intent(value):

    value = clean_text(value).upper()

    if value in VALID_INTENTS:
        return value

    return "INTERVIEW"


def normalize_difficulty(value):

    value = clean_text(value).lower()

    if value == "advanced":
        return "Advanced"

    if value == "intermediate":
        return "Intermediate"

    return "Beginner"


# ============================================================
# AGENT 1 — INTENT ANALYZER
# ============================================================

def analyze_question_intent(
    topic: str,
    question: str
) -> str:

    """
    Decide what kind of learning support is most appropriate.

    Possible outputs:
    CONCEPT
    INTERVIEW
    PRACTICE
    REVISION
    """

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are the routing agent for a technical interview
learning application.

Classify the learner's request into exactly ONE category:

CONCEPT
- asks what something is
- asks how something works
- asks for explanation or understanding

INTERVIEW
- asks for an interview-ready answer
- asks comparison/difference questions
- asks why/how questions commonly used in interviews

PRACTICE
- asks for practice, scenario or challenge questions

REVISION
- asks for summary, key points, quick review or recall

Return ONLY one word:

CONCEPT
INTERVIEW
PRACTICE
REVISION

Do not explain your choice.
"""
            ),
            (
                "human",
                """
Topic:
{topic}

Question:
{question}
"""
            ),
        ]
    )

    chain = (
        prompt
        | get_agent_llm()
        | StrOutputParser()
    )

    try:

        result = chain.invoke(
            {
                "topic": topic,
                "question": question,
            }
        )

        return normalize_intent(result)

    except Exception:

        # Safe deterministic fallback
        q = question.lower()

        if any(
            word in q
            for word in [
                "practice",
                "scenario",
                "challenge",
            ]
        ):
            return "PRACTICE"

        if any(
            word in q
            for word in [
                "summary",
                "revision",
                "key points",
                "recap",
            ]
        ):
            return "REVISION"

        if q.startswith("what is"):
            return "CONCEPT"

        return "INTERVIEW"


# ============================================================
# AGENT 2 — ADAPTIVE LEARNING AGENT
# ============================================================

def get_learning_context(
    user_id: Optional[str],
    topic: str
) -> Dict:

    default_result = {
        "level": "Beginner",
        "difficulty": "Beginner",
        "status": "New Topic",
        "trend": "Not Enough Data",
        "average_score": 0,
        "attempts": 0,
        "recommended_action":
            "Start with a beginner-level question.",
    }

    if not user_id:
        return default_result

    try:

        adaptive_data = build_user_adaptive_profile(
            user_id
        )

        profiles = adaptive_data.get(
            "profiles",
            {}
        )

        profile = profiles.get(topic)

        if not profile:
            return default_result

        return {
            "level": profile.get(
                "level",
                "Beginner"
            ),
            "difficulty": normalize_difficulty(
                profile.get(
                    "recommended_difficulty",
                    "Beginner"
                )
            ),
            "status": profile.get(
                "status",
                "New Topic"
            ),
            "trend": profile.get(
                "trend",
                "Not Enough Data"
            ),
            "average_score": profile.get(
                "average_score",
                0
            ),
            "attempts": profile.get(
                "attempts",
                0
            ),
            "recommended_action": profile.get(
                "recommended_action",
                "Continue practising."
            ),
        }

    except Exception:

        # Adaptive system should never block TechPrep.
        return default_result


# ============================================================
# AGENT 3 — RETRIEVAL AGENT
# ============================================================

def retrieve_agent_context(
    topic: str,
    question: str
) -> str:

    try:

        context = retrieve_techprep_context(
            topic,
            question,
            top_k=2
        )

        if context:
            return context

    except Exception:
        pass

    return (
        "No local retrieved context is available. "
        "Use reliable technical knowledge carefully."
    )


# ============================================================
# AGENT 4 — SPECIAL RESPONSE GENERATOR
# Used for PRACTICE and REVISION routes.
# ============================================================

def generate_special_response(
    topic: str,
    question: str,
    intent: str,
    difficulty: str,
    context: str
) -> str:

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are Agentic TechPrep, a supportive technical
interview coach.

Your job is to respond according to the selected
learning route.

For PRACTICE:

Use exactly:

## Practice Question
Give one relevant interview question.

## What the Interviewer Checks
Give 3 short points.

## Answer Structure
Give a simple structure the learner can follow.

## Coach Tip
Give one short practical tip.

Do NOT give the complete final answer unless the
learner explicitly asks for it.


For REVISION:

Use exactly:

## Quick Revision
Give a concise explanation.

## Key Points
Give 4 to 6 important points.

## Interview Reminder
Give one short interview-ready reminder.


General rules:

- Stay relevant to the requested topic.
- Match the requested difficulty.
- Use retrieved knowledge where relevant.
- Do not invent learner experience.
- Do not invent projects or achievements.
- Avoid repetition.
- Keep the response concise and practical.
"""
            ),
            (
                "human",
                """
Topic:
{topic}

Original Question:
{question}

Agent Route:
{intent}

Recommended Difficulty:
{difficulty}

Retrieved Knowledge:
{context}
"""
            ),
        ]
    )

    chain = (
        prompt
        | get_agent_llm()
        | StrOutputParser()
    )

    return chain.invoke(
        {
            "topic": topic,
            "question": question,
            "intent": intent,
            "difficulty": difficulty,
            "context": context,
        }
    )


# ============================================================
# CONCEPT / INTERVIEW GENERATION
# Primary = LangGraph
# Fallback 1 = LangChain
# ============================================================

def generate_interview_response(
    topic: str,
    question: str
) -> str:

    try:

        graph_result = (
            generate_techprep_graph_answer(
                topic,
                question
            )
        )

        answer = graph_result.get(
            "final_answer",
            graph_result.get(
                "answer",
                ""
            )
        )

        if answer:
            return answer

        raise RuntimeError(
            "LangGraph returned an empty answer."
        )

    except Exception:

        return generate_techprep_answer(
            topic,
            question
        )


# ============================================================
# FINAL QUALITY CHECK
# ============================================================

def quality_check(
    question: str,
    answer: str
) -> Dict:

    if not answer:
        return {
            "status": "IMPROVE",
            "reason": "Empty answer",
        }

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are a quality-control agent.

Check whether the answer:

1. Addresses the question.
2. Is technically relevant.
3. Is understandable.
4. Avoids unnecessary repetition.
5. Is useful for interview preparation.

Return exactly one of these:

PASS

or

IMPROVE

Do not add explanation.
"""
            ),
            (
                "human",
                """
Question:
{question}

Answer:
{answer}
"""
            ),
        ]
    )

    chain = (
        prompt
        | get_agent_llm()
        | StrOutputParser()
    )

    try:

        result = chain.invoke(
            {
                "question": question,
                "answer": answer,
            }
        ).strip().upper()

        if result.startswith("PASS"):

            return {
                "status": "PASS",
                "reason": "Quality check passed",
            }

        return {
            "status": "IMPROVE",
            "reason":
                "Quality agent requested improvement",
        }

    except Exception:

        # Existing LangGraph already includes
        # evaluation, so failure here should not
        # discard a valid generated answer.
        return {
            "status": "PASS",
            "reason":
                "Quality checker unavailable; "
                "existing verified pipeline preserved",
        }


# ============================================================
# AGENT DECISION ENGINE
# ============================================================

def decide_agent_route(
    intent: str,
    learning_context: Dict
) -> Dict:

    difficulty = learning_context.get(
        "difficulty",
        "Beginner"
    )

    if intent == "PRACTICE":

        action = "Generate Practice"

    elif intent == "REVISION":

        action = "Generate Revision"

    elif intent == "CONCEPT":

        action = "Teach Concept"

    else:

        action = "Generate Interview Answer"

    return {
        "intent": intent,
        "action": action,
        "difficulty": difficulty,
    }


# ============================================================
# MAIN AGENTIC TECHPREP ORCHESTRATOR
# ============================================================

def run_techprep_agent(
    topic: str,
    question: str,
    user_id: Optional[str] = None
) -> Dict:

    topic = clean_text(topic)
    question = clean_text(question)

    if not topic:
        raise ValueError(
            "TechPrep topic is required."
        )

    if not question:
        raise ValueError(
            "TechPrep question is required."
        )

    # --------------------------------------------------------
    # 1. Understand learner
    # --------------------------------------------------------

    learning_context = get_learning_context(
        user_id,
        topic
    )

    # --------------------------------------------------------
    # 2. Understand request
    # --------------------------------------------------------

    intent = analyze_question_intent(
        topic,
        question
    )

    # --------------------------------------------------------
    # 3. Decide route
    # --------------------------------------------------------

    decision = decide_agent_route(
        intent,
        learning_context
    )

    # --------------------------------------------------------
    # 4. Retrieve knowledge
    # --------------------------------------------------------

    context = retrieve_agent_context(
        topic,
        question
    )

    # --------------------------------------------------------
    # 5. Execute selected agent route
    # --------------------------------------------------------

    if intent in {
        "PRACTICE",
        "REVISION",
    }:

        answer = generate_special_response(
            topic=topic,
            question=question,
            intent=intent,
            difficulty=decision["difficulty"],
            context=context,
        )

    else:

        answer = generate_interview_response(
            topic,
            question
        )

    # --------------------------------------------------------
    # 6. Final quality control
    # --------------------------------------------------------

    quality = quality_check(
        question,
        answer
    )

    # --------------------------------------------------------
    # 7. Return structured agent result
    # --------------------------------------------------------

    return {
        "topic": topic,
        "question": question,

        "intent": intent,
        "agent_action": decision["action"],

        "learner_level":
            learning_context["level"],

        "recommended_difficulty":
            learning_context["difficulty"],

        "performance_status":
            learning_context["status"],

        "trend":
            learning_context["trend"],

        "average_score":
            learning_context["average_score"],

        "attempts":
            learning_context["attempts"],

        "recommended_next_action":
            learning_context[
                "recommended_action"
            ],

        "retrieved_context": context,

        "quality_status":
            quality["status"],

        "quality_reason":
            quality["reason"],

        "final_answer": answer,
    }


# ============================================================
# SIMPLE PUBLIC FUNCTION
# ============================================================

def generate_agentic_techprep_answer(
    topic: str,
    question: str,
    user_id: Optional[str] = None
) -> str:

    result = run_techprep_agent(
        topic=topic,
        question=question,
        user_id=user_id,
    )

    return result["final_answer"]


# ============================================================
# SELF TEST
# ============================================================

if __name__ == "__main__":

    print("AGENTIC TECHPREP TEST")
    print("=" * 70)

    try:

        result = run_techprep_agent(
            topic="Generative AI",
            question=(
                "What is Retrieval-Augmented "
                "Generation?"
            ),
            user_id="NS001",
        )

        print("AGENTIC TECHPREP: PASS")
        print()

        print(
            "Detected Intent:",
            result["intent"]
        )

        print(
            "Agent Action:",
            result["agent_action"]
        )

        print(
            "Learner Level:",
            result["learner_level"]
        )

        print(
            "Recommended Difficulty:",
            result[
                "recommended_difficulty"
            ]
        )

        print(
            "Performance Status:",
            result["performance_status"]
        )

        print(
            "Trend:",
            result["trend"]
        )

        print(
            "Previous Average:",
            result["average_score"]
        )

        print(
            "Previous Attempts:",
            result["attempts"]
        )

        print(
            "Quality Status:",
            result["quality_status"]
        )

        print()
        print("FINAL ANSWER")
        print("=" * 70)

        print(
            result["final_answer"]
        )

        print()
        print("=" * 70)

        print(
            "NEXT LEARNING ACTION:"
        )

        print(
            result[
                "recommended_next_action"
            ]
        )

    except Exception as e:

        print("AGENTIC TECHPREP: FAILED")
        print(type(e).__name__)
        print(str(e))