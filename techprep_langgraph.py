import os
from typing import TypedDict

from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openrouter import ChatOpenRouter

from techprep_rag import retrieve_techprep_context


load_dotenv()


# ============================================================
# STATE
# ============================================================

class TechPrepState(TypedDict, total=False):
    topic: str
    question: str
    context: str
    answer: str
    evaluation: str
    final_answer: str
    needs_improvement: bool


# ============================================================
# LLM
# ============================================================

def get_llm():

    api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not available."
        )

    return ChatOpenRouter(
        model="openrouter/free",
        api_key=api_key,
        temperature=0.2,
    )


# ============================================================
# NODE 1 — RETRIEVE
# ============================================================

def retrieve_node(state: TechPrepState):

    context = retrieve_techprep_context(
        state["topic"],
        state["question"],
        top_k=2
    )

    return {
        "context": context
    }


# ============================================================
# NODE 2 — GENERATE
# ============================================================

def generate_node(state: TechPrepState):

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are TechPrep, a supportive technical interview coach.

Use the retrieved context to answer the interview question.

Use exactly these sections:

## தமிழ் விளக்கம்
Explain in simple natural Tamil.
Keep standard technical terms in English where useful.

## Simple Interview English
Give a short, natural English interview answer.

## Professional Interview Answer
Give a concise and technically accurate professional answer.

## Key Points to Remember
Give 4 to 6 short revision points.

Rules:
- Ground the answer in retrieved context.
- Do not invent personal experience.
- Do not repeat sentences.
- English sections must remain English only.
"""
            ),
            (
                "human",
                """
Topic:
{topic}

Question:
{question}

Retrieved Context:
{context}
"""
            ),
        ]
    )

    chain = (
        prompt
        | get_llm()
        | StrOutputParser()
    )

    answer = chain.invoke(
        {
            "topic": state["topic"],
            "question": state["question"],
            "context": state.get("context", "")
        }
    )

    return {
        "answer": answer
    }


# ============================================================
# NODE 3 — EVALUATE
# ============================================================

def evaluate_node(state: TechPrepState):

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are a strict quality checker for technical interview answers.

Check the answer for:

1. Technical relevance
2. Clear explanation
3. Interview usefulness
4. No unnecessary repetition
5. Required sections are present

Return only one word:

PASS

or

IMPROVE
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
        | get_llm()
        | StrOutputParser()
    )

    result = chain.invoke(
        {
            "question": state["question"],
            "answer": state["answer"]
        }
    ).strip().upper()

    needs_improvement = not result.startswith("PASS")

    return {
        "evaluation": result,
        "needs_improvement": needs_improvement
    }


# ============================================================
# CONDITIONAL ROUTER
# ============================================================

def quality_router(state: TechPrepState):

    if state.get("needs_improvement", False):
        return "improve"

    return "finalize"


# ============================================================
# NODE 4 — IMPROVE
# ============================================================

def improve_node(state: TechPrepState):

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
Improve the technical interview answer.

Preserve exactly these four sections:

## தமிழ் விளக்கம்
## Simple Interview English
## Professional Interview Answer
## Key Points to Remember

Requirements:
- Correct technical meaning.
- Clear and concise.
- No repetition.
- Natural Tamil in the Tamil section.
- English only in both English answer sections.
- Do not invent personal experience.
"""
            ),
            (
                "human",
                """
Question:
{question}

Retrieved Context:
{context}

Current Answer:
{answer}

Rewrite the answer with better quality.
"""
            ),
        ]
    )

    chain = (
        prompt
        | get_llm()
        | StrOutputParser()
    )

    improved_answer = chain.invoke(
        {
            "question": state["question"],
            "context": state.get("context", ""),
            "answer": state["answer"]
        }
    )

    return {
        "answer": improved_answer,
        "final_answer": improved_answer
    }


# ============================================================
# NODE 5 — FINALIZE
# ============================================================

def finalize_node(state: TechPrepState):

    return {
        "final_answer": state["answer"]
    }


# ============================================================
# BUILD GRAPH
# ============================================================

def build_techprep_graph():

    graph = StateGraph(TechPrepState)

    graph.add_node(
        "retrieve",
        retrieve_node
    )

    graph.add_node(
        "generate",
        generate_node
    )

    graph.add_node(
        "evaluate",
        evaluate_node
    )

    graph.add_node(
        "improve",
        improve_node
    )

    graph.add_node(
        "finalize",
        finalize_node
    )

    graph.add_edge(
        START,
        "retrieve"
    )

    graph.add_edge(
        "retrieve",
        "generate"
    )

    graph.add_edge(
        "generate",
        "evaluate"
    )

    graph.add_conditional_edges(
        "evaluate",
        quality_router,
        {
            "improve": "improve",
            "finalize": "finalize"
        }
    )

    graph.add_edge(
        "improve",
        END
    )

    graph.add_edge(
        "finalize",
        END
    )

    return graph.compile()


# ============================================================
# PUBLIC FUNCTION
# ============================================================

def generate_techprep_graph_answer(topic, question):

    workflow = build_techprep_graph()

    result = workflow.invoke(
        {
            "topic": topic,
            "question": question
        }
    )

    return result


# ============================================================
# SELF TEST
# ============================================================

if __name__ == "__main__":

    print("LANGGRAPH TECHPREP TEST")
    print("=" * 70)

    try:

        result = generate_techprep_graph_answer(
            "Generative AI",
            "What is Retrieval-Augmented Generation?"
        )

        print("LANGGRAPH WORKFLOW: PASS")
        print("Evaluation:", result.get("evaluation"))
        print(
            "Improvement Route:",
            result.get("needs_improvement")
        )

        print("\nFINAL ANSWER")
        print("=" * 70)

        print(
            result.get(
                "final_answer",
                result.get("answer", "")
            )
        )

    except Exception as e:

        print("LANGGRAPH WORKFLOW: FAILED")
        print(type(e).__name__)
        print(str(e))