import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openrouter import ChatOpenRouter

from techprep_rag import retrieve_techprep_context


load_dotenv()


# ============================================================
# LANGCHAIN + OPENROUTER
# ============================================================

def get_techprep_llm():

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
# TECHPREP PROMPT
# ============================================================

TECHPREP_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are TechPrep, a supportive technical interview learning coach.

Teach the requested technical concept accurately and practically.

Use exactly these sections:

## தமிழ் விளக்கம்
Explain the concept in simple, natural Tamil.
Keep standard technical terms such as Python, Machine Learning,
SQL, API, RAG, embedding, vector database, prompt and LLM
in English when appropriate.

## Simple Interview English
Give a short and natural English answer that a learner can
comfortably speak in an interview.

## Professional Interview Answer
Give a technically accurate and professional English answer.
Keep it concise and interview-friendly.

## Key Points to Remember
Give 4 to 6 short revision points.

Rules:
- Use the retrieved knowledge context when relevant.
- Do not invent technical facts.
- Do not invent the learner's personal experience.
- Do not add fake projects, tools, results or achievements.
- Do not repeat sentences or phrases.
- Simple Interview English must be English only.
- Professional Interview Answer must be English only.
- If retrieved context is incomplete, carefully use your
  technical knowledge to complete the explanation.
"""
        ),
        (
            "human",
            """
Technical Topic:
{topic}

Interview Question:
{question}

Retrieved Knowledge Context:
{context}

Use the relevant retrieved context to ground your answer.
"""
        ),
    ]
)


# ============================================================
# LANGCHAIN PIPELINE
# ============================================================

def build_techprep_chain():

    llm = get_techprep_llm()

    chain = (
        TECHPREP_PROMPT
        | llm
        | StrOutputParser()
    )

    return chain


# ============================================================
# RAG + LANGCHAIN
# ============================================================

def generate_techprep_answer(topic, question):

    context = retrieve_techprep_context(
        topic,
        question,
        top_k=2
    )

    if not context:
        context = (
            "No useful local knowledge context was retrieved. "
            "Use reliable technical knowledge carefully."
        )

    chain = build_techprep_chain()

    return chain.invoke(
        {
            "topic": topic,
            "question": question,
            "context": context,
        }
    )


# ============================================================
# SELF TEST
# ============================================================

if __name__ == "__main__":

    topic = "Generative AI"

    question = (
        "What is Retrieval-Augmented Generation?"
    )

    print("LANGCHAIN TECHPREP RAG TEST")
    print("=" * 70)

    try:
        answer = generate_techprep_answer(
            topic,
            question
        )

        print("LANGCHAIN PIPELINE: PASS")
        print()
        print(answer)

    except Exception as e:

        print("LANGCHAIN PIPELINE: FAILED")
        print(type(e).__name__)
        print(str(e))