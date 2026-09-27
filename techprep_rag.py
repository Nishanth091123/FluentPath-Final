import math
import re
import os
import requests
from collections import Counter
from dotenv import load_dotenv

load_dotenv()


TECHPREP_KNOWLEDGE = [
    {
        "topic": "Python",
        "text": """
Python is a high-level, interpreted, general-purpose programming language.
It is widely used because of its readable syntax, large ecosystem and strong
support for automation, data analysis, machine learning and AI. Core Python
concepts include data types, mutable and immutable objects, functions,
exception handling, comprehensions, object-oriented programming, iterators
and generators.
"""
    },
    {
        "topic": "Machine Learning",
        "text": """
Machine Learning is a branch of Artificial Intelligence where systems learn
patterns from data to make predictions or decisions. Important concepts
include supervised and unsupervised learning, regression, classification,
training-validation-test splitting, overfitting, underfitting, bias and
variance, feature engineering, feature selection, preprocessing,
cross-validation and model evaluation. Classification metrics include
accuracy, precision, recall and F1 score.
"""
    },
    {
        "topic": "Data Science",
        "text": """
Data Science combines data collection, cleaning, exploratory data analysis,
statistics, visualization and machine learning to generate useful insights
and solutions. Real-world Data Science work includes handling missing values,
outliers, structured and unstructured data, feature engineering, model
evaluation and communicating findings to technical and non-technical
stakeholders.
"""
    },
    {
        "topic": "Statistics",
        "text": """
Statistics helps describe data and make conclusions from samples. Important
concepts include mean, median, mode, variance, standard deviation,
probability, distributions, populations and samples, hypothesis testing,
p-values, confidence intervals, covariance, correlation and Type I and
Type II errors.
"""
    },
    {
        "topic": "SQL",
        "text": """
SQL is used to store, retrieve and manipulate data in relational databases.
Important concepts include SELECT, WHERE, HAVING, GROUP BY, aggregate
functions, joins, subqueries, common table expressions, primary and foreign
keys, normalization, indexes, window functions and duplicate detection.
"""
    },
    {
        "topic": "Power BI",
        "text": """
Power BI is a business intelligence and visualization platform. Important
concepts include Power Query, DAX, measures, calculated columns, relationships,
cardinality, star schema, filter context, data refresh, dashboard design and
report performance optimization.
"""
    },
    {
        "topic": "Excel",
        "text": """
Excel is widely used for data analysis and reporting. Important interview
concepts include XLOOKUP, VLOOKUP, INDEX and MATCH, PivotTables, IF and IFS,
SUMIFS, COUNTIFS, conditional formatting, relative and absolute references,
data cleaning, dashboards and Power Query.
"""
    },
    {
        "topic": "Artificial Intelligence",
        "text": """
Artificial Intelligence refers to systems designed to perform tasks that
normally require human intelligence. AI includes areas such as Machine
Learning, Deep Learning, Natural Language Processing and Computer Vision.
Important practical considerations include training, inference, evaluation,
bias, responsible AI, limitations and deployment.
"""
    },
    {
        "topic": "Deep Learning",
        "text": """
Deep Learning is a subset of Machine Learning based on multi-layer neural
networks. Important concepts include neurons, weights, biases, activation
functions, forward propagation, backpropagation, loss functions, optimizers,
CNNs, RNNs, LSTMs, dropout and transfer learning.
"""
    },
    {
        "topic": "NLP",
        "text": """
Natural Language Processing enables computers to work with human language.
Important concepts include tokenization, stop-word handling, stemming,
lemmatization, Bag of Words, TF-IDF, embeddings, sentiment analysis,
Named Entity Recognition, attention and transformers.
"""
    },
    {
        "topic": "Generative AI",
        "text": """
Generative AI creates new content such as text, images or code. Large Language
Models commonly use transformer architectures. Important concepts include
tokens, context windows, prompting, temperature, hallucination, embeddings,
vector search, Retrieval-Augmented Generation and fine-tuning. RAG retrieves
relevant external knowledge and supplies it as context to a generative model,
which can improve grounding without retraining the model.
"""
    },
    {
        "topic": "Project Questions",
        "text": """
Technical project explanations should clearly cover the problem statement,
business objective, data source, preprocessing, technology or model choice,
evaluation, challenges, deployment or application layer, practical results
and possible future improvements. Personal project details must not be
invented when they have not been supplied by the learner.
"""
    },
]


def _tokens(text):
    return re.findall(r"[a-z0-9]+", str(text).lower())


def _vector(text):
    return Counter(_tokens(text))


def _cosine_similarity(a, b):
    va = _vector(a)
    vb = _vector(b)

    if not va or not vb:
        return 0.0

    common = set(va) & set(vb)

    dot = sum(va[word] * vb[word] for word in common)
    norm_a = math.sqrt(sum(value * value for value in va.values()))
    norm_b = math.sqrt(sum(value * value for value in vb.values()))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot / (norm_a * norm_b)



# ============================================================
# REAL EMBEDDING RAG
# ============================================================

EMBEDDING_MODEL = "openai/text-embedding-3-small"
EMBEDDING_URL = "https://openrouter.ai/api/v1/embeddings"

_embedding_cache = {}


def _get_embeddings(texts):
    """
    Generate real neural embeddings through OpenRouter.
    Cached in memory to avoid unnecessary repeat API calls.
    """

    if isinstance(texts, str):
        texts = [texts]

    missing = [
        text for text in texts
        if text not in _embedding_cache
    ]

    if missing:
        api_key = os.getenv("OPENROUTER_API_KEY")

        if not api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is not available."
            )

        response = requests.post(
            EMBEDDING_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": EMBEDDING_MODEL,
                "input": missing
            },
            timeout=60
        )

        response.raise_for_status()
        data = response.json()

        returned = sorted(
            data["data"],
            key=lambda item: item["index"]
        )

        if len(returned) != len(missing):
            raise RuntimeError(
                "Embedding API returned an unexpected vector count."
            )

        for text, item in zip(missing, returned):
            _embedding_cache[text] = item["embedding"]

    return [_embedding_cache[text] for text in texts]


def _vector_cosine(a, b):

    if not a or not b or len(a) != len(b):
        return 0.0

    dot = sum(x * y for x, y in zip(a, b))

    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot / (norm_a * norm_b)


def _embedding_retrieve(topic, question, top_k=2):
    """
    Real embedding-based semantic retrieval.
    Exact topic is preferred for known TechPrep topics.
    Custom topics search the complete knowledge base.
    """

    normalized_topic = str(topic).strip().lower()

    exact_topic_items = [
        item for item in TECHPREP_KNOWLEDGE
        if item["topic"].strip().lower() == normalized_topic
    ]

    candidates = (
        exact_topic_items
        if exact_topic_items
        else TECHPREP_KNOWLEDGE
    )

    query = f"{topic}. {question}".strip()

    document_texts = [
        f'{item["topic"]}. {item["text"].strip()}'
        for item in candidates
    ]

    # One batch call for query + candidate documents.
    vectors = _get_embeddings(
        [query] + document_texts
    )

    query_vector = vectors[0]
    document_vectors = vectors[1:]

    scored = []

    for item, vector in zip(candidates, document_vectors):
        similarity = _vector_cosine(
            query_vector,
            vector
        )

        scored.append((similarity, item))

    scored.sort(
        key=lambda pair: pair[0],
        reverse=True
    )

    selected = [
        item
        for score, item in scored[:max(1, top_k)]
        if score > 0
    ]

    if not selected:
        return ""

    return "\n\n---\n\n".join(
        f"Knowledge Topic: {item['topic']}\n"
        f"{item['text'].strip()}"
        for item in selected
    )


def retrieve_techprep_context(topic, question, top_k=2):
    """
    Lightweight semantic-style retrieval for TechPrep.

    Returns the most relevant knowledge passages for the selected
    topic/question. This is deliberately dependency-light so the
    existing FluentPath application remains stable.
    """

    # Primary retrieval: real neural embeddings.
    try:
        embedding_context = _embedding_retrieve(
            topic,
            question,
            top_k=top_k
        )

        if embedding_context:
            return embedding_context

    except Exception:
        # Safe fallback:
        # FluentPath TechPrep must continue working even when the
        # embedding service is temporarily unavailable.
        pass

    # Secondary retrieval: existing dependency-light lexical vectors.
    query = f"{topic} {question}".strip()

    scored = []

    normalized_topic = str(topic).strip().lower()

    # First preference: knowledge from the exact selected topic.
    exact_topic_items = [
        item for item in TECHPREP_KNOWLEDGE
        if item["topic"].strip().lower() == normalized_topic
    ]

    if exact_topic_items:
        candidates = exact_topic_items
    else:
        # Custom/unknown topics can search the full knowledge base.
        candidates = TECHPREP_KNOWLEDGE

    for item in candidates:
        similarity = _cosine_similarity(
            query,
            f'{item["topic"]} {item["text"]}'
        )

        scored.append((similarity, item))

    scored.sort(key=lambda x: x[0], reverse=True)

    selected = [
        item for score, item in scored[:max(1, top_k)]
        if score > 0
    ]

    if not selected:
        return ""

    parts = []

    for item in selected:
        parts.append(
            f"Knowledge Topic: {item['topic']}\n"
            f"{item['text'].strip()}"
        )

    return "\n\n---\n\n".join(parts)


def build_rag_user_prompt(topic, question):
    context = retrieve_techprep_context(
        topic,
        question,
        top_k=2
    )

    if context:
        return f"""
Topic:
{topic}

Interview Question:
{question}

Retrieved Knowledge Context:
{context}

Use the retrieved context as grounding.
Answer the interview question directly.
Do not copy irrelevant information from the context.
If the context does not fully cover the question, use your technical
knowledge carefully and do not invent facts.
""".strip()

    return f"""
Topic:
{topic}

Interview Question:
{question}

No useful local knowledge context was retrieved.
Answer using reliable technical knowledge and do not invent facts.
""".strip()


if __name__ == "__main__":
    tests = [
        (
            "Machine Learning",
            "What are overfitting and underfitting?"
        ),
        (
            "Generative AI",
            "What is Retrieval-Augmented Generation?"
        ),
        (
            "SQL",
            "What is the difference between WHERE and HAVING?"
        ),
    ]

    print("\nTECHPREP RAG SELF-TEST\n")

    for topic, question in tests:
        print("=" * 70)
        print("TOPIC:", topic)
        print("QUESTION:", question)
        print()
        print(retrieve_techprep_context(topic, question))
        print()