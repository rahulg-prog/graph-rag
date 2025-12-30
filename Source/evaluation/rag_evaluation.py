from langsmith import Client
from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict, Annotated

client = Client()

# ======================================================
# 1. CORRECTNESS (uses reference_outputs)
# ======================================================
class CorrectnessGrade(TypedDict):
    explanation: Annotated[str, "Explanation"]
    correct: Annotated[bool, "True if answer is correct"]

CORRECTNESS_PROMPT = """You are a strict grader.

Given:
- QUESTION
- GROUND TRUTH ANSWER
- STUDENT ANSWER

Judge ONLY factual correctness relative to the ground truth.
Explain your reasoning step by step.
"""

correctness_llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0
).with_structured_output(
    CorrectnessGrade,
    method="json_schema",
    strict=True
)

def correctness(inputs: dict, outputs: dict, reference_outputs: dict) -> bool:
    assert reference_outputs is not None, "Missing ground truth"

    prompt = f"""
QUESTION:
{inputs['question']}

GROUND TRUTH ANSWER:
{reference_outputs['answer']}

STUDENT ANSWER:
{outputs['answer']}
"""

    grade = correctness_llm.invoke([
        {"role": "system", "content": CORRECTNESS_PROMPT},
        {"role": "user", "content": prompt},
    ])

    return grade["correct"]


# ======================================================
# 2. RELEVANCE (answer vs question)
# ======================================================
class RelevanceGrade(TypedDict):
    explanation: Annotated[str, "Explanation"]
    relevant: Annotated[bool, "Answer addresses the question"]

RELEVANCE_PROMPT = """You are grading relevance.

Determine whether the STUDENT ANSWER directly and concisely
addresses the QUESTION.
"""

relevance_llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0
).with_structured_output(
    RelevanceGrade,
    method="json_schema",
    strict=True
)

def relevance(inputs: dict, outputs: dict) -> bool:
    prompt = f"""
QUESTION:
{inputs['question']}

STUDENT ANSWER:
{outputs['answer']}
"""

    grade = relevance_llm.invoke([
        {"role": "system", "content": RELEVANCE_PROMPT},
        {"role": "user", "content": prompt},
    ])

    return grade["relevant"]


# ======================================================
# 3. GROUNDEDNESS (answer vs retrieved docs)
# ======================================================
class GroundedGrade(TypedDict):
    explanation: Annotated[str, "Explanation"]
    grounded: Annotated[bool, "Answer supported by documents"]

GROUNDED_PROMPT = """You are grading groundedness.

Determine whether the STUDENT ANSWER is fully supported
by the provided FACTS and contains no hallucinations.
"""

grounded_llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0
).with_structured_output(
    GroundedGrade,
    method="json_schema",
    strict=True
)

def groundedness(inputs: dict, outputs: dict) -> bool:
    docs_text = "\n\n".join(doc.page_content for doc in outputs["documents"])

    prompt = f"""
FACTS:
{docs_text}

STUDENT ANSWER:
{outputs['answer']}
"""

    grade = grounded_llm.invoke([
        {"role": "system", "content": GROUNDED_PROMPT},
        {"role": "user", "content": prompt},
    ])

    return grade["grounded"]


# ======================================================
# 4. RETRIEVAL RELEVANCE (docs vs question)
# ======================================================
class RetrievalRelevanceGrade(TypedDict):
    explanation: Annotated[str, "Explanation"]
    relevant: Annotated[bool, "Docs relevant to question"]

RETRIEVAL_PROMPT = """You are grading document relevance.

Determine whether the retrieved FACTS are relevant
to answering the QUESTION.
"""

retrieval_llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0
).with_structured_output(
    RetrievalRelevanceGrade,
    method="json_schema",
    strict=True
)

def retrieval_relevance(inputs: dict, outputs: dict) -> bool:
    docs_text = "\n\n".join(doc.page_content for doc in outputs["documents"])

    prompt = f"""
QUESTION:
{inputs['question']}

FACTS:
{docs_text}
"""

    grade = retrieval_llm.invoke([
        {"role": "system", "content": RETRIEVAL_PROMPT},
        {"role": "user", "content": prompt},
    ])

    return grade["relevant"]


# ======================================================
# RUN EVALUATION
# ======================================================
def target(inputs: dict) -> dict:
    return rag_bot(inputs["question"])

client.evaluate(
    target,
    data="YOUR_DATASET_NAME",
    evaluators=[
        correctness,
        relevance,
        groundedness,
        retrieval_relevance,
    ],
    experiment_prefix="rag-full-eval",
)
