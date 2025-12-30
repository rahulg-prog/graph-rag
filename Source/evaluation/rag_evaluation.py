from langsmith import Client
from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict, Annotated

client = Client()

# -------------------------
# Correctness evaluator
# -------------------------
class CorrectnessGrade(TypedDict):
    explanation: Annotated[str, "Explain reasoning"]
    correct: Annotated[bool, "True if answer is correct"]

correctness_prompt = """You are a teacher grading a quiz.
Grade the STUDENT ANSWER based only on factual accuracy relative to the GROUND TRUTH ANSWER.
Explain your reasoning step by step."""

correctness_llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0
).with_structured_output(
    CorrectnessGrade,
    method="json_schema",
    strict=True
)

def correctness(inputs: dict, outputs: dict, reference_outputs: dict) -> bool:
    text = f"""
QUESTION: {inputs['question']}
GROUND TRUTH ANSWER: {reference_outputs['answer']}
STUDENT ANSWER: {outputs['answer']}
"""
    grade = correctness_llm.invoke([
        {"role": "system", "content": correctness_prompt},
        {"role": "user", "content": text},
    ])
    return grade["correct"]


# -------------------------
# Relevance evaluator
# -------------------------
class RelevanceGrade(TypedDict):
    explanation: Annotated[str, "Explain reasoning"]
    relevant: Annotated[bool, "Answer addresses the question"]

relevance_prompt = """You are a teacher grading a quiz.
Determine whether the STUDENT ANSWER is concise and relevant to the QUESTION."""

relevance_llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0
).with_structured_output(
    RelevanceGrade,
    method="json_schema",
    strict=True
)

def relevance(inputs: dict, outputs: dict) -> bool:
    text = f"""
QUESTION: {inputs['question']}
STUDENT ANSWER: {outputs['answer']}
"""
    grade = relevance_llm.invoke([
        {"role": "system", "content": relevance_prompt},
        {"role": "user", "content": text},
    ])
    return grade["relevant"]


# -------------------------
# Groundedness evaluator
# -------------------------
class GroundedGrade(TypedDict):
    explanation: Annotated[str, "Explain reasoning"]
    grounded: Annotated[bool, "Answer grounded in documents"]

grounded_prompt = """You are a teacher grading a quiz.
Determine whether the STUDENT ANSWER is fully supported by the FACTS."""

grounded_llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0
).with_structured_output(
    GroundedGrade,
    method="json_schema",
    strict=True
)

def groundedness(inputs: dict, outputs: dict) -> bool:
    docs = "\n\n".join(doc.page_content for doc in outputs["documents"])
    text = f"""
FACTS:
{docs}

STUDENT ANSWER:
{outputs['answer']}
"""
    grade = grounded_llm.invoke([
        {"role": "system", "content": grounded_prompt},
        {"role": "user", "content": text},
    ])
    return grade["grounded"]


# -------------------------
# Retrieval relevance evaluator
# -------------------------
class RetrievalRelevanceGrade(TypedDict):
    explanation: Annotated[str, "Explain reasoning"]
    relevant: Annotated[bool, "Docs relevant to question"]

retrieval_prompt = """You are a teacher grading a quiz.
Determine whether the FACTS are relevant to the QUESTION."""

retrieval_llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0
).with_structured_output(
    RetrievalRelevanceGrade,
    method="json_schema",
    strict=True
)

def retrieval_relevance(inputs: dict, outputs: dict) -> bool:
    docs = "\n\n".join(doc.page_content for doc in outputs["documents"])
    text = f"""
QUESTION: {inputs['question']}
FACTS:
{docs}
"""
    grade = retrieval_llm.invoke([
        {"role": "system", "content": retrieval_prompt},
        {"role": "user", "content": text},
    ])
    return grade["relevant"]


# -------------------------
# Run evaluation
# -------------------------
def target(inputs: dict) -> dict:
    return rag_bot(inputs["question"])

client.evaluate(
    target,
    data="Lilian Weng Blogs Q&A",
    evaluators=[
        correctness,
        relevance,
        groundedness,
        retrieval_relevance,
    ],
    experiment_prefix="rag-eval",
)
