# Quick Start

Get started with RM-Gallery in 5 minutes. This guide walks you through installation, environment setup, and running your first evaluation.

RM-Gallery evaluates AI responses using two approaches: **LLM-as-a-Judge** for subjective quality assessment, and **rule-based execution** for objective metrics like code correctness. You provide a query-response pair, and a grader returns a score with an explanation.

---

## Installation

**pip:**

    ```bash
    pip install rm-gallery
    ```

**Poetry:**

```bash
git clone https://github.com/modelscope/RM-Gallery.git
cd RM-Gallery
poetry install
```

> **Tip:** RM-Gallery requires Python 3.10 or higher.

---

## Configure Environment

For LLM-based graders, you need to configure API credentials. RM-Gallery uses the OpenAI-compatible API format.

**Option 1: Environment Variables (Recommended)**

Set environment variables in your terminal:

**OpenAI:**

```bash
export OPENAI_API_KEY="sk-your-api-key"
export OPENAI_BASE_URL="https://api.openai.com/v1"
```

**DashScope (Qwen):**

```bash
export OPENAI_API_KEY="sk-your-dashscope-key"
export OPENAI_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
```

**Option 2: Pass Directly in Code**

You can also pass credentials when creating the model:

```python
from rm_gallery.core.models import OpenAIChatModel

model = OpenAIChatModel(
    model="qwen3-32b",
    api_key="sk-your-api-key",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)
```

> **Note:** Environment variables are more secure and convenient. The model will automatically use `OPENAI_API_KEY` and `OPENAI_BASE_URL` if set.

---

## Prepare Your Data

RM-Gallery graders expect simple dictionaries with `query` and `response` fields:

```python
data = {
    "query": "What is the capital of France?",
    "response": "The capital of France is Paris.",
}
```

For correctness evaluation, include a `ground_truth` field:

```python
data = {
    "query": "What is 2 + 2?",
    "response": "The answer is 4.",
    "ground_truth": "4",
}
```

---

## Initialize a Grader

RM-Gallery provides two types of **graders**:

- **LLM-based graders**: Use language models to judge response quality (e.g., harmfulness, relevance)
- **Rule-based graders**: Use algorithms and execution to evaluate responses (e.g., code execution, syntax checking)

For LLM-based graders, first create the model that powers the evaluation:

**OpenAI:**

```python
from rm_gallery.core.models import OpenAIChatModel

model = OpenAIChatModel(model="qwen3-32b")
```

**DashScope (Qwen):**

```python
from rm_gallery.core.models import OpenAIChatModel

model = OpenAIChatModel(model="qwen-plus")
```

Then choose a grader based on what you want to evaluate:

**Harmfulness:** Detects harmful, offensive, or inappropriate content in responses.

```python
from rm_gallery.core.graders.common.harmfulness import HarmfulnessGrader

grader = HarmfulnessGrader(model=model)
```


**Code Execution:** Tests code against test cases (rule-based, no LLM needed).

```python
from rm_gallery.core.graders.code.code_excution import CodeExecutionGrader

grader = CodeExecutionGrader(continuous=True, timeout=5)
```

> **Note:** Code and math graders don't require an LLM model. They use rule-based execution and validation.

---

## Run Your First Evaluation

All graders use async/await for efficient API calls. Here's how to run evaluations:

**Single Sample:**

Evaluate a single response with `aevaluate()`. The grader sends the query and response to the LLM judge and returns a score:

```python
import asyncio
from rm_gallery.core.models import OpenAIChatModel
from rm_gallery.core.graders.common.relevance import RelevanceGrader

async def main():
    model = OpenAIChatModel(model="qwen3-32b")
    grader = RelevanceGrader(model=model)

    # Prepare data
    data = {
        "query": "What is machine learning?",
        "response": "Machine learning is a subset of AI that enables computers to learn from data.",
    }

    # Evaluate using the data
    result = await grader.aevaluate(**data)

    print(f"Score: {result.score}")  # Score: 5
    print(f"Reason: {result.reason}")

asyncio.run(main())
```

**Batch Evaluation:**

For datasets, use `GradingRunner` to evaluate multiple samples concurrently with automatic progress tracking:

```python
import asyncio
from rm_gallery.core.models import OpenAIChatModel
from rm_gallery.core.graders.common.relevance import RelevanceGrader
from rm_gallery.core.runner.grading_runner import GradingRunner, GraderConfig

async def main():
    model = OpenAIChatModel(model="qwen3-32b")
    grader = RelevanceGrader(model=model)

    # Configure the runner
    runner = GradingRunner(
        grader_configs={"relevance": GraderConfig(grader=grader)}
    )

    # Prepare dataset (same format as single samples)
    dataset = [
        {
            "query": "What is Python?",
            "response": "Python is a programming language."
        },
        {
            "query": "Explain gravity",
            "response": "Gravity is a force that attracts objects."
        },
    ]

    # Run batch evaluation
    results = await runner.arun(dataset)

    # Print results
    for i, result in enumerate(results["relevance"]):
        print(f"Sample {i}: Score={result.score}")

asyncio.run(main())
```

---

## Understanding Results

Graders return different result types depending on their mode:

### GraderScore (Pointwise Mode)

For evaluating individual responses, graders return a `GraderScore` object:

| Field | Description |
|-------|-------------|
| `score` | Numerical score (e.g., 0-1 or 1-5 scale) |
| `reason` | Explanation for the score |
| `name` | Name of the grader used |
| `metadata` | Additional details (e.g., threshold settings) |

```python
result = await grader.aevaluate(query="...", response="...")

print(result.score)     # 5
print(result.reason)    # "Response directly addresses the query..."
print(result.name)      # "relevance"
print(result.metadata)  # {"threshold": 0.7}
```

### GraderRank (Listwise Mode)

For ranking multiple responses, graders return a `GraderRank` object:

| Field | Description |
|-------|-------------|
| `rank` | Ranking of responses (e.g., [1, 3, 2] means 1st is best, 3rd is second, 2nd is worst) |
| `reason` | Explanation for the ranking |
| `name` | Name of the grader used |
| `metadata` | Additional ranking details |

```python
result = await grader.aevaluate(
    query="...",
    response_1="...",
    response_2="...",
    response_3="..."
)

print(result.rank)      # [1, 3, 2]
print(result.reason)    # "First response is most comprehensive..."
print(result.name)      # "relevance_ranker"
```

---

## Next Steps

- [Core Concepts](core-concepts.md) — Understand graders, modes, and scoring
- [Built-in Graders](../graders/overview.md) — Explore 35+ pre-built graders
- [Code & Math Graders](../graders/code-math.md) — Evaluate code execution, syntax, and style
- [Create Custom Graders](../building-graders/custom-graders.md) — Build your own evaluation logic
