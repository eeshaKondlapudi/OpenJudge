# Build Reward for Training

Build comprehensive reward signals for RLHF and post-training by combining multiple graders. This guide walks through a complete workflow from environment setup to aggregated reward computation.

A single grader evaluates one aspect (e.g., helpfulness). For training, you typically need a **composite reward** that balances safety, relevance, accuracy, and other criteria.

---

## Environment Setup

Install RM-Gallery and configure API credentials:

**Installation:**

```bash
pip install rm-gallery
```

**API Configuration:**

```bash
# Create .env file for API keys
echo "OPENAI_API_KEY=your_api_key_here" > .env
echo "OPENAI_BASE_URL=your_base_url_here" >> .env
```

```python
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Verify configuration
assert os.getenv("OPENAI_API_KEY"), "OPENAI_API_KEY not set"
```

> **Note:** Different model providers require different environment variables. See [Model Configuration](../integrations/models.md) for details.

---

## Data Preparation

Prepare your training dataset with required fields. Each sample should contain the query, response, and optional ground truth:

```python
# Training dataset structure
# Most graders only need query and response
dataset = [
    {
        "query": "What is the capital of France?",
        "response": "The capital of France is Paris.",
    },
    {
        "query": "Explain quantum computing",
        "response": "Quantum computing uses quantum-mechanical phenomena...",
    },
    {
        "query": "How to write a loop in Python?",
        "response": "Use for loop: for i in range(10): print(i)",
    },
]

# If you need correctness evaluation with reference answers,
# use a mapper to provide reference_response for CorrectnessGrader
dataset_with_references = [
    {
        "query": "What is the capital of France?",
        "response": "The capital of France is Paris.",
        "ground_truth": "Paris",  # Will be mapped to reference_response
    },
]
```

**Field Requirements:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `query` | str | Yes | Input question or prompt |
| `response` | str | Yes | Model's generated response |

> **Tip:** Structure your data as a list of dictionaries. Most graders (HarmfulnessGrader, RelevanceGrader) only need `query` and `response`. For graders requiring additional fields (like CorrectnessGrader needing `reference_response`), use GraderConfig's mapper feature to transform your data fields.

---

## Build Graders

### Single Grader Setup

Create individual graders to evaluate specific aspects:

```python
from rm_gallery.core.models import OpenAIChatModel
from rm_gallery.core.graders.common.relevance import RelevanceGrader

# Initialize model
model = OpenAIChatModel(model="qwen3-32b")

# Create a single grader
relevance_grader = RelevanceGrader(model=model)

# Test on one sample
result = await relevance_grader.aevaluate(
    query="What is the capital of France?",
    response="Paris is the capital of France.",
)

print(f"Score: {result.score}")  # 0.0-1.0
print(f"Reason: {result.reason}")
```

### Multi-Grader Setup

Configure multiple graders to evaluate different aspects:

```python
import asyncio
from rm_gallery.core.models import OpenAIChatModel
from rm_gallery.core.graders.common.harmfulness import HarmfulnessGrader
from rm_gallery.core.graders.common.relevance import RelevanceGrader
from rm_gallery.core.graders.common.correctness import CorrectnessGrader
from rm_gallery.core.runner.grading_runner import GradingRunner, GraderConfig

async def main():
    model = OpenAIChatModel(model="qwen3-32b")

    # Configure multiple graders
    # Use mapper to transform field names for graders that need different fields
    grader_configs = {
        "harmfulness": GraderConfig(grader=HarmfulnessGrader(model=model)),
        "relevance": GraderConfig(grader=RelevanceGrader(model=model)),
        "correctness": GraderConfig(
            grader=CorrectnessGrader(model=model),
            mapper={"reference_response": "ground_truth"}  # Map ground_truth to reference_response
        ),
    }
```

**Available Graders:**

| Grader | Evaluates | Output Range |
|--------|-----------|--------------|
| `HarmfulnessGrader` | Safety and toxicity | 0.0 (safe) - 1.0 (harmful) |
| `RelevanceGrader` | Topical relevance | 0.0 (off-topic) - 1.0 (relevant) |
| `CorrectnessGrader` | Factual accuracy | 0.0 (incorrect) - 1.0 (correct) |
| `InstructionFollowingGrader` | Instruction adherence | 0.0 (not followed) - 1.0 (fully followed) |

> **Tip:** Start with 2-3 graders for initial experiments. Add more as you refine your reward signal.

---

## Run GradingRunner

Execute graders on your dataset using `GradingRunner`:

### Basic Execution

```python
from rm_gallery.core.runner.grading_runner import GradingRunner

# Create runner
runner = GradingRunner(grader_configs=grader_configs)

# Run evaluation on dataset
results = await runner.arun(dataset)

# Access individual grader scores
for i in range(len(dataset)):
    print(f"Sample {i}:")
    print(f"  Harmfulness: {results['harmfulness'][i].score}")
    print(f"  Relevance: {results['relevance'][i].score}")
    print(f"  Correctness: {results['correctness'][i].score}")

asyncio.run(main())
```

**Result Structure:**

```python
{
    "harmfulness": [
        GraderScore(score=0.1, reason="Safe content"),
        GraderScore(score=0.0, reason="No harmful content"),
    ],
    "relevance": [
        GraderScore(score=0.9, reason="Directly addresses query"),
        GraderScore(score=0.85, reason="Highly relevant"),
    ],
    "correctness": [
        GraderScore(score=1.0, reason="Matches ground truth"),
        GraderScore(score=0.8, reason="Mostly accurate"),
    ],
}
```

### High-Concurrency Execution

For large datasets, control parallel API calls with `max_concurrency`:

```python
runner = GradingRunner(
    grader_configs=grader_configs,
    max_concurrency=64,  # Up to 64 parallel evaluations
)

# Evaluate thousands of samples efficiently
large_dataset = [{"query": f"Q{i}", "response": f"A{i}"} for i in range(10000)]
results = await runner.arun(large_dataset)
```

| max_concurrency | Use Case |
|-----------------|----------|
| `8-16` | Rate-limited APIs, free tier |
| `32-64` | Standard production usage |
| `128+` | High-throughput, dedicated endpoints |

> **Warning:** Set `max_concurrency` below your API provider's rate limit to avoid throttling errors.

---

## Aggregate Scores

Combine scores from multiple graders into a single reward signal using aggregators:

### Weighted Sum Aggregation

Combine scores with custom weights to prioritize certain criteria:

```python
from rm_gallery.core.runner.aggregator.weighted_sum_aggregator import WeightedSumAggregator

# Define weights for each grader
aggregator = WeightedSumAggregator(
    name="training_reward",
    weights={
        "harmfulness": 0.4,   # Safety is critical
        "relevance": 0.3,
        "correctness": 0.3,
    }
)

runner = GradingRunner(
    grader_configs=grader_configs,
    aggregators=aggregator,
)

results = await runner.arun(dataset)

# Access aggregated reward
for i in range(len(dataset)):
    reward = results["training_reward"][i].score
    print(f"Sample {i} reward: {reward}")
```

### Equal Weights

When no weights are specified, all graders contribute equally:

```python
from rm_gallery.core.runner.aggregator.weighted_sum_aggregator import WeightedSumAggregator

# Equal weights by default
aggregator = WeightedSumAggregator(name="training_reward")

runner = GradingRunner(
    grader_configs=grader_configs,
    aggregators=aggregator,
)

results = await runner.arun(dataset)
```

### Custom Aggregation

Create custom aggregation logic by implementing a callable:

```python
from rm_gallery.core.graders.schema import GraderScore

def min_aggregator(results):
    """Use the minimum score as reward (conservative approach)."""
    scores = [r.score for r in results.values() if hasattr(r, "score")]
    min_score = min(scores) if scores else 0.0
    return GraderScore(
        name="min_reward",
        score=min_score,
        reason=f"Minimum of {len(scores)} grader scores",
    )

runner = GradingRunner(
    grader_configs=grader_configs,
    aggregators=min_aggregator,
)
```

> **Tip:** For safety-critical applications, assign higher weights to `harmfulness`. For knowledge-intensive tasks, prioritize `correctness`.

---

## Complete Workflow Example

Putting it all together for a training reward pipeline:

```python
import asyncio
import os
from dotenv import load_dotenv

from rm_gallery.core.models import OpenAIChatModel
from rm_gallery.core.graders.common.harmfulness import HarmfulnessGrader
from rm_gallery.core.graders.common.relevance import RelevanceGrader
from rm_gallery.core.graders.common.correctness import CorrectnessGrader
from rm_gallery.core.runner.grading_runner import GradingRunner, GraderConfig
from rm_gallery.core.runner.aggregator.weighted_sum_aggregator import WeightedSumAggregator

async def compute_training_rewards(dataset):
    """Complete workflow: Environment → Data → Graders → Runner → Aggregation."""

    # 1. Environment Setup
    load_dotenv()
    assert os.getenv("OPENAI_API_KEY"), "API key not configured"

    # 2. Initialize Model
    model = OpenAIChatModel(model="qwen3-32b")

    # 3. Configure Graders
    grader_configs = {
        "harmfulness": GraderConfig(grader=HarmfulnessGrader(model=model)),
        "relevance": GraderConfig(grader=RelevanceGrader(model=model)),
        "correctness": GraderConfig(
            grader=CorrectnessGrader(model=model),
            mapper={"reference_response": "ground_truth"}  # Map ground_truth to reference_response
        ),
    }

    # 4. Configure Aggregation
    aggregator = WeightedSumAggregator(
        name="reward",
        weights={"harmfulness": 0.4, "relevance": 0.3, "correctness": 0.3},
    )

    # 5. Run GradingRunner
    runner = GradingRunner(
        grader_configs=grader_configs,
        aggregators=aggregator,
        max_concurrency=32,
    )

    results = await runner.arun(dataset)

    # 6. Extract Rewards
    # Find the aggregator result key
    aggregator_key = aggregator.__name__()
    rewards = [r.score for r in results[aggregator_key]]
    return rewards

# Usage
dataset = [
    {
        "query": "Explain gravity",
        "response": "Gravity is a force that attracts objects with mass...",
        "ground_truth": "Gravity is a fundamental force of nature...",
    },
    {
        "query": "What is 2+2?",
        "response": "4",
        "ground_truth": "4",
    },
]

rewards = asyncio.run(compute_training_rewards(dataset))
print(f"Training rewards: {rewards}")
```

**Expected Output:**

```
Training rewards: [0.87, 0.95]
```

---

## Next Steps

- [Run Grading Tasks](../running-graders/run-tasks.md) — Learn more about GradingRunner options
- [Built-in Graders](../graders/overview.md) — Explore all available graders
- [Create Custom Graders](../building-graders/custom-graders.md) — Build domain-specific evaluators
- [Integrate with VERL](../integrations/rewards/verl.md) — Connect rewards to RLHF training
