# Select and Rank Models

Compare multiple model versions using pairwise evaluation to determine which performs best. This approach eliminates the need for absolute scoring by directly comparing responses head-to-head.

**When to use:**
- Comparing different model versions (v1 vs v2 vs v3)
- A/B testing prompt variations
- Selecting the best model for deployment
- Benchmarking against competitors

---

## How It Works

Pairwise evaluation compares every pair of model outputs and determines a winner. The final ranking is based on **win rates** — how often each model wins against others.

```
Model A vs Model B → Winner: A
Model A vs Model C → Winner: C
Model B vs Model C → Winner: B

Win Rates: A=50%, B=50%, C=50%
```

To eliminate position bias, each pair is evaluated twice with swapped order (A vs B and B vs A).

---

## Three-Step Pipeline

The evaluation follows a clear three-step pipeline:

| Step | Function | Description |
|------|----------|-------------|
| 1 | `prepare_comparison_data()` | Generate all pairwise combinations |
| 2 | `run_pairwise_evaluation()` | Run LLM-based comparisons |
| 3 | `analyze_and_rank_models()` | Compute win rates and rankings |

---

## Quick Start

```python
import asyncio
from tutorials.cookbooks.evaluation_cases.pairwise_evaluation import evaluate_task

async def main():
    instruction = "Write a short poem about artificial intelligence"

    model_outputs = {
        "model_v1": "Silicon minds awake at dawn, thinking thoughts not yet withdrawn.",
        "model_v2": "Circuits pulse with electric thought, patterns learned, connections wrought.",
        "model_v3": "Binary dreams and neural nets, learning more with no regrets.",
    }

    results = await evaluate_task(instruction, model_outputs)

    # Best model
    print(f"Best: {results['pairwise'].best_model}")

    # Full rankings
    for rank, (model, win_rate) in enumerate(results['pairwise'].rankings, 1):
        print(f"{rank}. {model}: {win_rate:.1%}")

asyncio.run(main())
```

**Output:**

```
Best: model_v2
1. model_v2: 66.7%
2. model_v1: 50.0%
3. model_v3: 33.3%
```

---

## Step-by-Step Guide

**Step 1: Prepare Comparison Data**

Generate all pairwise combinations from your model outputs:

```python
from tutorials.cookbooks.evaluation_cases.pairwise_evaluation import prepare_comparison_data

instruction = "Explain quantum computing in simple terms"

model_outputs = {
    "gpt-4": "Quantum computers use qubits that can be 0 and 1 simultaneously...",
    "claude": "Think of quantum computing like a maze solver...",
    "gemini": "Classical computers use bits, quantum computers use qubits...",
}

dataset, model_names = prepare_comparison_data(instruction, model_outputs)

print(f"Models: {model_names}")
print(f"Comparisons: {len(dataset)}")  # 6 comparisons for 3 models
```

For N models, this generates N×(N-1) comparisons (each pair evaluated twice to eliminate position bias).

**Step 2: Run Pairwise Evaluation**

Execute the comparisons using an LLM judge:

```python
from tutorials.cookbooks.evaluation_cases.pairwise_evaluation import run_pairwise_evaluation

grader_results = await run_pairwise_evaluation(
    dataset,
    max_concurrency=10,  # Parallel comparisons
)

print(f"Completed: {len(grader_results)} evaluations")
```

The grader returns:
- `score=1.0` → Response A wins
- `score=0.0` → Response B wins

**Step 3: Analyze and Rank**

Compute win rates and generate rankings:

```python
from tutorials.cookbooks.evaluation_cases.pairwise_evaluation import analyze_and_rank_models

analysis = analyze_and_rank_models(dataset, grader_results, model_names)

print(f"Best model: {analysis.best_model}")
print(f"Worst model: {analysis.worst_model}")

# Win rates
for model, rate in analysis.win_rates.items():
    print(f"{model}: {rate:.1%}")

# Win matrix (who beats whom)
for model_a, opponents in analysis.win_matrix.items():
    for model_b, rate in opponents.items():
        print(f"{model_a} beats {model_b}: {rate:.1%}")
```

---

## Understanding Results

The `PairwiseAnalysisResult` contains:

| Field | Type | Description |
|-------|------|-------------|
| `rankings` | List[Tuple] | Models sorted by win rate (best first) |
| `win_rates` | Dict[str, float] | Win rate for each model (0.0-1.0) |
| `win_matrix` | Dict[str, Dict] | Head-to-head win rates |
| `best_model` | str | Model with highest win rate |
| `worst_model` | str | Model with lowest win rate |
| `total_comparisons` | int | Number of pairwise comparisons |

**Win Matrix Example:**

```
           gpt-4    claude   gemini
gpt-4       --       0.75     0.50
claude     0.25       --      0.75
gemini     0.50      0.25      --
```

This shows gpt-4 beats claude 75% of the time, but only beats gemini 50% of the time.

---

## Configuration Options

**Concurrency:**

```python
results = await evaluate_task(
    instruction,
    model_outputs,
    max_concurrency=20,  # More parallel comparisons
)
```

**Custom Judge Model:**

To use a different judge model, modify `run_pairwise_evaluation()`:

```python
from rm_gallery.core.models import OpenAIChatModel

model = OpenAIChatModel(model="qwen3-32b")  # Your judge model
```

---

## Complete Example

```python
import asyncio
from tutorials.cookbooks.evaluation_cases.pairwise_evaluation import (
    prepare_comparison_data,
    run_pairwise_evaluation,
    analyze_and_rank_models,
)

async def compare_models():
    # Define task and model outputs
    instruction = "Write a professional email declining a meeting request"

    model_outputs = {
        "baseline": "I can't make the meeting. Sorry.",
        "v1_improved": "Thank you for the invitation. Unfortunately, I have a conflict...",
        "v2_polished": "I appreciate you thinking of me for this meeting. Regrettably...",
    }

    # Step 1: Prepare data
    dataset, model_names = prepare_comparison_data(instruction, model_outputs)
    print(f"Prepared {len(dataset)} comparisons")

    # Step 2: Run evaluation
    results = await run_pairwise_evaluation(dataset, max_concurrency=10)
    print(f"Completed {len(results)} evaluations")

    # Step 3: Analyze results
    analysis = analyze_and_rank_models(dataset, results, model_names)

    # Display rankings
    print("\n=== Model Rankings ===")
    for rank, (model, win_rate) in enumerate(analysis.rankings, 1):
        print(f"{rank}. {model}: {win_rate:.1%}")

    print(f"\nBest model: {analysis.best_model}")

    return analysis

# Run
analysis = asyncio.run(compare_models())
```

---

## Best Practices

1. **Use sufficient models**: At least 3 models for meaningful comparisons
2. **Consistent instructions**: Use the same task for all models
3. **Position bias elimination**: The pipeline automatically swaps order
4. **Adequate concurrency**: Set `max_concurrency` based on your API limits
5. **Save results**: Use `save_evaluation_results()` for reproducibility

---

## Next Steps

- [Refine Data Quality](data-refinement.md) — Filter and improve training data
- [Build Reward for Training](../get_started/build-reward.md) — Use rankings for RLHF
- [General Graders](../graders/general.md) — Available evaluation criteria

