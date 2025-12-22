# Validate on RewardBench2

Validate your graders against RewardBench2, a comprehensive benchmark for evaluating response quality across multiple domains. RewardBench2 provides standardized test cases covering factuality, focus, safety, math, instruction following, and specialized domains.

---

## What is RewardBench2?

RewardBench2 is a benchmark dataset designed to evaluate reward models and LLM judges. It tests grader performance across diverse scenarios to measure evaluation quality and identify systematic biases.

**Key Features:**

- **Multi-Domain Coverage** — Factuality, focus, safety, math, precise instruction following
- **Multiple Evaluation Modes** — Ranking (best of 4) and absolute rating (1-10 scale)
- **Standardized Ground Truth** — Expert-curated correct answers
- **Bias Testing** — Position bias, length bias, and adversarial cases
- **Public Leaderboard** — Compare your grader with state-of-the-art models

### Dataset Statistics

| Subset | Samples | Task Type | Ground Truth |
|--------|---------|-----------|--------------|
| **Factuality** | 475 | Four-way ranking | Best response among 4 candidates |
| **Focus** | 495 | Four-way ranking | Best response among 4 candidates |
| **Math** | 183 | Four-way ranking | Best response among 4 candidates |
| **Precise IF** | 160 | Four-way ranking | Best response among 4 candidates |
| **Safety** | 450 | Four-way ranking | Best response among 4 candidates |
| **Ties** | 102 | Absolute rating | Multiple correct answers (1-26 per sample) |
| **Total** | **1,865** | - | - |

---

## Evaluation Modes

RewardBench2 uses two complementary evaluation approaches:

### Four-Way Comparison (Default)

Present 4 candidate responses and select the best one.

```
Query: "Explain quantum computing"

Candidates:
├─ A: "Quantum computing leverages quantum mechanics..." ← Best (Ground Truth)
├─ B: "It's a type of advanced computing..."
├─ C: "Computers that use quantum physics..."
└─ D: "I'm not sure about that."

Grader Task: Select the best response (A/B/C/D)
Scoring: Correct if grader selects A, incorrect otherwise
```

**Features:**
- Random shuffling to prevent position bias
- Tests comparative judgment ability
- Binary outcome (correct/incorrect)

### Ties Absolute Rating

Rate each response independently on a 1-10 scale.

```
Query: "Write a creative poem about nature"

Candidates with Ground Truth Ratings:
├─ A: "The forest whispers..." → Rating: 9/10 ✓ Winner
├─ B: "Trees and flowers..." → Rating: 9/10 ✓ Winner (tie)
├─ C: "Nature is nice..." → Rating: 5/10
└─ D: "Roses are red..." → Rating: 6/10

Grader Task: Rate each response (1-10)
Scoring: Correct if any highest-rated response is a ground truth winner
```

**Features:**
- Allows multiple correct answers (ties)
- Tests absolute quality assessment
- More nuanced evaluation than binary ranking

---

## Quick Start

### Prerequisites

```bash
# Install RM-Gallery
pip install rm-gallery

# Install additional dependencies
pip install pandas pyarrow loguru
```

### Download Dataset

```python
from datasets import load_dataset

# Load RewardBench2 dataset
dataset = load_dataset("allenai/reward-bench-2", split="test")

# Save as parquet for faster loading
dataset.to_parquet("rewardbench2_test.parquet")
```

### Run Validation

```python
import asyncio
from rm_gallery.core.models import OpenAIChatModel
from tutorials.cookbooks.grader_validation.rewardbench2 import (
    RewardBench2Grader,
    RewardBench2Analyzer,
    load_rewardbench2_data
)

async def validate():
    # 1. Load data
    data = load_rewardbench2_data(
        "rewardbench2_test.parquet",
        max_samples=100  # Start small for testing
    )

    # 2. Initialize grader
    model = OpenAIChatModel(
        model="qwen3-32b",
        api_key="your-api-key"
    )
    grader = RewardBench2Grader(model=model)

    # 3. Run evaluation
    results = []
    for sample in data:
        result = await grader.aevaluate(
            query=sample["query"],
            answers=sample["answers"],
            subset=sample["subset"],
            chosen_indices=sample["chosen_indices"]
        )
        results.append(result)

    # 4. Analyze results
    analyzer = RewardBench2Analyzer()
    report = analyzer.analyze(dataset=data, grader_results=results)

    print(f"Overall Accuracy: {report.metadata['accuracy']:.2%}")
    for subset, metrics in report.metadata["subset_accuracy"].items():
        print(f"  {subset}: {metrics['accuracy']:.2%}")

# Run
asyncio.run(validate())
```

---

## Implementation Overview

The RewardBench2 validation implementation consists of three main components:

### Core Components

**1. RewardBench2Grader**
- Extends `BaseGrader` from RM-Gallery
- Automatically switches between four-way comparison and Ties rating based on subset
- Implements position shuffling to prevent bias
- Returns `GraderScore` with prediction and accuracy

**2. Response Parsers**
- `parse_four_way_response()`: Extracts [[A]]/[[B]]/[[C]]/[[D]] from LLM output
- `parse_ties_rating()`: Extracts numerical rating (1-10) using regex

**3. RewardBench2Analyzer**
- Extends `BaseAnalyzer` for result analysis
- Computes overall accuracy and per-subset breakdown
- Generates validation reports with detailed metrics

### Evaluation Flow

```
Input Sample → RewardBench2Grader
                    ↓
            Check subset type
                    ↓
        ┌───────────┴───────────┐
        ↓                       ↓
Four-Way Comparison       Ties Rating
(4 candidates)           (N candidates)
        ↓                       ↓
Shuffle & Prompt         Rate each independently
        ↓                       ↓
Parse [[X]] format       Parse numeric rating
        ↓                       ↓
        └───────────┬───────────┘
                    ↓
            Compare with ground truth
                    ↓
            Return GraderScore
```

### Key Implementation Details

**Position Bias Prevention:**
- Randomly shuffle candidate positions before evaluation
- Track shuffle mapping in metadata
- Map predicted position back to original index

**Error Handling:**
- Graceful fallback for parsing failures
- Invalid ratings default to -1 (excluded from analysis)
- Concurrent evaluation with semaphore control

**Complete implementation:** `tutorials/cookbooks/grader_validation/rewardbench2.py`

---

## Running Full Validation

Follow these steps to run complete validation:

```bash
# 1. Navigate to validation directory
cd tutorials/cookbooks/grader_validation

# 2. Run validation script
python rewardbench2.py \
    --data-path /path/to/rewardbench2_test.parquet \
    --model qwen3-32b \
    --max-samples 100 \
    --max-concurrency 8
```

**Key Configuration Options:**
- `--data-path`: Path to RewardBench2 parquet file
- `--model`: Model name (e.g., qwen3-32b, gpt-4)
- `--max-samples`: Number of samples to validate (-1 for all)
- `--max-concurrency`: Concurrent API calls (default: 8)

**Output:**
```
RewardBench2 Validation Report
=====================================
Overall Accuracy: 78.5%
Correct: 1465/1865

Per-Subset Performance:
  Factuality  : 82.3% (391/475)
  Focus       : 78.8% (390/495)
  ...
```

Results are saved to `rewardbench2_results.json` with detailed metrics.

**Complete script:** `tutorials/cookbooks/grader_validation/rewardbench2.py`

---

## Interpreting Results

### Overall Accuracy

The primary metric is overall accuracy across all subsets:

```
Overall Accuracy: 78.5%
Correct: 785/1000
```

**Interpretation:**
- **> 80%** — Excellent: Grader performs well across domains
- **70-80%** — Good: Reliable for most use cases
- **60-70%** — Fair: May need refinement for production use
- **< 60%** — Poor: Requires significant improvement

### Per-Subset Breakdown

```
Per-Subset Performance:
  Factuality      : 82.3% (391/475)
  Focus           : 78.8% (390/495)
  Math            : 65.0% (119/183)
  Precise IF      : 71.9% (115/160)
  Safety          : 88.4% (398/450)
  Ties            : 76.5% ( 78/102)
```

**Analysis:**
- **Strengths**: High Safety (88.4%) and Factuality (82.3%) accuracy
- **Weaknesses**: Lower Math (65.0%) suggests difficulty with mathematical reasoning
- **Action**: Review failed Math cases, consider adding domain-specific examples to prompt

### Common Error Patterns

```python
# Analyze errors by category
errors_by_subset = {}
for sample, result in zip(validation_data, results):
    if result.score < 1.0:  # Incorrect
        subset = sample["subset"]
        if subset not in errors_by_subset:
            errors_by_subset[subset] = []
        errors_by_subset[subset].append({
            "query": sample["query"],
            "predicted": result.metadata.get("predicted_letter"),
            "correct": result.metadata.get("correct_letter"),
            "reason": result.reason
        })

# Review errors
for subset, errors in errors_by_subset.items():
    print(f"\n{subset} Errors ({len(errors)}):")
    for error in errors[:3]:  # Show first 3
        print(f"  Query: {error['query'][:80]}...")
        print(f"  Predicted: {error['predicted']}, Correct: {error['correct']}")
```

---

## Optimizing Your Grader

### Improve Four-Way Comparison

```python
# Enhanced system prompt with examples
IMPROVED_SYSTEM_PROMPT = """
Please act as an impartial judge and evaluate the quality of the responses provided by four AI assistants.

Focus on:
- Helpfulness: Does it answer the question?
- Accuracy: Is the information correct?
- Clarity: Is it well-explained?
- Completeness: Does it cover all aspects?

Important guidelines:
- Ignore response length (longer ≠ better)
- Ignore response position (first shown ≠ best)
- Look for factual errors or misconceptions
- Prefer specific, actionable answers over vague ones

Output your final verdict: [[A]], [[B]], [[C]], or [[D]]
"""
```

### Improve Ties Rating

```python
# More calibrated rating prompt
IMPROVED_TIES_PROMPT = """
Rate this response on a 1-10 scale:

1-3: Poor (incorrect, unhelpful, or harmful)
4-5: Below average (partially correct, lacks depth)
6-7: Good (correct and helpful)
8-9: Excellent (comprehensive, well-explained)
10: Outstanding (exceptional quality, insights)

Query: {query}
Response: {response}

Provide your reasoning, then output your rating (1-10) on the last line.
"""
```

### Reduce Position Bias

```python
# Test for position bias
def test_position_bias(grader, test_samples):
    """Check if grader favors certain positions."""
    position_counts = {"A": 0, "B": 0, "C": 0, "D": 0}

    for sample in test_samples:
        result = await grader.aevaluate(
            query=sample["query"],
            answers=sample["answers"],
            subset=sample["subset"],
            chosen_indices=sample["chosen_indices"]
        )

        predicted = result.metadata.get("predicted_letter")
        position_counts[predicted] += 1

    total = sum(position_counts.values())
    for pos, count in position_counts.items():
        print(f"Position {pos}: {count/total:.1%}")

    # Expected: ~25% each if no bias
    # If one position > 35%, likely position bias exists
```

---

## Advanced Usage

### Custom Subsets

Add domain-specific validation data by creating custom samples with the same format:

```python
# Add custom samples to validation set
custom_samples = [{
    "query": "Your domain question",
    "answers": [answer1, answer2, answer3, answer4],
    "subset": "custom_domain",
    "chosen_indices": [0],
    "id": "custom_001"
}]

# Merge and validate
full_set = rewardbench2_data + custom_samples
results = await run_validation(grader, full_set)
```

### Ensemble & Comparison

Compare multiple graders to find the best performer:

```python
async def compare_graders():
    """Compare multiple graders on the same dataset."""
    graders = [
        RewardBench2Grader(model=model_a),
        RewardBench2Grader(model=model_b),
        YourCustomGrader()
    ]

    for grader in graders:
        # Run evaluation
        results = []
        for sample in data:
            result = await grader.aevaluate(
                query=sample["query"],
                answers=sample["answers"],
                subset=sample.get("subset"),
                chosen_indices=sample.get("chosen_indices")
            )
            results.append(result)

        # Analyze results
        analyzer = RewardBench2Analyzer()
        report = analyzer.analyze(dataset=data, grader_results=results)
        print(f"{grader.name}: {report.metadata['accuracy']:.2%}")
```

### Error Analysis

Analyze disagreements between graders or failed cases to identify systematic issues. See [Overview](overview.md#advanced-validation-techniques) for detailed error analysis techniques.

---

## Troubleshooting

### Common Issues

| Issue | Symptoms | Solutions |
|-------|----------|-----------|
| **Low Ties Accuracy** | Poor on absolute rating | Use structured output (Pydantic models) for guaranteed numeric ratings |
| **Position Bias** | Favors first/last position | Add explicit anti-bias instructions in system prompt; verify shuffling |
| **Parsing Failures** | Many default/fallback values | Implement multi-strategy parser (check [[X]], "Answer X", standalone letter) |
| **Inconsistent Results** | Different outputs for same input | Lower temperature (0.0), use structured output, add few-shot examples |
| **Low Overall Accuracy** | <60% across all subsets | Review failed cases, refine prompts, consider using stronger model |

### Debugging Steps

1. **Analyze error patterns** by subset to identify systematic issues
2. **Test on small sample** (10-20) with logging enabled
3. **Review LLM responses** for formatting or reasoning problems
4. **Check position distribution** - should be ~25% for each position (A/B/C/D)
5. **Compare with baseline** - random selection should get ~25% accuracy

For detailed troubleshooting techniques, see [Overview - Troubleshooting](overview.md#troubleshooting-low-accuracy)

---

## Next Steps

### Improve Your Grader
- **[Overview](overview.md)** — Learn validation concepts and best practices
- **[Create Custom Graders](../building-graders/custom-graders.md)** — Refine your grader implementation
- **[Train Reward Models](../building-graders/training/overview.md)** — Train models on RewardBench2 data

### Run More Validations
- **MT-Bench Validation** — Coming soon
- **AlpacaEval Validation** — Coming soon
- **Custom Domain Validation** — Adapt RewardBench2 code for your domain

### Deploy Validated Graders
- **[Running Graders](../running-graders/run-tasks.md)** — Set up production evaluation pipelines
- **[Integration Guides](../integrations/)** — Deploy in your applications
- **[Monitor Performance](../running-graders/evaluation-reports.md)** — Track grader accuracy over time

---

## Reference

### Full Implementation

The complete implementation is available at:
```
tutorials/cookbooks/grader_validation/rewardbench2.py
```

### Key Classes

- `RewardBench2Grader` — Main grader implementation
- `RewardBench2Analyzer` — Results analyzer
- `load_rewardbench2_data()` — Data loader
- `parse_four_way_response()` — Four-way parser
- `parse_ties_rating()` — Ties rating parser

### Dataset Format

```python
{
    "query": str,              # User question
    "answers": List[List[Dict]],  # Message format: [{"role": "user", "content": ...}, {"role": "assistant", "content": ...}]
    "subset": str,             # "Factuality", "Focus", "Math", "Precise IF", "Safety", "Ties"
    "chosen_indices": List[int],  # Indices of correct/best answers
    "id": str                  # Sample identifier
}
```

