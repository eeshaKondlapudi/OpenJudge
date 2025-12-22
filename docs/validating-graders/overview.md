# Validating Graders

Ensure your graders produce accurate, reliable evaluation results by validating them against benchmark datasets or custom test sets. Grader validation helps you measure evaluation quality before deploying graders in production.

---

## Why Validate Graders?

Graders are themselves evaluation systems that need validation. Without validation, you risk deploying unreliable evaluators that produce misleading results.

**Key Benefits:**

- **Measure Accuracy** — Quantify how often your grader agrees with ground truth labels
- **Compare Approaches** — Benchmark LLM judges vs. trained models vs. rule-based logic
- **Build Confidence** — Validate graders meet quality thresholds before production use
- **Debug Issues** — Identify systematic errors or biases in evaluation logic
- **Track Improvements** — Monitor grader performance over time as you refine them

---

## What is Grader Validation?

Grader validation evaluates your grader's evaluation quality by comparing its judgments against known ground truth.

### Validation Workflow

```
┌─────────────────────────────────────────────────┐
│  Validation Dataset                             │
│  ├─ Query: "What is quantum computing?"         │
│  ├─ Candidates: [Response A, B, C, D]          │
│  └─ Ground Truth: "Response A is best"         │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│  Your Grader Under Test                         │
│  ├─ Evaluates all candidate responses          │
│  └─ Prediction: "Response A is best"           │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│  Validation Analyzer                            │
│  ├─ Compare: Prediction vs. Ground Truth       │
│  ├─ Compute: Accuracy, F1, correlation         │
│  └─ Report: Overall and per-category metrics   │
└─────────────────────────────────────────────────┘
```

### Key Components

| Component | Purpose | Example |
|-----------|---------|---------|
| **Validation Dataset** | Known ground truth for evaluation | RewardBench2, MT-Bench, custom test sets |
| **Grader Under Test** | The grader you want to validate | Your custom LLM judge or trained model |
| **Validation Analyzer** | Computes accuracy metrics | Accuracy, precision, recall, correlation |
| **Validation Report** | Human-readable results | Per-category breakdown, error analysis |

---

## Validation Approaches

RM-Gallery supports two validation approaches depending on your needs:

### 1. Benchmark Validation

Validate against public benchmarks with standardized ground truth.

**When to Use:**
- ✅ Want to compare with published baselines
- ✅ Need reproducible validation results
- ✅ Evaluating general-purpose graders
- ✅ Quick validation without data collection

**Available Benchmarks:**
- **[RewardBench2](rewardbench2.md)** — Multi-domain response quality evaluation
- **MT-Bench** — Multi-turn conversation quality (coming soon)
- **AlpacaEval** — Instruction-following evaluation (coming soon)

### 2. Custom Validation

Build validation pipelines tailored to your domain and evaluation criteria.

**When to Use:**
- ✅ Domain-specific evaluation (legal, medical, finance)
- ✅ Proprietary test sets with internal standards
- ✅ Non-standard evaluation tasks
- ✅ Need control over validation methodology

---

## Quick Start: Validate a Grader

### Step 1: Prepare Validation Data

Your validation dataset needs:
1. **Queries** — Input prompts or questions
2. **Candidate Responses** — Multiple responses to evaluate
3. **Ground Truth** — Known correct answers or rankings

```python
validation_data = [
    {
        "query": "Explain quantum computing",
        "candidates": [
            "Quantum computing uses quantum mechanics...",  # Best response
            "It's a type of computer...",                   # Good response
            "Computers that are very fast...",              # Poor response
            "I don't know."                                 # Worst response
        ],
        "ground_truth_rank": [0]  # Index 0 is the best response
    },
    # ... more samples
]
```

### Step 2: Create Your Grader

```python
from rm_gallery.core.graders.llm_grader import LLMGrader
from rm_gallery.core.models import OpenAIChatModel

model = OpenAIChatModel(model="qwen3-32b")

grader = LLMGrader(
    name="my_grader",
    model=model,
    template="""
    Evaluate the quality of this response:

    Query: {query}
    Response: {response}

    Return JSON: {{"score": <0.0-1.0>, "reason": "<explanation>"}}
    """
)
```

### Step 3: Run Validation

```python
from rm_gallery.core.runner import GradingRunner
from rm_gallery.core.analyzer import ValidationAnalyzer

# Initialize runner
runner = GradingRunner(
    grader_configs={"my_grader": grader},
    max_concurrency=8,
    show_progress=True
)

# Build dataset for runner
dataset = []
for sample in validation_data:
    for candidate in sample["candidates"]:
        dataset.append({
            "query": sample["query"],
            "response": candidate
        })

# Run grading on validation set
results = await runner.arun(dataset)

# Extract grader results
grader_results = results["my_grader"]

# Analyze results
analyzer = ValidationAnalyzer()
report = analyzer.analyze(
    dataset=validation_data,
    grader_results=grader_results
)

print(f"Accuracy: {report.metadata['accuracy']:.2%}")
print(f"Valid Samples: {report.metadata['valid_samples']}")
```

### Step 4: Review Results

```python
# Overall metrics
print(f"Overall Accuracy: {report.metadata['accuracy']:.2%}")
print(f"Correct: {report.metadata['correct_count']}/{report.metadata['total_samples']}")

# Per-category breakdown (if available)
if "subset_accuracy" in report.metadata:
    for category, metrics in report.metadata["subset_accuracy"].items():
        print(f"{category}: {metrics['accuracy']:.2%} ({metrics['correct_count']}/{metrics['total_samples']})")
```

---

## Building Custom Validation

Create domain-specific validation pipelines by implementing three key components.

### Component 1: Custom Grader

Implement your evaluation logic using `BaseGrader`:

```python
from rm_gallery.core.graders.base_grader import BaseGrader, GraderMode
from rm_gallery.core.graders.schema import GraderScore
from typing import List, Any

class MyCustomGrader(BaseGrader):
    """Custom grader for domain-specific evaluation."""

    def __init__(self, model, name="custom_grader"):
        super().__init__(
            name=name,
            mode=GraderMode.LISTWISE,  # or POINTWISE/PAIRWISE
            description="My custom evaluation logic"
        )
        self.model = model

    async def aevaluate(
        self,
        query: str,
        candidates: List[str],
        **kwargs
    ) -> GraderScore:
        """
        Evaluate candidates and return best choice.

        Args:
            query: Input prompt
            candidates: List of responses to evaluate

        Returns:
            GraderScore with predicted ranking
        """
        # Your evaluation logic here
        scores = []
        for candidate in candidates:
            # Evaluate each candidate
            result = await self.model.achat([
                {"role": "user", "content": f"Rate this: {candidate}"}
            ])
            score = self._parse_score(result)
            scores.append(score)

        best_index = scores.index(max(scores))

        return GraderScore(
            name=self.name,
            score=float(best_index),  # Return predicted best index
            reason=f"Selected candidate {best_index}",
            metadata={"scores": scores}
        )
```

### Component 2: Data Loader

Load and format your validation dataset:

```python
import pandas as pd
from typing import List, Dict

def load_validation_data(file_path: str) -> List[Dict]:
    """
    Load validation data from file.

    Expected format:
    - query: str
    - candidates: List[str]
    - ground_truth: int or List[int]
    - category: str (optional, for subset analysis)
    """
    df = pd.read_parquet(file_path)  # or .csv, .json

    data = []
    for _, row in df.iterrows():
        data.append({
            "query": row["query"],
            "candidates": row["candidates"],
            "ground_truth": row["ground_truth"],
            "category": row.get("category", "default")
        })

    return data
```

### Component 3: Validation Analyzer

Compute accuracy metrics and generate reports:

```python
from rm_gallery.core.analyzer.base_analyzer import BaseAnalyzer, AnalysisResult
from typing import List, Any

class MyValidationAnalyzer(BaseAnalyzer):
    """Analyzer for custom validation metrics."""

    name: str = "Custom Validation"

    def analyze(
        self,
        dataset: List[dict],
        grader_results: List[Any],
        **kwargs
    ) -> AnalysisResult:
        """
        Analyze grader results against ground truth.

        Args:
            dataset: Validation dataset with ground truth
            grader_results: Grader predictions

        Returns:
            AnalysisResult with accuracy metrics
        """
        correct = 0
        total = 0
        category_stats = {}

        for sample, result in zip(dataset, grader_results):
            if not result:
                continue

            # Compare prediction with ground truth
            predicted = int(result.score)  # Predicted best index
            ground_truth = sample["ground_truth"]

            is_correct = predicted == ground_truth

            if is_correct:
                correct += 1
            total += 1

            # Track per-category stats
            category = sample.get("category", "default")
            if category not in category_stats:
                category_stats[category] = {"correct": 0, "total": 0}

            category_stats[category]["total"] += 1
            if is_correct:
                category_stats[category]["correct"] += 1

        # Compute metrics
        accuracy = correct / total if total > 0 else 0.0

        category_accuracy = {
            cat: stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
            for cat, stats in category_stats.items()
        }

        return AnalysisResult(
            name=self.name,
            metadata={
                "accuracy": accuracy,
                "correct_count": correct,
                "total_samples": total,
                "category_accuracy": category_accuracy,
                "category_stats": category_stats
            }
        )
```

---

## Complete Validation Example

Combine all components into a full validation pipeline:

```python
import asyncio
from rm_gallery.core.models import OpenAIChatModel
from rm_gallery.core.runner import GradingRunner

async def validate_grader():
    """Complete grader validation workflow."""

    # 1. Load validation data
    validation_data = load_validation_data("validation_set.parquet")
    print(f"Loaded {len(validation_data)} validation samples")

    # 2. Initialize grader
    model = OpenAIChatModel(model="qwen3-32b")
    grader = MyCustomGrader(model=model)

    # 3. Run evaluation
    runner = GradingRunner(
        grader_configs={"custom_grader": grader},
        max_concurrency=8,
        show_progress=True
    )

    # Build dataset for runner
    dataset = []
    for sample in validation_data:
        dataset.append({
            "query": sample["query"],
            "candidates": sample["candidates"]
        })

    # Run evaluation
    results = await runner.arun(dataset)
    grader_results = results["custom_grader"]

    # 4. Analyze results
    analyzer = MyValidationAnalyzer()
    report = analyzer.analyze(
        dataset=validation_data,
        grader_results=grader_results
    )

    # 5. Print results
    print(f"\n{'='*50}")
    print(f"Validation Report: {analyzer.name}")
    print(f"{'='*50}")
    print(f"Overall Accuracy: {report.metadata['accuracy']:.2%}")
    print(f"Correct: {report.metadata['correct_count']}/{report.metadata['total_samples']}")
    print(f"\nPer-Category Accuracy:")

    for category, accuracy in report.metadata["category_accuracy"].items():
        stats = report.metadata["category_stats"][category]
        print(f"  {category}: {accuracy:.2%} ({stats['correct']}/{stats['total']})")

    return report

# Run validation
report = asyncio.run(validate_grader())
```

---

## Validation Metrics

Choose metrics based on your evaluation task:

### Ranking Metrics

For graders that rank or select best responses:

| Metric | When to Use | Interpretation |
|--------|-------------|----------------|
| **Accuracy** | Binary classification (correct/incorrect) | % of times grader selects correct answer |
| **Top-K Accuracy** | Multiple acceptable answers | % of times correct answer is in top K predictions |
| **Mean Reciprocal Rank (MRR)** | Ranking quality | Average reciprocal rank of correct answer |
| **Kendall's Tau** | Full ranking correlation | Agreement between predicted and true rankings |

### Scoring Metrics

For graders that output continuous scores:

| Metric | When to Use | Interpretation |
|--------|-------------|----------------|
| **Pearson Correlation** | Linear relationship | How well scores correlate with ground truth |
| **Spearman Correlation** | Ranking correlation | Agreement in relative ordering |
| **Mean Absolute Error (MAE)** | Score accuracy | Average distance from ground truth scores |
| **F1 Score** | Binary threshold (pass/fail) | Balance between precision and recall |

---

## Best Practices

### Data Quality

1. **Diverse Test Cases** — Include edge cases, ambiguous queries, domain-specific content
2. **Sufficient Size** — Aim for 100+ samples for reliable accuracy estimates
3. **Balanced Categories** — Ensure each category has enough samples (20+ minimum)
4. **Clear Ground Truth** — Use high-agreement human annotations or verified labels
5. **Hold-Out Sets** — Never validate on training data for trained models

### Validation Design

1. **Match Production** — Validation should mirror real-world use cases
2. **Control Bias** — Randomize answer positions to prevent position bias
3. **Multiple Runs** — Run validation multiple times with different random seeds
4. **Error Analysis** — Review failed cases to identify systematic issues
5. **Threshold Tuning** — Adjust score thresholds based on validation results

### Interpretation

1. **Context Matters** — 70% accuracy may be excellent or poor depending on task difficulty
2. **Compare Baselines** — Validate against random/majority baselines and existing graders
3. **Per-Category Analysis** — Overall accuracy may hide category-specific weaknesses
4. **Statistical Significance** — Use confidence intervals for small validation sets
5. **Human Agreement** — Compare grader accuracy to inter-annotator agreement

---

## Advanced Validation Techniques

### Cross-Validation

Validate robustness by testing on multiple data splits:

```python
from sklearn.model_selection import KFold

def cross_validate_grader(data, grader, n_splits=5):
    """Run k-fold cross-validation."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    accuracies = []

    for train_idx, test_idx in kf.split(data):
        test_data = [data[i] for i in test_idx]

        # Run validation on this fold
        results = await validate_on_fold(grader, test_data)
        accuracies.append(results["accuracy"])

    print(f"Mean Accuracy: {np.mean(accuracies):.2%} ± {np.std(accuracies):.2%}")
```

### Adversarial Testing

Test robustness to input perturbations:

```python
# Add challenging cases
adversarial_cases = [
    {"query": "...", "candidates": [...], "type": "similar_responses"},
    {"query": "...", "candidates": [...], "type": "negation_sensitivity"},
    {"query": "...", "candidates": [...], "type": "position_bias"},
]

# Analyze per adversarial type
for adv_type in ["similar_responses", "negation_sensitivity", "position_bias"]:
    type_cases = [c for c in adversarial_cases if c["type"] == adv_type]
    accuracy = validate_subset(grader, type_cases)
    print(f"{adv_type}: {accuracy:.2%}")
```

### Confidence Calibration

Assess whether grader confidence matches actual accuracy:

```python
def analyze_calibration(results, ground_truth):
    """Check if high-confidence predictions are more accurate."""
    bins = {"high": [], "medium": [], "low": []}

    for result, truth in zip(results, ground_truth):
        confidence = result.metadata.get("confidence", result.score)
        is_correct = (result.score >= 0.5) == truth

        if confidence > 0.8:
            bins["high"].append(is_correct)
        elif confidence > 0.5:
            bins["medium"].append(is_correct)
        else:
            bins["low"].append(is_correct)

    for level, correct_list in bins.items():
        if correct_list:
            acc = sum(correct_list) / len(correct_list)
            print(f"{level} confidence: {acc:.2%} ({len(correct_list)} samples)")
```

---

## Troubleshooting Low Accuracy

### Common Issues and Solutions

| Issue | Symptoms | Solutions |
|-------|----------|-----------|
| **Position Bias** | Grader favors first/last responses | Randomize candidate order, add position bias penalty |
| **Length Bias** | Prefers longer/shorter responses | Normalize by length, add length controls to prompt |
| **Prompt Mismatch** | Good on some categories, poor on others | Refine prompts per category, use category-specific graders |
| **Inconsistency** | Different results on same input | Reduce temperature, use structured output, add examples |
| **Overfitting** | High on training data, low on validation | Collect more training data, simplify model, regularize |

### Debugging Workflow

```python
# 1. Analyze failed cases
failed_cases = [
    (sample, result)
    for sample, result in zip(validation_data, results)
    if result.score < 0.5  # Assuming binary threshold
]

print(f"Failed {len(failed_cases)}/{len(validation_data)} cases")

# 2. Review error patterns
for sample, result in failed_cases[:10]:  # Review first 10
    print(f"\nQuery: {sample['query'][:100]}")
    print(f"Predicted: {result.score}, Ground Truth: {sample['ground_truth']}")
    print(f"Reason: {result.reason}")

# 3. Test hypothesis fixes
# Add position shuffling, refine prompts, adjust thresholds, etc.
```

---

## Next Steps

Start validating your graders:

### Use Public Benchmarks
- **[RewardBench2 Validation](rewardbench2.md)** — Validate on multi-domain response quality benchmark
- **MT-Bench Validation** — Coming soon
- **AlpacaEval Validation** — Coming soon

### Build Custom Validation
- **[Create Custom Graders](../building-graders/custom-graders.md)** — Build graders to validate
- **[Running Graders](../running-graders/run-tasks.md)** — Set up batch evaluation pipelines
- **[Statistical Analysis](../running-graders/evaluation-reports.md)** — Generate detailed reports

### Improve Your Graders
- **[Train Reward Models](../building-graders/training/overview.md)** — Train models on your data
- **[LLM Grader Best Practices](../building-graders/custom-graders.md#llm-grader-tips)** — Optimize LLM judges
- **[Grader Ensemble Methods](../applications/select-rank.md)** — Combine multiple graders

### Deploy with Confidence
- **[Integration Guides](../integrations/)** — Deploy validated graders in production
- **[Monitor Performance](../running-graders/evaluation-reports.md)** — Track grader accuracy over time

