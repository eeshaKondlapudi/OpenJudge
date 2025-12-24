# Deep Research Agent Evaluation

A comprehensive evaluation framework for deep research agents, assessing report quality, factual accuracy, trajectory efficiency, and information gain through six specialized graders.

## Overview

| Grader | Purpose | Input | Score Calculation | LLM |
|--------|---------|-------|-------------------|-----|
| **Financial Report Resolution** | Multi-dimensional report quality (precision, completeness, relevance, timeliness, logic, data, structure) | `messages`, `chat_date` | Weighted avg of 7 dimensions (25%, 20%, 10%, 10%, 10%, 15%, 10%) | ✅ |
| **Financial Trajectory Faithfulness** | Factual accuracy verification via tuple extraction | `messages` | `1 - (error_tuples / total_tuples)` | ✅ |
| **Rubrics-Based Performance** | Custom criteria evaluation with check points | `messages`, `rubrics` | Average pass rate across dimensions | ✅ |
| **Trajectory Comprehensive** | Step-by-step contribution assessment | `messages` | Average of step scores (4 dimensions per step) | ✅ |
| **Observation Information Gain** | Information redundancy detection | `messages` | Avg info gain with exponential penalty | ❌ |
| **Action Loop Detection** | Repetitive action detection | `messages` | `1 - (similar_pairs / total_pairs)` | ❌ |

## Quick Start

```python
import asyncio
from rm_gallery.core.models.openai_chat_model import OpenAIChatModel
from rm_gallery.core.runner.grading_runner import GradingRunner
from tutorials.deep_research.deep_research_evaluation import create_grader_configs

# Initialize model and configs
model = OpenAIChatModel(model="gpt-4o", temperature=0.0)
grader_configs = create_grader_configs(model)

# Create runner
runner = GradingRunner(
    grader_configs=grader_configs,
    max_concurrency=6,
    show_progress=True,
)

# Run evaluation
dataset = [{
    "messages": [...],           # Required: conversation history
    "chat_date": "2025-07-15",   # Optional: for report resolution
    "rubrics": [...],            # Optional: for rubrics evaluation
}]
results = await runner.arun(dataset)
```

## Input Format

### Messages Structure (Required)

Standard conversation format with tool calls:

```python
{
    "messages": [
        {"role": "user", "content": "分析贵州茅台2025年上半年的财务表现"},
        {
            "role": "assistant",
            "tool_calls": [{
                "function": {
                    "name": "search_financial_data",
                    "arguments": '{"company": "贵州茅台", "period": "2025H1"}'
                }
            }]
        },
        {"role": "tool", "content": "截至2025年6月30日，贵州茅台实现营业收入789.5亿元..."},
        {"role": "assistant", "content": "根据财务数据，贵州茅台2025年上半年表现优异..."}
    ]
}
```

### Rubrics Structure (Optional)

Custom evaluation criteria for rubrics-based grader:

```python
{
    "rubrics": [
        {
            "dimension": "数据准确性",
            "description": "报告中的数据是否准确无误",
            "check_points": ["营业收入数据正确", "净利润数据正确", "增长率计算准确"]
        }
    ]
}
```

## Grader Details

### 1. Financial Report Resolution

**Purpose**: Evaluates report quality across 7 weighted dimensions (1-5 scale normalized to 0-1).

**Dimensions**:
- Precision (25%): Direct addressing of query
- Completeness (20%): Coverage of key points
- Relevance (10%): Alignment with question
- Timeliness (10%): Use of latest data
- Logic (10%): Reasoning rigor
- Data Support (15%): Evidence backing
- Structure (10%): Organization and readability

**Output Example**:
```python
GraderScore(
    score=0.8875,  # Weighted average
    reason="【精准性】5/5分: ...\n【完整性】4/5分: ...",
    metadata={
        "dimension_scores": {"precision": {"raw": 5, "normalized": 1.0}, ...},
        "is_resolved": True,
        "resolution_threshold": 0.9
    }
)
```

### 2. Financial Trajectory Faithfulness

**Purpose**: Verifies factual accuracy by extracting financial tuples (subject, indicator, value, time) and checking against search results.

**Error Types**:
- `time_error`: Wrong year/quarter/date
- `value_error`: Incorrect numbers, precision loss
- `indicator_error`: Metric confusion
- `subject_error`: Entity confusion

**Output Example**:
```python
GraderScore(
    score=0.8333,  # 1 - (1 error / 6 tuples)
    reason="发现 1 个错误的六元组（总共 6 个）：\n  1. 【贵州茅台】净利润...",
    metadata={
        "total_tuples": 6,
        "error_count": 1,
        "error_tuples": [...],
        "all_tuples": [...]
    }
)
```

### 3. Rubrics-Based Performance

**Purpose**: Evaluates against custom criteria with check points marked as passed/failed.

**Calculation**: For each dimension, score = passed_checks / total_checks. Overall score = average across dimensions.

**Output Example**:
```python
GraderScore(
    score=0.8333,
    reason="总分: 0.8333\n【数据准确性】分数: 0.67 (通过: 2/3)\n  理由: ...",
    metadata={
        "dimension_scores": {"数据准确性": 0.67, "分析完整性": 1.0},
        "dimension_evaluations": [...]
    }
)
```

### 4. Trajectory Comprehensive

**Purpose**: Assesses each tool call step on 4 dimensions (1-5 scale):
- **Contribution**: Impact on problem solving
- **Relevance**: Alignment with query
- **Accuracy**: Information correctness
- **Efficiency**: Necessity and optimization

**Calculation**: Step score = avg(4 dimensions). Overall = avg(all steps).

**Output Example**:
```python
GraderScore(
    score=0.8125,
    reason="Step 0: 该步骤直接获取了核心财务数据...\nStep 1: ...",
    metadata={
        "avg_contribution": 0.875,
        "avg_relevance": 0.875,
        "avg_accuracy": 0.75,
        "avg_efficiency": 0.75,
        "step_evaluations": [...]
    }
)
```

### 5. Observation Information Gain

**Purpose**: Measures information redundancy (rule-based, no LLM).

**Algorithm**:
1. For each observation, calculate max similarity to previous observations
2. Info score = 1 - max_similarity
3. Apply exponential penalty: reward = info_score × exp(-2 × similarity_excess)
4. Final score = average reward

**Output Example**:
```python
GraderScore(
    score=0.7234,
    reason="Average info gain score across 3 observation steps: 0.723",
    metadata={
        "observation_count": 3,
        "each_turn_rewards": [1.0, 0.8, 0.38],
        "each_turn_similarity": [0.0, 0.2, 0.75],
        "similarity_threshold": 0.5
    }
)
```

### 6. Action Loop Detection

**Purpose**: Detects repetitive actions (rule-based, no LLM).

**Algorithm**:
1. Extract action signatures (function name + arguments)
2. Compare all action pairs for similarity
3. Count similar pairs (similarity >= threshold)
4. Score = 1 - (similar_pairs / total_pairs)

**Output Example**:
```python
GraderScore(
    score=0.8333,  # 1 - (1/6)
    reason="Loop detection: 1/6 pairs are similar (threshold=1.0)",
    metadata={
        "action_count": 4,
        "similar_pair_count": 1,
        "total_pair_count": 6,
        "similar_pairs": [...]
    }
)
```

## Mapper Configuration

The tutorial uses **lambda functions** for clean data mapping:

```python
grader_configs = {
    # Messages + optional date
    "report_resolution": GraderConfig(
        grader=FinancialReportResolutionGrader(model=model),
        mapper=lambda data: {
            "messages": data["messages"],
            "chat_date": data.get("chat_date")  # None if not present
        }
    ),
    
    # Messages only
    "trajectory_faithfulness": GraderConfig(
        grader=FinancialTrajectoryFaithfulGrader(model=model),
        mapper=lambda data: {"messages": data["messages"]}
    ),
    
    # Messages + optional rubrics
    "rubrics_performance": GraderConfig(
        grader=RubricsBasedTrajectoryPerformance(model=model),
        mapper=lambda data: {
            "messages": data["messages"],
            "rubrics": data.get("rubrics", [])  # Empty list if not present
        }
    ),
}
```

**Design Principles**:
- Use `.get()` for optional fields with sensible defaults
- Keep mappers simple and readable
- Ensure type consistency with grader signatures
- Use named functions only for complex transformations

## Running the Tutorial

```bash
# Set API key
export OPENAI_API_KEY="your-api-key"

# Run evaluation
python tutorials/deep_research/deep_research_evaluation.py
```

**Output**:
```
Starting deep research agent evaluation...
Grading: 100%|████████████████| 6/6 [00:15<00:00,  2.5s/it]

================================================================================
EVALUATION RESULTS
================================================================================

────────────────────────────────────────────────────────────────────────────────
📊 REPORT_RESOLUTION
────────────────────────────────────────────────────────────────────────────────

Sample 1:
  Score: 0.8875
  Reason: 【精准性】5/5分: 报告完全精准地回答了用户问题...
  Metadata: ['dimension_scores', 'is_resolved', 'resolution_threshold']

...
```

## Customization

### Adding Custom Graders

```python
from rm_gallery.core.graders.base_grader import BaseGrader, GraderScore

class CustomGrader(BaseGrader):
    async def aevaluate(self, messages, **kwargs) -> GraderScore:
        # Your evaluation logic
        return GraderScore(name=self.name, score=0.95, reason="...")

# Add to configs
grader_configs["custom"] = GraderConfig(
    grader=CustomGrader(),
    mapper=lambda data: {"messages": data["messages"]}
)
```

### Custom Rubrics

```python
custom_rubrics = [
    {
        "dimension": "风险提示",
        "description": "报告是否包含充分的风险提示",
        "check_points": ["提及市场风险", "提及政策风险", "提及经营风险"]
    }
]
```

## Best Practices

1. **Concurrency**: Set `max_concurrency` based on API rate limits (6 works well for OpenAI)
2. **Language**: Use consistent language across all LLM-based graders
3. **Error Handling**: Check for `GraderError` in results before accessing scores
4. **Progress**: Enable `show_progress=True` for long-running evaluations
5. **Optional Fields**: Always provide defaults for optional mapper fields

## Architecture

```
GradingRunner (concurrent execution)
    ├── LLM-Based Graders (structured output)
    │   ├── FinancialReportResolutionGrader
    │   ├── FinancialTrajectoryFaithfulGrader
    │   ├── RubricsBasedTrajectoryPerformance
    │   └── TrajectoryComprehensiveGrader
    └── Rule-Based Graders (deterministic)
        ├── ObservationInformationGainGrader
        └── ActionLoopDetectionGrader
```

## References

- [GradingRunner](../../rm_gallery/core/runner/grading_runner.py)
- [Financial Graders](../../rm_gallery/core/graders/agent/deep_research/)
- [Trajectory Graders](../../rm_gallery/core/graders/agent/trajectory/)
- [Action & Observation Graders](../../rm_gallery/core/graders/agent/)
