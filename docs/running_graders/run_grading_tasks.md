# Run Grading Tasks
> **Tip:** Grading is the process of evaluating how good responses from an AI model are. Think of it like having different teachers grade student essays - each teacher focuses on a different aspect like grammar, content, or creativity.
>

## What is a Runner?
In RM-Gallery, a Runner is an execution engine that orchestrates the evaluation process. It manages how evaluators (called "Graders") are executed against datasets, handles concurrency, transforms data, and organizes results.

The Runner takes care of orchestrating execution of multiple graders across your dataset, controlling how many evaluations happen simultaneously to optimize resource usage. It also handles mapping dataset fields to the format expected by each grader and collecting results in a predictable format.

With this foundational understanding of what a Runner does, let's explore RM-Gallery's primary implementation, the GradingRunner, which brings these capabilities together in a powerful and flexible package.

## Introducing GradingRunner
The [GradingRunner](../../rm_gallery/core/runner/grading_runner.py) is RM-Gallery's primary implementation of a Runner, specifically designed for executing grading tasks. Whether you're evaluating a few samples or processing thousands, understanding how to effectively use the GradingRunner is key to getting the most out of RM-Gallery.

The GradingRunner encapsulates the complexity of managing concurrent evaluations, data transformations, and result aggregation. It provides a clean interface that allows you to focus on defining what you want to evaluate rather than worrying about how to execute those evaluations efficiently. With built-in support for asynchronous processing, the GradingRunner maximizes throughput when working with resource-intensive graders like large language models.

## Getting Started with GradingRunner
Setting up an evaluation workflow with the GradingRunner involves configuring the graders you want to use and how they should be executed. The runner takes care of executing these graders across your dataset and collecting the results in an organized manner.

The configuration process involves defining grader instances and mapping your dataset fields to the inputs expected by each grader. This mapping is crucial because your data format rarely matches exactly what graders expect. Additionally, you can configure aggregators to combine results from multiple graders into composite scores.

Let's begin with a simple example to understand how GradingRunner works:

```python
# Comprehensive configuration
from rm_gallery.core.runner.grading_runner import GradingRunner
from rm_gallery.core.runner.aggregator.weighted_sum import WeightedSumAggregator
from rm_gallery.core.graders.common.helpfulness import HelpfulnessGrader
from rm_gallery.core.graders.common.accuracy import AccuracyGrader
from rm_gallery.core.graders.common.relevance import RelevanceGrader

runner = GradingRunner(
    grader_configs={
        "helpfulness": {
            "grader": HelpfulnessGrader(),
            "mapper": {"query": "question", "response": "answer"}
        },
        "accuracy": {
            "grader": AccuracyGrader(),
            "mapper": {"question": "question", "response": "answer"}
        },
        "relevance": {
            "grader": RelevanceGrader(),
            "mapper": {
                "query": "question",
                "response": "answer",
                "reference": "reference_answer"
            }
        }
    },
    max_concurrency=10,
    aggregators=WeightedSumAggregator(
        name="overall_score",
        weights={
            "helpfulness": 0.5,
            "accuracy": 0.3,
            "relevance": 0.2
        }
    )
)

# Create and run the evaluation
runner = GradingRunner(grader_configs=grader_configs)
results = await runner.arun(dataset)

# Process the results
for grader_name, grader_results in results.items():
    print(f"\nResults from {grader_name}:")
    for i, result in enumerate(grader_results):
        if hasattr(result, 'score'):
            print(f"  Sample {i+1}: Score = {result.score}")
        # Handle other result types...
```

This basic workflow works well for straightforward evaluations, but real-world scenarios often require more sophisticated handling. Let's look at some common challenges and how the GradingRunner addresses them.

## Bridging Data and Graders
One common challenge is that your data rarely matches exactly what graders expect. Real-world datasets come in all shapes and sizes, and rarely conform to the exact input format that graders expect. The GradingRunner provides flexible mechanisms to bridge this gap through its mapper system. Mappers transform your data into the format required by each grader, allowing you to use the same dataset with different graders that have varying input requirements.

### Field Mapping
Field mapping is the simplest form of data transformation, where you map fields from your dataset to the field names expected by graders. This is particularly useful when your dataset uses different naming conventions than what graders expect.

When your field names don't align with grader expectations:

```python
# Your data structure - notice the field names differ from what graders expect
dataset = [
    {
        "question": "What is the capital of France?",
        "answer": "The capital of France is Paris.",
        "reference_answer": "Paris"
    }
]

# Map your fields to what graders expect
# This tells the runner how to convert your data format to what graders need
grader_configs = {
    "helpfulness": {
        "grader": HelpfulnessGrader(),
        "mapper": {
            "query": "question",      # Grader expects "query", your data has "question"
            "response": "answer"      # Grader expects "response", your data has "answer"
        }
    },
    "relevance": {
        "grader": RelevanceGrader(),
        "mapper": {
            "query": "question",
            "response": "answer",
            "reference": "reference_answer"  # Grader expects "reference", your data has "reference_answer"
        }
    }
}
```

While field mapping solves many common cases, sometimes you need more flexibility to handle complex data transformations.

### Complex Transformations
For more complex data structures, custom mapper functions provide the flexibility needed to handle nested data, computed fields, or other transformations that simple field mapping cannot address. Custom mappers are functions that take a sample from your dataset and return a dictionary with the fields expected by the grader.

Custom mappers allow you to perform operations like extracting nested fields, combining multiple fields, or applying transformations to your data before it's passed to the grader.

For more complex data structures, custom mapper functions provide flexibility:

```python
# Nested data structure - more complex than what graders expect
dataset = [
    {
        "input": {"question": "What is the capital of France?"},
        "output": {"answer": "The capital of France is Paris."}
    }
]

# Custom transformation function to flatten the data
def custom_mapper(sample):
    return {
        "query": sample["input"]["question"],
        "response": sample["output"]["answer"]
    }

grader_configs = {
    "helpfulness": {
        "grader": HelpfulnessGrader(),
        "mapper": custom_mapper
    }
}
```

## Optimizing Performance
When dealing with large datasets or resource-intensive graders, performance becomes critical. The GradingRunner provides several mechanisms to optimize performance, primarily through concurrency control and efficient resource utilization.

Performance optimization is especially important when working with large language model-based graders, which can be slow and resource-intensive. The right balance of concurrency can dramatically reduce evaluation time without overwhelming your system resources.

### Controlling Concurrency
Concurrency control allows you to specify how many evaluations should run simultaneously. This is important because different types of graders have different resource requirements. Lightweight function-based graders can handle high concurrency, while heavyweight LLM-based graders might require lower concurrency to avoid resource exhaustion.

Finding the optimal concurrency level depends on your hardware resources, the types of graders you're using, and any external rate limits (like API quotas). Experimentation is often needed to find the sweet spot for your specific use case.

Adjust concurrency based on your resources and constraints:

```python
# For resource-intensive graders (e.g., LLM-based)
# Lower concurrency to avoid overwhelming resources like GPU or API limits
runner = GradingRunner(
    grader_configs=grader_configs,
    max_concurrency=5  # Process 5 samples at a time
)

# For fast, lightweight graders
# Higher concurrency for faster processing
runner = GradingRunner(
    grader_configs=grader_configs,
    max_concurrency=32  # Process 32 samples at a time
)
```

### Combining Results with Aggregators
Often you'll want to combine multiple grader results into unified scores. Aggregators provide a mechanism to synthesize results from multiple graders into composite scores, weighted averages, or other combinations that give you a holistic view of your model's performance.

Different aggregation strategies serve different purposes. A weighted sum aggregator might be appropriate when you want to compute an overall quality score, while other aggregators might be better suited for specific analytical tasks.

Often you'll want to combine multiple grader results into unified scores:

```python
from rm_gallery.core.runner.aggregator.weighted_sum import WeightedSumAggregator

# Combine multiple perspectives into a single quality score
# Like calculating a final grade based on different subject scores
aggregator = WeightedSumAggregator(
    name="overall_score",
    weights={
        "helpfulness": 0.6,  # Weight helpfulness at 60%
        "relevance": 0.4     # Weight relevance at 40%
    }
)

runner = GradingRunner(
    grader_configs=grader_configs,
    aggregators=aggregator
)
```

# Next Steps
Once you've mastered running grading tasks, you'll want to [generate validation reports](../validating_graders/generate_validation_reports.md) to assess the quality of your evaluations or [refine data quality](../applications/refine_data_quality.md) using your evaluation insights.

