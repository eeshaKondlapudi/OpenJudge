# 🚀 Core Concepts
Understanding RM-Gallery's **core concepts** will help you effectively evaluate AI models and extend the system for your specific needs. This guide walks you through the fundamental components and how they work together.

Whether you're new to RM-Gallery or looking to build custom evaluation workflows, this guide covers everything you need to know about integrating RM-Gallery with your systems and extending it with custom components.

> **Tip:** This guide provides a comprehensive overview of RM-Gallery's architecture and components.
>

## 🗃 1. From Evaluator/Reward to Grader
In the era of advanced AI systems, especially large language models (LLMs), having robust evaluation and reward mechanisms is critical for both measuring performance and guiding improvements.

### Why Evaluators Matter
AI models, regardless of their sophistication, need systematic evaluation to quantify performance, identify weaknesses, enable comparisons, and ensure quality control in production deployments. Without proper evaluation systems, it becomes nearly impossible to understand whether a model is truly capable or just appearing competent on the surface.

### Why Reward Systems Are Essential
Beyond evaluation, reward systems play a pivotal role in reinforcement learning by providing crucial signals for training improved models through RLHF, enabling automated optimization where systems can self-improve without constant human intervention, and creating objective functions that establish quantifiable targets aligning model behavior with desired outcomes. Traditional metrics often fall short in capturing nuanced aspects of model behavior, making sophisticated reward systems vital for developing truly capable AI assistants.

### The Challenge and Solution
However, creating effective evaluators and reward systems presents significant challenges including diverse evaluation criteria across domains, subjective aspects that are hard to quantify, scalability requirements for large-scale testing, and consistency and reliability in assessments. This is where RM-Gallery's **Grader** abstraction comes into play. A Grader is a modular, standardized component that can function as either an evaluator or a reward generator depending on your use case. By unifying these concepts under a single abstraction, RM-Gallery simplifies the process of building, managing, and deploying evaluation and reward systems while ensuring consistency across both domains.

## 🎯 2. What is a Grader?
Now that we understand why we need evaluators and reward systems, let's dive deep into what a Grader is and how it works.

### 2.a. Grader Definition
A Grader is the fundamental building block in RM-Gallery. It's a standardized component that takes input data (like questions and model responses) and produces evaluation results. You can think of it as a black box that transforms model outputs into meaningful scores or rankings.

Graders are designed to be modular, where each grader focuses on one specific aspect of evaluation, standardized so all graders follow the same interface making them interchangeable, and reusable since once created, graders can be used across different projects and datasets.

### 2.b. Using Graders for Evaluation
Graders work by taking your data and producing scores or rankings based on specific criteria. Let's look at how this process works:

#### Input Data
Graders accept structured input data that varies based on the evaluation criteria. Common input fields include `question`/`query` as the input prompt or question, `answer`/`response` as the model's response to evaluate, `reference` as ground truth or reference answer, and `context` as additional context for the evaluation. Flexible data mapping ensures your existing data formats can be easily adapted.

#### Scoring Approaches
Graders can evaluate your data in different ways through various scoring modes and implementation methods.

##### Scoring Modes
Pointwise evaluation assesses individual samples independently, such as scoring how helpful a single response is, while listwise evaluation ranks multiple samples relative to each other, like ranking several responses from best to worst.

##### Implementation Methods
Rule-based graders use predefined functions or algorithms to compute scores, whereas LLM-based graders leverage large language models to perform sophisticated evaluations.

#### Output Results
Graders produce standardized results in the form of **GraderResult** objects including GraderScore for numerical scores representing the quality or correctness of a response (e.g., 0.0 to 1.0) and GraderRank for relative rankings when comparing multiple responses to the same query. Both output types maintain consistency across different grader implementations, making it easy to combine and analyze results.

### 2.c. Built-in Predefined Graders
RM-Gallery comes with a rich collection of pre-built graders organized by domain:

+ **common/**: General-purpose graders (helpfulness, hallucination, harmfulness, compliance)
+ **agent/**: Agent capability evaluation (planning, tool usage, memory, etc.)
+ **code/**: Code-related evaluation (execution correctness, style, patch similarity)
+ **format/**: Format compliance checking (JSON validation, structure verification)
+ **multimodal/**: Multimodal content evaluation (image-text alignment, visual helpfulness)
+ **math/**: Mathematical computation and reasoning evaluation
+ **text/**: Text similarity and quality measurements

These graders are ready to use and cover most common evaluation scenarios.

## 🔧 3. Building Graders
While RM-Gallery provides many built-in graders, you'll often need to create custom graders for your specific use cases. There are several different approaches to building graders, each with its own advantages and use cases.

### 3.a. Custom Implementation
Creating custom graders gives you the most control over the evaluation logic.

#### Rule-based Graders
For objective, deterministic evaluations, you can create rule-based graders using the [FunctionGrader](../../rm_gallery/core/graders/function_grader.py):

```python
# A simple function grader that checks if response contains reference answer
def contains_reference(response, reference):
    return float(reference.lower() in response.lower())

contains_grader = FunctionGrader(contains_reference)
```

These graders are fast and deterministic, easy to debug and test, and perfect for objective metrics like exact match or format validation.

#### LLM-based Graders
For subjective or complex evaluations, you can create LLM-based graders using the [LLMGrader](../../rm_gallery/core/graders/llm_grader.py):

```python
# An LLM grader that evaluates helpfulness of responses
helpfulness_grader = HelpfulnessGrader(model=OpenAIChatModel("gpt-4"))
```

These graders leverage natural language understanding for nuanced assessments, flexible prompt templates for customization, and powerful language models for sophisticated reasoning. Key components for LLM-based graders include prompt templates for consistent evaluation queries ([PromptTemplate](../../rm_gallery/core/models/schema/prompt_template.py)) and models as interfaces to various LLM providers ([BaseChatModel](../../rm_gallery/core/models/base_chat_model.py), [OpenAIChatModel](../../rm_gallery/core/models/openai_chat_model.py)).

### 3.b. Automated Generation
Instead of manually creating graders, you can use automated tools to generate them from data. The [generator module](../../rm_gallery/core/generator/) provides tools to automatically create graders including LLMGraderGenerator which generates LLM-based graders by using language models to create evaluation rubrics from your data, and Iterative Rubric Generation which refines evaluation criteria through iterative feedback. Benefits of automated generation include reducing manual effort in grader creation, learning evaluation criteria from examples, and bootstrapping domain-specific evaluation logic.

### 3.c. Model-based Training
For the most sophisticated scenarios, you can train specialized graders using machine learning techniques. Trainable graders involve supervised learning on human-labeled evaluation data, creating specialized reward models, and continuous improvement through feedback. Approaches include fine-tuning pre-trained models for specific evaluation tasks, training neural networks to predict quality scores, and using reinforcement learning to optimize evaluation criteria. Benefits include adaptability to specific domains or tasks, continuous improvement through feedback, and potential for capturing complex evaluation patterns.

## 🚀 4. Running and Analyzing with GradingRunner and Analyzer
Once you have your graders built, you need a way to run them efficiently across your data and analyze the results. This is where the [runner module](../../rm_gallery/core/runner/) and [analyzer module](../../rm_gallery/core/analyzer/) come into play.

### 4.a. GradingRunner
The [GradingRunner](../../rm_gallery/core/runner/grading_runner.py) is the central execution engine of RM-Gallery that orchestrates the execution of multiple graders, handles data mapping between your dataset and grader inputs, manages parallel execution for efficiency, and combines results from multiple graders. It acts as the conductor of an orchestra, coordinating all the different graders to create a harmonious evaluation process.

Let's walk through a simple example to illustrate how this works in practice:

```python
from rm_gallery.core.runner.grading_runner import GradingRunner
from rm_gallery.core.runner.aggregator.weighted_sum_aggregator import WeightedSumAggregator
from rm_gallery.core.graders.common.helpfulness import HelpfulnessGrader
from rm_gallery.core.graders.common.relevance import RelevanceGrader

# Prepare your data in whatever format works for you
data = [
    {
        "query": "What is the capital of France?",
        "response": "Paris",
        "reference_answer": "Paris"
    },
    {
        "query": "Who wrote Romeo and Juliet?",
        "response": "Shakespeare",
        "reference_answer": "William Shakespeare"
    }
]

# Configure graders with mappers to connect your data fields
graders = {
    "helpfulness": {
        "grader": HelpfulnessGrader(),
        "mapper": {"question": "query", "answer": "response"}
    },
    "relevance": {
        "grader": RelevanceGrader(),
        "mapper": {"q": "query", "a": "response", "ref": "reference_answer"}
    }
}

# Configure aggregators to combine results
aggregators = [
    WeightedSumAggregator(weights={"helpfulness": 0.5, "relevance": 0.5})
]

# Run evaluation
runner = GradingRunner(graders, aggregators=aggregators, max_concurrency=5)
results = await runner.arun(data)

# Results contain scores from all graders that you can use for analysis or training
```

#### Data Mapping
The GradingRunner's mapper functionality allows you to transform your data fields to match the parameter names expected by your graders. Since your input data may not have the exact field names that your graders expect, mappers provide a way to map between your data structure and the grader's expected inputs. Types of mappers include dictionary mappers for simple key-value mappings (e.g., `{"question": "query", "answer": "response"}`) and callable mappers for custom functions that transform data in more complex ways.

#### Aggregators
After running multiple graders, you might want to combine their results into a single score. The [aggregator submodule](../../rm_gallery/core/runner/aggregator/) provides components that take multiple grader results and combine them into a unified result including WeightedSumAggregator for combining results using weighted averages, MaxAggregator for taking the maximum score among all graders, and MinAggregator for taking the minimum score among all graders.

#### Parallel Execution
The GradingRunner is designed for high-performance evaluation by executing graders concurrently to maximize throughput, managing resource utilization efficiently, and handling error cases gracefully.

### 4.b. Analyzer
After running evaluations with the **GradingRunner**, you can use the [analyzer module](../../rm_gallery/core/analyzer/) to process the results and gain deeper insights. Analyzers are optional components that help you understand your evaluation results better. [BaseAnalyzer](../../rm_gallery/core/analyzer/base_analyzer.py) defines the interface for all analyzers. As a user, you can apply analyzers to your evaluation results to gain insights. Types of analyzers include statistical analyzers that compute statistics on evaluation results (e.g., [DistributionAnalyzer](../../rm_gallery/core/analyzer/statistical/distribution_analyzer.py)) and validation analyzers that compare evaluation results with reference labels (e.g., [AccuracyAnalyzer](../../rm_gallery/core/analyzer/validation/accuracy_analyzer.py), [F1ScoreAnalyzer](../../rm_gallery/core/analyzer/validation/f1_score_analyzer.py)).

## Next Steps
For training scenarios, you can skip this step and directly use the **GradingRunner's** output as training signals.

## Next Steps
+ [Building custom graders](../building_graders/create_custom_graders.md) for specialized feedback
+ [Validating graders](../validating_graders/validation_workflow.md) to ensure feedback quality
+ [Training reward models](../building_graders/train_a_grader/) to automate feedback generation

