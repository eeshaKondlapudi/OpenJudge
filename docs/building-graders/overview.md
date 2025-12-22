# Building Custom Graders

Extend RM-Gallery beyond built-in evaluators by creating custom graders or training reward models. Build domain-specific evaluation logic that seamlessly integrates with RM-Gallery's evaluation pipeline.

---

## Why Build Custom Graders?

While RM-Gallery provides 50+ pre-built graders, custom graders unlock specialized evaluation capabilities:

- **Domain Specialization** — Evaluate industry-specific criteria (legal, medical, financial)
- **Business Requirements** — Implement proprietary scoring logic and evaluation rules
- **Data-Driven Evaluation** — Train models that learn from your preference data
- **Cost Optimization** — Replace expensive API judges with self-hosted models
- **Consistent Standards** — Maintain stable evaluation criteria across applications

---

## Building Approaches

RM-Gallery supports two paths for creating custom graders:

| Approach | Best For | Time to Deploy | Scalability | Cost |
|----------|----------|----------------|-------------|------|
| **[Create Custom Graders](custom-graders.md)** | Quick prototyping, rule-based logic, LLM-as-judge | Minutes | High | Low (API-based) |
| **[Train Reward Models](training/overview.md)** | Learning from data, high-volume evaluation, cost reduction | Hours-Days | Very High | High (training), Low (inference) |

---

## Approach 1: Create Custom Graders

Define evaluation logic using LLM judges or rule-based functions. No training required—start evaluating immediately.

### Implementation Methods

**LLM-based Graders:**
```python
from rm_gallery.core.graders.llm_grader import LLMGrader
from rm_gallery.core.models import OpenAIChatModel

model = OpenAIChatModel(model="qwen3-32b")

grader = LLMGrader(
    name="domain_expert",
    model=model,
    template="""
    Evaluate the medical accuracy of this response:

    Query: {query}
    Response: {response}

    Return JSON: {{"score": <0.0-1.0>, "reason": "<explanation>"}}
    """
)
```

**Rule-based Graders:**
```python
from rm_gallery.core.graders.function_grader import FunctionGrader
from rm_gallery.core.graders.schema import GraderScore

async def compliance_checker(response: str) -> GraderScore:
    """Check for required compliance statements."""
    required_terms = ["disclaimer", "terms", "conditions"]
    found = sum(term in response.lower() for term in required_terms)
    score = found / len(required_terms)

    return GraderScore(
        name="compliance_check",
        score=score,
        reason=f"Found {found}/{len(required_terms)} required terms"
    )

grader = FunctionGrader(func=compliance_checker, name="compliance")
```

### When to Use

- ✅ Need evaluation logic immediately
- ✅ Rule-based criteria are well-defined
- ✅ Moderate evaluation volume (<1M queries/month)
- ✅ Access to powerful LLM APIs (GPT-4, Claude)
- ❌ High evaluation costs becoming prohibitive
- ❌ Need to capture nuanced preferences from data

**Read more:** [Create Custom Graders Guide](custom-graders.md)

---

## Approach 2: Train Reward Models

Train neural network models on preference data to learn evaluation criteria. Higher upfront cost, but enables cost-effective large-scale evaluation.

### Training Methods

RM-Gallery supports multiple training paradigms via VERL framework:

| Method | Training Data | Best For | Example |
|--------|---------------|----------|---------|
| **[Bradley-Terry](training/bradley-terry.md)** | Preference pairs (chosen/rejected) | Binary preference learning | "Response A > Response B" |
| **[Generative Pointwise](training/generative-pointwise.md)** | Absolute scores (0-5 scale) | Direct quality scoring | "Response quality: 4/5" |
| **[Generative Pairwise](training/generative-pairwise.md)** | Comparison decisions (A/B/tie) | Ranking responses | "Prefer A over B" |
| **[SFT](training/sft.md)** | Multi-turn conversations | Model initialization | "Supervised fine-tuning" |

### Training Architecture

```
┌─────────────────────────────────────────────────┐
│  Preference Data Collection                     │
│  ├─ Human annotations                           │
│  ├─ Existing grader outputs                     │
│  └─ LLM-generated preferences                   │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│  Training with VERL Framework                   │
│  ├─ Multi-GPU/Multi-node (FSDP)                │
│  ├─ Bradley-Terry / Generative objectives      │
│  └─ Ray-based distributed training             │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│  Trained Reward Model                           │
│  ├─ Self-hosted inference                       │
│  ├─ Integrated as RM-Gallery grader            │
│  └─ Cost-effective large-scale evaluation      │
└─────────────────────────────────────────────────┘
```

### Quick Start: Train a Model

```bash
# 1. Prepare training data
python -m rm_gallery.core.generator.export \
    --dataset helpsteer2 \
    --output-dir ./data \
    --format parquet

# 2. Choose training method and run
cd tutorials/cookbooks/training_reward_model/bradley-terry
bash run_bt.sh

# 3. Integrate trained model
```

```python
from rm_gallery.core.models import OpenAIChatModel
from rm_gallery.core.graders.common import RelevanceGrader

# Load your trained model
model = OpenAIChatModel(
    model="./checkpoints/my-reward-model",
    is_local=True
)

# Use as a grader
grader = RelevanceGrader(model=model)
result = await grader.aevaluate(query="...", response="...")
```

### When to Use

- ✅ Have preference/score data (>1K examples)
- ✅ High evaluation volume (>1M queries/month)
- ✅ Need consistent evaluation criteria
- ✅ Want to reduce API costs long-term
- ✅ Can invest in training infrastructure
- ❌ Need results immediately (training takes hours-days)
- ❌ Don't have sufficient training data

**Read more:** [Training Graders Guide](training/overview.md)

---

## Decision Framework

Choose your approach based on requirements:

```
                    START
                      │
                      ▼
          ┌─────────────────────┐
          │ Need evaluation now?│
          └──────┬───────┬──────┘
                 │       │
             YES │       │ NO, can train
                 │       │
                 ▼       ▼
        ┌────────────┐  ┌──────────────────┐
        │ Rule-based │  │ Have training    │
        │ logic?     │  │ data (>1K)?      │
        └─────┬──────┘  └────┬─────────┬───┘
              │              │         │
          YES │ NO      YES  │         │ NO
              │              │         │
              ▼              ▼         ▼
     ┌──────────────┐  ┌─────────┐  ┌──────────┐
     │ FunctionGrader│  │ Train   │  │ Collect  │
     │ or BaseGrader │  │ Reward  │  │ data or  │
     │              │  │ Model   │  │ use      │
     └──────────────┘  └─────────┘  │ LLMGrader│
                                     └──────────┘
              │              │             │
              ▼              ▼             ▼
     ┌───────────────────────────────────────┐
     │     Use in evaluation pipeline        │
     │     (GradingRunner, batch eval)       │
     └───────────────────────────────────────┘
```

### Comparison Matrix

| Factor | Custom Graders | Trained Models |
|--------|---------------|----------------|
| **Setup Time** | < 1 hour | 1-3 days |
| **Data Required** | None | 1K-100K examples |
| **Per-Query Cost** | $0.001-$0.01 (API) | $0.0001-$0.001 (self-hosted) |
| **Evaluation Speed** | Fast (API latency) | Very Fast (local inference) |
| **Flexibility** | High (change prompts) | Medium (requires retraining) |
| **Consistency** | Medium (LLM variance) | High (deterministic) |
| **Domain Adaptation** | Manual prompt engineering | Automatic from data |

---

## Integration with RM-Gallery

Both approaches produce graders that work identically in RM-Gallery:

### Single Evaluation

```python
result = await grader.aevaluate(
    query="What is machine learning?",
    response="ML is a subset of AI..."
)
print(result.score, result.reason)
```

### Batch Evaluation

```python
from rm_gallery.core.runner import GradingRunner

runner = GradingRunner(
    grader_configs={
        "custom": custom_grader,
        "trained": trained_grader
    }
)
results = await runner.arun([
    {"query": "Q1", "response": "A1"},
    {"query": "Q2", "response": "A2"}
])
```

### Multi-Grader Evaluation

```python
runner = GradingRunner(
    grader_configs={
        "relevance": RelevanceGrader(),          # Built-in
        "custom_llm": custom_llm_grader,         # Custom LLM-based
        "trained": trained_reward_model          # Trained model
    }
)
results = await runner.arun([{"query": "...", "response": "..."}])
```

---

## Best Practices

### For Custom Graders

1. **Start Simple** — Begin with rule-based graders, add LLM judges as needed
2. **Validate Prompts** — Test LLM-based graders on diverse inputs
3. **Handle Errors** — Implement robust error handling for production use
4. **Version Control** — Track prompt versions for reproducibility
5. **Monitor Costs** — Set usage limits for API-based graders

### For Trained Models

1. **Data Quality** — Prioritize high-quality preference data over quantity
2. **Validation Set** — Hold out 10-20% for evaluation
3. **Start Small** — Begin with smaller models (1B-7B parameters)
4. **Iterate Quickly** — Run short training runs to validate setup
5. **Monitor Drift** — Track evaluation consistency over time

---

## Complete Example: Build an Evaluation Pipeline

Combine multiple custom graders for comprehensive assessment:

```python
from rm_gallery.core.graders.llm_grader import LLMGrader
from rm_gallery.core.graders.function_grader import FunctionGrader
from rm_gallery.core.graders.schema import GraderScore
from rm_gallery.core.runner import GradingRunner
from rm_gallery.core.models import OpenAIChatModel

# 1. Rule-based grader: Length check
async def length_check(response: str) -> GraderScore:
    length = len(response)
    score = 1.0 if 50 <= length <= 500 else 0.5
    return GraderScore(
        name="length_check",
        score=score,
        reason=f"Length: {length} chars"
    )

length_grader = FunctionGrader(func=length_check, name="length")

# 2. LLM-based grader: Domain accuracy
model = OpenAIChatModel(model="qwen3-32b")
accuracy_grader = LLMGrader(
    name="accuracy",
    model=model,
    template="""
    Rate technical accuracy (0.0-1.0):
    Query: {query}
    Response: {response}
    Return JSON: {{"score": <score>, "reason": "<reason>"}}
    """
)

# 3. Trained model: Custom preferences (example)
# trained_model = OpenAIChatModel(model="./checkpoints/my-model", is_local=True)
# preference_grader = RelevanceGrader(model=trained_model)

# 4. Combine in evaluation pipeline
runner = GradingRunner(
    grader_configs={
        "length": length_grader,
        "accuracy": accuracy_grader
        # Add preference_grader when ready
    }
)

# 5. Run evaluation
results = await runner.arun([
    {"query": "Explain quantum computing", "response": "Quantum computing uses..."},
    {"query": "What is AI?", "response": "Artificial Intelligence is..."}
])

for result in results:
    print(f"Scores: {result}")
```

---

## Next Steps

Start building your custom evaluation pipeline:

### Create Custom Graders
- **[Create Custom Graders Guide](custom-graders.md)** — LLM-based and rule-based graders
- **[Built-in Graders Reference](../graders/overview.md)** — Explore existing graders to customize

### Train Reward Models
- **[Training Overview](training/overview.md)** — Compare training methods
- **[Bradley-Terry Training](training/bradley-terry.md)** — Start with preference pairs
- **[Generative Training](training/generative-pointwise.md)** — Train with score labels

### Deploy at Scale
- **[Run Grading Tasks](../running-graders/run-tasks.md)** — Batch evaluation workflows
- **[Generate Validation Reports](../running-graders/validation-reports.md)** — Quality assurance
- **[Integration Guides](../integrations/)** — Connect with LangSmith, LlamaIndex

### Applications
- **[Refine Data Quality](../applications/refine-data-quality.md)** — Filter training data
- **[Select & Rank Responses](../applications/select-rank.md)** — Build response selection systems

