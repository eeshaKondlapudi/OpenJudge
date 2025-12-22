# Generative Training (Pairwise)

Train reward models to generate comparative evaluations between two responses using reinforcement learning. This method teaches models to explain which response is better and why.

---

## Overview

Generative pairwise training uses **Group Relative Policy Optimization (GRPO)** to train language models to compare two responses and output structured preference decisions. Unlike Bradley-Terry which learns implicit rankings, pairwise training generates explicit reasoning.

**When to use:**
- You have labeled comparison data (A vs B preferences)
- Need explainable preference decisions with reasoning
- Building judges for human preference alignment
- Want natural language feedback alongside preferences

**Training approach:**

The model generates preference analysis and receives binary rewards:

\[
\text{reward} =
\begin{cases}
1.0 & \text{if predicted = true preference} \\
0.0 & \text{otherwise}
\end{cases}
\]

---

## Data Format

### Required Structure

Pairwise training expects Parquet files with:

| Column | Type | Description |
|--------|------|-------------|
| `input` | List[Dict] | Chat messages (query) |
| `output` | List[Dict] | Two responses to compare |
| `metadata` | Dict | Preference label and strength |

**Example:**

```python
{
    "input": [
        {"role": "user", "content": "What are the benefits of exercise?"}
    ],
    "output": [
        {
            "content": "Regular exercise improves cardiovascular health, boosts mood through endorphin release, increases energy levels, helps maintain healthy weight, and reduces risk of chronic diseases like diabetes and heart disease."
        },
        {
            "content": "Exercise is good for you."
        }
    ],
    "metadata": {
        "preferred": "A",  # "A", "B", or "tie"
        "preference_strength": 2  # 0 (weak) to 2 (strong)
    }
}
```

### Preference Labels

- **"A"**: First response is better
- **"B"**: Second response is better
- **"tie"**: Both responses are equal quality

---

## Data Preparation

### Option 1: Convert from HelpSteer2

```python
from datasets import load_dataset
import pandas as pd

dataset = load_dataset("nvidia/HelpSteer2")

def convert_to_pairwise(examples, threshold=1.0):
    """Create pairwise comparisons from scored examples"""
    pairwise_data = []

    # Group by prompt
    prompts = {}
    for ex in examples:
        prompt = ex["prompt"]
        if prompt not in prompts:
            prompts[prompt] = []
        prompts[prompt].append(ex)

    # Create pairs
    for prompt, responses in prompts.items():
        if len(responses) < 2:
            continue

        # Compare first two responses
        r1, r2 = responses[0], responses[1]
        score_diff = r1["helpfulness"] - r2["helpfulness"]

        if abs(score_diff) < 0.5:
            preferred = "tie"
            strength = 0
        elif score_diff > 0:
            preferred = "A"
            strength = 1 if abs(score_diff) < threshold else 2
        else:
            preferred = "B"
            strength = 1 if abs(score_diff) < threshold else 2

        pairwise_data.append({
            "input": [{"role": "user", "content": prompt}],
            "output": [
                {"content": r1["response"]},
                {"content": r2["response"]}
            ],
            "metadata": {
                "preferred": preferred,
                "preference_strength": strength
            }
        })

    return pairwise_data

train_data = convert_to_pairwise(dataset["train"])
test_data = convert_to_pairwise(dataset["test"])

pd.DataFrame(train_data).to_parquet("./data/pairwise_train.parquet")
pd.DataFrame(test_data).to_parquet("./data/pairwise_test.parquet")
```

### Option 2: From Human Preferences

```python
import pandas as pd

# Your annotated preference data
preferences = [
    {
        "input": [{"role": "user", "content": "Explain quantum computing"}],
        "output": [
            {"content": "Detailed technical explanation..."},
            {"content": "Simple but incomplete explanation..."}
        ],
        "metadata": {"preferred": "A", "preference_strength": 2}
    },
    # More examples...
]

df = pd.DataFrame(preferences)
df.to_parquet("./data/custom_pairwise.parquet")
```

---

## Training Setup

### 1. Custom Dataset Class

```python
# dataset.py
from typing import Any, Dict, List
import json
from tutorials.cookbooks.training_reward_model.base import BaseTrainDataset

class HelpfulnessPairwiseTrainDataset(BaseTrainDataset):
    """Dataset for pairwise comparison training"""

    def _build_messages(self, example: Dict[str, Any]) -> List[Dict[str, str]]:
        """Build comparison prompt"""
        input_messages = example.get('input', [])
        output_data = example.get('output', [])

        # Extract query
        query = input_messages[0]['content'] if input_messages else ""

        # Extract two responses
        if len(output_data) >= 2:
            response_a = output_data[0].get('content', '')
            response_b = output_data[1].get('content', '')
        else:
            response_a = response_b = ""

        # Build comparison prompt
        prompt = f"""Compare these two responses to the query and determine which is better.

Query: {query}

Response A:
{response_a}

Response B:
{response_b}

Evaluate based on helpfulness, accuracy, completeness, and clarity.

Provide your evaluation in this format:
<analysis>Brief explanation of comparison</analysis>
<preference>A or B or tie</preference>"""

        return [{"role": "user", "content": prompt}]

    def _apply_chat_template(self, messages: List[Dict[str, str]]) -> str:
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

    def _extract_ground_truth(self, row_dict: Dict[str, Any]) -> str:
        """Extract preference label"""
        metadata = row_dict.get('metadata', {})

        # Handle JSON string metadata
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except (json.JSONDecodeError, ValueError):
                metadata = {}

        return metadata.get('preferred', 'tie')

    def _get_data_source(self, row_dict: Dict[str, Any]) -> str:
        return row_dict.get('data_source', 'unknown')
```

### 2. Reward Function

```python
# reward_fn.py
import re
import json
from typing import Any, Dict, Optional
from template import PairwiseComparisonTemplate

def extract_preference_from_response(response_text: str) -> str:
    """Extract preference decision from model output"""
    if not isinstance(response_text, str):
        response_text = str(response_text)

    # Parse using template
    try:
        parsed = PairwiseComparisonTemplate.parse(response_text)
        return parsed.preference or "unknown"
    except Exception:
        pass

    # Fallback: Extract from XML tags
    pattern = r"<preference>(.*?)</preference>"
    match = re.search(pattern, response_text, re.DOTALL | re.IGNORECASE)

    if match:
        preference = match.group(1).strip().upper()
        if preference == "A" or "RESPONSE A" in preference:
            return "A"
        elif preference == "B" or "RESPONSE B" in preference:
            return "B"
        elif "TIE" in preference or "EQUAL" in preference:
            return "tie"

    return "unknown"

def calculate_pairwise_reward(
    predicted_preference: str,
    true_preference: Optional[str]
) -> float:
    """Calculate binary reward for preference prediction"""
    if true_preference is None or predicted_preference == "unknown":
        return 0.0

    return 1.0 if predicted_preference == true_preference else 0.0

def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any = None,
    extra_info: Optional[Dict[str, Any]] = None,
    **kwargs: Any
) -> Dict[str, Any]:
    """Compute reward for pairwise comparison"""
    try:
        # Extract predicted preference
        predicted_preference = extract_preference_from_response(solution_str)

        # Extract true preference from metadata
        true_preference = "tie"
        preference_strength = 0

        if extra_info and isinstance(extra_info, dict):
            metadata = extra_info.get("metadata", {})

            # Handle JSON string
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except (json.JSONDecodeError, ValueError):
                    metadata = {}

            if isinstance(metadata, dict):
                true_preference = metadata.get("preferred", "tie")
                preference_strength = metadata.get("preference_strength", 0)

        # Calculate reward
        reward = calculate_pairwise_reward(predicted_preference, true_preference)
        accuracy = 1.0 if (predicted_preference == true_preference and
                          predicted_preference != "unknown") else 0.0

        return {
            "score": reward,
            "predicted_preference": predicted_preference,
            "true_preference": true_preference,
            "preference_strength": preference_strength,
            "accuracy": accuracy,
            "task_type": "pairwise",
            "data_source": data_source
        }

    except Exception as exc:
        return {
            "score": 0.0,
            "predicted_preference": "unknown",
            "true_preference": "tie",
            "accuracy": 0.0,
            "error": str(exc),
            "task_type": "pairwise",
            "data_source": data_source
        }
```

### 3. Output Template

```python
# template.py
import re
from typing import Optional
from pydantic import BaseModel, Field

class PairwiseComparisonTemplate(BaseModel):
    """Template for pairwise comparison parsing"""

    analysis: Optional[str] = Field(
        default=None,
        description="Explanation of the comparison"
    )
    preference: Optional[str] = Field(
        default=None,
        description="Preference decision: A, B, or tie"
    )

    @classmethod
    def parse(cls, text: str) -> "PairwiseComparisonTemplate":
        """Extract analysis and preference from output"""
        contents = {}

        # Extract analysis
        analysis_pattern = r"<analysis>(.*?)</analysis>"
        match = re.search(analysis_pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            contents["analysis"] = match.group(1).strip()

        # Extract preference
        pref_pattern = r"<preference>(.*?)</preference>"
        match = re.search(pref_pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            pref = match.group(1).strip().upper()
            if "A" in pref:
                contents["preference"] = "A"
            elif "B" in pref:
                contents["preference"] = "B"
            elif "TIE" in pref:
                contents["preference"] = "tie"

        return cls(**contents)
```

---

## Launch Training

### Training Script (run_pairwise.sh)

```bash
#!/bin/bash
TIMESTAMP=$(date "+%m%dT%H%M")

# Configuration
TRAIN_FILE=./data/helpsteer2_pairwise_train.parquet
VAL_FILE=./data/helpsteer2_pairwise_test.parquet
MODEL_PATH=Qwen/Qwen2.5-14B-Instruct

PROJECT_NAME=pairwise_train
EXPERIMENT_NAME=rm-gallery-pairwise-${TIMESTAMP}

CUSTOM_REWARD_FUNCTION_PATH=./reward_fn.py
CUSTOM_DATASET_PATH=./dataset.py
CUSTOM_DATASET_NAME=HelpfulnessPairwiseTrainDataset

DEFAULT_LOCAL_DIR=./checkpoints/${TIMESTAMP}

N_GPUS_PER_NODE=8
N_NODES=1

# Launch with Ray
ray job submit --address="http://127.0.0.1:8265" \
    --runtime-env=./runtime_env.yaml \
    -- \
    python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$TRAIN_FILE" \
    data.val_files="$VAL_FILE" \
    data.train_batch_size=96 \
    data.val_batch_size=192 \
    data.max_prompt_length=4096 \
    data.max_response_length=2048 \
    data.custom_cls.path="${CUSTOM_DATASET_PATH}" \
    data.custom_cls.name="${CUSTOM_DATASET_NAME}" \
    reward_model.reward_manager=naive \
    custom_reward_function.path=${CUSTOM_REWARD_FUNCTION_PATH} \
    custom_reward_function.name=compute_score \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=24 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.rollout.tensor_model_parallel_size=8 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=4 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.n_gpus_per_node=${N_GPUS_PER_NODE} \
    trainer.nnodes=${N_NODES} \
    trainer.save_freq=20 \
    trainer.test_freq=10 \
    trainer.total_epochs=1 \
    trainer.default_local_dir=${DEFAULT_LOCAL_DIR}
```

### Run Training

```bash
# 1. Start Ray cluster
ray start --head --port=8265

# 2. Navigate to training directory
cd tutorials/cookbooks/training_reward_model/generative/pairwise

# 3. Launch training
bash run_pairwise.sh

# 4. Monitor progress
open http://127.0.0.1:8265
```

---

## Key Parameters

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| `algorithm.adv_estimator` | RL algorithm | `grpo` |
| `data.train_batch_size` | Samples per step | 96 |
| `data.max_response_length` | Max output tokens | 2048 |
| `actor_rollout_ref.actor.optim.lr` | Learning rate | 1e-6 |
| `actor_rollout_ref.actor.kl_loss_coef` | KL regularization | 0.001 |
| `actor_rollout_ref.rollout.n` | Rollouts per prompt | 4 |
| `trainer.total_epochs` | Training epochs | 1-2 |

---

## Monitoring

### Metrics

Track during training:

- **Reward Score**: Average binary reward (0.0-1.0)
- **Accuracy**: Preference prediction accuracy
- **Preference Distribution**: % of A/B/tie predictions
- **KL Divergence**: Distance from reference model

### Expected Progress

```
Epoch 0.1: reward ~0.35, accuracy ~45%
Epoch 0.5: reward ~0.70, accuracy ~75%
Epoch 1.0: reward ~0.85, accuracy ~85%
```

### Analysis

```python
# Analyze predictions by preference strength
import pandas as pd

results = pd.read_csv("validation_results.csv")

# Group by preference strength
by_strength = results.groupby('preference_strength')['accuracy'].mean()
print(by_strength)

# Strong preferences should have higher accuracy
# Strength 0 (tie): ~60%
# Strength 1 (weak): ~75%
# Strength 2 (strong): ~90%
```

---

## Using Trained Models

### As Comparison Judge

```python
from rm_gallery.core.models import OpenAIChatModel

model = OpenAIChatModel(
    model="./checkpoints/pairwise-final",
    is_local=True
)

# Compare two responses
query = "What are the benefits of exercise?"
response_a = "Detailed comprehensive answer..."
response_b = "Brief incomplete answer..."

prompt = f"""Compare these responses:

Query: {query}

Response A: {response_a}

Response B: {response_b}

<analysis>Your analysis</analysis>
<preference>A or B or tie</preference>"""

result = await model.acomplete(prompt)
print(result)
```

### In GradingRunner

```python
from rm_gallery.core.graders.common import RelevanceGrader
from rm_gallery.core.runner import GradingRunner, GraderConfig

grader = RelevanceGrader(model=model)

runner = GradingRunner(
    grader_configs={
        "preference": GraderConfig(grader=grader)
    }
)

# Evaluate dataset
results = await runner.arun(your_comparison_data)
```

---

## Advanced Techniques

### Weighted Preferences

Account for preference strength in rewards:

```python
def calculate_weighted_reward(predicted, true, strength):
    """Weight reward by preference strength"""
    base_reward = 1.0 if predicted == true else 0.0

    # Strong preferences (strength=2) get full weight
    # Weak preferences (strength=1) get reduced weight
    # Ties (strength=0) get minimal weight
    weight = 1.0 if strength == 2 else (0.7 if strength == 1 else 0.3)

    return base_reward * weight
```

### Multi-Aspect Comparison

Compare across multiple dimensions:

```python
class MultiAspectTemplate(BaseModel):
    helpfulness_preference: str = Field(description="A, B, or tie")
    correctness_preference: str = Field(description="A, B, or tie")
    clarity_preference: str = Field(description="A, B, or tie")
    overall_preference: str = Field(description="A, B, or tie")
```

### Confidence Scores

Add confidence to predictions:

```python
class ConfidentPairwiseTemplate(BaseModel):
    analysis: str
    preference: str
    confidence: float = Field(
        description="Confidence level 0.0-1.0"
    )
```

---

## Next Steps

- [Generative Pointwise](generative-pointwise.md) — Train with absolute scores
- [Bradley-Terry Training](bradley-terry.md) — Implicit ranking approach
- [SFT for Reward Models](sft.md) — Initialize with supervised learning
- [Training Overview](overview.md) — Compare all methods






