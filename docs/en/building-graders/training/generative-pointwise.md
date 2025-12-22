# Generative Training (Pointwise)

Train reward models to generate absolute quality scores for individual responses using reinforcement learning. This method teaches models to evaluate responses on a fixed scale (e.g., 0-4 helpfulness).

---

## Overview

Generative pointwise training uses **Group Relative Policy Optimization (GRPO)** to train language models to output structured evaluation scores. Unlike Bradley-Terry which learns rankings, pointwise training learns to assign absolute quality ratings.

**When to use:**
- You have labeled data with quality scores (e.g., helpfulness: 0-4)
- Need fine-grained scoring rather than binary comparisons
- Want natural language explanations alongside scores
- Building domain-specific evaluators

**Training approach:**

The model generates responses with scores, and receives rewards based on prediction accuracy:

\[
\text{reward} = \exp\left(-k \times \frac{|\text{predicted} - \text{true}|}{\text{max\_score}}\right)
\]

This exponential decay rewards accurate predictions and penalizes errors.

---

## Data Format

### Required Structure

Pointwise training expects Parquet files with:

| Column | Type | Description |
|--------|------|-------------|
| `input` | List[Dict] | Chat messages (query) |
| `output` | Dict | Response with quality label |
| `data_source` | str | Dataset identifier |

**Example:**

```python
{
    "input": [
        {"role": "user", "content": "How can I improve my sleep quality?"}
    ],
    "output": {
        "content": "To improve sleep: maintain consistent schedule, avoid screens before bed, keep room cool and dark, limit caffeine after 2pm, exercise regularly.",
        "label": {
            "helpfulness": 4  # Score 0-4
        }
    },
    "data_source": "helpsteer2"
}
```

### Label Schema

Typical scoring scales:

- **Helpfulness**: 0 (not helpful) to 4 (very helpful)
- **Correctness**: 0 (wrong) to 4 (perfect)
- **Relevance**: 0 (off-topic) to 4 (highly relevant)

---

## Data Preparation

### Option 1: Use HelpSteer2 Dataset

HelpSteer2 provides pre-labeled helpfulness scores:

```python
from datasets import load_dataset
import pandas as pd

# Load HelpSteer2
dataset = load_dataset("nvidia/HelpSteer2")

# Convert to pointwise format
def convert_to_pointwise(example):
    return {
        "input": [{"role": "user", "content": example["prompt"]}],
        "output": {
            "content": example["response"],
            "label": {"helpfulness": int(example["helpfulness"])}
        },
        "data_source": "helpsteer2"
    }

train_data = [convert_to_pointwise(ex) for ex in dataset["train"]]
test_data = [convert_to_pointwise(ex) for ex in dataset["test"]]

# Save as parquet
pd.DataFrame(train_data).to_parquet("./data/train_pointwise.parquet")
pd.DataFrame(test_data).to_parquet("./data/test_pointwise.parquet")
```

### Option 2: Create Custom Labeled Data

```python
import pandas as pd

# Your custom evaluation data
data = [
    {
        "input": [{"role": "user", "content": "What is Python?"}],
        "output": {
            "content": "Python is a high-level programming language known for readability and versatility.",
            "label": {"helpfulness": 4}
        },
        "data_source": "custom"
    },
    {
        "input": [{"role": "user", "content": "What is Python?"}],
        "output": {
            "content": "It's a snake.",
            "label": {"helpfulness": 1}
        },
        "data_source": "custom"
    }
]

df = pd.DataFrame(data)
df.to_parquet("./data/custom_pointwise.parquet")
```

---

## Training Setup

### 1. Custom Dataset Class

Create a dataset class for pointwise evaluation:

```python
# dataset.py
from typing import Any, Dict, List
from tutorials.cookbooks.training_reward_model.base import BaseTrainDataset

class HelpfulnessPointwiseTrainDataset(BaseTrainDataset):
    """Dataset for pointwise helpfulness training"""

    def _build_messages(self, example: Dict[str, Any]) -> List[Dict[str, str]]:
        """Build evaluation prompt messages"""
        input_messages = example.get('input', [])
        output_data = example.get('output', [])

        # Extract query
        query = input_messages[0]['content'] if input_messages else ""

        # Extract response
        if isinstance(output_data, list) and len(output_data) > 0:
            response = output_data[0].get('content', '')
        elif isinstance(output_data, dict):
            response = output_data.get('content', '')
        else:
            response = str(output_data)

        # Build evaluation prompt
        prompt = f"""Evaluate the helpfulness of this response on a scale of 0-4.

Query: {query}

Response: {response}

Provide your evaluation in this format:
<score>X</score>

Where X is a number from 0 (not helpful) to 4 (very helpful)."""

        return [{"role": "user", "content": prompt}]

    def _apply_chat_template(self, messages: List[Dict[str, str]]) -> str:
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

    def _extract_ground_truth(self, row_dict: Dict[str, Any]) -> Dict[str, int]:
        """Extract helpfulness score"""
        output_data = row_dict.get('output', [])

        if isinstance(output_data, list) and len(output_data) > 0:
            label = output_data[0].get('label', {})
        elif isinstance(output_data, dict):
            label = output_data.get('label', {})
        else:
            label = {}

        return {
            "helpfulness": label.get("helpfulness", 0)
        }

    def _get_data_source(self, row_dict: Dict[str, Any]) -> str:
        return row_dict.get('data_source', 'unknown')
```

### 2. Reward Function

Define how model outputs are scored:

```python
# reward_fn.py
import math
from typing import Any, Dict, Optional
from template import PointwiseTrainTemplate

def calculate_helpfulness_reward(
    predicted_score: int,
    true_score: Optional[int]
) -> float:
    """
    Calculate reward based on prediction accuracy.
    Uses exponential decay for error tolerance.
    """
    if true_score is None:
        return 0.0

    # Calculate absolute error
    abs_error = abs(predicted_score - true_score)
    max_possible_error = 4  # Score range 0-4

    # Exponential decay: reward = exp(-k * error_ratio)
    k = 2.0  # Decay parameter
    error_ratio = abs_error / max_possible_error
    reward = math.exp(-k * error_ratio)

    return float(reward)

def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: Optional[Dict[str, Any]] = None,
    **kwargs: Any
) -> Dict[str, Any]:
    """Compute reward score for model prediction"""
    try:
        # Parse predicted score from model output
        parsed_result = PointwiseTrainTemplate.parse(solution_str)
        predicted_helpfulness = parsed_result.score

        # Validate range
        if not 0 <= predicted_helpfulness <= 4:
            predicted_helpfulness = 0

        # Extract true score
        if isinstance(ground_truth, dict):
            true_helpfulness = ground_truth.get("helpfulness", 0)
        elif isinstance(ground_truth, (int, float)):
            true_helpfulness = int(ground_truth)
        else:
            true_helpfulness = 0

        # Calculate reward
        reward = calculate_helpfulness_reward(
            predicted_helpfulness,
            true_helpfulness
        )

        accuracy = 1 if predicted_helpfulness == true_helpfulness else 0

        return {
            "score": reward,
            "predicted_helpfulness": predicted_helpfulness,
            "true_helpfulness": true_helpfulness,
            "accuracy": accuracy,
            "data_source": data_source
        }

    except Exception:
        return {
            "score": 0.0,
            "predicted_helpfulness": 0,
            "true_helpfulness": 0,
            "accuracy": 0,
            "data_source": data_source
        }
```

### 3. Output Template

Define structured output format:

```python
# template.py
import re
from typing import Any, Dict
from pydantic import BaseModel, Field

class PointwiseTrainTemplate(BaseModel):
    """Template for pointwise score parsing"""

    score: int = Field(
        default=...,
        description="score of helpfulness from 0 to 4"
    )

    @classmethod
    def parse(cls, text: str) -> "PointwiseTrainTemplate":
        """Extract score from model output"""
        # Match <score>X</score>
        pattern = r"<score>(.*?)</score>"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)

        if match:
            score_str = match.group(1).strip()
            score = int(score_str)
            return cls(score=score)

        raise ValueError("Failed to parse score from output")
```

---

## Launch Training

### Training Script (run_pointwise.sh)

```bash
#!/bin/bash
TIMESTAMP=$(date "+%m%dT%H%M")

# Configuration
TRAIN_FILE=./data/helpsteer2_pointwise_train.parquet
VAL_FILE=./data/helpsteer2_pointwise_test.parquet
MODEL_PATH=Qwen/Qwen2.5-7B-Instruct

PROJECT_NAME=pointwise_train
EXPERIMENT_NAME=rm-gallery-pointwise-${TIMESTAMP}

CUSTOM_REWARD_FUNCTION_PATH=./reward_fn.py
CUSTOM_DATASET_PATH=./dataset.py
CUSTOM_DATASET_NAME=HelpfulnessPointwiseTrainDataset

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

### Runtime Environment (runtime_env.yaml)

```yaml
# runtime_env.yaml
working_dir: "."
py_modules:
  - ./reward_fn.py
  - ./dataset.py
  - ./template.py

pip:
  - torch>=2.0.0
  - transformers>=4.30.0
  - datasets>=2.14.0
  - verl
```

### Run Training

```bash
# 1. Start Ray cluster
ray start --head --port=8265

# 2. Navigate to training directory
cd tutorials/cookbooks/training_reward_model/generative/pointwise

# 3. Launch training
bash run_pointwise.sh

# 4. Monitor via Ray dashboard
open http://127.0.0.1:8265
```

---

## Key Parameters

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| `algorithm.adv_estimator` | RL algorithm | `grpo` (Group Relative PO) |
| `data.train_batch_size` | Samples per training step | 96 |
| `data.max_response_length` | Max generated tokens | 2048 |
| `actor_rollout_ref.actor.optim.lr` | Actor learning rate | 1e-6 |
| `actor_rollout_ref.actor.use_kl_loss` | KL divergence regularization | true |
| `actor_rollout_ref.actor.kl_loss_coef` | KL loss weight | 0.001 |
| `actor_rollout_ref.rollout.n` | Samples per prompt | 4 |
| `trainer.total_epochs` | Training epochs | 1-2 |

---

## Monitoring

### Metrics

Track these key metrics during training:

- **Reward Score**: Average reward from scoring function
- **Accuracy**: Exact match rate (predicted == true score)
- **Mean Absolute Error (MAE)**: Average |predicted - true|
- **KL Divergence**: Distance from reference model

### Expected Progress

```
Epoch 0.1: reward ~0.4, accuracy ~30%
Epoch 0.5: reward ~0.7, accuracy ~55%
Epoch 1.0: reward ~0.85, accuracy ~70%
```

### Viewing Results

```bash
# Console output
tail -f logs/training.log

# WandB dashboard
wandb login
# View at https://wandb.ai/your-project

# Ray dashboard
open http://127.0.0.1:8265
```

---

## Using Trained Models

### Integrate with RM-Gallery

```python
from rm_gallery.core.models import OpenAIChatModel
from rm_gallery.core.graders.common import RelevanceGrader

# Load trained model
model = OpenAIChatModel(
    model="./checkpoints/rm-gallery-pointwise-final",
    is_local=True
)

# Use as grader
grader = RelevanceGrader(model=model)

result = await grader.aevaluate(
    query="How can I improve my sleep?",
    response="Maintain a consistent sleep schedule and avoid screens before bed."
)

print(f"Score: {result.score}")
print(f"Reason: {result.reason}")
```

### Batch Evaluation

```python
from rm_gallery.core.runner import GradingRunner, GraderConfig

runner = GradingRunner(
    grader_configs={
        "helpfulness": GraderConfig(grader=RelevanceGrader(model=model))
    }
)

dataset = [
    {"query": "What is AI?", "response": "AI is..."},
    {"query": "Benefits of exercise?", "response": "Exercise..."}
]

results = await runner.arun(dataset)
```

---

## Advanced Configuration

### Multi-Scale Scoring

Support multiple quality dimensions:

```python
class MultiDimensionTrainTemplate(BaseModel):
    helpfulness: int = Field(description="0-4 helpfulness")
    correctness: int = Field(description="0-4 correctness")
    relevance: int = Field(description="0-4 relevance")
```

### Custom Reward Functions

Implement domain-specific reward logic:

```python
def compute_custom_reward(predicted, true):
    # Weight recent data more
    time_weight = get_time_weight(data_source)

    # Penalize extreme errors more
    error = abs(predicted - true)
    penalty = error ** 2

    return (1.0 - penalty / 16) * time_weight
```

---


## Next Steps

- [Generative Pairwise](generative-pairwise.md) — Train with comparison preferences
- [Bradley-Terry Training](bradley-terry.md) — Simpler preference-based approach
- [SFT for Reward Models](sft.md) — Initialize with supervised fine-tuning
- [Training Overview](overview.md) — Compare training methods






