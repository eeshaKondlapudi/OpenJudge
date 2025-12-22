# Supervised Fine-Tuning for Reward Models

Train models on multi-turn conversation data to establish strong baseline capabilities before specialized reward training. This guide focuses on the SFT training pipeline using the VERL framework.

> **Note:** For an overview of when to use SFT vs. other methods, see [Training Overview](overview.md).

---

## Data Format

SFT training expects **Parquet files** with a `messages` column containing multi-turn conversations:

```python
{
    "messages": [
        {"role": "user", "content": "What is Python?"},
        {"role": "assistant", "content": "Python is a high-level programming language..."},
        {"role": "user", "content": "What are its main uses?"},
        {"role": "assistant", "content": "Python is widely used for..."}
    ]
}
```

**Supported roles:** `user`, `assistant`, `system` (optional)

---

## Prepare Your Data

### Convert Existing Datasets

```python
from datasets import load_dataset
import pandas as pd

# Load and convert ShareGPT-style data
dataset = load_dataset("anon8231489123/ShareGPT_Vicuna_unfiltered")

def to_messages(example):
    return {"messages": [
        {"role": "assistant" if msg["from"] == "gpt" else "user",
         "content": msg["value"]}
        for msg in example["conversations"]
    ]}

train_data = dataset["train"].map(to_messages)
pd.DataFrame(train_data)[['messages']].to_parquet("./data/sft_train.parquet")
```

### Create Evaluation-Focused Data

```python
import pandas as pd

conversations = [{
    "messages": [
        {"role": "user", "content": "Evaluate: Is 'Python is a language' helpful?"},
        {"role": "assistant", "content": "Score: 2/4. Correct but lacks detail..."}
    ]
}, {
    "messages": [
        {"role": "user", "content": "Compare Response A vs B..."},
        {"role": "assistant", "content": "Preference: B. More comprehensive..."}
    ]
}]

pd.DataFrame(conversations).to_parquet("./data/eval_sft.parquet")
```

---

## Training Configuration

### Key Parameters

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| `MODEL_PATH` | Base model to fine-tune | `Qwen/Qwen2.5-7B-Instruct` |
| `train_batch_size` | Global batch size across all GPUs | 96 |
| `micro_batch_size` | Batch size per GPU | 12 |
| `max_length` | Max sequence length (tokens) | 8192 |
| `lr` | Learning rate | 2e-5 |
| `total_epochs` | Training epochs | 1 |
| `ulysses_sequence_parallel_size` | Split sequences across N GPUs | 8 |

### Configuration File (trainer.yaml)

```yaml
model:
  partial_pretrain: Qwen/Qwen2.5-7B-Instruct
  enable_gradient_checkpointing: true
  fsdp_config:
    model_dtype: bf16

data:
  train_files: ./data/train.parquet
  val_files: ./data/test.parquet
  train_batch_size: 96
  micro_batch_size: 12
  max_length: 8192
  multiturn:
    enable: true
    messages_key: messages

optim:
  lr: 2e-5
  warmup_steps_ratio: 0.03

trainer:
  total_epochs: 1
  save_freq: 500
  test_freq: 500
  default_local_dir: ./checkpoints/sft

ulysses_sequence_parallel_size: 8
use_remove_padding: true
```

---

## Run Training

### Training Script (sft_rm.sh)

```bash
#!/bin/bash
set -x
TIMESTAMP=$(date "+%m%dT%H%M")

# Distributed training configuration
N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-8}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-29500}
NNODES=${WORLD_SIZE:-2}
NODE_RANK=${RANK}

echo "=== SFT Training Configuration ==="
echo "MASTER_ADDR: $MASTER_ADDR"
echo "NODE_RANK: $NODE_RANK"
echo "N_NODES: $NNODES"
echo "N_GPUS_PER_NODE: $N_GPUS_PER_NODE"
echo "==================================="

# Model and data paths
SAVE_PATH=./checkpoints/sft
MODEL_PATH=Qwen/Qwen2.5-7B-Instruct
TRAIN_FILE=./data/train.parquet
VAL_FILE=./data/test.parquet

PROJECT_NAME=rm-gallery-sft
EXPERIMENT_NAME=sft-${TIMESTAMP}

# Launch with torchrun
python -m torch.distributed.run \
    --nnodes=$NNODES \
    --nproc_per_node=$N_GPUS_PER_NODE \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$TRAIN_FILE \
    data.val_files=$VAL_FILE \
    data.train_batch_size=96 \
    data.multiturn.enable=true \
    data.multiturn.messages_key=messages \
    data.multiturn.tools_key="null" \
    data.micro_batch_size=12 \
    data.truncation=right \
    model.enable_gradient_checkpointing=true \
    model.partial_pretrain=$MODEL_PATH \
    model.fsdp_config.cpu_offload=false \
    model.fsdp_config.model_dtype="bf16" \
    trainer.default_local_dir=$SAVE_PATH \
    trainer.project_name=$PROJECT_NAME \
    data.max_length=8192 \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.logger=['console','swanlab'] \
    trainer.total_epochs=1 \
    trainer.default_hdfs_dir=null \
    ulysses_sequence_parallel_size=8 \
    use_remove_padding=true
```

### Execute Training

```bash
# Single-node (8 GPUs)
cd tutorials/cookbooks/training_reward_model/sft
export N_GPUS_PER_NODE=8
bash sft_rm.sh

# Multi-node (4 nodes × 8 GPUs)
export MASTER_ADDR=192.168.1.10
export NNODES=4
export NODE_RANK=0  # Set 0,1,2,3 on each node
bash sft_rm.sh
```

---

## Optimization Options

### Long Sequence Training

```bash
ulysses_sequence_parallel_size=8  # Split across 8 GPUs
data.max_length=16384             # Support 16K tokens
```

### Memory Optimization

```bash
model.enable_gradient_checkpointing=true  # Reduce memory usage
use_remove_padding=true                   # Efficient attention
```

### Speed Optimization

```bash
model.use_liger=true  # Use optimized Liger kernel
```

---

## Monitor Training

**Key metrics:**
- `train/loss` — Should decrease (typical: 2.5 → 0.9)
- `val/loss` — Validation performance
- `tokens/sec` — Training throughput

**Logging options:**

```bash
trainer.logger=['console']              # Console only
trainer.logger=['console','wandb']      # Add WandB
trainer.logger=['console','swanlab']    # Add SwanLab
```

---

## Use Trained Models

### Build Grader from SFT Checkpoint

```python
from rm_gallery.core.models import OpenAIChatModel
from rm_gallery.core.graders.common import RelevanceGrader

model = OpenAIChatModel(
    model="./checkpoints/sft/qwen-sft-final",
    is_local=True
)

grader = RelevanceGrader(model=model)
result = await grader.aevaluate(query="...", response="...")
```

### Use as Initialization for Reward Training

```bash
# Step 1: SFT
cd sft && bash sft_rm.sh

# Step 2: Continue with reward training
cd ../bradley-terry
MODEL_PATH=../sft/checkpoints/qwen-sft-final
bash run_bt.sh
```

---

## Training Tips

- **Data Quality**: Use diverse, multi-turn conversations; filter low-quality content
- **Epochs**: Use 1 epoch to avoid overfitting
- **Sequence Length**: Match your use case (4K-8K for evaluation, 16K+ for documents)
- **Validation**: Monitor `val/loss` — stop if it plateaus or increases

---

## Complete Training Pipeline

**Recommended workflow:**

```bash
# 1. Prepare conversation data
python prepare_sft_data.py --source sharegpt --output ./sft_data

# 2. SFT training
cd tutorials/cookbooks/training_reward_model/sft
bash sft_rm.sh

# 3. Prepare reward data
python prepare_reward_data.py --source helpsteer2 --output ./reward_data

# 4. Reward training (using SFT checkpoint)
cd ../bradley-terry
MODEL_PATH=../sft/checkpoints/sft-final bash run_bt.sh
```

**Why:** SFT establishes baseline capabilities → Reward training specializes for evaluation

---

## Next Steps

- [Bradley-Terry Training](bradley-terry.md) — Train with preference pairs after SFT
- [Generative Pointwise](generative-pointwise.md) — Score-based training after SFT
- [Generative Pairwise](generative-pairwise.md) — Comparison training after SFT
- [Training Overview](overview.md) — Complete training strategy guide

