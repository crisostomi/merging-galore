# Merging Galore

A challenge on **weight-space model merging** for multi-task CLIP models.

## The Challenge

Given a pretrained CLIP vision encoder and a set of fine-tuned models (one per task), your goal is to **merge them into a single model** that performs well across all tasks simultaneously — without any additional training.

All pretrained and fine-tuned models are hosted on HuggingFace and downloaded automatically.

## Setup

```bash
# Install dependencies
uv sync

# Create a .env file with the path where model checkpoints will be cached
echo "MODELS_PATH=/path/to/your/models" > .env
```

## How It Works

### The Pipeline

1. A pretrained CLIP encoder (e.g., ViT-B-32) is loaded from HuggingFace
2. Fine-tuned models for each task in the benchmark are loaded
3. Your **Merger** combines the pretrained + fine-tuned models into a single encoder
4. The merged encoder is evaluated on each task's test set
5. Results are reported as average accuracy and normalized accuracy

### Running the Evaluation

```bash
# Run with default settings (Task Arithmetic on N8 benchmark)
uv run python scripts/evaluate_multitask_merging.py

# Use a different merger
uv run python scripts/evaluate_multitask_merging.py merger=weight_avg

# Use a different benchmark size
uv run python scripts/evaluate_multitask_merging.py benchmark=N20

# Use a different encoder
uv run python scripts/evaluate_multitask_merging.py nn/encoder=b16

# Combine overrides
uv run python scripts/evaluate_multitask_merging.py merger=task_arithmetic benchmark=N14 nn/encoder=l14
```

## Implementing Your Merger

### 1. Create a new merger class

Create a file in `src/model_merging/merger/`, e.g. `my_merger.py`:

```python
from model_merging.merger.merger import TaskVectorBasedMerger
from model_merging.model.encoder import ImageEncoder


class MyMerger(TaskVectorBasedMerger):

    def __init__(self, my_param=1.0, device="cuda"):
        super().__init__()
        self.my_param = my_param
        self.device = device

    def merge(self, base_model: ImageEncoder, finetuned_models: dict) -> ImageEncoder:
        """
        Args:
            base_model: The pretrained (zero-shot) ImageEncoder.
            finetuned_models: Dict mapping dataset configs to state_dicts
                              of fine-tuned models.

        Returns:
            A merged ImageEncoder.
        """
        # Your merging logic here!
        # ...
        return merged_encoder
```

### 2. Add a Hydra config

Create `conf/merger/my_merger.yaml`:

```yaml
_target_: model_merging.merger.my_merger.MyMerger
my_param: 1.0
device: ${device}
```

### 3. Run it

```bash
uv run python scripts/evaluate_multitask_merging.py merger=my_merger
```

## Provided Baselines

| Merger | Config | Description |
|--------|--------|-------------|
| `DummyMerger` | `dummy` | Returns the pretrained model unchanged (lower bound) |
| `WeightAverageMerger` | `weight_avg` | Simple average of all fine-tuned weights |
| `TaskArithmeticMerger` | `task_arithmetic` | Weighted sum of task vectors: `base + alpha * sum(ft_i - base)` |

## Benchmarks

| Benchmark | # Tasks | Datasets |
|-----------|---------|----------|
| N2 | 2 | RESISC45, Cars |
| N8 | 8 | SUN397, Cars, RESISC45, EuroSAT, SVHN, GTSRB, MNIST, DTD |
| N14 | 14 | N8 + Flowers102, PCAM, FER2013, OxfordIIITPet, STL10, CIFAR100 |
| N20 | 20 | N14 + CIFAR10, Food101, FashionMNIST, EMNIST, KMNIST, RenderedSST2 |

## Available Encoders

| Config | Model |
|--------|-------|
| `b32` (default) | ViT-B-32 |
| `b16` | ViT-B-16 |
| `l14` | ViT-L-14 |

## Key Concepts

- **Task Vector**: The difference between a fine-tuned model's weights and the pretrained weights: `tv = finetuned - pretrained`
- **Merging**: Combining multiple task vectors into a single set of weights that works across all tasks
- The `merging/task_vectors.py` module provides `compute_task_vector()` for computing task vectors
- The `utils/utils.py` module provides `compute_task_dict()`, `sum_task_dict()`, and `apply_dict_to_model()` helpers

## Project Structure

```
merging-galore/
├── src/model_merging/          # Core library
│   ├── model/                  # ImageEncoder, ClassificationHead, ImageClassifier
│   ├── merger/                 # Merger base class + baselines (add yours here!)
│   ├── merging/                # Task vector computation
│   ├── data/                   # Dataset loading, CLIP templates
│   └── utils/                  # Utilities, model I/O
├── scripts/
│   └── evaluate_multitask_merging.py  # Main evaluation script
├── conf/                       # Hydra configs
│   ├── multitask.yaml          # Main config
│   ├── merger/                 # Merger configs
│   ├── benchmark/              # Task set definitions
│   ├── dataset/                # Per-dataset configs
│   └── nn/encoder/             # Model architecture configs
├── results/finetuning/         # Fine-tuning accuracy upper bounds
└── pyproject.toml
```
