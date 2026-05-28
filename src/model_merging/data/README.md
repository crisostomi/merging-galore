# Data Module

This module provides a data loading pipeline that wraps HuggingFace datasets into PyTorch-compatible formats for image classification tasks.

## Core Components

### `_HFImageTorchDataset`

A PyTorch `Dataset` wrapper around a HuggingFace dataset split. It:
- Applies torchvision-style transforms to images
- Optionally remaps labels (e.g., reordering class indices to match a specific order)
- Returns `(image, label)` tuples

### `HFImageClassification`

The main adapter class that:
- Takes an **already-loaded** `DatasetDict` from HuggingFace
- Creates `train_dataset`, `val_dataset`, and `test_dataset` as PyTorch datasets
- Creates `train_loader`, `val_loader`, and `test_loader` as PyTorch DataLoaders
- **Automatically reserves 10% of the test set for validation** (random split with seed for reproducibility, no contamination)
- Extracts `classnames` from the dataset's `ClassLabel` feature (or uses an override)

**Key parameters:**

| Parameter | Description |
|-----------|-------------|
| `hf_ds` | A HuggingFace `DatasetDict` with train/test splits |
| `preprocess` | Image transform (e.g., CLIP's preprocessing) |
| `ft_epochs` | Number of finetuning epochs (stored as metadata) |
| `split_map` | Maps logical splits to actual HF split names (e.g., `{"train": "train", "test": "validation"}`) |
| `batch_size` | Batch size for DataLoaders (default: 128) |
| `label_map` | Remaps integer labels to a different ordering |
| `classnames_override` | Manually specify class names instead of extracting from dataset |
| `val_fraction` | Fraction of test set to reserve for validation (default: 0.1) |
| `seed` | Random seed for reproducible val/test split (default: 42) |

## Utility Functions

| Function | Purpose |
|----------|---------|
| `convert()` | Converts numpy arrays to PIL images |
| `load_fer2013()` | Special loader for FER2013 emotion dataset (renames columns) |
| `emnist_preprocess_fn()` | Adds rotation/flip transforms for EMNIST (handwritten letters) |
| `compute_label_map_from_names()` | Computes a label remapping array given current vs. desired class orderings |
| `load_hf_dataset_filtered()` | Loads only standard splits (train/test/val), avoiding extras like "extra" or "unlabeled" |
| `load_dataset()` | High-level factory function that instantiates everything from Hydra configs |
| `maybe_dictionarize()` | Converts tuple batches to dict format with `x_key` and `y_key` |

## Resulting Dataset Structure

After using `load_dataset()` or `HFImageClassification`, you get an object with:

```python
dataset.train_dataset  # PyTorch Dataset → yields (preprocessed_image_tensor, label_int)
dataset.val_dataset    # PyTorch Dataset → 10% of original test set (random, no contamination)
dataset.test_dataset   # PyTorch Dataset → 90% of original test set (random, no contamination)

dataset.train_loader   # DataLoader(train_dataset, batch_size=128, shuffle=True, ...)
dataset.val_loader     # DataLoader(val_dataset, batch_size=128, shuffle=False, ...)
dataset.test_loader    # DataLoader(test_dataset, batch_size=128, shuffle=False, ...)

dataset.classnames     # List[str], e.g., ["cat", "dog", "bird", ...]
dataset.ft_epochs      # int, number of finetuning epochs (stored metadata)
```

**Each batch from the DataLoader:**

```python
images, labels = next(iter(dataset.train_loader))
# images: Tensor of shape [batch_size, C, H, W] (preprocessed for CLIP/ViT)
# labels: Tensor of shape [batch_size] (integer class indices)
```

## Usage Example

```python
from model_merging.data.dataset import load_dataset

dataset = load_dataset(
    name="CIFAR10",
    hf_dataset={"_target_": "datasets.load_dataset", "path": "cifar10"},
    preprocess_fn=encoder.val_preprocess,
    ft_epochs=10,
    batch_size=64,
)

# Access data
for images, labels in dataset.train_loader:
    # Training loop
    pass

# Get class names
print(dataset.classnames)  # ['airplane', 'automobile', 'bird', ...]
```

## Label Remapping

When class orderings differ between datasets and model heads, use `compute_label_map_from_names()`:

```python
from model_merging.data.dataset import compute_label_map_from_names

# Dataset has: ["cat", "dog", "bird"]
# Model expects: ["bird", "cat", "dog"]
label_map = compute_label_map_from_names(
    current_names=["cat", "dog", "bird"],
    desired_order=["bird", "cat", "dog"]
)
# label_map = [1, 2, 0]  →  old_label 0 ("cat") becomes new_label 1
```
