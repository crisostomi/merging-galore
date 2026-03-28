from collections import OrderedDict
import json
import logging
import os
import random
from typing import Dict, List, Optional, Union

import hydra
import numpy as np
import psutil
import pytorch_lightning as pl
import torch
from omegaconf import ListConfig
from pytorch_lightning import Callback

pylogger = logging.getLogger(__name__)


def print_memory(context):
    process = psutil.Process(os.getpid())
    pylogger.warning(
        f"{context} -- memory in MB: { process.memory_info().rss / 1024**2}",
    )


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [
        float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
        for k in topk
    ]


def get_finetuning_accuracies(path):
    with open(path, "rb") as f:
        finetuning_accuracies = json.load(f)
    return finetuning_accuracies


def compute_avg_accuracy(results) -> Dict:
    total_acc = 0
    total_normalized_acc = 0
    count = 0

    for dataset_name, metrics in results.items():
        for m in metrics:
            total_acc += m[f"acc/test/{dataset_name}"]
            total_normalized_acc += m[f"normalized_acc/test/{dataset_name}"]
            count += 1

    average_acc = total_acc / count if count > 0 else 0
    average_normalized_acc = total_normalized_acc / count if count > 0 else 0

    return {
        "acc/test/avg": average_acc,
        "normalized_acc/test/avg": average_normalized_acc,
    }


def torch_save(model, save_path):
    if os.path.dirname(save_path) != "":
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.cpu(), save_path)


def torch_load(save_path, device=None):
    model = torch.load(save_path, weights_only=False)
    if device is not None:
        model = model.to(device)
    return model


def apply_dict_to_model(task_vector_dict, model, coefficient: float = 1.0, device="cuda"):
    """
    Applies a task vector dictionary to a model.
    """
    with torch.no_grad():
        model.to(device)
        new_state_dict = model.state_dict()

        for key, value in task_vector_dict.items():
            new_key = key.replace("encoder.", "")
            if new_key not in new_state_dict:
                pylogger.warning(
                    f"Key {new_key} is present in the task vector but not in the model"
                )
                continue
            else:
                new_state_dict[new_key] += coefficient * value.to(device)

        model.load_state_dict(new_state_dict, strict=False)
    return model


def sum_task_dict(task_vector_dict_1, task_vector_dict_2):
    """
    Sums two task vector dictionaries.
    """
    for key, value in task_vector_dict_2.items():
        if key in task_vector_dict_1:
            task_vector_dict_1[key] += value
        else:
            task_vector_dict_1[key] = value
    return task_vector_dict_1


@torch.no_grad()
def compute_task_dict(pretrained, finetuned):
    new_state_dict = OrderedDict()

    for key in pretrained:
        if pretrained[key].dtype in [torch.int64, torch.uint8]:
            pylogger.info(f"Skipping key {key}")
            continue

        difference = finetuned[key] - pretrained[key]
        new_state_dict[key] = difference

    return new_state_dict


def build_callbacks(cfg: ListConfig, verbose=False) -> List[Callback]:
    """Instantiate the callbacks given their configuration."""
    callbacks: List[Callback] = []

    for callback in cfg:
        if verbose:
            pylogger.info(f"Adding callback <{callback['_target_'].split('.')[-1]}>")
        callbacks.append(hydra.utils.instantiate(callback, _recursive_=False))

    return callbacks


max_seed_value = np.iinfo(np.uint32).max
min_seed_value = np.iinfo(np.uint32).min


def seed_everything(seed: Optional[int] = None) -> int:
    if seed is None:
        env_seed = os.environ.get("PL_GLOBAL_SEED")
        if env_seed is None:
            seed = random.randint(min_seed_value, max_seed_value)
            pylogger.warning(f"No seed found, seed set to {seed}")
        else:
            try:
                seed = int(env_seed)
            except ValueError:
                seed = random.randint(min_seed_value, max_seed_value)
                pylogger.warning(
                    f"Invalid seed found: {env_seed!r}, seed set to {seed}"
                )
    elif not isinstance(seed, int):
        seed = int(seed)

    if not (min_seed_value <= seed <= max_seed_value):
        pylogger.warning(
            f"{seed} is out of bounds. Selecting a new seed."
        )
        seed = random.randint(min_seed_value, max_seed_value)

    pylogger.info(f"Seed set to {seed}")
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    return seed
