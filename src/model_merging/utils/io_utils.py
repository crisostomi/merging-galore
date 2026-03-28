import logging
import tempfile

from huggingface_hub import hf_hub_download

import torch

from model_merging.model.encoder import ImageEncoder

pylogger = logging.getLogger(__name__)


def load_model_from_hf(model_name, dataset_name="base") -> ImageEncoder:

    model_path = f"crisostomi/{model_name}-{dataset_name}"

    ckpt_path = hf_hub_download(repo_id=model_path, filename="pytorch_model.bin")
    state_dict = torch.load(ckpt_path, map_location="cpu")

    model = ImageEncoder(model_name)
    model.load_state_dict(state_dict)
    return model
