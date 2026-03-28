import copy
import logging
from typing import Dict
import torch
from model_merging.merger.merger import TaskVectorBasedMerger
from model_merging.model.encoder import ImageEncoder
from model_merging.utils.utils import (
    apply_dict_to_model,
    compute_task_dict,
    sum_task_dict,
)

pylogger = logging.getLogger(__name__)


class TaskArithmeticMerger(TaskVectorBasedMerger):

    def __init__(self, optimal_alpha, device="cuda"):
        super().__init__()

        self.optimal_alpha = optimal_alpha
        self.device = torch.device(device)

    def merge(
        self, base_model: ImageEncoder, finetuned_models: Dict[str, dict]
    ) -> ImageEncoder:

        comulative_dict = {}

        base_model.to(self.device)

        datasets = list(finetuned_models.keys())
        pretrained_model = copy.deepcopy(base_model)

        base_state = base_model.state_dict()
        for dataset in datasets:
            ft_state = finetuned_models[dataset]
            ft_state = {k: v.to(self.device) for k, v in ft_state.items()}
            comulative_dict = sum_task_dict(
                comulative_dict,
                compute_task_dict(base_state, ft_state),
            )
            del finetuned_models[dataset]
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        merged_encoder = apply_dict_to_model(
            comulative_dict, pretrained_model, coefficient=self.optimal_alpha,
            device=self.device,
        )

        return merged_encoder
