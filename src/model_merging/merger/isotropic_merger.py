import copy
import logging

import torch

from model_merging.merger.merger import TaskVectorBasedMerger
from model_merging.merging.structured import get_svd_dict, isotropic_sum
from model_merging.model.encoder import ImageEncoder
from model_merging.utils.utils import (
    apply_dict_to_model,
    compute_task_dict,
    print_memory,
)

pylogger = logging.getLogger(__name__)


class IsotropicMerger(TaskVectorBasedMerger):

    def __init__(
        self,
        optimal_alphas,
        svd_path,
        svd_compress_factor,
        model_name,
        alpha=None,
        device="cuda",
    ):
        super().__init__()
        self.alpha = alpha
        self.optimal_alphas = optimal_alphas
        self.svd_path = svd_path
        self.svd_compress_factor = svd_compress_factor
        self.model_name = model_name
        self.device = device

    @torch.no_grad()
    def merge(self, base_model, finetuned_models) -> ImageEncoder:
        task_dicts = {}
        datasets = list(finetuned_models.keys())
        num_tasks = str(len(datasets))

        for dataset in datasets:
            task_dicts[dataset] = compute_task_dict(
                base_model.state_dict(), finetuned_models[dataset]
            )
            del finetuned_models[dataset]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        print_memory("after computing task dicts")

        svd_dict = get_svd_dict(
            task_dicts, datasets, self.svd_path, self.svd_compress_factor
        )

        multi_task_vector = isotropic_sum(
            ref_state_dict=copy.deepcopy(base_model.state_dict()),
            svd_dict=svd_dict,
            device=self.device,
        )

        if self.alpha is not None:
            coefficient = self.alpha
        elif (
            self.model_name in self.optimal_alphas
            and num_tasks in self.optimal_alphas[self.model_name]
        ):
            coefficient = self.optimal_alphas[self.model_name][num_tasks]
        else:
            raise ValueError(
                f"No alpha provided and no optimal alpha found for model {self.model_name} "
                f"with {num_tasks} tasks"
            )

        merged_encoder: ImageEncoder = copy.deepcopy(base_model)
        merged_encoder = apply_dict_to_model(
            multi_task_vector,
            merged_encoder,
            coefficient=coefficient,
            device=self.device,
        )

        return merged_encoder
