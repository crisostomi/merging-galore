import logging
from pathlib import Path
from typing import Tuple

import torch
from tqdm import tqdm

from model_merging.utils.utils import is_matrix

pylogger = logging.getLogger(__name__)


@torch.no_grad()
def isotropic_sum(ref_state_dict, svd_dict, device="cuda"):
    aggregated_model_dict = ref_state_dict
    layer_names = list(aggregated_model_dict.keys())

    datasets = list(svd_dict.keys())

    for layer_name in tqdm(layer_names, desc="Summing SVD"):
        is_layer_matrix = aggregated_model_dict[layer_name].dim() == 2

        for i, dataset in enumerate(datasets):
            if "text_projection" in layer_name:
                continue

            if is_layer_matrix:
                delta_layer_svd = svd_dict[dataset][layer_name]

                u, s, v = (
                    delta_layer_svd["u"].to(device),
                    delta_layer_svd["s"].to(device),
                    delta_layer_svd["v"].to(device),
                )
                delta = u @ torch.diag_embed(s) @ v

                if i == 0:
                    summed = torch.zeros_like(delta)

                summed += delta

            else:
                delta_layer = svd_dict[datasets[i]][layer_name]["dim1"].to(device)

                if i == 0:
                    aggregated_model_dict[layer_name] = delta_layer
                else:
                    aggregated_model_dict[layer_name] += (
                        delta_layer - aggregated_model_dict[layer_name]
                    ) / (i + 1)

        if "text_projection" in layer_name or not is_layer_matrix:
            continue

        u, s, v = torch.linalg.svd(summed, full_matrices=False)
        iso_factor = torch.mean(s)
        aggregated_model_dict[layer_name] = iso_factor * u @ v

    return aggregated_model_dict


@torch.no_grad()
def aggregate_decomposed_task_vectors(
    ref_state_dict,
    decomposed_task_vectors,
    device="cuda",
    non_matrix_params_aggregation="base_model",
):
    """Concatenate per-task SVD factors and re-orthogonalize for the TSV merger."""

    aggregated_model_dict = ref_state_dict
    layer_names = list(aggregated_model_dict.keys())

    datasets = list(decomposed_task_vectors.keys())

    for layer_name in tqdm(layer_names, desc="Summing SVD"):
        is_layer_matrix = aggregated_model_dict[layer_name].dim() == 2
        new_key = layer_name
        offset = 0

        for i, dataset in enumerate(datasets):
            if "text_projection" in layer_name:
                continue

            if is_layer_matrix:
                delta_layer_svd = decomposed_task_vectors[dataset][new_key]

                u, s, v = (
                    delta_layer_svd["u"].to(device),
                    delta_layer_svd["s"].to(device),
                    delta_layer_svd["v"].to(device),
                )

                if i == 0:
                    total_rank = sum(
                        decomposed_task_vectors[d][new_key]["s"].shape[0]
                        for d in datasets
                    )
                    sum_u = torch.zeros(u.shape[0], total_rank, device=device)
                    sum_s = torch.zeros(total_rank, device=device)
                    sum_v = torch.zeros(total_rank, v.shape[1], device=device)

                rank_i = s.shape[0]
                sum_u[:, offset : offset + rank_i] = u
                sum_s[offset : offset + rank_i] = s
                sum_v[offset : offset + rank_i, :] = v
                offset += rank_i

            else:
                delta_layer = decomposed_task_vectors[datasets[i]][new_key]["dim1"].to(
                    device
                )

                if non_matrix_params_aggregation == "mean":
                    if i == 0:
                        aggregated_model_dict[layer_name] = delta_layer
                    else:
                        aggregated_model_dict[layer_name] += (
                            delta_layer - aggregated_model_dict[layer_name]
                        ) / (i + 1)
                else:
                    aggregated_model_dict[layer_name] = torch.zeros_like(delta_layer)

        if "text_projection" in layer_name or not is_layer_matrix:
            continue

        u_u, s_u, v_u = torch.linalg.svd(sum_u, full_matrices=False)
        u_v, s_v, v_v = torch.linalg.svd(sum_v, full_matrices=False)

        aggregated_model_dict[layer_name] = torch.linalg.multi_dot(
            (u_u, v_u, torch.diag(sum_s), u_v, v_v)
        ).to(device)

    return aggregated_model_dict


def compute_svd_and_compress(
    matrix, compress_ratio
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    u, s, v = torch.linalg.svd(matrix, full_matrices=False)
    reduced_index_s = int(s.shape[0] * compress_ratio)
    return u[:, :reduced_index_s], s[:reduced_index_s], v[:reduced_index_s, :]


def decompose_task_vectors(task_dicts, compress_rate: float):
    with torch.no_grad():
        svd_dict = {}

        for dataset, task_dict in tqdm(
            task_dicts.items(), desc="Computing and compressing SVD"
        ):
            svd_dict[dataset] = {}

            for key, layer in task_dict.items():
                if is_matrix(layer):
                    u, s, v = compute_svd_and_compress(layer, compress_rate)
                    svd_dict[dataset][key] = {
                        "u": u.detach().cpu(),
                        "s": s.detach().cpu(),
                        "v": v.detach().cpu(),
                    }
                else:
                    svd_dict[dataset][key] = {"dim1": layer.detach().cpu()}

        return svd_dict


def get_svd_dict(
    task_dicts,
    datasets,
    svd_path: str = None,
    compression_factor: float = None,
):
    """Load SVD dict from disk if it matches the requested datasets; otherwise compute and cache."""

    compression_factor = compression_factor or len(datasets)
    compression_ratio = 1 / compression_factor
    pylogger.info(f"Using compression ratio: {compression_ratio:.4f}")

    if svd_path is not None:
        svd_path = str(Path(svd_path))
        if svd_path.endswith(".pt"):
            svd_path = svd_path[:-3]
        svd_path = f"{svd_path}_compress_{compression_factor}.pt"

        if Path(svd_path).exists():
            pylogger.info(f"Loading precomputed SVD dictionary from: {svd_path}")
            svd_dict = torch.load(svd_path, map_location="cuda", weights_only=False)

            if set(svd_dict.keys()) == set(datasets):
                return svd_dict

            pylogger.warning("Mismatch in datasets. Recomputing SVD dictionary...")
        else:
            pylogger.info("No precomputed SVD dictionary found. Computing from scratch...")
    else:
        pylogger.info("SVD caching disabled. Computing SVD from scratch...")

    svd_dict = decompose_task_vectors(task_dicts, compression_ratio)

    if svd_path is not None:
        torch.save(svd_dict, svd_path)
        pylogger.info(f"SVD dictionary saved at: {svd_path}")

    return svd_dict
