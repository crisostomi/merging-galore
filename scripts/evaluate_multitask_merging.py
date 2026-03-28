import json
import logging
from pathlib import Path
from typing import Dict, List

import hydra
import omegaconf
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

# Force the execution of __init__.py if this file is executed directly.
import model_merging  # noqa
from model_merging import PROJECT_ROOT
from model_merging.model.encoder import ImageEncoder
from model_merging.model.image_classifier import ImageClassifier
from model_merging.model.heads import get_classification_head
from model_merging.utils.io_utils import load_model_from_hf
from model_merging.utils.utils import (
    build_callbacks,
    get_finetuning_accuracies,
    compute_avg_accuracy,
    print_memory,
    seed_everything,
)

pylogger = logging.getLogger(__name__)

torch.set_float32_matmul_precision("high")


def run(cfg: DictConfig) -> str:

    seed_everything(cfg.seed_index)

    num_tasks = len(cfg.benchmark.datasets)

    omegaconf.OmegaConf.set_struct(cfg, False)
    cfg.num_tasks = num_tasks
    omegaconf.OmegaConf.set_struct(cfg, True)

    # upperbound accuracies, used for logging the normalized accuracy
    finetuned_accuracies: Dict[str, float] = get_finetuning_accuracies(
        cfg.misc.finetuned_accuracy_path
    )[cfg.nn.encoder.model_name]

    # only has vision encoder, no text transformer
    zeroshot_encoder: ImageEncoder = load_model_from_hf(
        model_name=cfg.nn.encoder.model_name
    )

    finetuned_models = {
        dataset: load_model_from_hf(
            model_name=cfg.nn.encoder.model_name, dataset_name=dataset.name
        ).state_dict()
        for dataset in cfg.benchmark.datasets
    }

    pylogger.info(f"Number of tasks: {cfg.num_tasks}")
    pylogger.info(f"Finetuned models: {list(finetuned_models.keys())}")

    merger = instantiate(cfg.merger)

    merged_encoder = merger.merge(zeroshot_encoder, finetuned_models)

    results = {}
    print_memory("before eval")
    for dataset_cfg in cfg.benchmark.datasets:

        dataset = instantiate(
            dataset_cfg, preprocess_fn=zeroshot_encoder.val_preprocess
        )

        classification_head = get_classification_head(
            cfg.nn.encoder.model_name,
            dataset_cfg.name,
            ckpt_path=cfg.misc.ckpt_path,
            openclip_cachedir=cfg.misc.openclip_cachedir,
            device=cfg.device,
        )

        model = ImageClassifier(
            encoder=merged_encoder,
            classifier=classification_head,
            x_key=cfg.conventions.x_key,
            y_key=cfg.conventions.y_key,
        )

        model.set_metrics(len(dataset.classnames))
        model.set_task(dataset_cfg.name)
        model.set_finetuning_accuracy(
            finetuned_accuracies[dataset_cfg.name]
        )

        callbacks: List[pl.Callback] = build_callbacks(cfg.train.callbacks)

        trainer = pl.Trainer(
            default_root_dir=cfg.core.storage_dir,
            logger=False,
            callbacks=callbacks,
            **cfg.train.trainer,
        )

        if cfg.eval_on_val:
            pylogger.info(f"Evaluating on {dataset_cfg.name} validation split")
            test_results = trainer.test(model=model, dataloaders=dataset.val_loader)
        else:
            pylogger.info(f"Evaluating on the {dataset_cfg.name} test set!")
            test_results = trainer.test(model=model, dataloaders=dataset.test_loader)

        results[dataset_cfg.name] = test_results

    avg = compute_avg_accuracy(results)
    results["avg"] = [avg]

    pylogger.info(results)

    results_path = Path(cfg.misc.results_path)
    results_path.mkdir(parents=True, exist_ok=True)
    with open(results_path / f"{len(cfg.benchmark.datasets)}.json", "w+") as f:
        json.dump(results, f, indent=4)

    pylogger.info(f"Results saved to {cfg.misc.results_path}")
    pylogger.info(f"Average accuracy: {avg['acc/test/avg']:.4f}")
    pylogger.info(f"Average normalized accuracy: {avg['normalized_acc/test/avg']:.4f}")


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="multitask.yaml")
def main(cfg: omegaconf.DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()
