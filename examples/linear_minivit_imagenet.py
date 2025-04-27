import argparse
import pathlib

import lightning as L
import torch

import sensa


def args_parsed():
    parser = argparse.ArgumentParser("MiniImagenet", add_help=False)

    parser.add_argument("--config", type=str, default="linear_minivit_imagenet")
    parser.add_argument("--path", type=str, default="./checkpoints")

    parser.add_argument("-a", "--auto", action="store_true")
    parser.add_argument("-d", "--delete", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = args_parsed()
    cfg = sensa.assets.load_yaml(f"{args.config}.yaml")

    if args.auto:
        mae_config_prefix = args.config.replace("linear_", "mae_")
        checkpoints = sorted(pathlib.Path(args.path).resolve().glob(f"{mae_config_prefix}_*"))
        if len(checkpoints):
            cfg["backbone"]["path_to_checkpoint"] = str(checkpoints[-1])

    pt_model = sensa.supervised.Classifier(**cfg)

    trainer = L.Trainer(
        log_every_n_steps=pt_model.params.trainer.logger_frequency,
        logger=False,
        devices=torch.cuda.device_count(),
        accelerator="auto",
        precision="bf16-mixed" if pt_model.params.trainer.mixed_precision else "fp32",
        strategy="auto",
        sync_batchnorm=torch.cuda.device_count() > 1,
        use_distributed_sampler=torch.cuda.device_count() > 1,
        benchmark=True,
        accumulate_grad_batches=pt_model.params.trainer.accumulate_grad_batches,
        max_epochs=pt_model.params.trainer.epochs,
        callbacks=[
            L.pytorch.callbacks.TQDMProgressBar(refresh_rate=20),
            L.pytorch.callbacks.ModelCheckpoint(
                dirpath=args.path,
                filename=args.config + "_{epoch}",
                monitor="epoch",
                mode="max",
                enable_version_counter=False,
            ),
        ],
    )

    trainer.fit(model=pt_model)
