import shutil
from pathlib import Path

import lightning as L
from lightning import Trainer
from torch_geometric.loader.dataloader import DataLoader

from camp.data.dataset import Dataset
from camp.data.transform import ConsecutiveAtomType
from camp.model.camp import CAMP
from camp.training.utils import get_args, instantiate_class, load_model


def get_dataset(filename: Path, atomic_number: list[int], r_cut: float):
    dataset = Dataset(
        filename=filename,
        target_names=("energy", "forces"),
        r_cut=r_cut,
        transform=ConsecutiveAtomType(atomic_number),
        log=False,
    )

    return dataset


def get_dataloaders(
    atomic_number,
    r_cut,
    valset_filename,
    testset_filename,
    val_batch_size,
    test_batch_size,
    **kwargs,
):
    if valset_filename is not None:
        valset = get_dataset(valset_filename, atomic_number, r_cut)
        val_loader = DataLoader(valset, batch_size=val_batch_size, shuffle=False)
    else:
        val_loader = None

    testset = get_dataset(testset_filename, atomic_number, r_cut)
    test_loader = DataLoader(testset, batch_size=test_batch_size, shuffle=False)

    return val_loader, test_loader


def main(config: dict):
    L.seed_everything(config["seed_everything"])

    # Get model
    restore_checkpoint = config.pop("restore_checkpoint")
    assert restore_checkpoint is not None

    model = load_model(CAMP, restore_checkpoint, map_location="cpu")
    print(f"Loading model from checkpoint: {restore_checkpoint}")

    # Load data
    data_config = config.pop("data")
    # set cutoff to be the one used in training
    data_config["r_cut"] = model.hparams["other_hparams"]["data"]["r_cut"]
    val_loader, test_loader = get_dataloaders(**data_config)

    try:
        logger = instantiate_class(config["trainer"].pop("logger"))
    except KeyError:
        logger = None

    trainer = Trainer(
        logger=logger,
        accelerator=config["trainer"].pop("accelerator"),
        inference_mode=False,
    )

    if val_loader is not None:
        out = trainer.validate(model, dataloaders=val_loader)
        print("Metrics on val set:", out)
    else:
        print("No validation set provided")

    out = trainer.test(model, dataloaders=test_loader)
    print("Metrics on test set:", out)


if __name__ == "__main__":
    # remove the processed data directory
    shutil.rmtree("./processed", ignore_errors=True)

    config_file = Path(__file__).parent / "configs" / "config_eval.yaml"
    config = get_args(config_file)
    main(config)
