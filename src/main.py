from pytorch_lightning.cli import LightningCLI
import wandb
import logging

from src.data.DataModule import SportDataModule
from src.models.model import CustomModel


def cli_main():
    LightningCLI(CustomModel, SportDataModule)


if __name__ == "__main__":
    logging.Logger("lightning", level=logging.INFO)
    cli_main()
