from lightning.pytorch.cli import LightningCLI
import logging

from src.data.DataModule import SportDataModule
from src.models.model import CustomModel
from lightning.pytorch.cli import SaveConfigCallback
from lightning.pytorch.loggers import Logger


class CustomSaveConfigCallback(SaveConfigCallback):
    def save_config(self, trainer, pl_module, stage):
        if isinstance(trainer.logger, Logger):
            config = self.parser.dump(self.config, skip_none=False)  # Required for proper reproducibility
            trainer.logger.log_hyperparams({"config": config})


def cli_main():
    LightningCLI(
        CustomModel,
        SportDataModule,
        save_config_callback=CustomSaveConfigCallback,
        save_config_kwargs={"save_to_log_dir": False},
    )


if __name__ == "__main__":
    logging.Logger("lightning", level=logging.INFO)
    cli_main()
