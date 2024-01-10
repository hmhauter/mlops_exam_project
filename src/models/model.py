# model.py
import torch
import timm
import pytorch_lightning as pl
from pytorch_lightning.cli import LightningCLI
from torchmetrics import F1Score, Accuracy
from torch.nn import CrossEntropyLoss
import wandb
from omegaconf import DictConfig
import logging


class CustomModel(pl.LightningModule):
    def __init__(self, num_classes, model_name):
        super(CustomModel, self).__init__()
        logging.info(f"Loaded Config:\n")
        # Extract values from the config
        num_classes = num_classes
        model_name = model_name
        # log hyperparameters with lightning
        self.save_hyperparameters()

        self.f1 = F1Score(task="multiclass", num_classes=num_classes)
        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
        self.ce_loss = CrossEntropyLoss()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def forward(self, inp):
        return self.model(inp)

    def training_step(self, batch, batch_idx):
        ims, gts = batch
        preds = self.model(ims)
        loss = self.ce_loss(preds, gts)

        # Train metrics
        pred_clss = torch.argmax(preds, dim=1)
        acc = self.accuracy(pred_clss, gts)
        f1 = self.f1(pred_clss, gts)

        # log with pytorch lightning
        self.log("train_loss", loss, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        self.log("train_acc", acc, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        self.log("train_f1", f1, on_step=False, on_epoch=True, logger=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        ims, gts = batch
        preds = self.model(ims)
        loss = self.ce_loss(preds, gts)

        # Train metrics
        pred_clss = torch.argmax(preds, dim=1)
        acc = self.accuracy(pred_clss, gts)
        f1 = self.f1(pred_clss, gts)

        # log with pytorch lightning
        self.log("val_loss", loss, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        self.log("val_f1", f1, on_step=False, on_epoch=True, logger=True, sync_dist=True)

        return loss


# def main(cfg: DictConfig) -> None:
#     print(f"Loaded Config:\n{cfg}")

#     # use WandB for logging
#     wandb.init()

#     # use Hydra to instantiate the Lightning model to inject hyperparameter configuration
#     lightning_model = CustomModel(cfg=cfg)

#     trainer = pl.Trainer(
#         max_epochs=cfg.trainer.max_epochs,
#         logger=pl.loggers.WandbLogger(),
#     )

#     # Training
#     trainer.fit(lightning_model)

#     # Access the hyperparameters from the instantiated model
#     print("Hyperparameters:", lightning_model.hparams)

if __name__ == "__main__":
    # wandb.init()
    # Use LightningCLI to automatically handle command-line arguments and configuration
    # python src/models/model.py fit --model.num_classes 10 --model.model_name resnet18
    logging.Logger("lightning", level=logging.INFO)
    cli = LightningCLI(CustomModel)
