# model.py
import torch
import timm
import lightning as pl
from torchmetrics import F1Score, Accuracy
from torch.nn import CrossEntropyLoss
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch import seed_everything
import torchvision.transforms as transforms


class CustomModel(pl.LightningModule):
    def __init__(self, num_classes, model_name, lr):
        super().__init__()
        # Extract values from the config
        self.num_classes = num_classes
        self.model_name = model_name
        self.lr = lr

        # log hyperparameters with lightning
        self.save_hyperparameters()
        # make sure model is deterministic and reproducable (deterministig flag in config has to be true)
        seed_everything(42, workers=True)

        self.f1 = F1Score(task="multiclass", num_classes=self.num_classes)
        self.accuracy = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.model = timm.create_model(self.model_name, pretrained=True, num_classes=self.num_classes)
        self.ce_loss = CrossEntropyLoss()

    def preprocess_input(self, image):
        """
        Apply the same transformations to the input as used in training.
        """
        self.transformations = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        if self.transformations is not None:
            return self.transformations(image)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), self.lr)

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

    def predict(self, image):
        self.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            processed_image = self.preprocess_input(image)
            # If using a GPU, you might need to move the processed image to GPU here
            processed_image = processed_image.to(self.device)
            prediction = self.forward(processed_image.unsqueeze(0))  # Unsqueeze to add batch dimension
            return prediction

    def configure_logging(self):
        wandb_logger = WandbLogger(
            log_model=True,  # Log the best model
            save_dir="wandb_logs",  # Save logs in a specific directory
        )
        return [wandb_logger]
