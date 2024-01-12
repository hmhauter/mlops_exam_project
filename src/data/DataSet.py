from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd
from PIL import Image
from typing import Union
import logging
import torchvision.transforms as transforms


class SportDataset(Dataset):
    def __init__(self, csv_file: str | Path, data_dir: str | Path, transformations=None):
        super().__init__()

        if not isinstance(csv_file, (str, Path)):
            logging.error("csv_file must be of type str or Path")
            raise TypeError("csv_file must be of type str or Path")
        if not isinstance(data_dir, (str, Path)):
            logging.error("data_dir must be of type str or Path")
            raise TypeError("data_dir must be of type str or Path")
        if transformations is not None and not isinstance(transformations, transforms.Compose):
            logging.error("transformations must be of type transforms.Compose or None")
            raise TypeError("transformations must be of type transforms.Compose or None")

        self.data_dir = Path(data_dir)
        csv_path = self.data_dir / Path(csv_file)

        self.transformations = transformations or transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        if not csv_path.exists():
            logging.error(f"CSV file '{csv_file}' does not exist")
            raise FileNotFoundError(f"CSV file '{csv_file}' does not exist")

        self.df = pd.read_csv(csv_path)
        self.labels = self.df["label"].unique()
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.labels)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if idx >= len(self):
            logging.error("Index out of range")
            raise IndexError("Index out of range")

        row = self.df.iloc[idx]
        image = Image.open(self.data_dir / Path("train") / row["image_ID"]).convert("RGB")
        if self.transformations:
            image = self.transformations(image)

        if self.labels is not None:
            label = self.class_to_idx.get(row["label"])
            if label is None:
                logging.error(f"Label '{row['label']}' not found in class_to_idx mapping")
                raise ValueError(f"Label '{row['label']}' not found in class_to_idx mapping")
            return image, label
        else:
            return image
