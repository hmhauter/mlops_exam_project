from typing import Optional
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from src.data.DataSet import SportDataset

class SportDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = './data', batch_size: int = 32, num_workers: int = 4, transformations=None, train_val_test_split=[0.7, 0.15, 0.15]):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transformations = transformations
        self.train_val_test_split = train_val_test_split

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: Optional[str] = None):
        # Load the full dataset
        full_dataset = SportDataset(csv_file="train.csv", data_dir=self.data_dir, transformations=self.transformations)

        # Calculate split lengths
        full_len = len(full_dataset)
        train_len = int(full_len * self.train_val_test_split[0])
        val_len = int(full_len * self.train_val_test_split[1])
        test_len = full_len - train_len - val_len

        # Split the dataset
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(full_dataset, [train_len, val_len, test_len])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
