import pytest
from src.data.DataModule import SportDataModule


@pytest.fixture
def datamodule():
    return SportDataModule(
        data_dir="./data", batch_size=32, num_workers=4, transformations=None, train_val_test_split=[0.7, 0.15, 0.15]
    )


def test_setup(datamodule):
    datamodule.setup()
    assert datamodule.train_dataset is not None
    assert datamodule.val_dataset is not None
    assert datamodule.test_dataset is not None


def test_train_dataloader(datamodule):
    datamodule.setup()
    train_dataloader = datamodule.train_dataloader()
    assert len(train_dataloader.dataset) > 0
    assert len(train_dataloader) > 0


def test_val_dataloader(datamodule):
    datamodule.setup()
    val_dataloader = datamodule.val_dataloader()
    assert len(val_dataloader.dataset) > 0
    assert len(val_dataloader) > 0


def test_test_dataloader(datamodule):
    datamodule.setup()
    test_dataloader = datamodule.test_dataloader()
    assert len(test_dataloader.dataset) > 0
    assert len(test_dataloader) > 0
