import pytest
from pathlib import Path
from PIL import Image
import torch
from torchvision import transforms
from src.data.DataSet import SportDataset


@pytest.fixture
def dataset():
    csv_file = Path("train.csv")
    data_dir = Path("data/")
    transformations = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return SportDataset(csv_file, data_dir, transformations)


def test_len(dataset):
    assert len(dataset) == 8227


def test_getitem_with_labels(dataset):
    image, label = dataset[0]
    assert isinstance(image, torch.Tensor)
    assert isinstance(label, int)


def test_transformations(dataset):
    image, _ = dataset[0]
    assert image.shape == torch.Size([3, 256, 256])


def test_csv_file_not_found():
    with pytest.raises(FileNotFoundError):
        SportDataset("nonexistent.csv", Path("/path/to/data"), None)


def test_image_file_not_found(dataset):
    with pytest.raises(IndexError):
        dataset[len(dataset)] 
