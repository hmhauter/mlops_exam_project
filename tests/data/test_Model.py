import pytest
import torch
from src.models.model import CustomModel


@pytest.fixture
def model():
    num_classes = 10
    model_name = "resnet18"
    lr = 0.001
    return CustomModel(num_classes, model_name, lr)


def test_model_forward_pass(model):
    batch_size = 16
    input_shape = (3, 224, 224)
    inp = torch.randn(batch_size, *input_shape)
    output = model(inp)
    assert output.shape == (batch_size, model.num_classes)


def test_model_training_step(model):
    batch_size = 8
    input_shape = (3, 224, 224)
    ims = torch.randn(batch_size, *input_shape)
    gts = torch.randint(0, model.num_classes, (batch_size,))
    loss = model.training_step((ims, gts), 0)
    assert isinstance(loss, torch.Tensor)
    assert loss.item() >= 0.0


def test_model_validation_step(model):
    batch_size = 8
    input_shape = (3, 224, 224)
    ims = torch.randn(batch_size, *input_shape)
    gts = torch.randint(0, model.num_classes, (batch_size,))
    loss = model.validation_step((ims, gts), 0)
    assert isinstance(loss, torch.Tensor)
    assert loss.item() >= 0.0


def test_model_configure_optimizers(model):
    optimizer = model.configure_optimizers()
    assert isinstance(optimizer, torch.optim.Optimizer)


def test_model_hyperparameters(model):
    assert model.num_classes == 10
    assert model.model_name == "resnet18"
    assert model.lr == 0.001
