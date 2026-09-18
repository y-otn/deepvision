import torch

from deepvision.layers import FusedMBConv
from deepvision.layers import MBConv
from deepvision.layers import TransformerEncoder


def _blocks():
    return [
        MBConv(
            input_filters=32,
            output_filters=32,
            backend="pytorch",
            expand_ratio=4,
            dropout=0.5,
        ),
        FusedMBConv(
            input_filters=32,
            output_filters=32,
            backend="pytorch",
            expand_ratio=4,
            dropout=0.5,
        ),
    ]


def test_conv_blocks_are_deterministic_in_eval_mode():
    inputs = torch.rand(2, 32, 16, 16)
    for block in _blocks():
        block.eval()
        with torch.no_grad():
            assert torch.equal(block(inputs), block(inputs))


def test_conv_blocks_are_stochastic_in_train_mode():
    inputs = torch.rand(2, 32, 16, 16)
    for block in _blocks():
        block.train()
        with torch.no_grad():
            assert not torch.equal(block(inputs), block(inputs))


def test_transformer_encoder_is_deterministic_in_eval_mode():
    encoder = TransformerEncoder(
        project_dim=32, num_heads=2, mlp_dim=64, mlp_dropout=0.5, backend="pytorch"
    )
    encoder.eval()
    inputs = torch.rand(2, 8, 32)
    with torch.no_grad():
        assert torch.equal(encoder(inputs), encoder(inputs))
