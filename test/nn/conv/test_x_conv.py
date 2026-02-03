import pytest
import torch

from torch_geometric.nn import XConv
from torch_geometric.testing import is_full_test, withPackage



# Fixed input shapes for all tests
NUM_NODES = 100
IN_CHANNELS = 64
OUT_CHANNELS = 64
NUM_EDGES = 500

pytestmark = pytest.mark.cuda


@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device('cuda')

@withPackage('torch_cluster')
def test_x_conv(device):
    x = torch.randn(8, 16)
    pos = torch.rand(8, 3)
    batch = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])

    conv = XConv(16, 32, dim=3, kernel_size=2, dilation=2)
    assert str(conv) == 'XConv(16, 32)'

    torch.manual_seed(12345)
    out1 = conv(x, pos)
    assert out1.size() == (8, 32)

    torch.manual_seed(12345)
    out2 = conv(x, pos, batch)
    assert out2.size() == (8, 32)

    if is_full_test():
        jit = torch.jit.script(conv)

        torch.manual_seed(12345)
        assert torch.allclose(jit(x, pos), out1)

        torch.manual_seed(12345)
        assert torch.allclose(jit(x, pos, batch), out2)
