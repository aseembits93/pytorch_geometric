import pytest
import torch

import torch_geometric.typing
from torch_geometric.nn import EGConv
from torch_geometric.testing import is_full_test
from torch_geometric.typing import SparseTensor
from torch_geometric.utils import to_torch_csc_tensor



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

def test_eg_conv_with_error(device):
    with pytest.raises(ValueError, match="must be divisible by the number of"):
        EGConv(16, 30, num_heads=8)

    with pytest.raises(ValueError, match="Unsupported aggregator"):
        EGConv(16, 32, aggregators=['xxx'])


@pytest.mark.parametrize('aggregators', [
    ['symnorm'],
    ['sum', 'symnorm', 'std'],
])
@pytest.mark.parametrize('add_self_loops', [True, False])
def test_eg_conv(aggregators, add_self_loops, device):
    x = torch.randn(4, 16)
    edge_index = torch.tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])
    adj1 = to_torch_csc_tensor(edge_index, size=(4, 4))

    conv = EGConv(
        in_channels=16,
        out_channels=32,
        aggregators=aggregators,
        add_self_loops=add_self_loops,
    )
    assert str(conv) == f"EGConv(16, 32, aggregators={aggregators})"
    out = conv(x, edge_index)
    assert out.size() == (4, 32)
    assert torch.allclose(conv(x, adj1.t()), out)

    if torch_geometric.typing.WITH_TORCH_SPARSE:
        adj2 = SparseTensor.from_edge_index(edge_index, sparse_sizes=(4, 4))
        assert torch.allclose(conv(x, adj2.t()), out)

    conv.cached = True
    assert torch.allclose(conv(x, edge_index), out)
    assert conv._cached_edge_index is not None
    assert torch.allclose(conv(x, edge_index), out)
    assert torch.allclose(conv(x, adj1.t()), out)

    if torch_geometric.typing.WITH_TORCH_SPARSE:
        assert torch.allclose(conv(x, adj2.t()), out)
        assert conv._cached_adj_t is not None
        assert torch.allclose(conv(x, adj2.t()), out)

    if is_full_test():
        jit = torch.jit.script(conv)
        assert torch.allclose(jit(x, edge_index), out)

        if torch_geometric.typing.WITH_TORCH_SPARSE:
            assert torch.allclose(jit(x, adj2.t()), out)


def test_eg_conv_with_sparse_input_feature(device):
    x = torch.randn(4, 16).to_sparse_coo()
    edge_index = torch.tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])

    conv = EGConv(16, 32)
    assert conv(x, edge_index).size() == (4, 32)
