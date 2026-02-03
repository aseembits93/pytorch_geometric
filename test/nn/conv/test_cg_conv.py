import pytest
import torch

import torch_geometric.typing
from torch_geometric.nn import CGConv
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

@pytest.mark.parametrize('batch_norm', [False, True])
def test_cg_conv(batch_norm, device):
    x1 = torch.randn(4, 8)
    x2 = torch.randn(2, 16)
    edge_index = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
    adj1 = to_torch_csc_tensor(edge_index, size=(4, 4))

    conv = CGConv(8, batch_norm=batch_norm)
    assert str(conv) == 'CGConv(8, dim=0)'
    out = conv(x1, edge_index)
    assert out.size() == (4, 8)
    assert torch.allclose(conv(x1, adj1.t()), out)
    if torch_geometric.typing.WITH_TORCH_SPARSE:
        adj2 = SparseTensor.from_edge_index(edge_index, sparse_sizes=(4, 4))
        assert torch.allclose(conv(x1, adj2.t()), out)

    if is_full_test():
        jit = torch.jit.script(conv)
        assert torch.allclose(jit(x1, edge_index), out)

        if torch_geometric.typing.WITH_TORCH_SPARSE:
            assert torch.allclose(jit(x1, adj2.t()), out)

    # Test bipartite message passing:
    adj1 = to_torch_csc_tensor(edge_index, size=(4, 2))

    conv = CGConv((8, 16))
    assert str(conv) == 'CGConv((8, 16), dim=0)'
    out = conv((x1, x2), edge_index)
    assert out.size() == (2, 16)
    assert torch.allclose(conv((x1, x2), adj1.t()), out)

    if torch_geometric.typing.WITH_TORCH_SPARSE:
        adj2 = SparseTensor.from_edge_index(edge_index, sparse_sizes=(4, 2))
        assert torch.allclose(conv((x1, x2), adj2.t()), out)

    if is_full_test():
        jit = torch.jit.script(conv)
        assert torch.allclose(jit((x1, x2), edge_index), out)

        if torch_geometric.typing.WITH_TORCH_SPARSE:
            assert torch.allclose(jit((x1, x2), adj2.t()), out)


def test_cg_conv_with_edge_features(device):
    x1 = torch.randn(4, 8)
    x2 = torch.randn(2, 16)
    edge_index = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
    value = torch.rand(edge_index.size(1), 3)

    conv = CGConv(8, dim=3)
    assert str(conv) == 'CGConv(8, dim=3)'
    out = conv(x1, edge_index, value)
    assert out.size() == (4, 8)

    if torch_geometric.typing.WITH_TORCH_SPARSE:
        adj = SparseTensor.from_edge_index(edge_index, value, (4, 4))
        assert torch.allclose(conv(x1, adj.t()), out)

    if is_full_test():
        jit = torch.jit.script(conv)
        assert torch.allclose(jit(x1, edge_index, value), out)

        if torch_geometric.typing.WITH_TORCH_SPARSE:
            assert torch.allclose(jit(x1, adj.t()), out)

    # Test bipartite message passing:
    conv = CGConv((8, 16), dim=3)
    assert str(conv) == 'CGConv((8, 16), dim=3)'
    out = conv((x1, x2), edge_index, value)
    assert out.size() == (2, 16)

    if torch_geometric.typing.WITH_TORCH_SPARSE:
        adj = SparseTensor.from_edge_index(edge_index, value, (4, 2))
        assert torch.allclose(conv((x1, x2), adj.t()), out)

    if is_full_test():
        jit = torch.jit.script(conv)
        assert torch.allclose(jit((x1, x2), edge_index, value), out)

        if torch_geometric.typing.WITH_TORCH_SPARSE:
            assert torch.allclose(jit((x1, x2), adj.t()), out)
