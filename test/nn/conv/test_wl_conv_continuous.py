import pytest
import torch

import torch_geometric.typing
from torch_geometric.nn import WLConvContinuous
from torch_geometric.testing import is_full_test
from torch_geometric.typing import SparseTensor



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

def test_wl_conv(device):
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    x = torch.tensor([[-1], [0], [1]], dtype=torch.float)

    conv = WLConvContinuous()
    assert str(conv) == 'WLConvContinuous()'

    out = conv(x, edge_index)
    assert out.tolist() == [[-0.5], [0.0], [0.5]]

    if torch_geometric.typing.WITH_TORCH_SPARSE:
        adj = SparseTensor.from_edge_index(edge_index, sparse_sizes=(3, 3))
        assert torch.allclose(conv(x, adj.t()), out)

    if is_full_test():
        jit = torch.jit.script(conv)
        assert torch.allclose(jit(x, edge_index), out)
        if torch_geometric.typing.WITH_TORCH_SPARSE:
            assert torch.allclose(jit(x, adj.t()), out)

    # Test bipartite message passing:
    x1 = torch.randn(4, 8)
    x2 = torch.randn(2, 8)
    edge_index = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
    edge_weight = torch.randn(edge_index.size(1))

    out1 = conv((x1, None), edge_index, edge_weight, size=(4, 2))
    assert out1.size() == (2, 8)

    out2 = conv((x1, x2), edge_index, edge_weight)
    assert out2.size() == (2, 8)

    if torch_geometric.typing.WITH_TORCH_SPARSE:
        adj = SparseTensor.from_edge_index(edge_index, edge_weight, (4, 2))
        assert torch.allclose(conv((x1, None), adj.t()), out1)
        assert torch.allclose(conv((x1, x2), adj.t()), out2)

    if is_full_test():
        assert torch.allclose(
            jit((x1, None), edge_index, edge_weight, size=(4, 2)), out1)
        assert torch.allclose(jit((x1, x2), edge_index, edge_weight), out2)

        if torch_geometric.typing.WITH_TORCH_SPARSE:
            assert torch.allclose(jit((x1, None), adj.t()), out1)
            assert torch.allclose(jit((x1, x2), adj.t()), out2)
