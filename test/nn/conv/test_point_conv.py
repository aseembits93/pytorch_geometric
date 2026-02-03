import pytest
import torch
from torch.nn import Linear as Lin
from torch.nn import ReLU
from torch.nn import Sequential as Seq

import torch_geometric.typing
from torch_geometric.nn import PointNetConv
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

def test_point_net_conv(device):
    x1 = torch.randn(4, 16)
    pos1 = torch.randn(4, 3)
    pos2 = torch.randn(2, 3)
    edge_index = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
    adj1 = to_torch_csc_tensor(edge_index, size=(4, 4))

    local_nn = Seq(Lin(16 + 3, 32), ReLU(), Lin(32, 32))
    global_nn = Seq(Lin(32, 32))
    conv = PointNetConv(local_nn, global_nn)
    assert str(conv) == (
        'PointNetConv(local_nn=Sequential(\n'
        '  (0): Linear(in_features=19, out_features=32, bias=True)\n'
        '  (1): ReLU()\n'
        '  (2): Linear(in_features=32, out_features=32, bias=True)\n'
        '), global_nn=Sequential(\n'
        '  (0): Linear(in_features=32, out_features=32, bias=True)\n'
        '))')
    out = conv(x1, pos1, edge_index)
    assert out.size() == (4, 32)
    assert torch.allclose(conv(x1, pos1, adj1.t()), out)

    if torch_geometric.typing.WITH_TORCH_SPARSE:
        adj2 = SparseTensor.from_edge_index(edge_index, sparse_sizes=(4, 4))
        assert torch.allclose(conv(x1, pos1, adj2.t()), out)

    if is_full_test():
        jit = torch.jit.script(conv)
        assert torch.allclose(jit(x1, pos1, edge_index), out)

        if torch_geometric.typing.WITH_TORCH_SPARSE:
            assert torch.allclose(jit(x1, pos1, adj2.t()), out)

    # Test bipartite message passing:
    adj1 = to_torch_csc_tensor(edge_index, size=(4, 2))

    out = conv(x1, (pos1, pos2), edge_index)
    assert out.size() == (2, 32)
    assert torch.allclose(conv((x1, None), (pos1, pos2), edge_index), out)
    assert torch.allclose(conv(x1, (pos1, pos2), adj1.t()), out)
    assert torch.allclose(conv((x1, None), (pos1, pos2), adj1.t()), out)

    if torch_geometric.typing.WITH_TORCH_SPARSE:
        adj2 = SparseTensor.from_edge_index(edge_index, sparse_sizes=(4, 2))
        assert torch.allclose(conv(x1, (pos1, pos2), adj2.t()), out)
        assert torch.allclose(conv((x1, None), (pos1, pos2), adj2.t()), out)

    if is_full_test():
        assert torch.allclose(jit((x1, None), (pos1, pos2), edge_index), out)

        if torch_geometric.typing.WITH_TORCH_SPARSE:
            assert torch.allclose(jit((x1, None), (pos1, pos2), adj2.t()), out)
