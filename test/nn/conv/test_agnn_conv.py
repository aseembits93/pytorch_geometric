import pytest
import torch

import torch_geometric.typing
from torch_geometric.nn import AGNNConv
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

@pytest.mark.parametrize('requires_grad', [True, False])
def test_agnn_conv(requires_grad, device):
    x = torch.randn(4, 16)
    edge_index = torch.tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])
    adj1 = to_torch_csc_tensor(edge_index, size=(4, 4))

    conv = AGNNConv(requires_grad=requires_grad)
    assert str(conv) == 'AGNNConv()'
    out = conv(x, edge_index)
    assert out.size() == (4, 16)
    assert torch.allclose(conv(x, adj1.t()), out)

    if torch_geometric.typing.WITH_TORCH_SPARSE:
        adj2 = SparseTensor.from_edge_index(edge_index, sparse_sizes=(4, 4))
        assert torch.allclose(conv(x, adj2.t()), out)

    if is_full_test():
        jit = torch.jit.script(conv)
        assert torch.allclose(jit(x, edge_index), out)

        if torch_geometric.typing.WITH_TORCH_SPARSE:
            assert torch.allclose(jit(x, adj2.t()), out)
