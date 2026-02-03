import pytest
import torch

import torch_geometric.typing
from torch_geometric.nn import PANConv
from torch_geometric.testing import withPackage
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

@withPackage('torch_sparse')  # TODO `PANConv` returns a `SparseTensor`.
def test_pan_conv(device):
    x = torch.randn(4, 16)
    edge_index = torch.tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])
    adj1 = to_torch_csc_tensor(edge_index, size=(4, 4))

    adj2 = SparseTensor.from_edge_index(edge_index, sparse_sizes=(4, 4))

    conv = PANConv(16, 32, filter_size=2)
    assert str(conv) == 'PANConv(16, 32, filter_size=2)'
    out1, M1 = conv(x, edge_index)
    assert out1.size() == (4, 32)

    out2, M2 = conv(x, adj1.t())
    assert torch.allclose(out1, out2)
    assert torch.allclose(M1.to_dense(), M2.to_dense())

    if torch_geometric.typing.WITH_TORCH_SPARSE:
        out3, M3 = conv(x, adj2.t())
        assert torch.allclose(out1, out3)
        assert torch.allclose(M1.to_dense(), M3.to_dense())
