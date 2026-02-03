import pytest
import torch

from torch_geometric.nn import FusedGATConv
from torch_geometric.testing import onlyCUDA, withPackage



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

def test_to_graph_format() -> None:
    edge_index = torch.tensor([[1, 0, 2, 3], [0, 0, 1, 1]])

    csr, csc, perm = FusedGATConv.to_graph_format(edge_index, size=(4, 4))

    assert csr[0].dtype == torch.int
    assert torch.equal(csr[0], torch.tensor([0, 1, 2, 3, 4], dtype=torch.int))
    assert csr[1].dtype == torch.int
    assert torch.equal(csr[1], torch.tensor([0, 0, 1, 1], dtype=torch.int))
    assert csc[0].dtype == torch.int
    assert torch.equal(csc[0], torch.tensor([0, 1, 2, 3], dtype=torch.int))
    assert csc[1].dtype == torch.int
    assert torch.equal(csc[1], torch.tensor([0, 2, 4, 4, 4], dtype=torch.int))
    assert perm.dtype == torch.int
    assert torch.equal(perm, torch.tensor([0, 1, 2, 3], dtype=torch.int))


@onlyCUDA
@withPackage('dgNN')
def test_fused_gat_conv() -> None:
    device = torch.device('cuda')

    x = torch.randn(4, 8, device=device)
    edge_index = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]], device=device)

    csr, csc, perm = FusedGATConv.to_graph_format(edge_index, size=(4, 4))

    conv = FusedGATConv(8, 32, heads=2, add_self_loops=False).to(device)
    assert str(conv) == 'FusedGATConv(8, 32, heads=2)'

    out = conv(x, csr, csc, perm)
    assert out.size() == (4, 64)
