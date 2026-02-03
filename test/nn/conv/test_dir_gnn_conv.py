import pytest
import torch

from torch_geometric.nn import DirGNNConv, SAGEConv



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

def test_dir_gnn_conv(device):
    x = torch.randn(4, 16)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])

    conv = DirGNNConv(SAGEConv(16, 32))
    assert str(conv) == 'DirGNNConv(SAGEConv(16, 32, aggr=mean), alpha=0.5)'

    out = conv(x, edge_index)
    assert out.size() == (4, 32)


def test_static_dir_gnn_conv(device):
    x = torch.randn(3, 4, 16)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])

    conv = DirGNNConv(SAGEConv(16, 32))

    out = conv(x, edge_index)
    assert out.size() == (3, 4, 32)
