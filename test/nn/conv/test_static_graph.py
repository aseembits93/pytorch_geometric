import pytest
import torch

from torch_geometric.data import Batch, Data
from torch_geometric.nn import ChebConv, GCNConv, MessagePassing



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

class MyConv(MessagePassing):
    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)


def test_static_graph(device):
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    x1, x2 = torch.randn(3, 8), torch.randn(3, 8)

    data1 = Data(edge_index=edge_index, x=x1)
    data2 = Data(edge_index=edge_index, x=x2)
    batch = Batch.from_data_list([data1, data2])

    x = torch.stack([x1, x2], dim=0)
    for conv in [MyConv(), GCNConv(8, 16), ChebConv(8, 16, K=2)]:
        out1 = conv(batch.x, batch.edge_index)
        assert out1.size(0) == 6
        conv.node_dim = 1
        out2 = conv(x, edge_index)
        assert out2.size()[:2] == (2, 3)
        assert torch.allclose(out1, out2.view(-1, out2.size(-1)))
