import pytest
import torch

import torch_geometric.typing
from torch_geometric.nn import GPSConv, SAGEConv
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

@pytest.mark.parametrize('attn_type', ['multihead', 'performer'])
@pytest.mark.parametrize('norm', [None, 'batch_norm', 'layer_norm'])
def test_gps_conv(norm, attn_type, device):
    x = torch.randn(4, 16)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]])
    batch = torch.tensor([0, 0, 1, 1])
    adj1 = to_torch_csc_tensor(edge_index, size=(4, 4))

    conv = GPSConv(16, conv=SAGEConv(16, 16), heads=4, norm=norm,
                   attn_type=attn_type)
    conv.reset_parameters()
    assert str(conv) == (f'GPSConv(16, conv=SAGEConv(16, 16, aggr=mean), '
                         f'heads=4, attn_type={attn_type})')

    out = conv(x, edge_index)
    assert out.size() == (4, 16)
    assert torch.allclose(conv(x, adj1.t()), out)

    if torch_geometric.typing.WITH_TORCH_SPARSE:
        adj2 = SparseTensor.from_edge_index(edge_index, sparse_sizes=(4, 4))
        assert torch.allclose(conv(x, adj2.t()), out)

    out = conv(x, edge_index, batch)
    assert out.size() == (4, 16)
    assert torch.allclose(conv(x, adj1.t(), batch), out)

    if torch_geometric.typing.WITH_TORCH_SPARSE:
        assert torch.allclose(conv(x, adj2.t(), batch), out)
