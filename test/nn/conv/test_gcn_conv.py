import pytest
import torch

import torch_geometric.typing
from torch_geometric.nn import GCNConv
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.testing import is_full_test
from torch_geometric.typing import WITH_PT21, SparseTensor
from torch_geometric.utils import to_torch_coo_tensor, to_torch_csc_tensor

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


def test_gcn_conv(device):
    x = torch.randn(NUM_NODES, IN_CHANNELS, dtype=torch.float32, device=device)
    edge_index = torch.randint(0, NUM_NODES, (2, NUM_EDGES), dtype=torch.long, device=device)
    value = torch.rand(NUM_EDGES, dtype=torch.float32, device=device)
    adj1 = to_torch_csc_tensor(edge_index, size=(NUM_NODES, NUM_NODES))
    adj2 = to_torch_csc_tensor(edge_index, value, size=(NUM_NODES, NUM_NODES))

    conv = GCNConv(IN_CHANNELS, OUT_CHANNELS).to(device)
    assert str(conv) == f'GCNConv({IN_CHANNELS}, {OUT_CHANNELS})'

    out1 = conv(x, edge_index)
    assert out1.size() == (NUM_NODES, OUT_CHANNELS)
    assert torch.allclose(conv(x, adj1.t()), out1)

    out2 = conv(x, edge_index, value)
    assert out2.size() == (NUM_NODES, OUT_CHANNELS)
    assert torch.allclose(conv(x, adj2.t()), out2)

    if torch_geometric.typing.WITH_TORCH_SPARSE:
        adj3 = SparseTensor.from_edge_index(edge_index, sparse_sizes=(NUM_NODES, NUM_NODES))
        adj4 = SparseTensor.from_edge_index(edge_index, value, (NUM_NODES, NUM_NODES))
        assert torch.allclose(conv(x, adj3.t()), out1)
        assert torch.allclose(conv(x, adj4.t()), out2)

    if is_full_test():
        jit = torch.jit.script(conv)
        assert torch.allclose(jit(x, edge_index), out1)
        assert torch.allclose(jit(x, edge_index, value), out2)

        if torch_geometric.typing.WITH_TORCH_SPARSE:
            assert torch.allclose(jit(x, adj3.t()), out1)
            assert torch.allclose(jit(x, adj4.t()), out2)

    conv.cached = True
    conv(x, edge_index)
    assert conv._cached_edge_index is not None
    assert torch.allclose(conv(x, edge_index), out1)
    assert torch.allclose(conv(x, adj1.t()), out1)

    if torch_geometric.typing.WITH_TORCH_SPARSE:
        conv(x, adj3.t())
        assert conv._cached_adj_t is not None
        assert torch.allclose(conv(x, adj3.t()), out1)


def test_gcn_conv_with_decomposed_layers(device):
    x = torch.randn(NUM_NODES, IN_CHANNELS, dtype=torch.float32, device=device)
    edge_index = torch.randint(0, NUM_NODES, (2, NUM_EDGES), dtype=torch.long, device=device)

    def hook(module, inputs):
        assert inputs[0]['x_j'].size(-1) == OUT_CHANNELS // module.decomposed_layers

    conv = GCNConv(IN_CHANNELS, OUT_CHANNELS).to(device)
    conv.register_message_forward_pre_hook(hook)
    out1 = conv(x, edge_index)

    conv.decomposed_layers = 2
    assert conv.propagate.__module__.endswith('message_passing')
    out2 = conv(x, edge_index)
    assert torch.allclose(out1, out2)

    # TorchScript should still work since it relies on class methods
    # (but without decomposition).
    torch.jit.script(conv)

    conv.decomposed_layers = 1
    assert conv.propagate.__module__.endswith('GCNConv_propagate')


def test_gcn_conv_with_sparse_input_feature(device):
    indices = torch.tensor([[0, 0, 1, 1], [0, 1, 0, 1]], device=device)
    values = torch.tensor([1., 1., 1., 1.], dtype=torch.float32, device=device)
    x = torch.sparse_coo_tensor(
        indices=indices,
        values=values,
        size=torch.Size([NUM_NODES, IN_CHANNELS]),
        device=device
    )
    edge_index = torch.randint(0, NUM_NODES, (2, NUM_EDGES), dtype=torch.long, device=device)

    conv = GCNConv(IN_CHANNELS, OUT_CHANNELS).to(device)
    assert conv(x, edge_index).size() == (NUM_NODES, OUT_CHANNELS)


def test_static_gcn_conv(device):
    batch_size = 8
    x = torch.randn(batch_size, NUM_NODES, IN_CHANNELS, dtype=torch.float32, device=device)
    edge_index = torch.randint(0, NUM_NODES, (2, NUM_EDGES), dtype=torch.long, device=device)

    conv = GCNConv(IN_CHANNELS, OUT_CHANNELS).to(device)
    out = conv(x, edge_index)
    assert out.size() == (batch_size, NUM_NODES, OUT_CHANNELS)


def test_gcn_conv_error(device):
    with pytest.raises(ValueError, match="does not support adding self-loops"):
        GCNConv(IN_CHANNELS, OUT_CHANNELS, normalize=False, add_self_loops=True)


def test_gcn_conv_flow(device):
    x = torch.randn(NUM_NODES, IN_CHANNELS, dtype=torch.float32, device=device)
    edge_index = torch.randint(0, NUM_NODES, (2, NUM_EDGES), dtype=torch.long, device=device)

    conv = GCNConv(IN_CHANNELS, OUT_CHANNELS, flow="source_to_target").to(device)
    out1 = conv(x, edge_index)
    conv.flow = "target_to_source"
    out2 = conv(x, edge_index.flip(0))
    assert torch.allclose(out1, out2)


@pytest.mark.parametrize('requires_grad', [False, True])
@pytest.mark.parametrize('layout', [torch.sparse_coo, torch.sparse_csr])
def test_gcn_norm_gradient(requires_grad, layout, device):
    edge_index = torch.randint(0, NUM_NODES, (2, NUM_EDGES), dtype=torch.long, device=device)
    edge_weight = torch.ones(NUM_EDGES, dtype=torch.float32, requires_grad=requires_grad, device=device)
    adj = to_torch_coo_tensor(edge_index, edge_weight)
    if layout == torch.sparse_csr:
        adj = adj.to_sparse_csr()

    # TODO Sparse CSR tensor doesn't inherit `requires_grad` for PyTorch < 2.1.
    if layout == torch.sparse_csr and not WITH_PT21:
        assert not gcn_norm(adj)[0].requires_grad
    else:
        assert adj.requires_grad == gcn_norm(adj)[0].requires_grad
