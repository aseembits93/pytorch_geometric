from typing import Optional, Tuple

import pytest
import torch
from torch import Tensor

import torch_geometric.typing
from torch_geometric.nn import GATConv
from torch_geometric.testing import is_full_test, withDevice
from torch_geometric.typing import Adj, Size, SparseTensor
from torch_geometric.utils import to_torch_csc_tensor

# Fixed input shapes for all tests
NUM_NODES = 100
IN_CHANNELS = 64
OUT_CHANNELS = 64
NUM_EDGES = 500
NUM_HEADS = 4

pytestmark = pytest.mark.cuda


@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device('cuda')


@pytest.mark.parametrize('residual', [False, True])
def test_gat_conv(residual, device):
    x1 = torch.randn(NUM_NODES, IN_CHANNELS, dtype=torch.float32, device=device)
    x2 = torch.randn(NUM_NODES // 2, IN_CHANNELS * 2, dtype=torch.float32, device=device)
    edge_index = torch.randint(0, NUM_NODES, (2, NUM_EDGES), dtype=torch.long, device=device)
    adj1 = to_torch_csc_tensor(edge_index, size=(NUM_NODES, NUM_NODES))

    conv = GATConv(IN_CHANNELS, OUT_CHANNELS, heads=NUM_HEADS, residual=residual).to(device)
    assert str(conv) == f'GATConv({IN_CHANNELS}, {OUT_CHANNELS}, heads={NUM_HEADS})'
    out = conv(x1, edge_index)
    assert out.size() == (NUM_NODES, OUT_CHANNELS * NUM_HEADS)
    assert torch.allclose(conv(x1, edge_index, size=(NUM_NODES, NUM_NODES)), out)
    assert torch.allclose(conv(x1, adj1.t()), out)

    if torch_geometric.typing.WITH_TORCH_SPARSE:
        adj2 = SparseTensor.from_edge_index(edge_index, sparse_sizes=(NUM_NODES, NUM_NODES))
        assert torch.allclose(conv(x1, adj2.t()), out)

    if is_full_test():

        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = conv

            def forward(
                self,
                x: Tensor,
                edge_index: Adj,
                size: Size = None,
            ) -> Tensor:
                return self.conv(x, edge_index, size=size)

        jit = torch.jit.script(MyModule())
        assert torch.allclose(jit(x1, edge_index), out)
        assert torch.allclose(jit(x1, edge_index, size=(4, 4)), out)

        if torch_geometric.typing.WITH_TORCH_SPARSE:
            assert torch.allclose(jit(x1, adj2.t()), out)

    # Test `return_attention_weights`.
    result = conv(x1, edge_index, return_attention_weights=True)
    assert torch.allclose(result[0], out)
    assert result[1][0].size() == (2, 7)
    assert result[1][1].size() == (7, 2)
    assert result[1][1].min() >= 0 and result[1][1].max() <= 1

    result = conv(x1, adj1.t(), return_attention_weights=True)
    assert torch.allclose(result[0], out)
    assert result[1][0].size() == torch.Size([NUM_NODES, NUM_NODES, NUM_HEADS])

    if torch_geometric.typing.WITH_TORCH_SPARSE:
        result = conv(x1, adj2.t(), return_attention_weights=True)
        assert torch.allclose(result[0], out)
        assert result[1].sizes() == [NUM_NODES, NUM_NODES, NUM_HEADS]

    if is_full_test():

        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = conv

            def forward(
                self,
                x: Tensor,
                edge_index: Tensor,
            ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
                return self.conv(x, edge_index, return_attention_weights=True)

        jit = torch.jit.script(MyModule())
        result = jit(x1, edge_index)
        assert torch.allclose(result[0], out)
        assert result[1][0].size() == (2, 7)
        assert result[1][1].size() == (7, 2)
        assert result[1][1].min() >= 0 and result[1][1].max() <= 1

        if torch_geometric.typing.WITH_TORCH_SPARSE:

            class MyModule(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.conv = conv

                def forward(
                    self,
                    x: Tensor,
                    edge_index: SparseTensor,
                ) -> Tuple[Tensor, SparseTensor]:
                    return self.conv(x, edge_index,
                                     return_attention_weights=True)

            jit = torch.jit.script(MyModule())
            result = jit(x1, adj2.t())
            assert torch.allclose(result[0], out)
            assert result[1].sizes() == [NUM_NODES, NUM_NODES, NUM_HEADS]

    # Test bipartite message passing:
    bipartite_edge_index = torch.randint(0, NUM_NODES // 2, (2, NUM_EDGES // 2), dtype=torch.long, device=device)
    bipartite_edge_index[0] = torch.randint(0, NUM_NODES, (NUM_EDGES // 2,), device=device)
    adj1_bip = to_torch_csc_tensor(bipartite_edge_index, size=(NUM_NODES, NUM_NODES // 2))

    conv = GATConv((IN_CHANNELS, IN_CHANNELS * 2), OUT_CHANNELS, heads=NUM_HEADS, residual=residual).to(device)
    assert str(conv) == f'GATConv(({IN_CHANNELS}, {IN_CHANNELS * 2}), {OUT_CHANNELS}, heads={NUM_HEADS})'

    out1 = conv((x1, x2), bipartite_edge_index)
    assert out1.size() == (NUM_NODES // 2, OUT_CHANNELS * NUM_HEADS)
    assert torch.allclose(conv((x1, x2), bipartite_edge_index, size=(NUM_NODES, NUM_NODES // 2)), out1)
    assert torch.allclose(conv((x1, x2), adj1_bip.t()), out1)

    out2 = conv((x1, None), bipartite_edge_index, size=(NUM_NODES, NUM_NODES // 2))
    assert out2.size() == (NUM_NODES // 2, OUT_CHANNELS * NUM_HEADS)
    assert torch.allclose(conv((x1, None), adj1_bip.t()), out2)

    if torch_geometric.typing.WITH_TORCH_SPARSE:
        adj2_bip = SparseTensor.from_edge_index(bipartite_edge_index, sparse_sizes=(NUM_NODES, NUM_NODES // 2))
        assert torch.allclose(conv((x1, x2), adj2_bip.t()), out1)
        assert torch.allclose(conv((x1, None), adj2_bip.t()), out2)

    if is_full_test():

        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = conv

            def forward(
                self,
                x: Tuple[Tensor, Optional[Tensor]],
                edge_index: Adj,
                size: Size = None,
            ) -> Tensor:
                return self.conv(x, edge_index, size=size)

        jit = torch.jit.script(MyModule())
        assert torch.allclose(jit((x1, x2), edge_index), out1)
        assert torch.allclose(jit((x1, x2), edge_index, size=(4, 2)), out1)
        assert torch.allclose(jit((x1, None), edge_index, size=(4, 2)), out2)

        if torch_geometric.typing.WITH_TORCH_SPARSE:
            assert torch.allclose(jit((x1, x2), adj2_bip.t()), out1)
            assert torch.allclose(jit((x1, None), adj2_bip.t()), out2)


def test_gat_conv_with_edge_attr(device):
    x = torch.randn(NUM_NODES, IN_CHANNELS, dtype=torch.float32, device=device)
    edge_index = torch.randint(0, NUM_NODES, (2, NUM_EDGES), dtype=torch.long, device=device)
    edge_weight = torch.randn(NUM_EDGES, dtype=torch.float32, device=device)
    edge_attr = torch.randn(NUM_EDGES, 8, dtype=torch.float32, device=device)

    conv = GATConv(IN_CHANNELS, OUT_CHANNELS, heads=NUM_HEADS, edge_dim=1, fill_value=0.5).to(device)
    out = conv(x, edge_index, edge_weight)
    assert out.size() == (NUM_NODES, OUT_CHANNELS * NUM_HEADS)
    if torch_geometric.typing.WITH_TORCH_SPARSE:
        adj1 = SparseTensor.from_edge_index(edge_index, edge_weight, (NUM_NODES, NUM_NODES))
        with pytest.raises(NotImplementedError):
            assert torch.allclose(conv(x, adj1.t()), out)

    conv = GATConv(IN_CHANNELS, OUT_CHANNELS, heads=NUM_HEADS, edge_dim=1, fill_value='mean').to(device)
    out = conv(x, edge_index, edge_weight)
    assert out.size() == (NUM_NODES, OUT_CHANNELS * NUM_HEADS)
    if torch_geometric.typing.WITH_TORCH_SPARSE:
        with pytest.raises(NotImplementedError):
            assert torch.allclose(conv(x, adj1.t()), out)

    conv = GATConv(IN_CHANNELS, OUT_CHANNELS, heads=NUM_HEADS, edge_dim=8, fill_value=0.5).to(device)
    out = conv(x, edge_index, edge_attr)
    assert out.size() == (NUM_NODES, OUT_CHANNELS * NUM_HEADS)
    if torch_geometric.typing.WITH_TORCH_SPARSE:
        adj2 = SparseTensor.from_edge_index(edge_index, edge_attr, (NUM_NODES, NUM_NODES))
        with pytest.raises(NotImplementedError):
            assert torch.allclose(conv(x, adj2.t()), out)

    conv = GATConv(IN_CHANNELS, OUT_CHANNELS, heads=NUM_HEADS, edge_dim=8, fill_value='mean').to(device)
    out = conv(x, edge_index, edge_attr)
    assert out.size() == (NUM_NODES, OUT_CHANNELS * NUM_HEADS)
    if torch_geometric.typing.WITH_TORCH_SPARSE:
        with pytest.raises(NotImplementedError):
            assert torch.allclose(conv(x, adj2.t()), out)


def test_gat_conv_empty_edge_index(device):
    x = torch.randn(0, IN_CHANNELS, dtype=torch.float32, device=device)
    edge_index = torch.empty(2, 0, dtype=torch.long, device=device)

    conv = GATConv(IN_CHANNELS, OUT_CHANNELS, heads=NUM_HEADS).to(device)
    out = conv(x, edge_index)
    assert out.size() == (0, OUT_CHANNELS * NUM_HEADS)
