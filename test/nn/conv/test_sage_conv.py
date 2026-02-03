import pytest
import torch

import torch_geometric.typing
from torch_geometric.nn import MLPAggregation, SAGEConv
from torch_geometric.testing import (
    assert_module,
    is_full_test,
    onlyLinux,
    withDevice,
    withPackage,
)
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


@pytest.mark.parametrize('project', [False, True])
@pytest.mark.parametrize('aggr', ['mean', 'sum'])
def test_sage_conv(project, aggr, device):
    x = torch.randn(NUM_NODES, IN_CHANNELS, dtype=torch.float32, device=device)
    edge_index = torch.randint(0, NUM_NODES, (2, NUM_EDGES), dtype=torch.long, device=device)

    conv = SAGEConv(IN_CHANNELS, OUT_CHANNELS, project=project, aggr=aggr).to(device)
    assert str(conv) == f'SAGEConv({IN_CHANNELS}, {OUT_CHANNELS}, aggr={aggr})'

    out = assert_module(conv, x, edge_index, expected_size=(NUM_NODES, OUT_CHANNELS))

    if is_full_test():
        jit = torch.jit.script(conv)
        assert torch.allclose(jit(x, edge_index), out)
        assert torch.allclose(jit(x, edge_index, size=(NUM_NODES, NUM_NODES)), out)

        if torch_geometric.typing.WITH_TORCH_SPARSE:
            adj = SparseTensor.from_edge_index(edge_index, sparse_sizes=(NUM_NODES, NUM_NODES))
            assert torch.allclose(jit(x, adj.t()), out)

    # Test bipartite message passing:
    x1 = torch.randn(NUM_NODES, IN_CHANNELS, dtype=torch.float32, device=device)
    x2 = torch.randn(NUM_NODES // 2, IN_CHANNELS * 2, dtype=torch.float32, device=device)
    bipartite_edge_index = torch.randint(0, NUM_NODES // 2, (2, NUM_EDGES // 2), dtype=torch.long, device=device)
    bipartite_edge_index[0] = torch.randint(0, NUM_NODES, (NUM_EDGES // 2,), device=device)

    conv = SAGEConv((IN_CHANNELS, IN_CHANNELS * 2), OUT_CHANNELS, project=project, aggr=aggr).to(device)
    assert str(conv) == f'SAGEConv(({IN_CHANNELS}, {IN_CHANNELS * 2}), {OUT_CHANNELS}, aggr={aggr})'

    out1 = assert_module(conv, (x1, x2), bipartite_edge_index, expected_size=(NUM_NODES // 2, OUT_CHANNELS))
    out2 = assert_module(conv, (x1, None), bipartite_edge_index, size=(NUM_NODES, NUM_NODES // 2),
                         expected_size=(NUM_NODES // 2, OUT_CHANNELS))

    if is_full_test():
        jit = torch.jit.script(conv)
        assert torch.allclose(jit((x1, x2), bipartite_edge_index), out1)
        assert torch.allclose(jit((x1, x2), bipartite_edge_index, size=(NUM_NODES, NUM_NODES // 2)), out1)
        assert torch.allclose(jit((x1, None), bipartite_edge_index, size=(NUM_NODES, NUM_NODES // 2)), out2)

        if torch_geometric.typing.WITH_TORCH_SPARSE:
            adj = SparseTensor.from_edge_index(bipartite_edge_index, sparse_sizes=(NUM_NODES, NUM_NODES // 2))
            assert torch.allclose(jit((x1, x2), adj.t()), out1)
            assert torch.allclose(jit((x1, None), adj.t()), out2)


@pytest.mark.parametrize('project', [False, True])
def test_lazy_sage_conv(project, device):
    x = torch.randn(NUM_NODES, IN_CHANNELS, dtype=torch.float32, device=device)
    edge_index = torch.randint(0, NUM_NODES, (2, NUM_EDGES), dtype=torch.long, device=device)

    if project:
        with pytest.raises(ValueError, match="does not support lazy"):
            SAGEConv(-1, OUT_CHANNELS, project=project)
    else:
        conv = SAGEConv(-1, OUT_CHANNELS, project=project).to(device)
        assert str(conv) == f'SAGEConv(-1, {OUT_CHANNELS}, aggr=mean)'

        out = conv(x, edge_index)
        assert out.size() == (NUM_NODES, OUT_CHANNELS)


def test_lstm_aggr_sage_conv(device):
    x = torch.randn(NUM_NODES, IN_CHANNELS, dtype=torch.float32, device=device)
    edge_index = torch.randint(0, NUM_NODES, (2, NUM_EDGES), dtype=torch.long, device=device)
    edge_index = edge_index[:, edge_index[1].argsort()]  # Sort by target nodes

    conv = SAGEConv(IN_CHANNELS, OUT_CHANNELS, aggr='lstm').to(device)
    assert str(conv) == f'SAGEConv({IN_CHANNELS}, {OUT_CHANNELS}, aggr=lstm)'

    assert_module(conv, x, edge_index, expected_size=(NUM_NODES, OUT_CHANNELS),
                  test_edge_permutation=False)


def test_mlp_sage_conv(device):
    x = torch.randn(NUM_NODES, IN_CHANNELS, dtype=torch.float32, device=device)
    edge_index = torch.randint(0, NUM_NODES, (2, NUM_EDGES), dtype=torch.long, device=device)

    conv = SAGEConv(
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS,
        aggr=MLPAggregation(
            in_channels=IN_CHANNELS,
            out_channels=IN_CHANNELS,
            max_num_elements=10,
            num_layers=1,
        ),
    ).to(device)

    out = conv(x, edge_index)
    assert out.size() == (NUM_NODES, OUT_CHANNELS)


@pytest.mark.parametrize('aggr_kwargs', [
    dict(mode='cat'),
    dict(mode='proj', mode_kwargs=dict(in_channels=IN_CHANNELS, out_channels=IN_CHANNELS * 2)),
    dict(mode='attn', mode_kwargs=dict(in_channels=IN_CHANNELS, out_channels=IN_CHANNELS * 2,
                                       num_heads=4)),
    dict(mode='sum'),
])
def test_multi_aggr_sage_conv(aggr_kwargs, device):
    x = torch.randn(NUM_NODES, IN_CHANNELS, dtype=torch.float32, device=device)
    edge_index = torch.randint(0, NUM_NODES, (2, NUM_EDGES), dtype=torch.long, device=device)

    aggr_kwargs['aggrs_kwargs'] = [{}, {}, {}, dict(learn=True, t=1)]
    conv = SAGEConv(IN_CHANNELS, OUT_CHANNELS, aggr=['mean', 'max', 'sum', 'softmax'],
                    aggr_kwargs=aggr_kwargs).to(device)

    assert_module(conv, x, edge_index, expected_size=(NUM_NODES, OUT_CHANNELS))


@onlyLinux
@withPackage('torch>=2.1.0')
def test_compile_multi_aggr_sage_conv(device):
    import torch._dynamo as dynamo

    x = torch.randn(NUM_NODES, IN_CHANNELS, dtype=torch.float32, device=device)
    edge_index = torch.randint(0, NUM_NODES, (2, NUM_EDGES), dtype=torch.long, device=device)

    conv = SAGEConv(
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS,
        aggr=['mean', 'sum', 'min', 'max', 'std'],
    ).to(device)

    explanation = dynamo.explain(conv)(x, edge_index)
    assert explanation.graph_break_count == 0

    compiled_conv = torch.compile(conv)

    expected = conv(x, edge_index)
    out = compiled_conv(x, edge_index)
    assert torch.allclose(out, expected)
