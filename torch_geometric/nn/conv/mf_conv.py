from typing import Tuple, Union

import torch
from torch import Tensor
from torch.nn import ModuleList

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptPairTensor, Size, SparseTensor
from torch_geometric.utils import degree, spmm


@torch.compile
class MFConv(MessagePassing):
    r"""The graph neural network operator from the
    `"Convolutional Networks on Graphs for Learning Molecular Fingerprints"
    <https://arxiv.org/abs/1509.09292>`_ paper.

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{W}^{(\deg(i))}_1 \mathbf{x}_i +
        \mathbf{W}^{(\deg(i))}_2 \sum_{j \in \mathcal{N}(i)} \mathbf{x}_j

    which trains a distinct weight matrix for each possible vertex degree.

    Args:
        in_channels (int or tuple): Size of each input sample, or :obj:`-1` to
            derive the size from the first input(s) to the forward method.
            A tuple corresponds to the sizes of source and target
            dimensionalities.
        out_channels (int): Size of each output sample.
        max_degree (int, optional): The maximum node degree to consider when
            updating weights (default: :obj:`10`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **inputs:**
          node features :math:`(|\mathcal{V}|, F_{in})` or
          :math:`((|\mathcal{V_s}|, F_{s}), (|\mathcal{V_t}|, F_{t}))`
          if bipartite,
          edge indices :math:`(2, |\mathcal{E}|)`
        - **outputs:** node features :math:`(|\mathcal{V}|, F_{out})` or
          :math:`(|\mathcal{V_t}|, F_{out})` if bipartite
    """
    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, max_degree: int = 10, bias=True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.max_degree = max_degree

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lins_l = ModuleList([
            Linear(in_channels[0], out_channels, bias=bias)
            for _ in range(max_degree + 1)
        ])

        self.lins_r = ModuleList([
            Linear(in_channels[1], out_channels, bias=False)
            for _ in range(max_degree + 1)
        ])

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        for lin in self.lins_l:
            lin.reset_parameters()
        for lin in self.lins_r:
            lin.reset_parameters()

    def forward(
        self,
        x: Union[Tensor, OptPairTensor],
        edge_index: Adj,
        size: Size = None,
    ) -> Tensor:

        if isinstance(x, Tensor):
            x = (x, x)

        x_r = x[1]

        deg = x[0]  # Dummy.
        if isinstance(edge_index, SparseTensor):
            deg = edge_index.storage.rowcount()
        elif isinstance(edge_index, Tensor):
            i = 1 if self.flow == 'source_to_target' else 0
            N = x[0].size(self.node_dim)
            N = size[1] if size is not None else N
            N = x_r.size(self.node_dim) if x_r is not None else N
            deg = degree(edge_index[i], N, dtype=torch.long)
        deg.clamp_(max=self.max_degree)

        # propagate_type: (x: OptPairTensor)
        h = self.propagate(edge_index, x=x, size=size)

        weight_l = torch.stack([lin.weight for lin in self.lins_l], dim=0)
        bias_l = torch.stack([
            lin.bias if lin.bias is not None else lin.weight.new_zeros((self.out_channels,))
            for lin in self.lins_l
        ], dim=0)
        weight_r = torch.stack([lin.weight for lin in self.lins_r], dim=0)

        node_dim = self.node_dim if self.node_dim >= 0 else (h.dim() + self.node_dim)
        h = h.movedim(node_dim, 0)

        N = h.size(0)
        in_l = h.size(-1)
        rest_shape = h.size()[1:-1]
        rest_numel = 1
        for s in rest_shape:
            rest_numel *= s
        
        if rest_numel == 1:
            h_flat = h.reshape(N, in_l)
            deg_expanded = deg
        else:
            h_flat = h.reshape(N * rest_numel, in_l)
            deg_expanded = deg.repeat_interleave(rest_numel)

        Wl_nodes = weight_l[deg_expanded]
        out_flat = torch.einsum('noi,ni->no', Wl_nodes, h_flat)
        out_flat += bias_l[deg_expanded]

        if x_r is not None:
            xr = x_r.movedim(node_dim, 0)
            xr_in = xr.size(-1)
            if rest_numel == 1:
                xr_flat = xr.reshape(N, xr_in)
            else:
                xr_flat = xr.reshape(N * rest_numel, xr_in)

            Wr_nodes = weight_r[deg_expanded]
            out_flat = out_flat + torch.einsum('noi,ni->no', Wr_nodes, xr_flat)

        if rest_numel == 1:
            out_moved = out_flat.reshape(N, self.out_channels)
        else:
            out_moved = out_flat.reshape((N, ) + rest_shape + (self.out_channels,))

        out = out_moved.movedim(0, node_dim)

        return out

    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def message_and_aggregate(self, adj_t: Adj, x: OptPairTensor) -> Tensor:
        if isinstance(adj_t, SparseTensor):
            adj_t = adj_t.set_value(None, layout=None)
        return spmm(adj_t, x[0], reduce=self.aggr)
