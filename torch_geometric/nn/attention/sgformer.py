from typing import Optional

import torch
from torch import Tensor

@torch.compile
class SGFormerAttention(torch.nn.Module):
    r"""The simple global attention mechanism from the
    `"SGFormer: Simplifying and Empowering Transformers for
    Large-Graph Representations"
    <https://arxiv.org/abs/2306.10759>`_ paper.

    Args:
        channels (int): Size of each input sample.
        heads (int, optional): Number of parallel attention heads.
            (default: :obj:`1.`)
        head_channels (int, optional): Size of each attention head.
            (default: :obj:`64.`)
        qkv_bias (bool, optional): If specified, add bias to query, key
            and value in the self attention. (default: :obj:`False`)
    """
    def __init__(
        self,
        channels: int,
        heads: int = 1,
        head_channels: int = 64,
        qkv_bias: bool = False,
    ) -> None:
        super().__init__()
        assert channels % heads == 0
        if head_channels is None:
            head_channels = channels // heads

        self.heads = heads
        self.head_channels = head_channels

        inner_channels = head_channels * heads
        self.q = torch.nn.Linear(channels, inner_channels, bias=qkv_bias)
        self.k = torch.nn.Linear(channels, inner_channels, bias=qkv_bias)
        self.v = torch.nn.Linear(channels, inner_channels, bias=qkv_bias)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """Forward pass.


        Args:
            x (torch.Tensor): Node feature tensor
                :math:`\mathbf{X} \in \mathbb{R}^{B \times N \times F}`, with
                batch-size :math:`B`, (maximum) number of nodes :math:`N` for
                each graph, and feature dimension :math:`F`.
            mask (torch.Tensor, optional): Mask matrix
                :math:`\mathbf{M} \in {\{ 0, 1 \}}^{B \times N}` indicating
                the valid nodes for each graph. (default: :obj:`None`)
        """
        B, N, *_ = x.shape
        qs, ks, vs = self.q(x), self.k(x), self.v(x)
        # reshape and permute q, k and v to proper shape
        # (b, n, num_heads * head_channels) to (b, n, num_heads, head_channels)
        qs = qs.reshape(B, N, self.heads, self.head_channels)
        ks = ks.reshape(B, N, self.heads, self.head_channels)
        vs = vs.reshape(B, N, self.heads, self.head_channels)


        if mask is not None:
            mask = mask[:, :, None, None]
            vs.masked_fill_(~mask, 0.)
        # replace 0's with epsilon
        epsilon = 1e-6
        qs.masked_fill_(qs == 0, epsilon)
        ks.masked_fill_(ks == 0, epsilon)
        # normalize input, shape not changed
        qs = qs / torch.linalg.norm(qs, ord=2, dim=-1, keepdim=True)
        ks = ks / torch.linalg.norm(ks, ord=2, dim=-1, keepdim=True)

        # numerator
        # kvs: (B, H, M, D) computed as sum over N of ks[b, n, h, m] * vs[b, n, h, d]
        ks_p = ks.permute(0, 2, 1, 3)  # B, H, N, M
        vs_p = vs.permute(0, 2, 1, 3)  # B, H, N, D
        kvs = torch.matmul(ks_p.transpose(-2, -1), vs_p)  # B, H, M, D

        # attention_num: for each head, qs[b, n, h, m] @ kvs[b, h, m, d] -> B, H, N, D
        qs_p = qs.permute(0, 2, 1, 3)  # B, H, N, M
        attention_num_p = torch.matmul(qs_p, kvs)  # B, H, N, D
        attention_num = attention_num_p.permute(0, 2, 1, 3)  # B, N, H, D

        attention_num += N * vs

        # denominator
        ks_sum = ks.sum(dim=1)  # B, H, M
        attention_normalizer = (qs * ks_sum[:, None, :, :]).sum(dim=-1)  # B, N, H
        # attentive aggregated results
        attention_normalizer = attention_normalizer.unsqueeze(-1)
        attention_normalizer = attention_normalizer + N
        attn_output = attention_num / attention_normalizer

        return attn_output.mean(dim=2)

    def reset_parameters(self):
        self.q.reset_parameters()
        self.k.reset_parameters()
        self.v.reset_parameters()

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}('
                f'heads={self.heads}, '
                f'head_channels={self.head_channels})')
