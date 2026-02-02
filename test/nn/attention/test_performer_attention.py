import pytest
import torch

from torch_geometric.nn.attention import PerformerAttention
from torch_geometric.nn.attention.performer import linear_attention


def test_performer_attention():
    x = torch.randn(1, 4, 16)
    mask = torch.ones([1, 4], dtype=torch.bool)
    attn = PerformerAttention(channels=16, heads=4)
    out = attn.forward(x, mask)
    # assert out.shape == (1, 4, 16)
    # assert str(attn) == ('PerformerAttention(heads=4, '
    #                      'head_channels=64 kernel=ReLU())')


@pytest.mark.parametrize('batch_size,num_heads,seq_len,head_dim', [(2, 4, 8, 16)])
def test_linear_attention_output_shape(batch_size, num_heads, seq_len, head_dim):
    if not torch.cuda.is_available():
        pytest.skip('CUDA not available')

    q = torch.randn(batch_size, num_heads, seq_len, head_dim,
                    dtype=torch.float32, device='cuda')
    k = torch.randn(batch_size, num_heads, seq_len, head_dim,
                    dtype=torch.float32, device='cuda')
    v = torch.randn(batch_size, num_heads, seq_len, head_dim,
                    dtype=torch.float32, device='cuda')

    out = linear_attention(q, k, v)

    assert out.shape == (batch_size, num_heads, seq_len, head_dim)
    assert out.dtype == torch.float32
    assert out.device.type == 'cuda'


@pytest.mark.parametrize('batch_size,num_heads,seq_len,head_dim', [(2, 4, 8, 16)])
def test_linear_attention_numerical_stability(batch_size, num_heads, seq_len,
                                               head_dim):
    if not torch.cuda.is_available():
        pytest.skip('CUDA not available')

    q = torch.randn(batch_size, num_heads, seq_len, head_dim,
                    dtype=torch.float32, device='cuda')
    k = torch.randn(batch_size, num_heads, seq_len, head_dim,
                    dtype=torch.float32, device='cuda')
    v = torch.randn(batch_size, num_heads, seq_len, head_dim,
                    dtype=torch.float32, device='cuda')

    out = linear_attention(q, k, v)

    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()
    assert torch.isfinite(out).all()


@pytest.mark.parametrize('batch_size,num_heads,seq_len,head_dim', [(2, 4, 8, 16)])
def test_linear_attention_computation(batch_size, num_heads, seq_len, head_dim):
    if not torch.cuda.is_available():
        pytest.skip('CUDA not available')

    q = torch.randn(batch_size, num_heads, seq_len, head_dim,
                    dtype=torch.float32, device='cuda')
    k = torch.randn(batch_size, num_heads, seq_len, head_dim,
                    dtype=torch.float32, device='cuda')
    v = torch.randn(batch_size, num_heads, seq_len, head_dim,
                    dtype=torch.float32, device='cuda')

    out = linear_attention(q, k, v)

    D_inv = 1.0 / (q @ k.sum(dim=-2).unsqueeze(-1))
    kv = k.transpose(-2, -1) @ v
    qkv = q @ kv
    expected = torch.einsum('...L,...Ld->...Ld', D_inv.squeeze(-1), qkv)

    assert torch.allclose(out, expected)


@pytest.mark.parametrize('batch_size,num_heads,seq_len,head_dim', [(2, 4, 8, 16)])
def test_linear_attention_gradient_flow(batch_size, num_heads, seq_len,
                                        head_dim):
    if not torch.cuda.is_available():
        pytest.skip('CUDA not available')

    q = torch.randn(batch_size, num_heads, seq_len, head_dim,
                    dtype=torch.float32, device='cuda', requires_grad=True)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim,
                    dtype=torch.float32, device='cuda', requires_grad=True)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim,
                    dtype=torch.float32, device='cuda', requires_grad=True)

    out = linear_attention(q, k, v)
    loss = out.sum()
    loss.backward()

    assert q.grad is not None
    assert k.grad is not None
    assert v.grad is not None
    assert not torch.isnan(q.grad).any()
    assert not torch.isnan(k.grad).any()
    assert not torch.isnan(v.grad).any()


@pytest.mark.parametrize('batch_size,num_heads,seq_len,head_dim', [(2, 4, 8, 16)])
def test_linear_attention_positive_values(batch_size, num_heads, seq_len,
                                          head_dim):
    if not torch.cuda.is_available():
        pytest.skip('CUDA not available')

    q = torch.randn(batch_size, num_heads, seq_len, head_dim,
                    dtype=torch.float32, device='cuda').abs()
    k = torch.randn(batch_size, num_heads, seq_len, head_dim,
                    dtype=torch.float32, device='cuda').abs()
    v = torch.randn(batch_size, num_heads, seq_len, head_dim,
                    dtype=torch.float32, device='cuda')

    out = linear_attention(q, k, v)

    assert out.shape == (batch_size, num_heads, seq_len, head_dim)
    assert not torch.isnan(out).any()


@pytest.mark.parametrize('batch_size,num_heads,seq_len,head_dim', [(2, 4, 8, 16)])
def test_linear_attention_zero_values(batch_size, num_heads, seq_len, head_dim):
    if not torch.cuda.is_available():
        pytest.skip('CUDA not available')

    q = torch.randn(batch_size, num_heads, seq_len, head_dim,
                    dtype=torch.float32, device='cuda')
    k = torch.randn(batch_size, num_heads, seq_len, head_dim,
                    dtype=torch.float32, device='cuda')
    v = torch.zeros(batch_size, num_heads, seq_len, head_dim,
                    dtype=torch.float32, device='cuda')

    out = linear_attention(q, k, v)

    assert torch.allclose(out, torch.zeros_like(out))


# Tests for PerformerAttention.forward()
@pytest.mark.parametrize('batch_size,seq_len,channels,heads', [(2, 8, 64, 4)])
def test_performer_attention_forward_output_shape(batch_size, seq_len, channels,
                                                  heads):
    if not torch.cuda.is_available():
        pytest.skip('CUDA not available')

    x = torch.randn(batch_size, seq_len, channels, dtype=torch.float32,
                    device='cuda')
    attn = PerformerAttention(channels=channels, heads=heads).cuda()

    out = attn.forward(x)

    assert out.shape == (batch_size, seq_len, channels)
    assert out.dtype == torch.float32
    assert out.device.type == 'cuda'


@pytest.mark.parametrize('batch_size,seq_len,channels,heads', [(2, 8, 64, 4)])
def test_performer_attention_forward_with_mask_output_shape(
        batch_size, seq_len, channels, heads):
    if not torch.cuda.is_available():
        pytest.skip('CUDA not available')

    x = torch.randn(batch_size, seq_len, channels, dtype=torch.float32,
                    device='cuda')
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device='cuda')
    attn = PerformerAttention(channels=channels, heads=heads).cuda()

    out = attn.forward(x, mask)

    assert out.shape == (batch_size, seq_len, channels)
    assert out.dtype == torch.float32
    assert out.device.type == 'cuda'


@pytest.mark.parametrize('batch_size,seq_len,channels,heads', [(2, 8, 64, 4)])
def test_performer_attention_forward_numerical_stability(batch_size, seq_len,
                                                         channels, heads):
    if not torch.cuda.is_available():
        pytest.skip('CUDA not available')

    x = torch.randn(batch_size, seq_len, channels, dtype=torch.float32,
                    device='cuda')
    attn = PerformerAttention(channels=channels, heads=heads).cuda()

    out = attn.forward(x)

    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()
    assert torch.isfinite(out).all()


@pytest.mark.parametrize('batch_size,seq_len,channels,heads', [(2, 8, 64, 4)])
def test_performer_attention_forward_with_mask_numerical_stability(
        batch_size, seq_len, channels, heads):
    if not torch.cuda.is_available():
        pytest.skip('CUDA not available')

    x = torch.randn(batch_size, seq_len, channels, dtype=torch.float32,
                    device='cuda')
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device='cuda')
    attn = PerformerAttention(channels=channels, heads=heads).cuda()

    out = attn.forward(x, mask)

    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()
    assert torch.isfinite(out).all()


@pytest.mark.parametrize('batch_size,seq_len,channels,heads', [(2, 8, 64, 4)])
def test_performer_attention_forward_gradient_flow(batch_size, seq_len,
                                                   channels, heads):
    if not torch.cuda.is_available():
        pytest.skip('CUDA not available')

    x = torch.randn(batch_size, seq_len, channels, dtype=torch.float32,
                    device='cuda', requires_grad=True)
    attn = PerformerAttention(channels=channels, heads=heads).cuda()

    out = attn.forward(x)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    assert not torch.isnan(x.grad).any()
    assert not torch.isinf(x.grad).any()


@pytest.mark.parametrize('batch_size,seq_len,channels,heads', [(2, 8, 64, 4)])
def test_performer_attention_forward_with_mask_gradient_flow(
        batch_size, seq_len, channels, heads):
    if not torch.cuda.is_available():
        pytest.skip('CUDA not available')

    x = torch.randn(batch_size, seq_len, channels, dtype=torch.float32,
                    device='cuda', requires_grad=True)
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device='cuda')
    attn = PerformerAttention(channels=channels, heads=heads).cuda()

    out = attn.forward(x, mask)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    assert not torch.isnan(x.grad).any()
    assert not torch.isinf(x.grad).any()


@pytest.mark.parametrize('batch_size,seq_len,channels,heads', [(2, 8, 64, 4)])
def test_performer_attention_forward_deterministic_with_seed(
        batch_size, seq_len, channels, heads):
    if not torch.cuda.is_available():
        pytest.skip('CUDA not available')

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    x1 = torch.randn(batch_size, seq_len, channels, dtype=torch.float32,
                     device='cuda')
    attn1 = PerformerAttention(channels=channels, heads=heads).cuda()
    out1 = attn1.forward(x1)

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    x2 = torch.randn(batch_size, seq_len, channels, dtype=torch.float32,
                     device='cuda')
    attn2 = PerformerAttention(channels=channels, heads=heads).cuda()
    out2 = attn2.forward(x2)

    assert torch.allclose(out1, out2)


@pytest.mark.parametrize('batch_size,seq_len,channels,heads', [(2, 8, 64, 4)])
def test_performer_attention_forward_mask_zeros_output(batch_size, seq_len,
                                                       channels, heads):
    if not torch.cuda.is_available():
        pytest.skip('CUDA not available')

    x = torch.randn(batch_size, seq_len, channels, dtype=torch.float32,
                    device='cuda')
    mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device='cuda')
    attn = PerformerAttention(channels=channels, heads=heads).cuda()

    out = attn.forward(x, mask)

    assert out.shape == (batch_size, seq_len, channels)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


@pytest.mark.parametrize('batch_size,seq_len,channels,heads', [(2, 8, 64, 4)])
def test_performer_attention_forward_partial_mask(batch_size, seq_len, channels,
                                                  heads):
    if not torch.cuda.is_available():
        pytest.skip('CUDA not available')

    x = torch.randn(batch_size, seq_len, channels, dtype=torch.float32,
                    device='cuda')
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device='cuda')
    mask[:, seq_len // 2:] = False
    attn = PerformerAttention(channels=channels, heads=heads).cuda()

    out = attn.forward(x, mask)

    assert out.shape == (batch_size, seq_len, channels)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


@pytest.mark.parametrize('batch_size,seq_len,channels,heads', [(2, 8, 64, 4)])
def test_performer_attention_forward_consistency_no_mask_vs_full_mask(
        batch_size, seq_len, channels, heads):
    if not torch.cuda.is_available():
        pytest.skip('CUDA not available')

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    x = torch.randn(batch_size, seq_len, channels, dtype=torch.float32,
                    device='cuda')
    attn = PerformerAttention(channels=channels, heads=heads).cuda()

    out_no_mask = attn.forward(x)

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    x = torch.randn(batch_size, seq_len, channels, dtype=torch.float32,
                    device='cuda')
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device='cuda')
    attn = PerformerAttention(channels=channels, heads=heads).cuda()

    out_with_mask = attn.forward(x, mask)

    assert torch.allclose(out_no_mask, out_with_mask)


@pytest.mark.parametrize('batch_size,seq_len,channels,heads', [(2, 8, 64, 4)])
def test_performer_attention_forward_zero_input(batch_size, seq_len, channels,
                                                heads):
    if not torch.cuda.is_available():
        pytest.skip('CUDA not available')

    x = torch.zeros(batch_size, seq_len, channels, dtype=torch.float32,
                    device='cuda')
    attn = PerformerAttention(channels=channels, heads=heads).cuda()

    out = attn.forward(x)

    assert out.shape == (batch_size, seq_len, channels)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


@pytest.mark.parametrize('batch_size,seq_len,channels,heads', [(2, 8, 64, 4)])
def test_performer_attention_forward_with_dropout(batch_size, seq_len, channels,
                                                  heads):
    if not torch.cuda.is_available():
        pytest.skip('CUDA not available')

    x = torch.randn(batch_size, seq_len, channels, dtype=torch.float32,
                    device='cuda')
    attn = PerformerAttention(channels=channels, heads=heads,
                              dropout=0.5).cuda()

    attn.train()
    out_train = attn.forward(x)

    attn.eval()
    out_eval = attn.forward(x)

    assert out_train.shape == (batch_size, seq_len, channels)
    assert out_eval.shape == (batch_size, seq_len, channels)
    assert not torch.isnan(out_train).any()
    assert not torch.isnan(out_eval).any()


@pytest.mark.parametrize('batch_size,seq_len,channels,heads', [(2, 8, 64, 4)])
def test_performer_attention_forward_with_bias_configurations(
        batch_size, seq_len, channels, heads):
    if not torch.cuda.is_available():
        pytest.skip('CUDA not available')

    x = torch.randn(batch_size, seq_len, channels, dtype=torch.float32,
                    device='cuda')

    attn_no_bias = PerformerAttention(channels=channels, heads=heads,
                                      qkv_bias=False,
                                      attn_out_bias=False).cuda()
    out_no_bias = attn_no_bias.forward(x)

    attn_with_bias = PerformerAttention(channels=channels, heads=heads,
                                        qkv_bias=True,
                                        attn_out_bias=True).cuda()
    out_with_bias = attn_with_bias.forward(x)

    assert out_no_bias.shape == (batch_size, seq_len, channels)
    assert out_with_bias.shape == (batch_size, seq_len, channels)
    assert not torch.isnan(out_no_bias).any()
    assert not torch.isnan(out_with_bias).any()
