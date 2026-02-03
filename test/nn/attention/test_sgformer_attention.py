import pytest
import torch

from torch_geometric.nn.attention import SGFormerAttention


def test_sgformer_attention_basic():
    x = torch.randn(1, 4, 16)
    mask = torch.ones([1, 4], dtype=torch.bool)
    attn = SGFormerAttention(channels=16, heads=4)
    out = attn.forward(x, mask)


@pytest.mark.parametrize('batch_size,seq_len,channels,heads', [(2, 8, 64, 4)])
def test_sgformer_attention_forward_output_shape(batch_size, seq_len, channels,
                                                  heads):
    if not torch.cuda.is_available():
        pytest.skip('CUDA not available')

    x = torch.randn(batch_size, seq_len, channels, dtype=torch.float32,
                    device='cuda')
    attn = SGFormerAttention(channels=channels, heads=heads).cuda()

    out = attn.forward(x)

    assert out.shape == (batch_size, seq_len, channels)
    assert out.dtype == torch.float32
    assert out.device.type == 'cuda'


@pytest.mark.parametrize('batch_size,seq_len,channels,heads', [(2, 8, 64, 4)])
def test_sgformer_attention_forward_with_mask_output_shape(
        batch_size, seq_len, channels, heads):
    if not torch.cuda.is_available():
        pytest.skip('CUDA not available')

    x = torch.randn(batch_size, seq_len, channels, dtype=torch.float32,
                    device='cuda')
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device='cuda')
    attn = SGFormerAttention(channels=channels, heads=heads).cuda()

    out = attn.forward(x, mask)

    assert out.shape == (batch_size, seq_len, channels)
    assert out.dtype == torch.float32
    assert out.device.type == 'cuda'


@pytest.mark.parametrize('batch_size,seq_len,channels,heads', [(2, 8, 64, 4)])
def test_sgformer_attention_forward_numerical_stability(batch_size, seq_len,
                                                        channels, heads):
    if not torch.cuda.is_available():
        pytest.skip('CUDA not available')

    x = torch.randn(batch_size, seq_len, channels, dtype=torch.float32,
                    device='cuda')
    attn = SGFormerAttention(channels=channels, heads=heads).cuda()

    out = attn.forward(x)

    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()
    assert torch.isfinite(out).all()


@pytest.mark.parametrize('batch_size,seq_len,channels,heads', [(2, 8, 64, 4)])
def test_sgformer_attention_forward_with_mask_numerical_stability(
        batch_size, seq_len, channels, heads):
    if not torch.cuda.is_available():
        pytest.skip('CUDA not available')

    x = torch.randn(batch_size, seq_len, channels, dtype=torch.float32,
                    device='cuda')
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device='cuda')
    attn = SGFormerAttention(channels=channels, heads=heads).cuda()

    out = attn.forward(x, mask)

    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()
    assert torch.isfinite(out).all()


@pytest.mark.parametrize('batch_size,seq_len,channels,heads', [(2, 8, 64, 4)])
def test_sgformer_attention_forward_gradient_flow(batch_size, seq_len,
                                                  channels, heads):
    if not torch.cuda.is_available():
        pytest.skip('CUDA not available')

    x = torch.randn(batch_size, seq_len, channels, dtype=torch.float32,
                    device='cuda', requires_grad=True)
    attn = SGFormerAttention(channels=channels, heads=heads).cuda()

    out = attn.forward(x)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    assert not torch.isnan(x.grad).any()
    assert not torch.isinf(x.grad).any()


@pytest.mark.parametrize('batch_size,seq_len,channels,heads', [(2, 8, 64, 4)])
def test_sgformer_attention_forward_with_mask_gradient_flow(
        batch_size, seq_len, channels, heads):
    if not torch.cuda.is_available():
        pytest.skip('CUDA not available')

    x = torch.randn(batch_size, seq_len, channels, dtype=torch.float32,
                    device='cuda', requires_grad=True)
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device='cuda')
    attn = SGFormerAttention(channels=channels, heads=heads).cuda()

    out = attn.forward(x, mask)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    assert not torch.isnan(x.grad).any()
    assert not torch.isinf(x.grad).any()


@pytest.mark.parametrize('batch_size,seq_len,channels,heads', [(2, 8, 64, 4)])
def test_sgformer_attention_forward_deterministic_with_seed(
        batch_size, seq_len, channels, heads):
    if not torch.cuda.is_available():
        pytest.skip('CUDA not available')

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    x1 = torch.randn(batch_size, seq_len, channels, dtype=torch.float32,
                     device='cuda')
    attn1 = SGFormerAttention(channels=channels, heads=heads).cuda()
    out1 = attn1.forward(x1)

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    x2 = torch.randn(batch_size, seq_len, channels, dtype=torch.float32,
                     device='cuda')
    attn2 = SGFormerAttention(channels=channels, heads=heads).cuda()
    out2 = attn2.forward(x2)

    assert torch.allclose(out1, out2)


@pytest.mark.parametrize('batch_size,seq_len,channels,heads', [(2, 8, 64, 4)])
def test_sgformer_attention_forward_mask_zeros_output(batch_size, seq_len,
                                                      channels, heads):
    if not torch.cuda.is_available():
        pytest.skip('CUDA not available')

    x = torch.randn(batch_size, seq_len, channels, dtype=torch.float32,
                    device='cuda')
    mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device='cuda')
    attn = SGFormerAttention(channels=channels, heads=heads).cuda()

    out = attn.forward(x, mask)

    assert out.shape == (batch_size, seq_len, channels)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


@pytest.mark.parametrize('batch_size,seq_len,channels,heads', [(2, 8, 64, 4)])
def test_sgformer_attention_forward_partial_mask(batch_size, seq_len, channels,
                                                 heads):
    if not torch.cuda.is_available():
        pytest.skip('CUDA not available')

    x = torch.randn(batch_size, seq_len, channels, dtype=torch.float32,
                    device='cuda')
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device='cuda')
    mask[:, seq_len // 2:] = False
    attn = SGFormerAttention(channels=channels, heads=heads).cuda()

    out = attn.forward(x, mask)

    assert out.shape == (batch_size, seq_len, channels)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


@pytest.mark.parametrize('batch_size,seq_len,channels,heads', [(2, 8, 64, 4)])
def test_sgformer_attention_forward_consistency_no_mask_vs_full_mask(
        batch_size, seq_len, channels, heads):
    if not torch.cuda.is_available():
        pytest.skip('CUDA not available')

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    x = torch.randn(batch_size, seq_len, channels, dtype=torch.float32,
                    device='cuda')
    attn = SGFormerAttention(channels=channels, heads=heads).cuda()

    out_no_mask = attn.forward(x)

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    x = torch.randn(batch_size, seq_len, channels, dtype=torch.float32,
                    device='cuda')
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device='cuda')
    attn = SGFormerAttention(channels=channels, heads=heads).cuda()

    out_with_mask = attn.forward(x, mask)

    assert torch.allclose(out_no_mask, out_with_mask)


@pytest.mark.parametrize('batch_size,seq_len,channels,heads', [(2, 8, 64, 4)])
def test_sgformer_attention_forward_zero_input(batch_size, seq_len, channels,
                                               heads):
    if not torch.cuda.is_available():
        pytest.skip('CUDA not available')

    x = torch.zeros(batch_size, seq_len, channels, dtype=torch.float32,
                    device='cuda')
    attn = SGFormerAttention(channels=channels, heads=heads).cuda()

    out = attn.forward(x)

    assert out.shape == (batch_size, seq_len, channels)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


@pytest.mark.parametrize('batch_size,seq_len,channels,heads', [(2, 8, 64, 4)])
def test_sgformer_attention_forward_with_bias_configurations(
        batch_size, seq_len, channels, heads):
    if not torch.cuda.is_available():
        pytest.skip('CUDA not available')

    x = torch.randn(batch_size, seq_len, channels, dtype=torch.float32,
                    device='cuda')

    attn_no_bias = SGFormerAttention(channels=channels, heads=heads,
                                     qkv_bias=False).cuda()
    out_no_bias = attn_no_bias.forward(x)

    attn_with_bias = SGFormerAttention(channels=channels, heads=heads,
                                       qkv_bias=True).cuda()
    out_with_bias = attn_with_bias.forward(x)

    assert out_no_bias.shape == (batch_size, seq_len, channels)
    assert out_with_bias.shape == (batch_size, seq_len, channels)
    assert not torch.isnan(out_no_bias).any()
    assert not torch.isnan(out_with_bias).any()
