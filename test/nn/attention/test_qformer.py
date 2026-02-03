import pytest
import torch

from torch_geometric.nn.attention import QFormer


def test_qformer_basic():
    x = torch.randn(1, 4, 16)
    attn = QFormer(input_dim=16, hidden_dim=16, output_dim=32, num_heads=4,
                   num_layers=2)
    out = attn.forward(x)


@pytest.mark.parametrize('batch_size,seq_len,input_dim,hidden_dim,output_dim,num_heads,num_layers',
                         [(2, 8, 64, 128, 32, 4, 2)])
def test_qformer_forward_output_shape(batch_size, seq_len, input_dim,
                                      hidden_dim, output_dim, num_heads,
                                      num_layers):
    if not torch.cuda.is_available():
        pytest.skip('CUDA not available')

    x = torch.randn(batch_size, seq_len, input_dim, dtype=torch.float32,
                    device='cuda')
    qformer = QFormer(input_dim=input_dim, hidden_dim=hidden_dim,
                      output_dim=output_dim, num_heads=num_heads,
                      num_layers=num_layers).cuda()

    out = qformer.forward(x)

    assert out.shape == (batch_size, seq_len, output_dim)
    assert out.dtype == torch.float32
    assert out.device.type == 'cuda'


@pytest.mark.parametrize('batch_size,seq_len,input_dim,hidden_dim,output_dim,num_heads,num_layers',
                         [(2, 8, 64, 128, 32, 4, 2)])
def test_qformer_forward_numerical_stability(batch_size, seq_len, input_dim,
                                             hidden_dim, output_dim, num_heads,
                                             num_layers):
    if not torch.cuda.is_available():
        pytest.skip('CUDA not available')

    x = torch.randn(batch_size, seq_len, input_dim, dtype=torch.float32,
                    device='cuda')
    qformer = QFormer(input_dim=input_dim, hidden_dim=hidden_dim,
                      output_dim=output_dim, num_heads=num_heads,
                      num_layers=num_layers).cuda()

    out = qformer.forward(x)

    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()
    assert torch.isfinite(out).all()


@pytest.mark.parametrize('batch_size,seq_len,input_dim,hidden_dim,output_dim,num_heads,num_layers',
                         [(2, 8, 64, 128, 32, 4, 2)])
def test_qformer_forward_gradient_flow(batch_size, seq_len, input_dim,
                                       hidden_dim, output_dim, num_heads,
                                       num_layers):
    if not torch.cuda.is_available():
        pytest.skip('CUDA not available')

    x = torch.randn(batch_size, seq_len, input_dim, dtype=torch.float32,
                    device='cuda', requires_grad=True)
    qformer = QFormer(input_dim=input_dim, hidden_dim=hidden_dim,
                      output_dim=output_dim, num_heads=num_heads,
                      num_layers=num_layers).cuda()

    out = qformer.forward(x)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    assert not torch.isnan(x.grad).any()
    assert not torch.isinf(x.grad).any()


@pytest.mark.parametrize('batch_size,seq_len,input_dim,hidden_dim,output_dim,num_heads,num_layers',
                         [(2, 8, 64, 128, 32, 4, 2)])
def test_qformer_forward_deterministic_with_seed(batch_size, seq_len,
                                                 input_dim, hidden_dim,
                                                 output_dim, num_heads,
                                                 num_layers):
    if not torch.cuda.is_available():
        pytest.skip('CUDA not available')

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    x1 = torch.randn(batch_size, seq_len, input_dim, dtype=torch.float32,
                     device='cuda')
    qformer1 = QFormer(input_dim=input_dim, hidden_dim=hidden_dim,
                       output_dim=output_dim, num_heads=num_heads,
                       num_layers=num_layers).cuda()
    out1 = qformer1.forward(x1)

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    x2 = torch.randn(batch_size, seq_len, input_dim, dtype=torch.float32,
                     device='cuda')
    qformer2 = QFormer(input_dim=input_dim, hidden_dim=hidden_dim,
                       output_dim=output_dim, num_heads=num_heads,
                       num_layers=num_layers).cuda()
    out2 = qformer2.forward(x2)

    assert torch.allclose(out1, out2)


@pytest.mark.parametrize('batch_size,seq_len,input_dim,hidden_dim,output_dim,num_heads,num_layers',
                         [(2, 8, 64, 128, 32, 4, 2)])
def test_qformer_forward_zero_input(batch_size, seq_len, input_dim, hidden_dim,
                                    output_dim, num_heads, num_layers):
    if not torch.cuda.is_available():
        pytest.skip('CUDA not available')

    x = torch.zeros(batch_size, seq_len, input_dim, dtype=torch.float32,
                    device='cuda')
    qformer = QFormer(input_dim=input_dim, hidden_dim=hidden_dim,
                      output_dim=output_dim, num_heads=num_heads,
                      num_layers=num_layers).cuda()

    out = qformer.forward(x)

    assert out.shape == (batch_size, seq_len, output_dim)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


@pytest.mark.parametrize('batch_size,seq_len,input_dim,hidden_dim,output_dim,num_heads,num_layers',
                         [(2, 8, 64, 128, 32, 4, 2)])
def test_qformer_forward_with_dropout(batch_size, seq_len, input_dim,
                                      hidden_dim, output_dim, num_heads,
                                      num_layers):
    if not torch.cuda.is_available():
        pytest.skip('CUDA not available')

    x = torch.randn(batch_size, seq_len, input_dim, dtype=torch.float32,
                    device='cuda')
    qformer = QFormer(input_dim=input_dim, hidden_dim=hidden_dim,
                      output_dim=output_dim, num_heads=num_heads,
                      num_layers=num_layers, dropout=0.5).cuda()

    qformer.train()
    out_train = qformer.forward(x)

    qformer.eval()
    out_eval = qformer.forward(x)

    assert out_train.shape == (batch_size, seq_len, output_dim)
    assert out_eval.shape == (batch_size, seq_len, output_dim)
    assert not torch.isnan(out_train).any()
    assert not torch.isnan(out_eval).any()


@pytest.mark.parametrize('batch_size,seq_len,input_dim,hidden_dim,output_dim,num_heads,num_layers',
                         [(2, 8, 64, 128, 32, 4, 2)])
def test_qformer_forward_various_layer_counts(batch_size, seq_len, input_dim,
                                              hidden_dim, output_dim, num_heads,
                                              num_layers):
    if not torch.cuda.is_available():
        pytest.skip('CUDA not available')

    x = torch.randn(batch_size, seq_len, input_dim, dtype=torch.float32,
                    device='cuda')

    for layers in [1, 2, 4]:
        qformer = QFormer(input_dim=input_dim, hidden_dim=hidden_dim,
                         output_dim=output_dim, num_heads=num_heads,
                         num_layers=layers).cuda()
        out = qformer.forward(x)

        assert out.shape == (batch_size, seq_len, output_dim)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()


@pytest.mark.parametrize('batch_size,seq_len,input_dim,hidden_dim,output_dim,num_heads,num_layers',
                         [(2, 8, 64, 128, 32, 4, 2)])
def test_qformer_forward_various_head_counts(batch_size, seq_len, input_dim,
                                             hidden_dim, output_dim, num_heads,
                                             num_layers):
    if not torch.cuda.is_available():
        pytest.skip('CUDA not available')

    x = torch.randn(batch_size, seq_len, input_dim, dtype=torch.float32,
                    device='cuda')

    for heads in [2, 4, 8]:
        qformer = QFormer(input_dim=input_dim, hidden_dim=hidden_dim,
                         output_dim=output_dim, num_heads=heads,
                         num_layers=num_layers).cuda()
        out = qformer.forward(x)

        assert out.shape == (batch_size, seq_len, output_dim)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()


@pytest.mark.parametrize('batch_size,seq_len,input_dim,hidden_dim,output_dim,num_heads,num_layers',
                         [(2, 8, 64, 128, 32, 4, 2)])
def test_qformer_forward_batch_size_one(batch_size, seq_len, input_dim,
                                        hidden_dim, output_dim, num_heads,
                                        num_layers):
    if not torch.cuda.is_available():
        pytest.skip('CUDA not available')

    x = torch.randn(1, seq_len, input_dim, dtype=torch.float32, device='cuda')
    qformer = QFormer(input_dim=input_dim, hidden_dim=hidden_dim,
                      output_dim=output_dim, num_heads=num_heads,
                      num_layers=num_layers).cuda()

    out = qformer.forward(x)

    assert out.shape == (1, seq_len, output_dim)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


@pytest.mark.parametrize('batch_size,seq_len,input_dim,hidden_dim,output_dim,num_heads,num_layers',
                         [(2, 8, 64, 128, 32, 4, 2)])
def test_qformer_forward_sequence_length_one(batch_size, seq_len, input_dim,
                                             hidden_dim, output_dim, num_heads,
                                             num_layers):
    if not torch.cuda.is_available():
        pytest.skip('CUDA not available')

    x = torch.randn(batch_size, 1, input_dim, dtype=torch.float32,
                    device='cuda')
    qformer = QFormer(input_dim=input_dim, hidden_dim=hidden_dim,
                      output_dim=output_dim, num_heads=num_heads,
                      num_layers=num_layers).cuda()

    out = qformer.forward(x)

    assert out.shape == (batch_size, 1, output_dim)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


@pytest.mark.parametrize('batch_size,seq_len,input_dim,hidden_dim,output_dim,num_heads,num_layers',
                         [(2, 8, 64, 128, 32, 4, 2)])
def test_qformer_forward_large_sequence(batch_size, seq_len, input_dim,
                                        hidden_dim, output_dim, num_heads,
                                        num_layers):
    if not torch.cuda.is_available():
        pytest.skip('CUDA not available')

    large_seq_len = 32
    x = torch.randn(batch_size, large_seq_len, input_dim, dtype=torch.float32,
                    device='cuda')
    qformer = QFormer(input_dim=input_dim, hidden_dim=hidden_dim,
                      output_dim=output_dim, num_heads=num_heads,
                      num_layers=num_layers).cuda()

    out = qformer.forward(x)

    assert out.shape == (batch_size, large_seq_len, output_dim)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()
