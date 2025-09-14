from torch import nn
import torch
import math
import copy


class Linear(nn.Module):
    def __init__(self, in_features, out_features):
        super(Linear, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        x = torch.matmul(x, self.weight.t()) + self.bias

        return x


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.linear1 = Linear(d_model, d_ff)
        self.linear2 = Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_h = d_model // num_heads

        self.W_q = Linear(d_model, d_model)
        self.W_k = Linear(d_model, d_model)
        self.W_v = Linear(d_model, d_model)
        self.W_o = Linear(d_model, d_model)

    def forward(self, q, k, v, attn_mask):
        # Create Q,K,V tensors
        Q = self.W_q(q)
        K = self.W_k(k)
        V = self.W_v(v)

        # Split into multi heads (batch_no, num_heads, seq_no, d_h)
        Q_mh = Q.reshape(Q.shape[0], Q.shape[1], self.num_heads, self.d_h).permute(0, 2, 1, 3)
        K_mh = K.reshape(K.shape[0], K.shape[1], self.num_heads, self.d_h).permute(0, 2, 1, 3)
        V_mh = V.reshape(V.shape[0], V.shape[1], self.num_heads, self.d_h).permute(0, 2, 1, 3)

        # Create attention mask
        causal_mask = (
            torch.tril(torch.ones(q.shape[1], k.shape[1]).bool()).unsqueeze(0).unsqueeze(1)
        )
        # padding_mask = (q_padding_mask.unsqueeze(-1) & k_padding_mask.unsqueeze(-2)).unsqueeze(1)
        # attn_mask = (causal_mask & padding_mask).expand(-1, self.num_heads, -1, -1)

        # Calculate attention matrices
        attn_raw = torch.matmul(Q_mh, K_mh.transpose(-1, -2)) / math.sqrt(self.d_h)
        attn_masked = attn_raw.masked_fill(~attn_mask, float("-inf"))
        attn = torch.softmax(attn_masked, -1)

        # Apply to value vectors and recombine
        A_mh = torch.matmul(attn, V_mh)
        A = A_mh.permute(0, 2, 1, 3).reshape(A_mh.shape[0], A_mh.shape[2], -1)

        # Project out
        O = self.W_o(A)

        return O


class Embedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super(Embedding, self).__init__()
        self.E = nn.Parameter(torch.rand(vocab_size, d_model))

    def forward(self, x):
        x = self.E[x]
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_length):
        super(PositionalEncoding, self).__init__()
        self.register_buffer("PE", self.create_pe(d_model, max_length))

    def create_pe(self, d_model, max_length):
        PE = torch.zeros((max_length, d_model))
        positions = torch.arange(0, max_length)
        dims = torch.arange(0, d_model // 2)

        evens = torch.sin(positions.unsqueeze(-1) / (10000 ** (2 * dims / d_model)).unsqueeze(0))
        odds = torch.cos(positions.unsqueeze(-1) / (10000 ** (2 * dims / d_model)).unsqueeze(0))

        PE[:, 0::2] = evens
        PE[:, 1::2] = odds

        return PE

    def forward(self, x):
        x = x + self.PE[: x.shape[1], :].unsqueeze(0)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.layer_norm_one = LayerNorm(d_model)
        self.feedforward = PositionWiseFeedForward(d_model, d_ff)
        self.layer_norm_two = LayerNorm(d_model=d_model)

    def forward(self, src, src_padding_mask):
        src = self.attention(src, src, src, src_padding_mask, src_padding_mask)
        src = self.layer_norm_one(src)
        src = self.feedforward(src)
        src = self.layer_norm_two(src)
        return src


class Encoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_enc_layers):
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff) for _ in range(num_enc_layers)]
        )

    def forward(self, src, src_padding_mask):
        for layer in self.layers:
            src = layer(src, src_padding_mask)

        return src
