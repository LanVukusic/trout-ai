import torch
from torch import nn
from torch.nn.functional import softmax


class MHA(nn.Module):
    def __init__(self, model_dim, n_heads, head_dim):

        super().__init__()

        self.d = model_dim
        self.head_dim = head_dim
        self.n_heads = n_heads

        # we get N(batch size) x 64 x model_dim

        self.Q_linear = nn.Linear(model_dim, n_heads * head_dim, bias=False)
        self.K_linear = nn.Linear(model_dim, n_heads * head_dim, bias=False)
        self.V_linear = nn.Linear(model_dim, n_heads * head_dim, bias=False)

        # if we have heads A, B, C, D..... fora je da stisnes al pa odstisnes nazaj na moduleDims
        self.fc = nn.Linear(n_heads * head_dim, model_dim, bias=False)

    def forward(self, x):
        # x.shape (N, 64, module_dim)

        N, S, d = x.shape

        q = self.Q_linear(x)
        k = self.K_linear(x)
        v = self.V_linear(x)
        # (N, 64, n_heads * head_dim). iz vsake pozicije zračuna vse
        # (N, squares, ...)

        # razbijemo na dejansko n_glav. prej bi naceloma lahko nardil for i in range(n_heads), sam je to vektorizirana verzija
        q = q.view(N, S, self.n_heads, self.head_dim)
        k = k.view(N, S, self.n_heads, self.head_dim)
        v = v.view(N, S, self.n_heads, self.head_dim)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        # vsaka je N, n_heads, S, head_dim

        attn = softmax((q @ k.transpose(2, 3)) / d**0.5, dim=-1)  # N, n_heads, S, S

        out = attn @ v  # N, n_heads, S, head_dim

        out = out.transpose(1, 2)  # N, S, n_heads, head_dim
        out = out.contiguous().view(
            N, S, self.n_heads * self.head_dim
        )  # N, S, n_heads * head_dim
        out = self.fc(out)  # N, S, model_dim

        return out


class TransformerBlock(nn.Module):
    def __init__(self, model_dim, n_heads, head_dim):
        super().__init__()

        self.ffn = nn.Sequential(
            nn.Linear(model_dim, 3 * model_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(3 * model_dim, model_dim),
            nn.Dropout(0.3),
        )

        # positional encoding
        self.pos_encode = SinusoisalPositionalEmbedding(model_dim)

        # multihead attention
        self.mha = MHA(model_dim=model_dim, n_heads=n_heads, head_dim=head_dim)

        # enacba ki normalizira * a + b (sta learnable)
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)

    def forward(self, x):
        # x.shape (N, 8, 8, 14)
        x = x + self.pos_encode(x)  # N, 64, model_dim
        x = x + self.mha(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, model_dim=512, n_heads=8, head_dim=64, n_blocks=4):
        super().__init__()

        self.embed = nn.Linear(14, model_dim)

        self.pos_encode = nn.Parameter(torch.zeros(64, model_dim))

        self.blocks = nn.Sequential(
            *[TransformerBlock(model_dim, n_heads, head_dim) for _ in range(n_blocks)]
        )

        self.head = nn.Sequential(
            nn.Linear(64 * model_dim, model_dim), nn.ReLU(), nn.Linear(model_dim, 1)
        )

    def forward(self, x):
        x = self.embed(x)
        x = x + self.pos_encode
        x = self.blocks(x)
        x = x.view(x.shape[0], -1)
        x = self.head(x)
        return x


class SinusoisalPositionalEmbedding(nn.Module):
    # https://medium.com/nlp-trend-and-review-en/positional-embeddings-7b168da36605

    def __init__(self, model_dim):
        super().__init__()
        self.dim = model_dim

    def sin_pos(self, pos, i):
        return torch.sin(pos / 10000 ** (2 * i / self.dim))

    def cos_pos(self, pos, i):
        return torch.cos(pos / 10000 ** (2 * i / self.dim))

    def forward(self, x):
        N, S, d = x.shape
        pos = torch.arange(S).unsqueeze(0).repeat(N, 1).float()
        x = torch.stack(
            [
                self.sin_pos(pos, i) if i % 2 == 0 else self.cos_pos(pos, i)
                for i in range(self.dim)
            ],
            dim=-1,
        )
        return x
