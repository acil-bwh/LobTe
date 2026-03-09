"""
File: simple_lobte.py
Author: Ariel H. Curiale
Github: https://gitlab.com/Curiale
Date: 2025-04-18
Description: LobTe implementation based on the Tensorflow version.
"""

import torch
import torch.nn

import einops
import einops.layers.torch

# helpers


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


def posemb_sincos_2d(h, w, dim, temperature: int = 10000, dtype=torch.float32):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    assert (
        dim % 4
    ) == 0, "feature dimension must be multiple of 4 for sincos emb"
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature**omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)


# classes


class FeedForward(torch.nn.Module):

    def __init__(self, dim, hidden_dim, dropout_rate=0):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, dim),
            torch.nn.Dropout(dropout_rate),
        )
        self.norm = torch.nn.LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        x = x + self.net(x)
        return x


class Attention(torch.nn.Module):

    def __init__(
        self, dim, num_heads, return_attention_scores=False, **kwargs
    ):
        super().__init__()
        embed_dim = dim * num_heads
        self.mha = torch.nn.MultiheadAttention(embed_dim, num_heads, **kwargs)
        self.to_qkv = torch.nn.Linear(dim, embed_dim, bias=False)
        self.to_out = torch.nn.Linear(embed_dim, dim, bias=False)
        self.norm = torch.nn.LayerNorm(dim)
        self.return_attention_scores = return_attention_scores

    def forward(self, x):
        x1 = self.norm(x)
        x1 = self.to_qkv(x1)
        attn_output, attn_scores = self.mha(query=x1, key=x1, value=x1)
        if self.return_attention_scores:
            # Cache the attention scores for plotting later.
            self.attn_scores = attn_scores
        else:
            self.attn_scores = None
        x = x + self.to_out(attn_output)
        return x


class Transformer(torch.nn.Module):

    def __init__(
        self,
        dim,
        depth,
        heads,
        mlp_dim,
        dropout_rate=0,
    ):
        super().__init__()
        self.norm = torch.nn.LayerNorm(dim)
        self.layers = torch.nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                torch.nn.ModuleList(
                    [
                        Attention(
                            dim=dim,
                            num_heads=heads,
                            dropout=dropout_rate,
                            batch_first=True,
                        ),
                        FeedForward(
                            dim,
                            mlp_dim,
                            dropout_rate=dropout_rate,
                        ),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x)
            x = ff(x)
        # Cache the last attention scores for plotting later
        self.last_attn_scores = self.layers[-1][0].attn_scores
        x = self.norm(x)
        return x


class SimpleLobTe(torch.nn.Module):

    def __init__(
        self,
        image_size,
        patch_size,
        outcomes,
        dim,
        depth,
        heads,
        mlp_dim,
        deepf_dim=5,
        in_channels=1,
        lobes=5,
        pool="cls",
        dropout_rate=0,
    ):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        self.dropout_rate = dropout_rate

        assert (
            image_height % patch_height == 0 and image_width % patch_width == 0
        ), "Image dimensions must be divisible by the patch size."

        assert pool in {
            "cls",
            "mean",
        }, "pool type must be either cls (cls token) or mean (mean pooling)"

        # self.input_attention = None
        patch_dim = in_channels * patch_height * patch_width

        self.to_patch_embedding = torch.nn.Sequential(
            einops.layers.torch.Rearrange(
                "b c l (h p1) (w p2) -> b (l h w) (p1 p2 c)",
                p1=patch_height,
                p2=patch_width,
            ),
            torch.nn.Linear(patch_dim, dim),
        )

        num_patches = (
            (image_height // patch_height)
            * (image_width // patch_width)
            * lobes
        )
        self.pool = pool
        self.cls_token = None
        if pool == "cls":
            # The model will use an extra class to model the global learned
            # representation
            self.cls_token = torch.nn.Parameter(torch.randn(1, 1, dim))
            num_patches += 1

        posem_dim = 4 * dim
        self.pos_embedding = posemb_sincos_2d(
            h=lobes * (image_height // patch_height),
            w=lobes * (image_width // patch_width),
            dim=posem_dim,
        )[:num_patches, :dim]

        self.masking_inputs = torch.nn.Dropout(dropout_rate)

        self.transformer = Transformer(
            dim,
            depth,
            heads,
            mlp_dim,
            dropout_rate=dropout_rate,
        )

        self.deepf = torch.nn.Sequential(
            torch.nn.Linear(dim, deepf_dim),
            torch.nn.GELU(),
            torch.nn.BatchNorm1d(deepf_dim),
        )

        self.outcomes = outcomes
        self.linear_head = torch.nn.ModuleDict(
            {k: torch.nn.Linear(deepf_dim, outcomes[k]) for k in outcomes}
        )

    def forward(self, img):
        device = img.device
        x = self.to_patch_embedding(img)
        if self.cls_token is not None:
            b = x.shape[0]
            cls_tokens = einops.repeat(self.cls_token, "1 1 d -> b 1 d", b=b)
            x = torch.cat((cls_tokens, x), dim=1)

        x += self.pos_embedding.to(device, dtype=x.dtype)
        x = self.masking_inputs(x)
        x = self.transformer(x)
        x = x.mean(dim=1) if self.pool == "mean" else x[:, 0]
        deepf = self.deepf(x)
        out = {k: self.linear_head[k](deepf) for k in self.linear_head}
        out["deepf"] = deepf
        return out
