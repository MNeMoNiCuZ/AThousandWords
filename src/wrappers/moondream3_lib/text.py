import torch
import torch.nn as nn

from torch.nn import functional as F
from torch.nn.attention.flex_attention import flex_attention
from typing import Optional

from .layers import layer_norm, mlp, QuantizedLinear, moe_mlp
from .rope import apply_rotary_emb, precompute_freqs_cis
from .config import TextConfig


def text_encoder(input_ids: torch.Tensor, w: nn.Module):
    return F.embedding(input_ids, w.wte)


def attn(
    x: torch.Tensor,
    w: nn.Module,
    freqs_cis: torch.Tensor,
    kv_cache: nn.Module,
    attn_mask: torch.Tensor,
    n_heads: int,
    n_kv_heads: int,
    position_ids: torch.Tensor,
    lora: Optional[dict] = None,
    flex_block_mask_slice=None,
):
    bsz, q_len, d_model = x.shape
    head_dim = d_model // n_heads

    qkv_out = w.qkv(x)  # shape: (bsz, q_len, (n_heads + 2*n_kv_heads)*head_dim)
    if lora is not None:
        qkv_out += F.linear(F.linear(x, lora["qkv"]["A"]), lora["qkv"]["B"])
    q_dim = n_heads * head_dim
    kv_dim = n_kv_heads * head_dim
    q, k, v = qkv_out.split([q_dim, kv_dim, kv_dim], dim=-1)

    q = q.view(bsz, q_len, n_heads, head_dim).transpose(1, 2)
    k = k.view(bsz, q_len, n_kv_heads, head_dim).transpose(1, 2)
    v = v.view(bsz, q_len, n_kv_heads, head_dim).transpose(1, 2)

    if hasattr(w, "tau") and w.tau is not None:
        tok_feat = F.gelu(qkv_out)
        tok_q = torch.tanh(torch.matmul(tok_feat, w.tau["wq"].t())).permute(0, 2, 1)
        tok_v = torch.tanh(torch.matmul(tok_feat, w.tau["wv"].t())).permute(0, 2, 1)
        pos = position_ids.to(q.dtype) + 1
        tau_pos = 1 + (
            torch.sigmoid(w.tau["alpha"][:, None] * pos.log()) - 0.5
        )  # (H,S)
        tau_q = (tok_q + tau_pos[None]).unsqueeze(-1)  # (B,H,S,1)
        tau_v = (tok_v + tau_pos[None]).unsqueeze(-1)
        q = q * tau_q
        v = v * tau_v

    q = apply_rotary_emb(q, freqs_cis, position_ids, n_heads)
    k = apply_rotary_emb(k, freqs_cis, position_ids, n_kv_heads)

    if kv_cache is not None:
        k, v = kv_cache.update(position_ids, k, v)

    if flex_block_mask_slice is not None:
        torch._assert(n_heads == n_kv_heads, "gqa not supported yet")
        out = flex_attention(q, k, v, block_mask=flex_block_mask_slice)
    else:
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, enable_gqa=n_heads != n_kv_heads
        )

    out = out.transpose(1, 2).reshape(bsz, q_len, d_model)

    out0 = w.proj(out)
    if lora is not None:
        out1 = F.linear(F.linear(x, lora["proj"]["A"]), lora["proj"]["B"])
        out = out0 + out1
    else:
        out = out0

    return out


def text_decoder(
    x: torch.Tensor,
    w: nn.Module,
    attn_mask: torch.Tensor,
    position_ids: torch.Tensor,
    config: TextConfig,
    lora: Optional[dict] = None,
    flex_block_mask_slice=None,
):
    for i, block in enumerate(w.blocks):
        if lora is not None:
            layer_lora = lora["text"]["blocks"][str(i)]
            mlp_lora = layer_lora["mlp"]
            attn_lora = layer_lora["attn"]
        else:
            mlp_lora = None
            attn_lora = None

        l_in = layer_norm(x, block.ln)
        l_attn = attn(
            l_in,
            block.attn,
            freqs_cis=w.freqs_cis,
            kv_cache=block.kv_cache,
            attn_mask=attn_mask,
            n_heads=config.n_heads,
            n_kv_heads=config.n_kv_heads,
            position_ids=position_ids,
            lora=attn_lora,
            flex_block_mask_slice=flex_block_mask_slice,
        )

        if config.moe is not None and i >= config.moe.start_layer:
            l_mlp = moe_mlp(l_in, block.mlp, config.moe.experts_per_token)
        else:
            l_mlp = mlp(l_in, block.mlp, lora=mlp_lora)

        x = x + l_attn + l_mlp

    return x


def lm_head(
    hidden_BTC: torch.Tensor, w: nn.Module, indices: Optional[torch.Tensor] = None
):
    hidden_BC = hidden_BTC[:, -1, :]
    hidden_BC = layer_norm(hidden_BC, w.post_ln)
    if indices is not None:
        # Only compute logits for specified token indices
        logits = hidden_BC @ w.lm_head.weight[indices].T + w.lm_head.bias[indices]
    else:
        logits = w.lm_head(hidden_BC)
    return logits


def build_dense_mlp(d_model, d_ffn, dtype, linear_cls):
    return nn.ModuleDict(
        {
            "fc1": linear_cls(d_model, d_ffn, dtype=dtype),
            "fc2": linear_cls(d_ffn, d_model, dtype=dtype),
        }
    )


def build_moe_mlp(d_model, d_ffn, n_experts, dtype):
    # For GeGLU, fc1 needs to output 2 * d_ffn (for gating)
    return nn.ModuleDict(
        {
            "router": nn.Linear(d_model, n_experts, dtype=dtype),
            "fc1": nn.ParameterDict(
                {
                    "weight": nn.Parameter(
                        torch.empty(n_experts, 2 * d_ffn, d_model, dtype=dtype)
                    )
                }
            ),
            "fc2": nn.ParameterDict(
                {
                    "weight": nn.Parameter(
                        torch.empty(n_experts, d_model, d_ffn, dtype=dtype)
                    )
                }
            ),
        }
    )


def build_text_model(config: TextConfig, dtype: torch.dtype) -> nn.Module:
    qkv_dim = int(config.dim * (1 + 2 * config.n_kv_heads / config.n_heads))
    linear_cls = QuantizedLinear if config.group_size is not None else nn.Linear

    text = nn.ModuleDict(
        {
            "blocks": nn.ModuleList(
                [
                    nn.ModuleDict(
                        {
                            "ln": nn.LayerNorm(config.dim, dtype=dtype),
                            "attn": nn.ModuleDict(
                                {
                                    "qkv": linear_cls(config.dim, qkv_dim, dtype=dtype),
                                    "proj": linear_cls(
                                        config.dim, config.dim, dtype=dtype
                                    ),
                                    "tau": nn.ParameterDict(
                                        {
                                            "wq": nn.Parameter(
                                                torch.empty(
                                                    config.n_heads, qkv_dim, dtype=dtype
                                                )
                                            ),
                                            "wv": nn.Parameter(
                                                torch.empty(
                                                    config.n_heads, qkv_dim, dtype=dtype
                                                )
                                            ),
                                            "alpha": nn.Parameter(
                                                torch.empty(config.n_heads, dtype=dtype)
                                            ),
                                        }
                                    ),
                                }
                            ),
                            "mlp": (
                                build_moe_mlp(
                                    config.dim,
                                    config.moe.expert_inner_dim,
                                    config.moe.num_experts,
                                    dtype,
                                )
                                if config.moe is not None
                                and layer_idx >= config.moe.start_layer
                                else build_dense_mlp(
                                    config.dim, config.ff_dim, dtype, linear_cls
                                )
                            ),
                        }
                    )
                    for layer_idx in range(config.n_layers)
                ]
            ),
            "post_ln": nn.LayerNorm(config.dim, dtype=dtype),
            "lm_head": nn.Linear(config.dim, config.vocab_size, dtype=dtype),
        }
    )
    text.wte = nn.Parameter(torch.empty(config.vocab_size, config.dim, dtype=dtype))
    text.register_buffer(
        "freqs_cis",
        precompute_freqs_cis(config.dim // (2 * config.n_heads), config.max_context),
        persistent=False,
    )

    return text
