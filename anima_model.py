"""
================================================================================
 anima_model.py  ::  Custom PyTorch implementation of the Anima architecture
================================================================================
 Matches the EXACT weight naming of `circlestone-labs/Anima` safetensors files
 (originally designed for ComfyUI). This is NOT a diffusers model — it's a
 from-scratch PyTorch implementation that loads the original weights directly.

 Key naming (after stripping `model.diffusion_model.` prefix):
   x_embedder.proj.1.weight                          : [2048, 68]
   t_embedder.1.linear_1.weight                      : [2048, 2048]
   t_embedder.1.linear_2.weight                      : [6144, 2048]
   t_embedding_norm.weight                           : [2048]
   blocks.X.adaln_modulation_self_attn.{1,2}.weight  : [256,2048], [6144,256]
   blocks.X.self_attn.{q,k,v}_proj.weight            : [2048, 2048]
   blocks.X.self_attn.output_proj.weight             : [2048, 2048]
   blocks.X.self_attn.{q,k}_norm.weight              : [128]
   blocks.X.cross_attn.{q}_proj.weight               : [2048, 2048]
   blocks.X.cross_attn.{k,v}_proj.weight             : [2048, 1024]
   blocks.X.cross_attn.output_proj.weight            : [2048, 2048]
   blocks.X.cross_attn.{q,k}_norm.weight             : [128]
   blocks.X.mlp.layer{1,2}.weight                    : [8192,2048], [2048,8192]
   final_layer.adaln_modulation.{1,2}.weight         : [256,2048], [4096,256]
   final_layer.linear.weight                         : [64, 2048]
   llm_adapter.embed.weight                          : [32128, 1024]
   llm_adapter.blocks.X.self_attn.{q,k,v,o}_proj     : [1024, 1024]
   llm_adapter.blocks.X.cross_attn.{q,k,v,o}_proj    : [1024, 1024]
   llm_adapter.blocks.X.{q,k}_norm.weight            : [64]
   llm_adapter.blocks.X.mlp.{0,2}.{weight,bias}      : [4096,1024]+[4096], [1024,4096]+[1024]
   llm_adapter.blocks.X.norm_{self,cross,mlp}.weight : [1024]
   llm_adapter.norm.weight                           : [1024]
   llm_adapter.out_proj.{weight,bias}                : [1024,1024], [1024]
================================================================================
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------
def rms_norm(x, weight, eps=1e-6):
    """RMSNorm over the last dim, then multiply by weight."""
    rms = x.pow(2).mean(dim=-1, keepdim=True).add(eps).rsqrt()
    return x * rms * weight


def timestep_embedding(t, dim=2048, max_period=10000):
    """Sinusoidal timestep embedding. t: [B] scalar → [B, dim]."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half
    )
    args = t.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb.to(weight_dtype_next)


# We'll set this global lazily so timestep_embedding can cast correctly.
weight_dtype_next = torch.float32


# ---------------------------------------------------------------------------
#  Main transformer attention (uses `output_proj` naming)
# ---------------------------------------------------------------------------
class MainAttention(nn.Module):
    """Multi-head attention with q/k RMSNorm and `output_proj` naming."""

    def __init__(self, q_dim, kv_dim, num_heads=16, head_dim=128):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        inner = num_heads * head_dim
        self.q_proj = nn.Linear(q_dim, inner, bias=False)
        self.k_proj = nn.Linear(kv_dim, inner, bias=False)
        self.v_proj = nn.Linear(kv_dim, inner, bias=False)
        self.output_proj = nn.Linear(inner, q_dim, bias=False)
        # q_norm and k_norm are parameters (not modules) of shape [head_dim]
        self.q_norm = nn.Parameter(torch.ones(head_dim))
        self.k_norm = nn.Parameter(torch.ones(head_dim))

    def forward(self, x, context=None):
        B, L, _ = x.shape
        ctx = context if context is not None else x
        Lctx = ctx.shape[1]

        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim)
        k = self.k_proj(ctx).view(B, Lctx, self.num_heads, self.head_dim)
        v = self.v_proj(ctx).view(B, Lctx, self.num_heads, self.head_dim)

        q = rms_norm(q, self.q_norm)
        k = rms_norm(k, self.k_norm)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).contiguous().view(B, L, self.num_heads * self.head_dim)
        return self.output_proj(out)


# ---------------------------------------------------------------------------
#  LLM adapter attention (uses `o_proj` naming)
# ---------------------------------------------------------------------------
class AdapterAttention(nn.Module):
    def __init__(self, q_dim, kv_dim, num_heads=16, head_dim=64):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        inner = num_heads * head_dim
        self.q_proj = nn.Linear(q_dim, inner, bias=False)
        self.k_proj = nn.Linear(kv_dim, inner, bias=False)
        self.v_proj = nn.Linear(kv_dim, inner, bias=False)
        self.o_proj = nn.Linear(inner, q_dim, bias=False)
        self.q_norm = nn.Parameter(torch.ones(head_dim))
        self.k_norm = nn.Parameter(torch.ones(head_dim))

    def forward(self, x, context=None):
        B, L, _ = x.shape
        ctx = context if context is not None else x
        Lctx = ctx.shape[1]
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim)
        k = self.k_proj(ctx).view(B, Lctx, self.num_heads, self.head_dim)
        v = self.v_proj(ctx).view(B, Lctx, self.num_heads, self.head_dim)
        q = rms_norm(q, self.q_norm)
        k = rms_norm(k, self.k_norm)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).contiguous().view(B, L, self.num_heads * self.head_dim)
        return self.o_proj(out)


# ---------------------------------------------------------------------------
#  Main transformer block (28 of these)
# ---------------------------------------------------------------------------
class MainBlock(nn.Module):
    """Block with self_attn + cross_attn + mlp, each with its own adaln_modulation."""

    def __init__(self, hidden=2048, num_heads=16, head_dim=128,
                 text_dim=1024, mlp_dim=8192, adaln_hidden=256):
        super().__init__()
        # adaln_modulation_self_attn is Sequential(SiLU, Linear, Linear)
        # so its state_dict keys are .1.weight and .2.weight
        self.adaln_modulation_self_attn = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden, adaln_hidden, bias=False),
            nn.Linear(adaln_hidden, 3 * hidden, bias=False),
        )
        self.self_attn = MainAttention(hidden, hidden, num_heads, head_dim)

        self.adaln_modulation_cross_attn = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden, adaln_hidden, bias=False),
            nn.Linear(adaln_hidden, 3 * hidden, bias=False),
        )
        self.cross_attn = MainAttention(hidden, text_dim, num_heads, head_dim)

        self.adaln_modulation_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden, adaln_hidden, bias=False),
            nn.Linear(adaln_hidden, 3 * hidden, bias=False),
        )

        # mlp is a Module with `layer1` and `layer2` Linear children (no bias)
        self.mlp = nn.Module()
        self.mlp.layer1 = nn.Linear(hidden, mlp_dim, bias=False)
        self.mlp.layer2 = nn.Linear(mlp_dim, hidden, bias=False)

    def forward(self, x, text_features, cond):
        """
        x: [B, L, hidden]
        text_features: [B, Ltext, text_dim]
        cond: [B, 3*hidden]  -- pre-computed by t_embedder, contains 3 chunks of hidden
                              one chunk per sublayer (self/cross/mlp)
        """
        cond_self, cond_cross, cond_mlp = cond.chunk(3, dim=-1)  # each [B, hidden]

        # Self-attn
        shift, scale, gate = self.adaln_modulation_self_attn(cond_self).chunk(3, dim=-1)
        h = rms_norm(x, scale + 1) + shift
        x = x + gate * self.self_attn(h)

        # Cross-attn
        shift, scale, gate = self.adaln_modulation_cross_attn(cond_cross).chunk(3, dim=-1)
        h = rms_norm(x, scale + 1) + shift
        x = x + gate * self.cross_attn(h, text_features)

        # MLP
        shift, scale, gate = self.adaln_modulation_mlp(cond_mlp).chunk(3, dim=-1)
        h = rms_norm(x, scale + 1) + shift
        h = self.mlp.layer2(F.silu(self.mlp.layer1(h)))
        x = x + gate * h

        return x


# ---------------------------------------------------------------------------
#  LLM adapter block (6 of these)
# ---------------------------------------------------------------------------
class AdapterBlock(nn.Module):
    def __init__(self, dim=1024, num_heads=16, head_dim=64, mlp_dim=4096):
        super().__init__()
        self.self_attn = AdapterAttention(dim, dim, num_heads, head_dim)
        self.cross_attn = AdapterAttention(dim, dim, num_heads, head_dim)

        # LayerNorms (weight-only, no bias) — but PyTorch LayerNorm has bias by default
        # We need to check if the safetensors has bias for these. From inspection, only
        # `norm_X.weight` keys exist (no `norm_X.bias`), so use LayerNorm(bias=False) via elementwise_affine
        # Actually PyTorch LayerNorm with elementwise_affine=True creates both weight and bias.
        # We need bias=False, but standard LayerNorm doesn't support that.
        # Use a custom LayerNorm without bias.
        self.norm_self_attn = LayerNormNoBias(dim)
        self.norm_cross_attn = LayerNormNoBias(dim)
        self.norm_mlp = LayerNormNoBias(dim)

        # mlp is Sequential(Linear, SiLU, Linear) — keys mlp.0.weight, mlp.0.bias, mlp.2.weight, mlp.2.bias
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim, bias=True),  # index 0
            nn.SiLU(),                           # index 1
            nn.Linear(mlp_dim, dim, bias=True),  # index 2
        )

    def forward(self, x, context):
        h = self.norm_self_attn(x)
        x = x + self.self_attn(h)

        h = self.norm_cross_attn(x)
        x = x + self.cross_attn(h, context)

        h = self.norm_mlp(x)
        h = self.mlp(h)
        x = x + h
        return x


class LayerNormNoBias(nn.Module):
    """LayerNorm with only weight (no bias). Matches `norm_X.weight` keys."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return F.layer_norm(x, (x.shape[-1],), self.weight, None, self.eps)


# ---------------------------------------------------------------------------
#  LLM Adapter (full)
# ---------------------------------------------------------------------------
class LLMAdapter(nn.Module):
    def __init__(self, vocab_size=32128, dim=1024, num_heads=16,
                 head_dim=64, mlp_dim=4096, num_layers=6):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([
            AdapterBlock(dim, num_heads, head_dim, mlp_dim)
            for _ in range(num_layers)
        ])
        self.norm = LayerNormNoBias(dim)
        self.out_proj = nn.Linear(dim, dim, bias=True)

    def forward(self, token_ids, qwen3_hidden):
        x = self.embed(token_ids)
        for block in self.blocks:
            x = block(x, qwen3_hidden)
        x = self.norm(x)
        return self.out_proj(x)


# ---------------------------------------------------------------------------
#  Timestep embedder (sits at t_embedder[1])
# ---------------------------------------------------------------------------
class TEmbedderMLP(nn.Module):
    """MLP that produces 3*hidden conditioning from sinusoidal timestep embedding."""

    def __init__(self, dim):
        super().__init__()
        self.linear_1 = nn.Linear(dim, dim, bias=False)
        self.linear_2 = nn.Linear(dim, 3 * dim, bias=False)

    def forward(self, t_emb):
        return self.linear_2(F.silu(self.linear_1(t_emb)))


# ---------------------------------------------------------------------------
#  Final layer
# ---------------------------------------------------------------------------
class FinalLayer(nn.Module):
    def __init__(self, hidden=2048, adaln_hidden=256, patch_out=64):
        super().__init__()
        # adaln_modulation outputs 2*hidden (shift, scale) — note: 2 chunks, not 3
        self.adaln_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden, adaln_hidden, bias=False),
            nn.Linear(adaln_hidden, 2 * hidden, bias=False),
        )
        self.linear = nn.Linear(hidden, patch_out, bias=False)

    def forward(self, x, cond):
        shift, scale = self.adaln_modulation(cond).chunk(2, dim=-1)
        h = rms_norm(x, scale + 1) + shift
        return self.linear(h)


# ---------------------------------------------------------------------------
#  Full Anima Transformer
# ---------------------------------------------------------------------------
class AnimaTransformer(nn.Module):
    def __init__(self, hidden=2048, num_heads=16, head_dim=128,
                 text_dim=1024, mlp_dim=8192, adaln_hidden=256,
                 num_layers=28, in_channels=16, patch_size=2,
                 llm_cfg=None):
        super().__init__()
        self.hidden = hidden
        self.patch_size = patch_size
        self.in_channels = in_channels

        # x_embedder.proj is Sequential(?, Linear(68, hidden))
        # The ? at index 0 is a patchify op (no params)
        x_in = in_channels * patch_size * patch_size + 4  # +4 for padding mask
        self.x_embedder = nn.Module()
        self.x_embedder.proj = nn.Sequential(
            nn.Identity(),                          # index 0 (placeholder for patchify)
            nn.Linear(x_in, hidden, bias=False),    # index 1
        )

        # t_embedder is Sequential(Identity, TEmbedderMLP)
        # Index 0 is Identity (placeholder for sinusoidal embedding step)
        # Index 1 is the MLP with linear_1 and linear_2
        self.t_embedder = nn.Sequential(
            nn.Identity(),
            TEmbedderMLP(hidden),
        )

        # t_embedding_norm: LayerNorm(hidden) — only weight, no bias
        self.t_embedding_norm = LayerNormNoBias(hidden)

        self.blocks = nn.ModuleList([
            MainBlock(hidden, num_heads, head_dim, text_dim, mlp_dim, adaln_hidden)
            for _ in range(num_layers)
        ])

        # final_layer.linear outputs in_channels * patch_size^2 = 64
        patch_out = in_channels * patch_size * patch_size
        self.final_layer = FinalLayer(hidden, adaln_hidden, patch_out)

        # LLM adapter
        ll = llm_cfg or {}
        self.llm_adapter = LLMAdapter(
            vocab_size=ll.get("vocab_size", 32128),
            dim=ll.get("dim", 1024),
            num_heads=ll.get("num_heads", 16),
            head_dim=ll.get("head_dim", 64),
            mlp_dim=ll.get("mlp_dim", 4096),
            num_layers=ll.get("num_layers", 6),
        )

    def forward(self, latents, timestep, token_ids, qwen3_hidden):
        """
        latents: [B, 16, H, W]
        timestep: [B]
        token_ids: [B, L]
        qwen3_hidden: [B, L, 1024]
        Returns: noise prediction [B, 16, H, W]
        """
        B, C, H, W = latents.shape
        p = self.patch_size

        # 1. Patchify
        patches = latents.view(B, C, H // p, p, W // p, p)
        patches = patches.permute(0, 2, 4, 1, 3, 5).contiguous()
        patches = patches.view(B, (H // p) * (W // p), C * p * p)

        # 2. Append 4-dim padding mask (zeros = no padding)
        pad_mask = torch.zeros(B, patches.shape[1], 4,
                               device=latents.device, dtype=latents.dtype)
        patches = torch.cat([patches, pad_mask], dim=-1)  # [B, L, 68]

        # 3. Project to hidden
        x = self.x_embedder.proj[1](patches)  # [B, L, hidden]

        # 4. Timestep → 3*hidden conditioning
        # t_embedder[0] is Identity (we pre-compute sinusoidal embedding externally)
        # t_embedder[1] is TEmbedderMLP
        t_emb = timestep_embedding(timestep, self.hidden)  # [B, hidden]
        t_emb = self.t_embedding_norm(t_emb)               # [B, hidden] normalized
        cond = self.t_embedder[1](t_emb)                   # [B, 3*hidden]

        # 5. LLM Adapter → text features
        text_features = self.llm_adapter(token_ids, qwen3_hidden)

        # 6. 28 transformer blocks
        for block in self.blocks:
            x = block(x, text_features, cond)

        # 7. Final layer (uses cond_self = first chunk of cond)
        cond_self = cond.chunk(3, dim=-1)[0]
        x = self.final_layer(x, cond_self)

        # 8. Unpatchify: [B, L, C*p*p] → [B, C, H, W]
        x = x.view(B, H // p, W // p, p, p, C)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        x = x.view(B, C, H, W)
        return x


# ---------------------------------------------------------------------------
#  Weight loading
# ---------------------------------------------------------------------------
def load_anima_transformer(weights_path, map_location="cpu", dtype=torch.float32):
    """Load the Anima transformer from a safetensors file."""
    print(f"Loading Anima weights from {weights_path}...")
    state_dict = load_file(weights_path, device=map_location)

    # Strip "model.diffusion_model." prefix
    new_sd = {}
    for k, v in state_dict.items():
        if k.startswith("model.diffusion_model."):
            new_sd[k[len("model.diffusion_model."):]] = v
        else:
            new_sd[k] = v

    model = AnimaTransformer()
    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    print(f"  Loaded {len(new_sd) - len(unexpected)}/{len(new_sd)} keys")
    print(f"  Missing keys (in model, not in weights): {len(missing)}")
    print(f"  Unexpected keys (in weights, not in model): {len(unexpected)}")
    if missing:
        print(f"  First 10 missing: {missing[:10]}")
    if unexpected:
        print(f"  First 10 unexpected: {unexpected[:10]}")

    if dtype is not None:
        model = model.to(dtype)
    model.eval()
    return model


if __name__ == "__main__":
    from huggingface_hub import hf_hub_download
    path = hf_hub_download("circlestone-labs/Anima",
                           "split_files/diffusion_models/anima-turbo-v1.0.safetensors")
    model = load_anima_transformer(path)
    print(f"\nModel loaded. Params: {sum(p.numel() for p in model.parameters())/1e9:.2f}B")
