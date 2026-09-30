"""
CRAFTv2 pipeline skeleton — single-frame, no temporal dimension.
Run directly to print shapes at each stage.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import deform_conv2d
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights


# ---------------------------------------------------------------------------
# Stage 1: SharedStem — ConvNeXt-Tiny backbone (stride-2+2 + stage-1 blocks, 96→64ch)
# ---------------------------------------------------------------------------

def _make_convnext_stem(out_ch=64):
    """
    Replace ConvNeXt stride-4 patchify with stride-2 + stride-2 to preserve
    small object features. ConvNeXt stage-1 blocks are kept for rich features.
    Output: 96ch @ H/4 × W/4  →  projected to out_ch
    """
    backbone = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
    # stride-2 conv replaces the original stride-4 patchify
    stride2a = nn.Sequential(
        nn.Conv2d(3, 96, kernel_size=3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(96),
        nn.GELU(),
    )
    stride2b = nn.Sequential(
        nn.Conv2d(96, 96, kernel_size=3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(96),
        nn.GELU(),
    )
    stem = nn.Sequential(
        stride2a,
        stride2b,
        backbone.features[1],   # stage 1: 3 ConvNeXt blocks, 96ch
    )
    proj = nn.Sequential(
        nn.Conv2d(96, out_ch, 1, bias=False),
        nn.BatchNorm2d(out_ch),
    )
    return stem, proj


class SharedStem(nn.Module):
    """
    Two stems: RGB (3ch) and IR (1ch).
    use_convnext=True  → ConvNeXt-Tiny pretrained backbone (run10+)
    use_convnext=False → lightweight 2-conv stem (run08/run09, backward compat)
    Output: f_rgb, f_ir  [B, out_ch, 80, 80]
    """

    def __init__(self, out_ch=64, use_convnext=True):
        super().__init__()
        self.use_convnext = use_convnext
        if use_convnext:
            rgb_stem, rgb_proj = _make_convnext_stem(out_ch)
            ir_stem,  ir_proj  = _make_convnext_stem(out_ch)
            self.rgb_stem = rgb_stem;  self.rgb_proj = rgb_proj
            self.ir_stem  = ir_stem;   self.ir_proj  = ir_proj
        else:
            self.rgb_stem = nn.Sequential(
                nn.Conv2d(3,      32, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(32), nn.GELU(),
                nn.Conv2d(32, out_ch, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_ch), nn.GELU(),
            )
            self.ir_stem = nn.Sequential(
                nn.Conv2d(1,      32, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(32), nn.GELU(),
                nn.Conv2d(32, out_ch, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_ch), nn.GELU(),
            )

    def forward(self, rgb, ir):
        if self.use_convnext:
            ir3 = ir.repeat(1, 3, 1, 1)
            return self.rgb_proj(self.rgb_stem(rgb)), self.ir_proj(self.ir_stem(ir3))
        else:
            return self.rgb_stem(rgb), self.ir_stem(ir)


# ---------------------------------------------------------------------------
# Stage 2: MSPAlign — bidirectional cross-attention, K/V saliency-pooled
# ---------------------------------------------------------------------------

class MSPAlign(nn.Module):
    """
    Multi-Scale Pooled Alignment.
    Q: full 80×80 (6400 tokens).  K/V: L2-energy pooled to 16×16 (256 tokens).
    """

    def __init__(self, ch=64, pool_size=16, num_heads=4):
        super().__init__()
        self.pool_size = pool_size
        self.num_heads = num_heads
        self.head_dim = ch // num_heads
        assert ch % num_heads == 0

        # Projections for RGB-attends-to-IR direction
        self.q_rgb = nn.Linear(ch, ch)
        self.k_ir  = nn.Linear(ch, ch)
        self.v_ir  = nn.Linear(ch, ch)
        self.out_rgb = nn.Linear(ch, ch)

        # Projections for IR-attends-to-RGB direction
        self.q_ir  = nn.Linear(ch, ch)
        self.k_rgb = nn.Linear(ch, ch)
        self.v_rgb = nn.Linear(ch, ch)
        self.out_ir = nn.Linear(ch, ch)

        self.norm_rgb = nn.LayerNorm(ch)
        self.norm_ir  = nn.LayerNorm(ch)

    def _saliency_pool(self, f):
        """Pool spatial dims to pool_size×pool_size using L2-energy weighting."""
        # f: [B, C, H, W]
        B, C, H, W = f.shape
        p = self.pool_size
        # Coarse grid via adaptive avg pool, then re-weight by L2 energy
        coarse = F.adaptive_avg_pool2d(f, (p, p))           # [B,C,p,p]
        energy = (f ** 2).sum(dim=1, keepdim=True)           # [B,1,H,W]
        energy_pool = F.adaptive_avg_pool2d(energy, (p, p))  # [B,1,p,p]
        weight = F.softmax(energy_pool.flatten(2), dim=-1)    # [B,1,p*p]
        weight = weight.view(B, 1, p, p)
        pooled = coarse * weight * (p * p)                   # re-normalise
        return pooled.flatten(2).transpose(1, 2)             # [B, p*p, C]

    def _mha(self, q_feat, kv_feat, q_proj, k_proj, v_proj, out_proj):
        """q_feat: [B, Nq, C], kv_feat: [B, Nkv, C] → [B, Nq, C]"""
        B, Nq, C = q_feat.shape
        Nkv = kv_feat.shape[1]
        H, D = self.num_heads, self.head_dim

        Q = q_proj(q_feat).view(B, Nq,  H, D).transpose(1, 2)   # [B,H,Nq,D]
        K = k_proj(kv_feat).view(B, Nkv, H, D).transpose(1, 2)
        V = v_proj(kv_feat).view(B, Nkv, H, D).transpose(1, 2)

        scale = D ** -0.5
        attn = (Q @ K.transpose(-2, -1)) * scale               # [B,H,Nq,Nkv]
        attn = attn.softmax(dim=-1)
        out = (attn @ V).transpose(1, 2).reshape(B, Nq, C)     # [B,Nq,C]
        return out_proj(out)

    def forward(self, f_rgb, f_ir, rgb_mask=None, ir_mask=None):
        # rgb_mask / ir_mask: optional [B,1,80,80], 1=real pixel, 0=letterbox padding
        B, C, H, W = f_rgb.shape

        # Zero out padding regions before saliency pool so constant-value
        # padding pixels don't inflate cross-attention similarity scores.
        if rgb_mask is not None:
            f_rgb = f_rgb * rgb_mask
        if ir_mask is not None:
            f_ir = f_ir * ir_mask

        # Flatten query tokens
        q_rgb_tokens = f_rgb.flatten(2).transpose(1, 2)  # [B, HW, C]
        q_ir_tokens  = f_ir.flatten(2).transpose(1, 2)

        # Saliency-pooled K/V tokens
        kv_ir  = self._saliency_pool(f_ir)   # [B, p*p, C]
        kv_rgb = self._saliency_pool(f_rgb)

        # Bidirectional cross-attention
        rgb_upd = self._mha(q_rgb_tokens, kv_ir,  self.q_rgb, self.k_ir,  self.v_ir,  self.out_rgb)
        ir_upd  = self._mha(q_ir_tokens,  kv_rgb, self.q_ir,  self.k_rgb, self.v_rgb, self.out_ir)

        # Residual + norm
        f_rgb_enh = self.norm_rgb(q_rgb_tokens + rgb_upd).transpose(1, 2).view(B, C, H, W)
        f_ir_enh  = self.norm_ir( q_ir_tokens  + ir_upd ).transpose(1, 2).view(B, C, H, W)

        return f_rgb_enh, f_ir_enh


# ---------------------------------------------------------------------------
# Stage 3: OffsetNet + DeformAlign
# ---------------------------------------------------------------------------

class DeformAlignBlock(nn.Module):
    """
    Predicts deformable offsets + modulation masks from concatenated features,
    then applies torchvision deform_conv2d to each branch.
    """

    def __init__(self, ch=64, groups=1):
        super().__init__()
        in_ch = ch * 2  # concat of rgb_enh + ir_enh
        self.offset_net = nn.Sequential(
            nn.Conv2d(in_ch, 128, 3, padding=1, bias=False),
            nn.GroupNorm(8, 128), nn.GELU(),
        )
        # Two heads, each for rgb and ir
        self.offset_rgb  = nn.Conv2d(128, 18, 3, padding=1)  # 2*9 offsets
        self.mask_rgb    = nn.Conv2d(128, 9,  3, padding=1)  # 9 weights
        self.offset_ir   = nn.Conv2d(128, 18, 3, padding=1)
        self.mask_ir     = nn.Conv2d(128, 9,  3, padding=1)

        # Zero-init offsets for identity start
        nn.init.zeros_(self.offset_rgb.weight); nn.init.zeros_(self.offset_rgb.bias)
        nn.init.zeros_(self.offset_ir.weight);  nn.init.zeros_(self.offset_ir.bias)

        self.weight_rgb = nn.Parameter(torch.randn(ch, ch, 3, 3) * 0.01)
        self.weight_ir  = nn.Parameter(torch.randn(ch, ch, 3, 3) * 0.01)
        self.norm_rgb   = nn.GroupNorm(8, ch)
        self.norm_ir    = nn.GroupNorm(8, ch)

    def forward(self, f_rgb_enh, f_ir_enh):
        x = torch.cat([f_rgb_enh, f_ir_enh], dim=1)  # [B,128,80,80]
        feat = self.offset_net(x)

        dp_rgb   = self.offset_rgb(feat)
        mask_rgb = self.mask_rgb(feat).sigmoid()
        dp_ir    = self.offset_ir(feat)
        mask_ir  = self.mask_ir(feat).sigmoid()

        offset_mag = (dp_rgb.abs().mean() + dp_ir.abs().mean()).item() / 2.0

        f_rgb_aligned = self.norm_rgb(
            deform_conv2d(f_rgb_enh, dp_rgb, self.weight_rgb, mask=mask_rgb, padding=1)
        )
        f_ir_aligned = self.norm_ir(
            deform_conv2d(f_ir_enh,  dp_ir,  self.weight_ir,  mask=mask_ir,  padding=1)
        )
        return f_rgb_aligned, f_ir_aligned, mask_rgb, mask_ir, offset_mag


# ---------------------------------------------------------------------------
# Stage 4: HGAFuse — hub-and-spoke GATv2 fusion
# ---------------------------------------------------------------------------

class GATv2Layer(nn.Module):
    """Single GATv2 layer over a 3-node graph: [rgb, ir, hub]."""

    def __init__(self, ch=64, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = ch // num_heads
        self.W  = nn.Linear(ch, ch, bias=False)
        self.a  = nn.Linear(2 * self.head_dim, 1, bias=False)
        self.out = nn.Linear(ch, ch)

    def forward(self, nodes):
        # nodes: list of [B, C, H, W], length N
        B, C, H, W = nodes[0].shape
        N = len(nodes)
        H_n, D = self.num_heads, self.head_dim

        tokens = torch.stack([n.flatten(2).transpose(1,2) for n in nodes], dim=1)
        # tokens: [B, N, HW, C]
        proj = self.W(tokens)                        # [B,N,HW,C]
        proj_h = proj.view(B, N, H*W, H_n, D)       # [B,N,HW,H,D]

        out_nodes = []
        for i in range(N):
            qi = proj_h[:, i]                        # [B,HW,H,D]
            agg = []
            for j in range(N):
                kj = proj_h[:, j]
                pair = torch.cat([qi, kj], dim=-1)   # [B,HW,H,2D]
                e = self.a(pair).squeeze(-1)          # [B,HW,H]
            att = torch.stack([
                self.a(torch.cat([qi, proj_h[:, j]], dim=-1)).squeeze(-1)
                for j in range(N)
            ], dim=-1)                               # [B,HW,H,N]
            att = att.softmax(dim=-1)
            # weighted sum over neighbours
            vals = proj_h.permute(0,2,3,1,4)         # [B,HW,H,N,D]
            node_out = (att.unsqueeze(-1) * vals).sum(dim=-2)  # [B,HW,H,D]
            node_out = node_out.reshape(B, H*W, C)
            out_nodes.append(self.out(node_out).transpose(1,2).view(B, C, H, W))

        return out_nodes


class HGAFuse(nn.Module):

    def __init__(self, ch=64, num_gat_layers=2):
        super().__init__()
        self.hub_proj = nn.Conv2d(ch * 2, ch, 1)
        self.gat = nn.ModuleList([GATv2Layer(ch) for _ in range(num_gat_layers)])
        # Spatial gate driven by the hub node: [B,C,H,W] → [B,1,H,W]
        self.gate_conv = nn.Conv2d(ch, 1, kernel_size=1)

    def forward(self, f_rgb_aligned, f_ir_aligned):
        hub = self.hub_proj(torch.cat([f_rgb_aligned, f_ir_aligned], dim=1))

        nodes = [f_rgb_aligned, f_ir_aligned, hub]
        for layer in self.gat:
            nodes = layer(nodes)
        f_rgb_out, f_ir_out, hub_out = nodes

        # Hub-conditioned spatial gate: α varies per location, not a global scalar
        alpha = torch.sigmoid(self.gate_conv(hub_out))  # [B,1,H,W]
        f_fused = alpha * f_rgb_out + (1 - alpha) * f_ir_out + hub_out
        return f_fused


# ---------------------------------------------------------------------------
# Stage 5: DetectionHead — anchor-free, two-scale (stride 2 + stride 4)
# ---------------------------------------------------------------------------

class _ScaleHead(nn.Module):
    """Single-scale branch: reg + cls convolutions."""
    def __init__(self, ch=64):
        super().__init__()
        self.reg = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False), nn.BatchNorm2d(ch), nn.GELU(),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False), nn.BatchNorm2d(ch), nn.GELU(),
            nn.Conv2d(ch, 4, 1),
        )
        self.cls = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False), nn.BatchNorm2d(ch), nn.GELU(),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False), nn.BatchNorm2d(ch), nn.GELU(),
            nn.Conv2d(ch, 1, 1),
        )

    def forward(self, f, stride):
        B, C, H, W = f.shape
        reg = self.reg(f)   # [B,4,H,W]
        cls = self.cls(f)   # [B,1,H,W]

        xs = (torch.arange(W, device=f.device) + 0.5) * stride
        ys = (torch.arange(H, device=f.device) + 0.5) * stride
        gy, gx = torch.meshgrid(ys, xs, indexing='ij')

        cx = gx + torch.tanh(reg[:, 0]) * stride
        cy = gy + torch.tanh(reg[:, 1]) * stride
        bw = torch.exp(reg[:, 2].clamp(-4, 4)) * stride
        bh = torch.exp(reg[:, 3].clamp(-4, 4)) * stride
        x  = cx - bw / 2
        y  = cy - bh / 2

        boxes  = torch.stack([x, y, bw, bh], dim=-1).flatten(1, 2)  # [B,N,4]
        scores = cls.squeeze(1).flatten(1).unsqueeze(-1)              # [B,N,1]
        return boxes, scores


class DetectionHead(nn.Module):
    """
    Two-scale anchor-free head.
    P2 (stride 2 → 160×160 = 25600 cells) for small drones.
    P3 (stride 4 →  80×80  =  6400 cells) for medium/large drones.
    Total: 32000 cells per image.
    P2 feature = bilinear upsample of fused + lateral conv.
    """

    def __init__(self, ch=64):
        super().__init__()
        self.lateral = nn.Conv2d(ch, ch, 1)   # adapt fused features for P2
        self.p2_head = _ScaleHead(ch)
        self.p3_head = _ScaleHead(ch)

    def forward(self, f_fused):
        # P3: 80×80 (stride 4)
        p3_boxes, p3_scores = self.p3_head(f_fused, stride=4)

        # P2: upsample to 160×160 (stride 2)
        p2 = F.interpolate(f_fused, scale_factor=2, mode='bilinear', align_corners=False)
        p2 = self.lateral(p2)
        p2_boxes, p2_scores = self.p2_head(p2, stride=2)

        boxes  = torch.cat([p2_boxes,  p3_boxes],  dim=1)  # [B, 32000, 4]
        scores = torch.cat([p2_scores, p3_scores], dim=1)  # [B, 32000, 1]
        return boxes, scores


# ---------------------------------------------------------------------------
# CRAFTv2 — top-level model
# ---------------------------------------------------------------------------

class CRAFTv2(nn.Module):

    def __init__(self, ch=64, use_convnext=True):
        super().__init__()
        self.stem        = SharedStem(out_ch=ch, use_convnext=use_convnext)
        self.msp_align   = MSPAlign(ch=ch)
        self.deform_align = DeformAlignBlock(ch=ch)
        self.hga_fuse    = HGAFuse(ch=ch)
        self.det_head    = DetectionHead(ch=ch)   # two-scale: P2+P3

    def forward(self, rgb, ir, rgb_mask=None, ir_mask=None, verbose=False):
        # rgb_mask / ir_mask: optional [B,1,80,80] padding masks from dataset
        def _p(name, *tensors):
            if verbose:
                for t in tensors:
                    print(f"  {name}: {tuple(t.shape)}")

        # Stage 1
        f_rgb, f_ir = self.stem(rgb, ir)
        _p("Stage1 SharedStem  f_rgb, f_ir", f_rgb, f_ir)

        # Stage 2 — masks bypass SharedStem, zero padding before K/V pool
        f_rgb_enh, f_ir_enh = self.msp_align(f_rgb, f_ir, rgb_mask=rgb_mask, ir_mask=ir_mask)
        _p("Stage2 MSPAlign    f_rgb_enh, f_ir_enh", f_rgb_enh, f_ir_enh)

        # Stage 3
        f_rgb_aligned, f_ir_aligned, mask_rgb, mask_ir, offset_mag = self.deform_align(f_rgb_enh, f_ir_enh)
        _p("Stage3 DeformAlign f_rgb_aligned, f_ir_aligned", f_rgb_aligned, f_ir_aligned)
        _p("Stage3             mask_rgb, mask_ir", mask_rgb, mask_ir)

        # Stage 4
        f_fused = self.hga_fuse(f_rgb_aligned, f_ir_aligned)
        _p("Stage4 HGAFuse     f_fused", f_fused)

        # Stage 5
        boxes, scores = self.det_head(f_fused)
        _p("Stage5 DetHead     boxes, scores", boxes, scores)

        return {
            "boxes":         boxes,          # [B, N, 4]  x,y,w,h in vis pixels
            "scores":        scores,         # [B, N, 1]
            "f_rgb_aligned": f_rgb_aligned,  # for L_ag / L_al
            "f_ir_aligned":  f_ir_aligned,
            "f_fused":       f_fused,        # for L_f
            "mask_rgb":      mask_rgb,       # for L_m
            "mask_ir":       mask_ir,
            "offset_mag":    offset_mag,     # monitoring only — not a loss term
        }


# ---------------------------------------------------------------------------
# Shape smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    B = 2
    rgb = torch.randn(B, 3, 320, 320)
    ir  = torch.randn(B, 1, 320, 320)

    model = CRAFTv2(ch=64)
    model.eval()

    print("=" * 60)
    print(f"Input  rgb : {tuple(rgb.shape)}")
    print(f"Input  ir  : {tuple(ir.shape)}")
    print("=" * 60)
    with torch.no_grad():
        out = model(rgb, ir, verbose=True)
    print("=" * 60)
    print("Output keys:", list(out.keys()))
    print(f"  boxes  : {tuple(out['boxes'].shape)}   (B, H*W, 4) x/y/w/h vis-px")
    print(f"  scores : {tuple(out['scores'].shape)}   (B, N, 1)")
    print("All shapes OK.")
