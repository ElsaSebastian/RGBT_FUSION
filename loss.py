"""
CRAFTv2 loss function — five terms, all coefficients default to 1.0.

Returns a dict so callers can log individual terms:
    {'loss': total, 'l_ag': ..., 'l_al': ..., 'l_m': ..., 'l_f': ..., 'l_det': ...}
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# CIoU (from scratch)
# ---------------------------------------------------------------------------

def _xywh_to_xyxy(boxes):
    """[..., 4] x,y,w,h → x1,y1,x2,y2"""
    x1 = boxes[..., 0]
    y1 = boxes[..., 1]
    x2 = boxes[..., 0] + boxes[..., 2]
    y2 = boxes[..., 1] + boxes[..., 3]
    return torch.stack([x1, y1, x2, y2], dim=-1)


def ciou(pred_xywh, gt_xywh):
    """
    CIoU between paired boxes, both in x,y,w,h format.
    pred_xywh, gt_xywh : [N, 4]
    Returns CIoU values [N], range (-inf, 1].
    """
    pred = _xywh_to_xyxy(pred_xywh)   # [N,4]
    gt   = _xywh_to_xyxy(gt_xywh)

    # Intersection
    ix1 = torch.max(pred[:, 0], gt[:, 0])
    iy1 = torch.max(pred[:, 1], gt[:, 1])
    ix2 = torch.min(pred[:, 2], gt[:, 2])
    iy2 = torch.min(pred[:, 3], gt[:, 3])
    inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)

    # Union
    pred_area = pred_xywh[:, 2] * pred_xywh[:, 3]
    gt_area   = gt_xywh[:, 2]   * gt_xywh[:, 3]
    union = pred_area + gt_area - inter + 1e-7
    iou   = inter / union

    # Enclosing box diagonal²
    ex1 = torch.min(pred[:, 0], gt[:, 0])
    ey1 = torch.min(pred[:, 1], gt[:, 1])
    ex2 = torch.max(pred[:, 2], gt[:, 2])
    ey2 = torch.max(pred[:, 3], gt[:, 3])
    c2  = (ex2 - ex1).pow(2) + (ey2 - ey1).pow(2) + 1e-7

    # Centre distance²
    pred_cx = pred_xywh[:, 0] + pred_xywh[:, 2] / 2
    pred_cy = pred_xywh[:, 1] + pred_xywh[:, 3] / 2
    gt_cx   = gt_xywh[:, 0]   + gt_xywh[:, 2]   / 2
    gt_cy   = gt_xywh[:, 1]   + gt_xywh[:, 3]   / 2
    rho2    = (pred_cx - gt_cx).pow(2) + (pred_cy - gt_cy).pow(2)

    # Aspect ratio term v
    v = (4 / (math.pi ** 2)) * (
        torch.atan(gt_xywh[:, 2]   / (gt_xywh[:, 3]   + 1e-7)) -
        torch.atan(pred_xywh[:, 2] / (pred_xywh[:, 3] + 1e-7))
    ).pow(2)

    with torch.no_grad():
        alpha = v / (1 - iou + v + 1e-7)

    return iou - rho2 / c2 - alpha * v


# ---------------------------------------------------------------------------
# Loss terms
# ---------------------------------------------------------------------------

def _l_ag(f_rgb_aligned, f_ir_aligned, n_patches=400):
    """
    Patch alignment loss.
    Sample 400 random spatial locations; compute mean(1 - cosine_sim).
    Always active.
    """
    B, C, H, W = f_rgb_aligned.shape
    device = f_rgb_aligned.device

    h_idx = torch.randint(0, H, (n_patches,), device=device)
    w_idx = torch.randint(0, W, (n_patches,), device=device)

    # [B, n_patches, C]
    rgb_patches = f_rgb_aligned[:, :, h_idx, w_idx].permute(0, 2, 1)
    ir_patches  = f_ir_aligned[:, :, h_idx, w_idx].permute(0, 2, 1)

    cos_sim = F.cosine_similarity(rgb_patches, ir_patches, dim=-1)  # [B, n_patches]
    return (1 - cos_sim).mean()


def _l_al(f_rgb_aligned, f_ir_aligned, box_ir, exist):
    """
    Local alignment loss — GT box region in IR feature space.
    box_ir: [B,4] in letterboxed 320×320 space (x,y,w,h).
    Active only for exist=1 samples.
    """
    device = f_rgb_aligned.device
    mask = exist.bool()                     # [B]
    if not mask.any():
        return torch.tensor(0.0, device=device)

    f_rgb = f_rgb_aligned[mask]             # [N,C,H,W]
    f_ir  = f_ir_aligned[mask]
    boxes = box_ir[mask]                    # [N,4]

    stride = 4.0
    _, C, H, W = f_rgb.shape

    # Convert from letterbox-320 space to feature-map space (stride 4)
    bx = (boxes[:, 0] / stride).clamp(0, W - 1)
    by = (boxes[:, 1] / stride).clamp(0, H - 1)
    bw = (boxes[:, 2] / stride).clamp(1, W)
    bh = (boxes[:, 3] / stride).clamp(1, H)

    x1 = bx.long()
    y1 = by.long()
    x2 = (bx + bw).long().clamp(max=W)
    y2 = (by + bh).long().clamp(max=H)

    cos_sims = []
    for i in range(f_rgb.shape[0]):
        rx1, ry1, rx2, ry2 = x1[i], y1[i], x2[i], y2[i]
        # Ensure at least a 1×1 region
        if rx2 <= rx1:
            rx2 = rx1 + 1
        if ry2 <= ry1:
            ry2 = ry1 + 1
        rgb_vec = f_rgb[i, :, ry1:ry2, rx1:rx2].mean(dim=(-2, -1))  # [C]
        ir_vec  = f_ir[i,  :, ry1:ry2, rx1:rx2].mean(dim=(-2, -1))
        cos_sims.append(F.cosine_similarity(rgb_vec.unsqueeze(0), ir_vec.unsqueeze(0)))

    cos_sim = torch.cat(cos_sims)           # [N]
    return (1 - cos_sim).mean()


def _l_m(mask_rgb, mask_ir):
    """
    Modulation mask regularisation.
    Penalise spatially flat masks by negating spatial variance.
    Always active.
    """
    var_rgb = mask_rgb.var(dim=[2, 3])      # [B, 9]
    var_ir  = mask_ir.var(dim=[2, 3])
    return (-var_rgb.mean() - var_ir.mean()).clamp(min=-0.15)


def _l_f(f_fused, f_rgb_aligned, f_ir_aligned):
    """
    Fused feature alignment loss.
    f_fused should be close to the mean of the two aligned features.
    Always active.
    """
    target = 0.5 * (f_rgb_aligned + f_ir_aligned)

    # Mean-pool over spatial dims → [B, C]
    fused_mean  = f_fused.mean(dim=(-2, -1))
    target_mean = target.mean(dim=(-2, -1))

    cos_sim = F.cosine_similarity(fused_mean, target_mean, dim=-1)  # [B]
    return (1 - cos_sim).mean()


def _build_grid_centres(H, W, stride, device):
    """Return grid cell centres in letterbox-320 space, shape [H*W, 2] (cx, cy)."""
    xs = (torch.arange(W, device=device).float() + 0.5) * stride
    ys = (torch.arange(H, device=device).float() + 0.5) * stride
    gy, gx = torch.meshgrid(ys, xs, indexing='ij')
    # [H*W, 2]
    return torch.stack([gx.flatten(), gy.flatten()], dim=-1)


def _focal_loss(pred_logits, targets, alpha=0.25, gamma=2.0):
    """
    Focal loss for dense binary classification (RetinaNet formulation).
    Down-weights easy negatives so the 1 positive cell per image is not
    overwhelmed by 6399 negatives.
      FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
      alpha_t = alpha for positives, (1 - alpha) for negatives.
    """
    p       = torch.sigmoid(pred_logits)
    p_t     = torch.where(targets == 1, p, 1 - p)
    alpha_t = torch.where(targets == 1,
                          torch.full_like(p, alpha),
                          torch.full_like(p, 1 - alpha))
    fl = -alpha_t * (1 - p_t).pow(gamma) * torch.log(p_t.clamp(min=1e-8))
    return fl.mean()


def _build_multiscale_grid(device):
    """
    Build concatenated grid centres for two-scale head:
      P2: stride 2 → 160×160 = 25600 cells
      P3: stride 4 →  80×80  =  6400 cells
    Returns [32000, 2] (cx, cy) in letterbox-320 space.
    """
    p2 = _build_grid_centres(160, 160, 2, device)   # [25600, 2]
    p3 = _build_grid_centres(80,   80, 4, device)   # [ 6400, 2]
    return torch.cat([p2, p3], dim=0)               # [32000, 2]


def _l_det(pred_boxes, pred_scores, box_vis, exist, stride=4,
           focal_alpha=0.25, focal_gamma=2.0):
    """
    Detection loss = Focal confidence loss + CIoU box loss.
    Supports both single-scale (N=6400) and two-scale (N=32000) heads.

    pred_boxes  : [B, N, 4]  letterboxed x,y,w,h
    pred_scores : [B, N, 1]  raw logits
    box_vis     : [B, 4]     GT box letterboxed x,y,w,h
    exist       : [B]
    """
    B, N, _ = pred_boxes.shape
    device   = pred_boxes.device

    # Build grid centres matching head output
    if N == 32000:
        grid_centres = _build_multiscale_grid(device)
    else:
        H = W = int(N ** 0.5)
        grid_centres = _build_grid_centres(H, W, stride, device)

    # --- Score targets: all zeros, set closest-cell to 1 for exist=1 ---
    score_target = torch.zeros(B, N, 1, device=device)
    pos_indices  = []

    exist_mask = exist.bool()
    for b in range(B):
        if not exist_mask[b]:
            continue
        gt = box_vis[b]
        gt_cx = gt[0] + gt[2] / 2
        gt_cy = gt[1] + gt[3] / 2
        dists = (grid_centres[:, 0] - gt_cx).pow(2) + \
                (grid_centres[:, 1] - gt_cy).pow(2)
        cell  = dists.argmin().item()
        score_target[b, cell, 0] = 1.0
        pos_indices.append((b, cell))

    focal = _focal_loss(pred_scores, score_target,
                        alpha=focal_alpha, gamma=focal_gamma)

    if not pos_indices:
        return focal, torch.tensor(0.0, device=device)

    # --- CIoU on positive cells only ---
    pb_list, gt_list = [], []
    for b, cell in pos_indices:
        pb_list.append(pred_boxes[b, cell])
        gt_list.append(box_vis[b])

    pb = torch.stack(pb_list)
    gt = torch.stack(gt_list)

    ciou_vals = ciou(pb, gt)
    ciou_loss = (1 - ciou_vals).mean()

    return focal, ciou_loss


# ---------------------------------------------------------------------------
# CRAFTv2Loss
# ---------------------------------------------------------------------------

class CRAFTv2Loss(nn.Module):
    """
    All five loss terms. Coefficients default to 1.0.

    forward() expects the output dict from CRAFTv2.forward() plus
    the batch GT fields from the dataloader.
    """

    def __init__(
        self,
        w_ag:  float = 0.0,
        w_al:  float = 0.0,
        w_m:   float = 0.0,
        w_f:   float = 0.0,
        w_det: float = 4.0,
        n_patches:   int   = 400,
        stride:      int   = 4,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
    ):
        super().__init__()
        self.w_ag  = w_ag
        self.w_al  = w_al
        self.w_m   = w_m
        self.w_f   = w_f
        self.w_det = w_det
        self.n_patches   = n_patches
        self.stride      = stride
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

    def forward(self, model_out: dict, batch: dict) -> dict:
        """
        model_out keys: boxes, scores, f_rgb_aligned, f_ir_aligned,
                        f_fused, mask_rgb, mask_ir
        batch keys:     box_vis [B,4], box_ir [B,4], exist [B]
        """
        f_rgb = model_out["f_rgb_aligned"]
        f_ir  = model_out["f_ir_aligned"]
        exist = batch["exist"]                  # [B]

        l_ag = _l_ag(f_rgb, f_ir, self.n_patches)

        l_al = _l_al(f_rgb, f_ir, batch["box_ir"], exist)

        l_m  = _l_m(model_out["mask_rgb"], model_out["mask_ir"])

        l_f  = _l_f(model_out["f_fused"], f_rgb, f_ir)

        focal, ciou_loss = _l_det(
            model_out["boxes"],
            model_out["scores"],
            batch["box_vis"],
            exist,
            self.stride,
            focal_alpha=self.focal_alpha,
            focal_gamma=self.focal_gamma,
        )
        l_det = focal + ciou_loss

        total = (self.w_ag  * l_ag
               + self.w_al  * l_al
               + self.w_m   * l_m
               + self.w_f   * l_f
               + self.w_det * l_det)

        return {
            "loss":  total,
            "l_ag":  l_ag.detach(),
            "l_al":  l_al.detach(),
            "l_m":   l_m.detach(),
            "l_f":   l_f.detach(),
            "l_det":   l_det.detach(),
            "l_focal": focal.detach(),
            "l_ciou":  ciou_loss.detach(),
        }


# ---------------------------------------------------------------------------
# Smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)
    B = 4

    # Simulate model output
    model_out = {
        "boxes":         torch.randn(B, 6400, 4),
        "scores":        torch.randn(B, 6400, 1),
        "f_rgb_aligned": torch.randn(B, 64, 80, 80),
        "f_ir_aligned":  torch.randn(B, 64, 80, 80),
        "f_fused":       torch.randn(B, 64, 80, 80),
        "mask_rgb":      torch.sigmoid(torch.randn(B, 9, 80, 80)),
        "mask_ir":       torch.sigmoid(torch.randn(B, 9, 80, 80)),
    }

    # Two exist=1, two exist=0
    batch = {
        "box_vis": torch.tensor([
            [200.0, 150.0, 40.0, 30.0],
            [100.0,  80.0, 20.0, 15.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]),
        "box_ir": torch.tensor([
            [80.0, 60.0, 16.0, 12.0],
            [40.0, 32.0,  8.0,  6.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]),
        "exist": torch.tensor([1.0, 1.0, 0.0, 0.0]),
    }

    criterion = CRAFTv2Loss()
    losses = criterion(model_out, batch)

    print("=" * 50)
    for k, v in losses.items():
        print(f"  {k:8s}: {v.item():.6f}")
    print("=" * 50)
    print("All loss terms finite:", all(v.isfinite() for v in losses.values()))

    # Edge case: all exist=0
    batch_no_pos = {**batch, "exist": torch.zeros(B)}
    losses_no_pos = criterion(model_out, batch_no_pos)
    print("\nAll exist=0 batch:")
    for k, v in losses_no_pos.items():
        print(f"  {k:8s}: {v.item():.6f}")
    print("l_al and l_ciou are 0.0:",
          losses_no_pos["l_al"].item() == 0.0 and losses_no_pos["l_ciou"].item() == 0.0)
