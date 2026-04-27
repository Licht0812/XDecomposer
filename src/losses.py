import torch
import torch.nn.functional as F
from itertools import permutations

def calculate_pairwise_sisdr(pred, target, eps=1e-8):
    """
    Calculate Pairwise SI-SDR matrix.
    Args:
        pred: [B, K, L]
        target: [B, K, L]
    Returns:
        pairwise_sisdr: [B, K, K] where (b, i, j) is SI-SDR(pred[b,i], target[b,j])
    """
    # Pred: [B, K, 1, L]
    # Target: [B, 1, K, L]
    pred_exp = pred.unsqueeze(2)
    target_exp = target.unsqueeze(1)
    
    # Dot product along L dimension
    # [B, K, K]
    dot = torch.sum(pred_exp * target_exp, dim=-1)
    
    # Target Energy: [B, 1, K]
    s_energy = torch.sum(target_exp ** 2, dim=-1) + eps
    
    # Check for zero-energy targets (padded sources)
    # [B, 1, K]
    target_is_silent = s_energy < 1e-6
    
    # Optimal scaling factor alpha
    # [B, K, K]
    # Add eps to denominator to avoid division by zero for silent targets
    alpha = dot / (s_energy + 1e-8)
    
    # e_target = alpha * target
    # [B, K, K, L] = [B, K, K, 1] * [B, 1, K, L]
    e_target = alpha.unsqueeze(-1) * target_exp
    
    # e_res = pred - e_target
    # [B, K, K, L] = [B, K, 1, L] - [B, K, K, L]
    e_res = pred_exp - e_target
    
    # SI-SDR = 10 * log10(||e_target||^2 / ||e_res||^2)
    # [B, K, K]
    signal_energy = torch.sum(e_target ** 2, dim=-1)
    noise_energy = torch.sum(e_res ** 2, dim=-1) + eps
    
    sisdr = 10 * torch.log10((signal_energy + 1e-8) / noise_energy)
    
    # Mask out silent targets: Set SI-SDR to 0.0 (or minimal impact)
    # target_is_silent: [B, 1, K] -> broadcast to [B, K, K]
    sisdr = torch.where(target_is_silent.expand_as(sisdr), torch.zeros_like(sisdr), sisdr)
    
    return sisdr


def _pairwise_geo_root_loss(pred_exp, target_exp, beta=0.5, eps=1e-8):
    """
    Physical geometry constraint in sqrt space (看清高角度弱峰).
    z = sqrt(y), L_Geo_Root = ||∇ẑ - ∇z||_1 + β·||∇²ẑ - ∇²z||_1
    pred_exp: [B, K, 1, L], target_exp: [B, 1, K, L]. Returns: [B, K, K]
    """
    pred_safe = torch.clamp(pred_exp, min=eps)
    target_safe = torch.clamp(target_exp, min=eps)
    z_pred = torch.sqrt(pred_safe)
    z_target = torch.sqrt(target_safe)

    # First order: ∇z (锁位置)
    grad_z_pred = z_pred[..., 1:] - z_pred[..., :-1]
    grad_z_target = z_target[..., 1:] - z_target[..., :-1]
    pairwise_grad1 = torch.abs(grad_z_pred - grad_z_target).mean(dim=-1)

    # Second order: ∇²z (锁峰宽)
    grad2_z_pred = z_pred[..., 2:] - 2 * z_pred[..., 1:-1] + z_pred[..., :-2]
    grad2_z_target = z_target[..., 2:] - 2 * z_target[..., 1:-1] + z_target[..., :-2]
    pairwise_grad2 = torch.abs(grad2_z_pred - grad2_z_target).mean(dim=-1)

    return pairwise_grad1 + beta * pairwise_grad2


def calculate_pit_loss(
    pred,
    target,
    mixture_ref=None,
    weights=None,
    alpha=10.0,
    lambda_sisdr=0.1,
    lambda_geo=0.1,
    beta=0.5,
    lambda_mix=0.0,
):
    """
    PIT Loss for separation:

    1. Dynamic Weighted L1 (L_Quant): |ŷ - y| × (1 + α·y)
    2. Scale-invariant shape (L_Shape): -SI-SDR(ŷ, y)
    3. Physical geometry (L_Geo_Root): z=√y, ||∇ẑ-∇z||_1 + β·||∇²ẑ-∇²z||_1

    Args:
        alpha: peak emphasis in L_Quant.
        lambda_sisdr: weight for L_Shape.
        lambda_geo: weight for L_Geo_Root.
        beta: weight for second-order term in L_Geo_Root.
    """
    B, K, L = pred.shape
    device = pred.device

    if target.shape[1] < K:
        diff = K - target.shape[1]
        target = torch.cat([target, torch.zeros(target.shape[0], diff, target.shape[2], device=device)], dim=1)

    pred_exp = pred.unsqueeze(2)
    target_exp = target.unsqueeze(1)

    # 1. L_Quant: Dynamic Weighted L1
    l1_diff = torch.abs(pred_exp - target_exp)
    focal_weights = 1.0 + alpha * target_exp
    pairwise_l1 = (l1_diff * focal_weights).mean(dim=-1)

    # 2. L_Shape: -SI-SDR
    if lambda_sisdr > 0:
        pairwise_sisdr = calculate_pairwise_sisdr(pred, target)
        pairwise_neg_sisdr = -pairwise_sisdr
    else:
        pairwise_neg_sisdr = torch.zeros_like(pairwise_l1)

    # 3. L_Geo_Root: sqrt-space geometry (一阶锁位置 + 二阶锁峰宽)
    if lambda_geo > 0:
        pairwise_geo = _pairwise_geo_root_loss(pred_exp, target_exp, beta=beta)
    else:
        pairwise_geo = torch.zeros_like(pairwise_l1)

    pairwise_losses = pairwise_l1 + lambda_sisdr * pairwise_neg_sisdr + lambda_geo * pairwise_geo
            
    # 5. Generate all permutations [P, K]
    perms = list(permutations(range(K)))
    P = len(perms)
    perms_tensor = torch.tensor(perms, device=device) # [P, K]
    
    # 6. Gather losses for all permutations
    losses_expanded = pairwise_losses.unsqueeze(1).expand(-1, P, -1, -1)
    perms_indices = perms_tensor.unsqueeze(0).unsqueeze(-1).expand(B, -1, -1, -1)
    gathered_losses = torch.gather(losses_expanded, 3, perms_indices).squeeze(-1)
    
    # Sum over K to get total loss for each permutation: [B, P]
    total_perm_losses = gathered_losses.sum(dim=2)
    
    # 7. Min over P: [B]
    min_loss, min_indices = torch.min(total_perm_losses, dim=1) # [B], [B]
    total_loss = min_loss.mean()
    
    # Get best permutations for activity labels
    best_perms = perms_tensor[min_indices] # [B, K]
    
    # 8. Mixture Consistency Loss
    if lambda_mix > 0 and mixture_ref is not None:
        if weights is not None:
            aligned_weights = torch.gather(weights, 1, best_perms)
            pred_mixture = (pred * aligned_weights.unsqueeze(-1)).sum(dim=1)
        else:
            pred_mixture = pred.sum(dim=1)
        total_loss = total_loss + lambda_mix * torch.abs(pred_mixture - mixture_ref).mean()

    return total_loss, best_perms

def calculate_pit_sisdr(pred, target):
    """
    Calculate the SI-SDR of the best permutation.
    Returns: Average SI-SDR [scalar]
    """
    B, K, L = pred.shape
    device = pred.device
    
    # [B, K, K]
    pairwise_sisdr = calculate_pairwise_sisdr(pred, target)
    
    # Permutations
    perms = list(permutations(range(K)))
    P = len(perms)
    perms_tensor = torch.tensor(perms, device=device) # [P, K]
    
    # Gather SI-SDR for all perms
    # Expand: [B, P, K, K] -> gather -> [B, P, K]
    sisdr_expanded = pairwise_sisdr.unsqueeze(1).expand(-1, P, -1, -1)
    perms_indices = perms_tensor.unsqueeze(0).unsqueeze(-1).expand(B, -1, -1, -1)
    gathered_sisdr = torch.gather(sisdr_expanded, 3, perms_indices).squeeze(-1)
    
    # Mean over K components: [B, P]
    # Note: SI-SDR is typically averaged over sources
    avg_perm_sisdr = gathered_sisdr.mean(dim=2)
    
    # Max over P: [B]
    best_sisdr, _ = torch.max(avg_perm_sisdr, dim=1)
    
    return best_sisdr.mean().item()

def compute_masked_mse_loss(pred, target, mask):
    """
    Compute MSE loss only on masked patches.
    Args:
        pred: [N, L, P] or [N, L]
        target: [N, L, P] or [N, L]
        mask: [N, L] or similar, 1 is masked (calculate loss), 0 is unmasked (ignore)
    """
    loss = (pred - target) ** 2
    
    # If pred is patches [N, L, P], average over P first
    if pred.dim() == 3:
        loss = loss.mean(dim=-1)  # [N, L]
        
    # Calculate loss only on masked patches
    # Avoid division by zero
    mask_sum = mask.sum()
    if mask_sum > 0:
        loss = (loss * mask).sum() / mask_sum
    else:
        loss = loss.sum() * 0.0
        
    return loss

def calculate_reconstruction_loss(pred, target, mask, alpha=10.0, lambda_cos=0.5, lambda_deriv=0.1):
    """
    Comprehensive Reconstruction Loss for MAE (Single Phase) - Optimized for Masked Patches Only
    """
    # Optimization: Flatten and select only masked patches
    # pred: [B, N, P] -> flatten [B*N, P]
    # mask: [B, N] -> flatten [B*N]
    
    B, N, P = pred.shape
    
    pred_flat = pred.reshape(-1, P)
    target_flat = target.reshape(-1, P)
    mask_flat = mask.reshape(-1) # Boolean mask where 1=masked
    
    # Boolean indexing to select only masked patches
    # We treat mask values > 0.5 as True
    bool_mask = mask_flat > 0.5
    
    # Check if we have any masked patches
    if not bool_mask.any():
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
        
    pred_masked = pred_flat[bool_mask]     # [M, P]
    target_masked = target_flat[bool_mask] # [M, P]
    
    # 2. Weighted L1 Loss (Focus on Peaks)
    l1_diff = torch.abs(pred_masked - target_masked)
    focal_weights = 1.0 + alpha * target_masked
    loss_l1 = (l1_diff * focal_weights).mean()
    
    # 3. Cosine Similarity Loss (Shape Constraint)
    # Now much faster as we only compute on M masked patches instead of B*N
    cos_sim = F.cosine_similarity(pred_masked, target_masked, dim=-1, eps=1e-8)
    loss_cos = (1.0 - cos_sim).mean()
    
    # 4. Derivative Loss (Slope Matching)
    pred_grad = pred_masked[:, 1:] - pred_masked[:, :-1]
    target_grad = target_masked[:, 1:] - target_masked[:, :-1]
    loss_deriv = torch.abs(pred_grad - target_grad).mean()
    
    # Total Loss
    total_loss = loss_l1 + lambda_cos * loss_cos + lambda_deriv * loss_deriv
    
    return total_loss
