import os
import torch
import json
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import re
import numpy as np
import sqlite3
import pickle
import warnings
import matplotlib.transforms as mtransforms

warnings.filterwarnings("ignore")
try:
    from pymatgen.core import Structure
    from pymatgen.analysis.diffraction.xrd import XRDCalculator
    xrd_calc = XRDCalculator(wavelength='CuKa')
except ImportError:
    xrd_calc = None

from data_utils import get_dataloaders, OnlineMixingConfig
from model import get_model
from metrics_utils import align_predictions, calculate_cosine_similarity

def build_reference_library(test_loader, device):
    """
    Builds a reference library of pure XRD patterns for all IDs in the test set.
    """
    print("Building reference library for visualization...")
    library = {}
    for batch in test_loader:
        xrds = batch['single_xrds'] # [B, K, L]
        pids = batch['phase_ids']   # [B, K]
        weights = batch['weights']  # [B, K]
        
        B, K, L = xrds.shape
        for i in range(B):
            for k in range(K):
                pid = int(pids[i, k])
                w = float(weights[i, k])
                if pid != -1 and w > 1e-6 and pid not in library:
                    # Pure pattern = scaled_xrd / weight
                    pure = xrds[i, k] / w
                    library[pid] = pure.cpu()
                    
    ref_ids = sorted(list(library.keys()))
    ref_patterns = torch.stack([library[rid] for rid in ref_ids])
    return ref_patterns.to(device), ref_ids

def retrieve_id(pred_xrd, ref_lib, ref_ids):
    """
    Retrieves the most similar crystal ID from the reference library.
    pred_xrd: [1, L]
    """
    if pred_xrd.max() < 1e-4:
        return -1, 0.0
    
    # Calculate cosine similarity: [1, NumRefs]
    similarities = calculate_cosine_similarity(pred_xrd, ref_lib)
    max_sim, max_idx = torch.max(similarities, dim=1)
    
    return ref_ids[max_idx.item()], max_sim.item()

def format_chem(chem):
    if isinstance(chem, (int, float)): return str(chem)
    # Clean name: remove .cif, space group, etc.
    clean_name = str(chem).replace(".cif", "").split('_')[0]
    # Escape existing underscores and handle subscripts for numbers
    safe_chem = clean_name.replace("_", "\\_")
    return "$\\mathrm{" + re.sub(r'(\d+)', r'_{\1}', safe_chem) + "}$"

def get_name_mapping():
    mapping_path = "/data/home/zdhs0019/Projects/xrd_baselines/XRD-AutoAnalyzer/id_to_ref_mapping_full.pkl"
    if os.path.exists(mapping_path):
        with open(mapping_path, 'rb') as f:
            return pickle.load(f)
    return {}

# 新增：组分绘图函数（复用目标代码逻辑）
def plot_component(ax, x, gt_scaled, pred, gt_label, pred_label, gt_w, pred_w, color_pred, cif_path=None, is_last=False):
    # 绘制理论XRD峰（如果有cif文件）
    if xrd_calc is not None and cif_path and os.path.exists(cif_path):
        try:
            struct = Structure.from_file(cif_path)
            pat = xrd_calc.get_pattern(struct, two_theta_range=(8, 82))
            max_h = max(gt_scaled.max(), pred.max())
            ax.vlines(pat.x, 0, (pat.y/100.0)*max_h, color='#FF7F0E', lw=1.5, label='Theory', alpha=0.8, zorder=1)
        except Exception:
            pass
            
    # GT和Pred曲线（+0.05避免贴底）
    ax.plot(x, gt_scaled + 0.05, color='#7F7F7F', lw=1.5, label=f'GT Content: {gt_w*100:.1f}%', zorder=2)
    ax.plot(x, pred + 0.05, color=color_pred, lw=1.5, ls='--', label=f'Pred Content: {pred_w*100:.1f}%', zorder=3)
    
    # 化学式标题
    title = f"GT: {gt_label}\nPred: {pred_label}"
    ax.text(0.02, 0.95, title, transform=ax.transAxes, fontsize=11, va='top', ha='left', linespacing=1.2)
    
    # 坐标轴配置
    ax.set(xlim=(8, 82), xticks=[10, 20, 30, 40, 50, 60, 70, 80], ylim=(-0.05, 1.15), yticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    for spine in ax.spines.values():
        spine.set(visible=True, linewidth=1.0, color='black')
    
    # 图例
    ax.legend(loc='upper right', frameon=False, fontsize=10)

    # 刻度样式
    if not is_last:
        ax.tick_params(axis='y', left=True, right=False, labelleft=True, labelright=False, direction='in')
        ax.tick_params(axis='x', direction='in', top=False, labelbottom=False)
    else:
        ax.tick_params(axis='y', left=True, right=False, labelleft=True, labelright=False, direction='in')
        ax.set_xlabel(r"$2\theta\ (^\circ)$", fontsize=15)
        ax.tick_params(axis='x', direction='in', top=False)

def visualize_samples(model_path='best_separation_model.pth', entries_dict_path='../../XQueryer/src/entries_dict.json', db_path='/data/group/project1/Crystal/UniqCryLabeled.db'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    db_path = '/data/group/project1/Crystal/UniqCry/mp20-xrd_data'
    
    name_mapping = get_name_mapping()

    config = OnlineMixingConfig(
        MIN_K=2, MAX_K=4, MIN_WEIGHT=0.15, XRD_LENGTH=3500,
        AUGMENT=False, NOISE_LEVEL=0.01, SEED=42 # Change seed to get different samples
    )
    
    # 1. Load Data & Build Library
    _, _, test_loader = get_dataloaders(db_path, config, batch_size=32, num_workers=4)
    ref_lib, ref_ids = build_reference_library(test_loader, device)
    
    # 2. Load Model
    model = get_model("baseline", out_channels=config.MAX_K).to(device)
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 3. Select samples for 2, 3, 4 phases
    selected_batches = []
    found_ks = set()
    
    print("Searching for 2, 3, 4 phase samples in test set...")
    for batch in test_loader:
        num_phases = batch['num_phases']
        for i in range(len(num_phases)):
            k = int(num_phases[i])
            if k in [2, 3, 4] and k not in found_ks:
                # Store a single-sample batch
                single_batch = {k: v[i:i+1] for k, v in batch.items()}
                selected_batches.append(single_batch)
                found_ks.add(k)
        if len(found_ks) == 3:
            break
            
    # Sort by k: 2, 3, 4
    selected_batches.sort(key=lambda x: int(x['num_phases'][0]))
    
    # 4. Plotting - 完全重构为目标风格
    # NIPS Style 全局配置
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'font.size': 12,
        'mathtext.fontset': 'stix',
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'axes.linewidth': 1.0,
        'legend.frameon': False
    })
    
    # 现代配色板（避开混合图的红色）
    colors_pred_palette = ['#1F77B4', '#2CA02C', '#9467BD', '#8C564B', '#17BECF']
    
    for row_idx, batch in enumerate(selected_batches):
        k_actual = int(batch['num_phases'][0])
        inputs = batch['multiphase_xrd'].to(device)
        targets = batch['single_xrds'].to(device)
        true_pids = batch['phase_ids'].to(device)
        true_weights = batch['weights'].to(device)
        
        with torch.no_grad():
            outputs = model(inputs) # [1, 4, L]
            best_perms = align_predictions(outputs, targets)
            best_perms_expanded = best_perms.unsqueeze(-1).expand(-1, -1, 3500)
            aligned_outputs = torch.gather(outputs, 1, best_perms_expanded)

        # Get data for sorting
        sample_data = []
        mix_np = inputs[0, 0].cpu().numpy()
        two_theta = np.linspace(5, 85, 3500)  # X轴坐标
        
        for col_idx in range(4):
            true_id = int(true_pids[0, col_idx])
            true_w = float(true_weights[0, col_idx])
            pred_xrd = aligned_outputs[0, col_idx]
            pred_xrd_np = pred_xrd.cpu().numpy()
            pred_w = pred_xrd_np.sum() / mix_np.sum() if mix_np.sum() > 0 else 0
            ret_id, sim = retrieve_id(pred_xrd.unsqueeze(0), ref_lib, ref_ids)
            gold_xrd = targets[0, col_idx].cpu().numpy()
            
            sample_data.append({
                'true_id': true_id,
                'true_w': true_w,
                'pred_w': pred_w,
                'pred_xrd_np': pred_xrd_np,
                'gold_xrd': gold_xrd,
                'ret_id': ret_id,
                'sim': sim,
                'gt_label': format_chem(name_mapping.get(true_id, str(true_id))),
                'pred_label': format_chem(name_mapping.get(ret_id, str(ret_id)))
            })
        
        # Sort by true weight (descending)
        sample_data = [d for d in sample_data if d['true_w'] > 1e-6]  # 过滤空组分
        sample_data.sort(key=lambda x: x['true_w'], reverse=True)
        
        # Calculate reconstructed mixture and residual
        reconstructed_y = np.zeros(3500)
        for data in sample_data:
            reconstructed_y += data['pred_xrd_np']
        
        # Normalize everything to [0, 1] relative to the max of mix_np
        max_val = np.max(mix_np) if np.max(mix_np) > 0 else 1.0
        mix_np_norm = mix_np / max_val
        reconstructed_y_norm = reconstructed_y / max_val
        residual_norm = mix_np_norm - reconstructed_y_norm
        
        # -------------------------- 绘图布局重构 --------------------------
        # 整体尺寸：宽16，高8（适配K=2/3/4）
        fig = plt.figure(figsize=(16.0, 8.0))
        
        # 2列布局：左列（混合+残差）、右列（组分分解）
        rows = k_actual + 1
        gs = gridspec.GridSpec(rows, 2, width_ratios=[1.25, 1.0], wspace=0.10, hspace=0.12)
        
        # 嵌套Gridspec：左列拆分为混合图+残差图，右列拆分为K个组分图
        gs_left = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[:, 0], height_ratios=[k_actual, 1], hspace=0.0)
        gs_mid  = gridspec.GridSpecFromSubplotSpec(k_actual, 1, subplot_spec=gs[:, 1], hspace=0.10)
        
        # -------------------------- 左列：混合+残差 --------------------------
        ax_mix = fig.add_subplot(gs_left[0, 0])     
        ax_res = fig.add_subplot(gs_left[1, 0], sharex=ax_mix) 
        
        # 混合图
        ax_mix.plot(two_theta, mix_np_norm, color='#7F7F7F', lw=2.5, alpha=0.7, label='Ground Truth', zorder=2)
        ax_mix.plot(two_theta, reconstructed_y_norm, color='#D62728', lw=1.5, label='Reconstructed', zorder=3)
        
        # 左侧垂直Intensity标签
        fig.text(0.06, 0.5, "Intensity (a.u.)", va='center', ha='center', rotation='vertical', fontsize=15)
        
        # 混合图图例和标题
        ax_mix.legend(loc="upper right", frameon=True, edgecolor='black').get_frame().set_linewidth(0.5)
        ax_mix.set_title("Overall Mixture and Residual", fontsize=16)
        
        # 混合图坐标轴配置
        ax_mix.set(xlim=(8, 82), xticks=[10, 20, 30, 40, 50, 60, 70, 80], ylim=(-0.02, 1.1), yticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax_mix.tick_params(axis='both', direction='in', top=False, right=False, labelbottom=False)
        for spine in ax_mix.spines.values():
            spine.set(linewidth=1.0, color='black')
        
        # 残差图
        ax_res.plot(two_theta, residual_norm, color='#7F7F7F', lw=1)
        ax_res.axhline(0, color='black', lw=0.8, ls='--')
        ax_res.set_xlabel(r"$2\theta\ (^\circ)$", fontsize=15)
        ax_res.set_xticks([10, 20, 30, 40, 50, 60, 70, 80])
        ax_res.tick_params(axis='both', direction='in', top=False, right=False)
        for spine in ax_res.spines.values():
            spine.set(linewidth=1.0, color='black')
        
        # 残差图Y轴范围
        res_max = max(np.abs(residual_norm).max(), 0.05) if np.abs(residual_norm).max() != 0 else 0.05
        ax_res.set(ylim=(-res_max*1.2, res_max*1.2), yticks=[-res_max, 0, res_max])
        ax_res.set_yticklabels([f"{-res_max:.2f}", "0", f"{res_max:.2f}"])
        
        # -------------------------- 右列：组分分解 --------------------------
        comp_axes = []
        for i in range(k_actual):
            ax_c = fig.add_subplot(gs_mid[i, 0], sharex=(comp_axes[0] if i > 0 else None))
            data = sample_data[i]
            
            # 缩放GT和Pred曲线（和目标代码保持一致）
            gt_scaled = (data['true_w'] * data['gold_xrd']) / max_val
            pred_scaled = (data['pred_w'] * data['pred_xrd_np']) / max_val
            
            # 绘制单个组分
            plot_component(
                ax=ax_c,
                x=two_theta,
                gt_scaled=gt_scaled,
                pred=pred_scaled,
                gt_label=data['gt_label'],
                pred_label=data['pred_label'],
                gt_w=data['true_w'],
                pred_w=data['pred_w'],
                color_pred=colors_pred_palette[i % len(colors_pred_palette)],
                is_last=(i == k_actual - 1)
            )
            comp_axes.append(ax_c)
        
        # -------------------------- 保存配置 --------------------------
        plt.subplots_adjust(left=0.09, right=0.92, top=0.94, bottom=0.10, wspace=0.10)
        save_path = f'xrd_separation_results_{k_actual}phase.pdf'
        plt.savefig(save_path, format='pdf', bbox_inches='tight', dpi=600)
        print(f"Visualization saved to {save_path}")
        plt.close()

if __name__ == "__main__":
    visualize_samples()