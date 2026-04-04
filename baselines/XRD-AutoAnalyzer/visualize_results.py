import os
import sqlite3
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import pickle
import random
import numpy as np
import tensorflow as tf
from autoXRD.spectrum_analysis import SpectrumAnalyzer, CustomDropout
from tensorflow.keras.utils import custom_object_scope
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import re
import torch
import torch.nn.functional as F
import warnings
import matplotlib.transforms as mtransforms
from scipy.optimize import linear_sum_assignment

# ========== 目标代码的样式依赖 ==========
warnings.filterwarnings("ignore")
try:
    from pymatgen.core import Structure
    from pymatgen.analysis.diffraction.xrd import XRDCalculator
    xrd_calc = XRDCalculator(wavelength='CuKa')
except ImportError:
    xrd_calc = None

NPZ_DIR = "/data/group/project1/Crystal/UniqCry/mp20-xrd_data/data"
REF_DIR = "Novel-Space/References"
MODEL_PATH = "Model_Reconstructor.h5"
MAPPING_PATH = "id_to_ref_mapping_full.pkl"
OUTPUT_DIR = "visualization_results_fixed"
TARGET_LENGTH = 3500
SEED = 7

# ========== 适配目标代码的CIF路径映射 ==========
def ref_to_cif_path(ref_name):
    if not ref_name or ref_name == "Unknown":
        return None
    cif_path = os.path.join(REF_DIR, f"{ref_name}.cif")
    return cif_path if os.path.exists(cif_path) else None

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_mapping():
    with open(MAPPING_PATH, 'rb') as f: return pickle.load(f)

def format_chem(chem):
    if isinstance(chem, (int, float)): 
        chem = str(chem)
    clean_name = str(chem).replace(".cif", "").split('_')[0].replace("_", "\\_")
    return "$\mathrm{" + re.sub(r'(\d+)', r'_{\1}', clean_name) + "}$"

def get_test_cids_aligned(mapping):
    all_ids = sorted(list(mapping.keys()))
    random.seed(SEED); np.random.seed(SEED)
    shuffled = all_ids.copy(); random.shuffle(shuffled)
    return set(shuffled[int(len(shuffled) * 0.9):])

def load_npz(label, s_idx):
    row_id = label + 1
    fpath = os.path.join(NPZ_DIR, f"crystal_{row_id}_sample_{s_idx:02d}.npz")
    try:
        data = np.load(fpath)
        y = data['y'] if 'y' in data else data['intensity']
        if len(y) != TARGET_LENGTH:
            y = np.interp(np.linspace(10, 80, TARGET_LENGTH), np.linspace(10, 80, len(y)), y)
        return y.astype(np.float32)
    except: return None

def generate_mixed_sample(test_cids, mapping, k):
    valid_test_cids = [cid for cid in test_cids if cid in mapping]
    selected_cids = random.sample(valid_test_cids, k) if len(valid_test_cids) >= k else random.choices(valid_test_cids, k=k)
    
    mixed_y = np.zeros(TARGET_LENGTH, dtype=np.float32)
    gt_components = []
    weights = np.random.dirichlet(np.ones(k))
    
    for i, cid in enumerate(selected_cids):
        y = load_npz(cid, random.randint(0, 19))
        if y is None:
            gt_components.append({'ref': 'Unknown', 'weight': 0.0, 'y': np.zeros(TARGET_LENGTH), 'cif_path': None, 'formula': 'Unknown'})
            continue
        
        y_norm = y / (np.max(y) + 1e-8)
        mixed_y += weights[i] * y_norm
        ref_name = mapping[cid][:-4]
        
        gt_components.append({
            'ref': ref_name, 
            'weight': weights[i], 
            'y': y_norm, # 存储归一化后的GT谱
            'cif_path': ref_to_cif_path(ref_name),
            'formula': ref_name.split('_')[0]
        })
    
    if np.max(mixed_y) > 0:
        mixed_y /= np.max(mixed_y)
    
    return mixed_y, gt_components

# ========== 重构后的组分绘图函数 ==========
def plot_component(ax, gt_info, pred_info, max_val, color_pred, is_last, x):
    gt_w, pred_w = 0.0, 0.0
    gt_name, pred_name = "None", "None"
    
    # --- GT 数据处理 ---
    if gt_info and gt_info['weight'] > 1e-6:
        gt_w = gt_info['weight']
        gt_name = format_chem(gt_info['formula'])
        gt_scaled = gt_info['y'] * gt_w # y已经是归一化的
        ax.plot(x, gt_scaled + 0.05, color='#7F7F7F', lw=1.5, label=f'GT Content: {gt_w*100:.1f}%', zorder=2)

    # --- Pred 数据处理 ---
    if pred_info and pred_info['scale'] > 0:
        pred_w = pred_info['weight'] # 使用归一化后的权重
        pred_name = format_chem(pred_info['ref'])
        # 预测谱需要根据其在混合物中的scale和整体max_val进行缩放
        pred_scaled = (pred_info['spectrum'] * pred_info['scale']) / max_val
        ax.plot(x, pred_scaled + 0.05, color=color_pred, lw=1.5, ls='--', label=f'Pred Content: {pred_w*100:.1f}%', zorder=3)

    # --- 理论峰 ---
    cif_path = (gt_info or {}).get('cif_path') or (pred_info or {}).get('cif_path')
    if xrd_calc and cif_path:
        try:
            struct = Structure.from_file(cif_path)
            pat = xrd_calc.get_pattern(struct, two_theta_range=(8, 82))
            
            # 决定理论峰的高度
            max_h = 0
            if gt_info and gt_info['weight'] > 1e-6:
                max_h = max(max_h, (gt_info['y'] * gt_info['weight']).max())
            if pred_info and pred_info['scale'] > 0:
                 max_h = max(max_h, ((pred_info['spectrum'] * pred_info['scale']) / max_val).max())
            max_h = max(max_h, 0.1) # 保证至少有高度

            ax.vlines(pat.x, 0, (pat.y/100.0)*max_h, color='#FF7F0E', lw=1.5, label='Theory', alpha=0.8, zorder=1)
        except Exception:
            pass

    # --- 文本和图例 ---
    title = f"GT: {gt_name}\nPred: {pred_name}"
    ax.text(0.02, 0.95, title, transform=ax.transAxes, fontsize=11, va='top', ha='left', linespacing=1.2)
    
    if gt_w > 1e-6 or pred_w > 1e-6:
        ax.legend(loc='upper right', frameon=False, fontsize=10)

    # --- 坐标轴样式 ---
    ax.set(xlim=(8, 82), xticks=[10, 20, 30, 40, 50, 60, 70, 80], ylim=(-0.05, 1.15), yticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    for spine in ax.spines.values(): spine.set(visible=True, linewidth=1.0, color='black')
    
    if not is_last:
        ax.tick_params(axis='y', left=True, right=False, labelleft=True, labelright=False, direction='in')
        ax.tick_params(axis='x', direction='in', top=False, labelbottom=False)
    else:
        ax.tick_params(axis='y', left=True, right=False, labelleft=True, labelright=False, direction='in')
        ax.set_xlabel(r"$2\theta\ (^\circ)$", fontsize=15)
        ax.tick_params(axis='x', direction='in', top=False)

def main():
    mapping = load_mapping()
    test_cids = get_test_cids_aligned(mapping)
    
    plt.rcParams.update({
        'font.family': 'serif', 'font.serif': ['Times New Roman'], 'font.size': 12,
        'mathtext.fontset': 'stix', 'pdf.fonttype': 42, 'ps.fonttype': 42,
        'axes.linewidth': 1.0, 'legend.frameon': False
    })

    for k in [2, 3, 4]:
        print(f"Testing {k}-phase mixture...")
        # 注意：mixed_y现在是[0,1]归一化的，gt_components里的y也是[0,1]归一化的
        mixed_y, gt_components = generate_mixed_sample(test_cids, mapping, k)
        
        # 混合谱在保存前要乘以100以匹配analyzer的预期
        max_raw_val = 100.0
        xy_path = f"temp_vis_{k}.xy"
        np.savetxt(xy_path, np.column_stack((np.linspace(10, 80, TARGET_LENGTH), mixed_y * max_raw_val)))
        
        analyzer = SpectrumAnalyzer(
            spectra_dir=".", spectrum_fname=xy_path, max_phases=4,
            cutoff_intensity=0.5, min_conf=10.0, reference_dir=REF_DIR, model_path=MODEL_PATH
        )
        mixtures, _, _, scalings, spectra = analyzer.suspected_mixtures
        os.remove(xy_path)
        
        pred_data = []
        if mixtures and mixtures[0]:
            for i, ref in enumerate(mixtures[0]):
                pred_data.append({
                    'ref': ref[:-4],
                    'scale': scalings[0][i],
                    'spectrum': spectra[0][i],
                    'cif_path': ref_to_cif_path(ref[:-4]),
                    'original_index': i
                })

        # --- 核心匹配与槽位分配逻辑 ---
        num_plot_slots = 4
        plot_slots = [{'gt': None, 'pred': None} for _ in range(num_plot_slots)]
        
        valid_gt_mask = np.array([c['weight'] for c in gt_components]) > 1e-6
        valid_gt_indices = np.where(valid_gt_mask)[0]
        
        unmatched_pred_indices = set(range(len(pred_data)))

        if len(valid_gt_indices) > 0 and len(pred_data) > 0:
            gt_xrds = np.array([gt_components[i]['y'] * gt_components[i]['weight'] for i in valid_gt_indices])
            pred_xrds = np.array([(p['spectrum'] * p['scale']) / max_raw_val for p in pred_data])

            cost_matrix = torch.cdist(torch.from_numpy(pred_xrds).float(), torch.from_numpy(gt_xrds).float(), p=2).numpy()
            pred_match_indices, gt_match_indices = linear_sum_assignment(cost_matrix)
            
            for pred_idx, gt_idx in zip(pred_match_indices, gt_match_indices):
                original_gt_idx = valid_gt_indices[gt_idx]
                plot_slots[original_gt_idx]['gt'] = gt_components[original_gt_idx]
                plot_slots[original_gt_idx]['pred'] = pred_data[pred_idx]
                if pred_idx in unmatched_pred_indices:
                    unmatched_pred_indices.remove(pred_idx)

        # 填充未匹配的GT
        for i in range(k):
            if not plot_slots[i]['gt']:
                plot_slots[i]['gt'] = gt_components[i]

        # 用未匹配的预测填充剩余的槽位
        next_slot = k
        for pred_idx in sorted(list(unmatched_pred_indices)):
            if next_slot < num_plot_slots:
                plot_slots[next_slot]['pred'] = pred_data[pred_idx]
                next_slot += 1
        
        # --- 归一化与重构 ---
        all_pred_scales = [p['scale'] for p in pred_data]
        sum_pred_scales = sum(all_pred_scales) if sum(all_pred_scales) > 0 else 1.0
        for slot in plot_slots:
            if slot['pred']:
                slot['pred']['weight'] = slot['pred']['scale'] / sum_pred_scales

        reconstructed_y = np.zeros(TARGET_LENGTH)
        if mixtures and mixtures[0]:
            for i in range(len(spectra[0])):
                reconstructed_y += spectra[0][i] * scalings[0][i]
        
        reconstructed_norm = reconstructed_y / max_raw_val
        residual_norm = mixed_y - reconstructed_norm
        two_theta = np.linspace(8, 82, TARGET_LENGTH)

        # --- 绘图 ---
        fig = plt.figure(figsize=(16.0, 8.0))
        gs = gridspec.GridSpec(num_plot_slots + 1, 2, width_ratios=[1.25, 1.0], wspace=0.10, hspace=0.12)
        
        gs_left = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[:, 0], height_ratios=[num_plot_slots, 1], hspace=0.0)
        ax_mix = fig.add_subplot(gs_left[0, 0])     
        ax_res = fig.add_subplot(gs_left[1, 0], sharex=ax_mix) 

        ax_mix.plot(two_theta, mixed_y, color='#7F7F7F', lw=2.5, alpha=0.7, label='Ground Truth', zorder=2)
        ax_mix.plot(two_theta, reconstructed_norm, color='#D62728', lw=1.5, label='Reconstructed', zorder=3)
        fig.text(0.06, 0.5, "Intensity (a.u.)", va='center', ha='center', rotation='vertical', fontsize=15)
        ax_mix.legend(loc="upper right", frameon=True, edgecolor='black').get_frame().set_linewidth(0.5)
        ax_mix.set_title("Overall Mixture and Residual", fontsize=16)
        ax_mix.set(xlim=(8, 82), xticks=range(10, 81, 10), ylim=(-0.02, 1.1), yticks=np.arange(0.0, 1.1, 0.2))
        ax_mix.tick_params(axis='both', direction='in', top=False, right=False, labelbottom=False)
        for spine in ax_mix.spines.values(): spine.set(linewidth=1.0, color='black')

        ax_res.plot(two_theta, residual_norm, color='#7F7F7F', lw=1)
        ax_res.axhline(0, color='black', lw=0.8, ls='--')
        ax_res.set_xlabel(r"$2\theta\ (^\circ)$", fontsize=15)
        ax_res.set_xticks(range(10, 81, 10))
        ax_res.tick_params(axis='both', direction='in', top=False, right=False)
        for spine in ax_res.spines.values(): spine.set(linewidth=1.0, color='black')
        res_max = max(np.abs(residual_norm).max(), 0.05)
        ax_res.set(ylim=(-res_max*1.2, res_max*1.2), yticks=[-res_max, 0, res_max])
        ax_res.set_yticklabels([f"{-res_max:.2f}", "0", f"{res_max:.2f}"])

        colors_pred_palette = ['#1F77B4', '#2CA02C', '#9467BD', '#8C564B', '#17BECF']
        gs_mid  = gridspec.GridSpecFromSubplotSpec(num_plot_slots, 1, subplot_spec=gs[:, 1], hspace=0.10)
        comp_axes = []
        
        for i in range(num_plot_slots):
            ax_c = fig.add_subplot(gs_mid[i, 0], sharex=(comp_axes[0] if i > 0 else None))
            is_last = (i == num_plot_slots - 1)
            
            plot_component(
                ax_c, plot_slots[i]['gt'], plot_slots[i]['pred'], max_raw_val,
                colors_pred_palette[i % len(colors_pred_palette)], is_last, two_theta
            )
            comp_axes.append(ax_c)

        plt.subplots_adjust(left=0.09, right=0.92, top=0.94, bottom=0.10, wspace=0.10)
        
        save_file = os.path.join(OUTPUT_DIR, f"vis_{k}_phases.pdf")
        plt.savefig(save_file, format='pdf', bbox_inches='tight', dpi=600)
        print(f"Saved: {save_file}")
        plt.close()

if __name__ == "__main__": 
    main()