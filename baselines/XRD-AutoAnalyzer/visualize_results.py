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

# ========== 新增：目标代码的样式依赖 ==========
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
OUTPUT_DIR = "visualization_results"
TARGET_LENGTH = 3500
SEED = 7

# ========== 新增：适配目标代码的CIF路径映射（需根据实际路径调整） ==========
# 请根据你的REF_DIR和mapping文件，补充ref名称到cif文件的映射逻辑
def ref_to_cif_path(ref_name):
    """从ref名称生成对应的CIF文件路径（需根据实际存储路径调整）"""
    cif_path = os.path.join(REF_DIR, f"{ref_name}.cif")  # 示例逻辑，需匹配你的实际路径
    return cif_path if os.path.exists(cif_path) else None

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_mapping():
    with open(MAPPING_PATH, 'rb') as f: return pickle.load(f)

def format_chem(chem):
    # 统一为目标代码的化学式格式化逻辑
    return "$\mathrm{" + re.sub(r'(\d+)', r'_{\1}', chem) + "}$"

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
    selected_cids = random.sample(valid_test_cids, k)
    mixed_y = np.zeros(TARGET_LENGTH, dtype=np.float32)
    gt_components = []
    weights = np.random.dirichlet(np.ones(k))
    for i, cid in enumerate(selected_cids):
        y = load_npz(cid, random.randint(0, 19))
        if y is None: continue
        y /= (np.max(y) + 1e-8)
        mixed_y += weights[i] * y
        ref_name = mapping[cid][:-4]
        # 新增：记录CIF路径
        cif_path = ref_to_cif_path(ref_name)
        gt_components.append({
            'ref': ref_name, 
            'weight': weights[i], 
            'y': y,
            'cif_path': cif_path,
            'formula': ref_name.split('_')[0]  # 从ref名称提取化学式（适配目标代码）
        })
    if np.max(mixed_y) > 0:
        scale = 100.0 / np.max(mixed_y)
        mixed_y *= scale
        for c in gt_components: c['y'] *= (scale * c['weight'])
    return mixed_y, gt_components

# ========== 新增：复用目标代码的组分绘图函数 ==========
def plot_component(ax, gt_scaled, pred, comp, cif_path, color_pred, gt_w, pred_w, is_last, x):
    # 绘制理论XRD峰（目标代码风格）
    if xrd_calc is not None and cif_path and os.path.exists(cif_path):
        try:
            struct = Structure.from_file(cif_path)
            pat = xrd_calc.get_pattern(struct, two_theta_range=(8, 82))
            max_h = max(gt_scaled.max(), pred.max()) if (gt_scaled.max() + pred.max()) > 0 else 1.0
            ax.vlines(pat.x, 0, (pat.y/100.0)*max_h, color='#FF7F0E', lw=1.5, label='Theory', alpha=0.8, zorder=1)
        except Exception as e:
            pass
    
    # GT曲线（目标代码配色+偏移）
    ax.plot(x, gt_scaled + 0.05, color='#7F7F7F', lw=1.5, 
            label=f'GT Content: {gt_w*100:.1f}%', zorder=2)
    # 预测曲线（目标代码配色+虚线）
    ax.plot(x, pred + 0.05, color=color_pred, lw=1.5, ls='--', 
            label=f'Pred Content: {pred_w*100:.1f}%', zorder=3)
    
    # 化学式标题（目标代码风格）
    title = f"GT: {format_chem(comp['formula'])}\nPred: {format_chem(comp['formula'])}"
    ax.text(0.02, 0.95, title, transform=ax.transAxes, fontsize=11, va='top', ha='left', linespacing=1.2)
    
    # 坐标轴样式（目标代码风格）
    ax.set(xlim=(8, 82), xticks=[10, 20, 30, 40, 50, 60, 70, 80], ylim=(-0.05, 1.15), yticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    for spine in ax.spines.values(): 
        spine.set(visible=True, linewidth=1.0, color='black')
    
    ax.legend(loc='upper right', frameon=False, fontsize=10)

    # 刻度样式（目标代码的inward风格）
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
    
    # 目标代码的样式配置（NIPS风格）
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

    for k in [2, 3, 4]:
        print(f"Testing {k}-phase mixture...")
        mixed_y, gt_components = generate_mixed_sample(test_cids, mapping, k)
        xy_path = f"temp_vis_{k}.xy"
        np.savetxt(xy_path, np.column_stack((np.linspace(10, 80, TARGET_LENGTH), mixed_y)))
        
        analyzer = SpectrumAnalyzer(
            spectra_dir=".", spectrum_fname=xy_path, max_phases=4,
            cutoff_intensity=0.5, min_conf=10.0, reference_dir=REF_DIR, model_path=MODEL_PATH
        )
        mixtures, confidences, _, scalings, spectra = analyzer.suspected_mixtures
        os.remove(xy_path)
        
        if not mixtures[0]: 
            print(f"No phases identified for {k}-phase mixture."); continue
            
        # 原有数据处理逻辑（保留）
        gt_components.sort(key=lambda x: x['weight'], reverse=True)
        pred_refs = [r[:-4] for r in mixtures[0]]
        pred_scales = scalings[0]
        pred_spectra = spectra[0]
        pred_confs = confidences[0]

        pred_indices = sorted(range(len(pred_scales)), key=lambda i: pred_scales[i], reverse=True)
        pred_refs = [pred_refs[i] for i in pred_indices]
        pred_scales = [pred_scales[i] for i in pred_indices]
        pred_spectra = [pred_spectra[i] for i in pred_indices]
        pred_confs = [pred_confs[i] for i in pred_indices]
        
        # 归一化逻辑（适配目标代码）
        two_theta = np.linspace(10, 80, TARGET_LENGTH)
        reconstructed_y = np.zeros(TARGET_LENGTH)
        for i in range(len(pred_refs)):
            reconstructed_y += pred_spectra[i] * pred_scales[i]

        # 统一归一化到[0,1]（目标代码风格）
        max_val = np.max(mixed_y) if np.max(mixed_y) > 0 else 1.0
        mixed_y_norm = mixed_y / max_val
        reconstructed_y_norm = reconstructed_y / max_val
        residual_norm = mixed_y_norm - reconstructed_y_norm

        # ========== 核心修改：替换绘图逻辑为目标代码风格 ==========
        K = k
        fig = plt.figure(figsize=(16.0, 8.0))
        
        # 嵌套Gridspec（目标代码的2列布局）
        rows = K + 1
        gs = gridspec.GridSpec(rows, 2, width_ratios=[1.25, 1.0], wspace=0.10, hspace=0.12)
        gs_left = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[:, 0], height_ratios=[K, 1], hspace=0.0)
        gs_mid  = gridspec.GridSpecFromSubplotSpec(K, 1, subplot_spec=gs[:, 1], hspace=0.10)

        # 左列：混合曲线 + 残差
        ax_mix = fig.add_subplot(gs_left[0, 0])     
        ax_res = fig.add_subplot(gs_left[1, 0], sharex=ax_mix) 

        # 1. 混合曲线绘制（目标代码配色和样式）
        ax_mix.plot(two_theta, mixed_y_norm, color='#7F7F7F', lw=2.5, alpha=0.7, label='Ground Truth', zorder=2)
        ax_mix.plot(two_theta, reconstructed_y_norm, color='#D62728', lw=1.5, label='Reconstructed', zorder=3)
        
        # 左侧垂直Intensity标签（目标代码风格）
        fig.text(0.06, 0.5, "Intensity (a.u.)", va='center', ha='center', rotation='vertical', fontsize=15)
        
        ax_mix.legend(loc="upper right", frameon=True, edgecolor='black').get_frame().set_linewidth(0.5)
        ax_mix.set_title("Overall Mixture and Residual", fontsize=16)
        
        # 坐标轴范围和样式（目标代码风格）
        ax_mix.set(xlim=(8, 82), xticks=[10, 20, 30, 40, 50, 60, 70, 80], ylim=(-0.02, 1.1), yticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax_mix.tick_params(axis='both', direction='in', top=False, right=False, labelbottom=False)
        for spine in ax_mix.spines.values(): 
            spine.set(linewidth=1.0, color='black')

        # 2. 残差图绘制（目标代码风格）
        ax_res.plot(two_theta, residual_norm, color='#7F7F7F', lw=1)
        ax_res.axhline(0, color='black', lw=0.8, ls='--')
        ax_res.set_xlabel(r"$2\theta\ (^\circ)$", fontsize=15)
        ax_res.set_xticks([10, 20, 30, 40, 50, 60, 70, 80])
        ax_res.tick_params(axis='both', direction='in', top=False, right=False)
        for spine in ax_res.spines.values(): 
            spine.set(linewidth=1.0, color='black')
        
        # 残差图ylim计算（目标代码逻辑）
        res_max = max(np.abs(residual_norm).max(), 0.05) if np.abs(residual_norm).max() != 0 else 0.05
        ax_res.set(ylim=(-res_max*1.2, res_max*1.2), yticks=[-res_max, 0, res_max])
        ax_res.set_yticklabels([f"{-res_max:.2f}", "0", f"{res_max:.2f}"])

        # 3. 右列：组分分解（目标代码风格）
        colors_pred_palette = ['#1F77B4', '#2CA02C', '#9467BD', '#8C564B', '#17BECF']  # 目标代码配色板
        comp_axes = []
        
        # 处理预测权重（归一化为百分比）
        sum_pred = sum(pred_scales) if sum(pred_scales) > 0 else 1.0
        pred_weights = [s / sum_pred for s in pred_scales]
        
        # 遍历每个组分绘制
        for i in range(K):
            ax_c = fig.add_subplot(gs_mid[i, 0], sharex=(comp_axes[0] if i > 0 else None))
            is_last = (i == K - 1)
            
            # 获取GT和预测数据
            gt_comp = gt_components[i] if i < len(gt_components) else {'y': np.zeros(TARGET_LENGTH), 'weight': 0, 'formula': 'None', 'cif_path': None}
            gt_scaled = gt_comp['y'] / max_val if max_val > 0 else np.zeros(TARGET_LENGTH)
            pred_spec = pred_spectra[i] * pred_scales[i] / max_val if (i < len(pred_spectra) and max_val > 0) else np.zeros(TARGET_LENGTH)
            pred_w = pred_weights[i] if i < len(pred_weights) else 0.0
            cif_path = gt_comp.get('cif_path')
            
            # 调用组分绘制函数
            plot_component(
                ax_c, gt_scaled, pred_spec, gt_comp, cif_path,
                colors_pred_palette[i % len(colors_pred_palette)],
                gt_comp['weight'], pred_w, is_last, two_theta
            )
            comp_axes.append(ax_c)

        # 最终布局调整（目标代码风格）
        plt.subplots_adjust(left=0.09, right=0.92, top=0.94, bottom=0.10, wspace=0.10)
        
        # 保存为PDF（目标代码格式）
        save_file = os.path.join(OUTPUT_DIR, f"vis_{k}_phases.pdf")
        plt.savefig(save_file, format='pdf', bbox_inches='tight', dpi=600)
        print(f"Saved: {save_file}")
        plt.close()  # 关闭画布避免内存泄漏

if __name__ == "__main__": 
    main()