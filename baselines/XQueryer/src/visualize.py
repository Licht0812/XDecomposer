import os
import json
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import re
import sqlite3
import pickle
import warnings
import matplotlib.transforms as mtransforms
from torch.utils.data import DataLoader
from scipy.optimize import linear_sum_assignment
from model.dataset import ASEDataset
from model.XQueryer import Xmodel
import random

warnings.filterwarnings("ignore")
try:
    from pymatgen.core import Structure
    from pymatgen.analysis.diffraction.xrd import XRDCalculator
    xrd_calc = XRDCalculator(wavelength='CuKa')
except ImportError:
    xrd_calc = None

def setup_plot_style():
    """Set plot defaults."""
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

def format_chem(chem):
    """Format a chemical formula."""
    if isinstance(chem, (int, float)):
        chem = str(chem)
    # Clean the name and add subscripts
    clean_name = str(chem).replace(".cif", "").split('_')[0].replace("_", "\\_")
    return "$\mathrm{" + re.sub(r'(\d+)', r'_{\1}', clean_name) + "}$"

def get_cif_path(mpid, cif_base_dir="data/cif_files"):
    """Return the CIF path for an mpid."""
    if not mpid or mpid == "Unknown":
        return None
    cif_path = os.path.join(cif_base_dir, f"{mpid}.cif")
    return cif_path if os.path.exists(cif_path) else None

def get_name_mapping():
    mapping_path = "data/id_to_ref_mapping_full.pkl"
    if os.path.exists(mapping_path):
        with open(mapping_path, 'rb') as f:
            return pickle.load(f)
    return {}

def visualize(model, dataset, device, entries_dict, save_dir='plots',
              db_path='data/UniqCryLabeled.db',
              cif_base_dir="data/cif_files"):
    os.makedirs(save_dir, exist_ok=True)
    setup_plot_style()
    model.eval()

    name_mapping = get_name_mapping()

    # Target number of phases to find
    target_phases = [2, 3, 4]
    found_samples = {} # Dictionary to store {num_phases: (index, batch_data)}

    print("Searching for samples with 2, 3, and 4 phases...")
    # Search through dataset to find one of each
    search_limit = 2000
    for i in range(min(len(dataset), search_limit)):
        batch = dataset[i]
        num_phases = (batch['gt_ratios'] > 1e-6).sum().item()
        if num_phases in target_phases and num_phases not in found_samples:
            found_samples[num_phases] = (i, batch)
            print(f"  Found {num_phases}-phase sample at index {i}")
        if len(found_samples) == len(target_phases):
            break

    two_theta = np.linspace(8, 82, 3500)

    for n_phase in target_phases:
        if n_phase not in found_samples:
            print(f"Warning: Could not find a sample with {n_phase} phases.")
            continue

        sample_idx, batch = found_samples[n_phase]

        # Prepare input
        intensity = batch['intensity'].unsqueeze(0).to(device)
        element = batch['element'].unsqueeze(0).to(device)
        gt_xrds = batch['gt_xrds'] # MAX_K, 3500
        gt_ratios = batch['gt_ratios'] # MAX_K
        gt_ids = batch['gt_ids'] # MAX_K

        with torch.no_grad():
            outputs = model(intensity, element)
            pred_xrds = outputs['xrds'][0].cpu() # num_slots, 3500
            pred_ratios = outputs['ratios'][0].cpu() # num_slots
            feat_logits = outputs['feat_logits'][0].cpu() # num_slots, num_classes

        # Hungarian Matching for visualization alignment
        valid_gt_mask = gt_ratios > 1e-6
        num_valid_gt = valid_gt_mask.sum().item()

        # Convert to float32 for cdist
        cost_matrix = torch.cdist(pred_xrds.float(), gt_xrds[valid_gt_mask].float(), p=2).numpy()
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        matched_slots = set(row_ind)

        # Prepare slot data for sorting
        num_slots = pred_xrds.shape[0]
        slot_data = []
        for slot_idx in range(num_slots):
            p_ratio = pred_ratios[slot_idx].item()
            slot_feat = outputs['features'][0, slot_idx]
            weight_bank = model.feat_cls_head.weight
            slot_feat_norm = F.normalize(slot_feat, p=2, dim=0)
            weight_bank_norm = F.normalize(weight_bank, p=2, dim=1)
            cos_sim = torch.matmul(weight_bank_norm, slot_feat_norm)
            retrieved_id = torch.argmax(cos_sim).item()
            max_sim = cos_sim[retrieved_id].item()
            retrieved_mpid = entries_dict.get(str(retrieved_id), {}).get("mpid", "Unknown")

            gt_info = None
            if slot_idx in matched_slots:
                match_pos = list(row_ind).index(slot_idx)
                gt_idx_in_valid = col_ind[match_pos]
                gt_idx = torch.where(valid_gt_mask)[0][gt_idx_in_valid]
                g_ratio = gt_ratios[gt_idx].item()
                g_id = gt_ids[gt_idx].item()
                gt_mpid = entries_dict.get(str(int(g_id)), {}).get("mpid", "Unknown")
                gt_info = {
                    'ratio': g_ratio,
                    'id': int(g_id),
                    'mpid': gt_mpid,
                    'xrd': gt_xrds[gt_idx].numpy(),
                    'cif_path': get_cif_path(gt_mpid, cif_base_dir)
                }

            slot_data.append({
                'slot_idx': slot_idx,
                'p_ratio': p_ratio,
                'p_id': retrieved_id,
                'p_mpid': retrieved_mpid,
                'p_sim': max_sim,
                'p_xrd': pred_xrds[slot_idx].numpy(),
                'gt': gt_info,
                'cif_path': get_cif_path(retrieved_mpid, cif_base_dir)
            })

        # Sort by GT ratio if available, otherwise by predicted ratio
        slot_data.sort(key=lambda x: x['gt']['ratio'] if x['gt'] else -1, reverse=True)

        # Calculate reconstructed mixture and residual
        reconstructed_y = np.zeros(3500)
        for data in slot_data:
            reconstructed_y += data['p_ratio'] * data['p_xrd']

        mixed_input_np = batch['intensity'].numpy()

        max_val = np.max(mixed_input_np) if np.max(mixed_input_np) > 0 else 1.0
        mixed_input_norm = mixed_input_np / max_val
        reconstructed_norm = reconstructed_y / max_val
        residual_norm = mixed_input_norm - reconstructed_norm

        fig = plt.figure(figsize=(16.0, 8.0))
        rows = num_slots
        gs = gridspec.GridSpec(rows + 1, 2, width_ratios=[1.25, 1.0], wspace=0.10, hspace=0.12)

        gs_left = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[:, 0], height_ratios=[rows, 1], hspace=0.0)
        ax_mix = fig.add_subplot(gs_left[0, 0])
        ax_res = fig.add_subplot(gs_left[1, 0], sharex=ax_mix)

        ax_mix.plot(two_theta, mixed_input_norm, color='#7F7F7F', lw=2.5, alpha=0.7, label='Ground Truth', zorder=2)
        ax_mix.plot(two_theta, reconstructed_norm, color='#D62728', lw=1.5, label='Reconstructed', zorder=3)

        fig.text(0.06, 0.5, "Intensity (a.u.)", va='center', ha='center', rotation='vertical', fontsize=15)

        ax_mix.legend(loc="upper right", frameon=True, edgecolor='black').get_frame().set_linewidth(0.5)
        ax_mix.set_title("Overall Mixture and Residual", fontsize=16)

        ax_mix.set(xlim=(8, 82), xticks=[10, 20, 30, 40, 50, 60, 70, 80],
                   ylim=(-0.02, 1.1), yticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax_mix.tick_params(axis='both', direction='in', top=False, right=False, labelbottom=False)
        for spine in ax_mix.spines.values():
            spine.set(linewidth=1.0, color='black')

        ax_res.plot(two_theta, residual_norm, color='#7F7F7F', lw=1)
        ax_res.axhline(0, color='black', lw=0.8, ls='--')
        ax_res.set_xlabel(r"$2\theta\ (^\circ)$", fontsize=15)
        ax_res.set_xticks([10, 20, 30, 40, 50, 60, 70, 80])
        ax_res.tick_params(axis='both', direction='in', top=False, right=False)
        for spine in ax_res.spines.values():
            spine.set(linewidth=1.0, color='black')

        res_max = max(np.abs(residual_norm).max(), 0.05) if np.abs(residual_norm).max() != 0 else 0.05
        ax_res.set(ylim=(-res_max*1.2, res_max*1.2), yticks=[-res_max, 0, res_max])
        ax_res.set_yticklabels([f"{-res_max:.2f}", "0", f"{res_max:.2f}"])

        colors_pred_palette = ['#1F77B4', '#2CA02C', '#9467BD', '#8C564B', '#17BECF']
        gs_mid = gridspec.GridSpecFromSubplotSpec(num_slots, 1, subplot_spec=gs[:, 1], hspace=0.10)
        comp_axes = []

        def plot_component(ax, slot_data, color_pred, is_last):

            gt_scaled = (slot_data['gt']['ratio'] * slot_data['gt']['xrd']) / max_val if slot_data['gt'] else np.zeros(3500)
            pred_scaled = (slot_data['p_ratio'] * slot_data['p_xrd']) / max_val
            gt_w = slot_data['gt']['ratio'] if slot_data['gt'] else 0
            pred_w = slot_data['p_ratio']

            cif_path = slot_data['gt']['cif_path'] if (slot_data['gt'] and slot_data['gt']['cif_path']) else slot_data['cif_path']
            if xrd_calc is not None and cif_path and os.path.exists(cif_path):
                try:
                    struct = Structure.from_file(cif_path)
                    pat = xrd_calc.get_pattern(struct, two_theta_range=(8, 82))
                    max_h = max(gt_scaled.max(), pred_scaled.max())
                    ax.vlines(pat.x, 0, (pat.y/100.0)*max_h,
                              color='#FF7F0E', lw=1.5, label='Theory', alpha=0.8, zorder=1)
                except Exception as e:
                    pass

            if slot_data['gt']:
                ax.plot(two_theta, gt_scaled + 0.05, color='#7F7F7F', lw=1.5,
                        label=f'GT Content: {gt_w*100:.1f}%', zorder=2)
            ax.plot(two_theta, pred_scaled + 0.05, color=color_pred, lw=1.5, ls='--',
                    label=f'Pred Content: {pred_w*100:.1f}%', zorder=3)

            gt_name = format_chem(name_mapping.get(slot_data['gt']['id'], "Unknown")) if slot_data['gt'] else "None"
            pred_name = format_chem(name_mapping.get(slot_data['p_id'], "Unknown"))
            title = f"GT: {gt_name}\nPred: {pred_name}"
            ax.text(0.02, 0.95, title, transform=ax.transAxes,
                    fontsize=11, va='top', ha='left', linespacing=1.2)

            ax.set(xlim=(8, 82), xticks=[10, 20, 30, 40, 50, 60, 70, 80],
                   ylim=(-0.05, 1.15), yticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
            for spine in ax.spines.values():
                spine.set(visible=True, linewidth=1.0, color='black')

            ax.legend(loc='upper right', frameon=False, fontsize=10)

            if not is_last:
                ax.tick_params(axis='y', left=True, right=False, labelleft=True, labelright=False, direction='in')
                ax.tick_params(axis='x', direction='in', top=False, labelbottom=False)
            else:
                ax.tick_params(axis='y', left=True, right=False, labelleft=True, labelright=False, direction='in')
                ax.set_xlabel(r"$2\theta\ (^\circ)$", fontsize=15)
                ax.tick_params(axis='x', direction='in', top=False)

        for i in range(num_slots):
            ax_c = fig.add_subplot(gs_mid[i, 0], sharex=(comp_axes[0] if i > 0 else None))
            is_last = (i == num_slots - 1)
            plot_component(ax_c, slot_data[i], colors_pred_palette[i % len(colors_pred_palette)], is_last)
            comp_axes.append(ax_c)

        plt.subplots_adjust(left=0.09, right=0.92, top=0.94, bottom=0.10, wspace=0.10)

        save_path = os.path.join(save_dir, f'vis_{n_phase}phase_sample.pdf')
        plt.savefig(save_path, format='pdf', bbox_inches='tight', dpi=600)
        print(f'Saved visualization to {save_path}')
        plt.close()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_path', default='checkpoints/xqueryer/latest.pth', type=str)
    parser.add_argument('--db_path', default='data/UniqCryLabeled.db', type=str)
    parser.add_argument('--npz_dir', default='data/UniqCry', type=str)
    parser.add_argument('--entries_dict', default='./src/entries_dict.json', type=str)
    parser.add_argument('--num_slots', default=4, type=int)
    parser.add_argument('--feature_dim', default=256, type=int)
    parser.add_argument('--num_classes', default=100315, type=int)
    parser.add_argument('--cif_base_dir', default="data/cif_files", type=str)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load entries dictionary
    with open(args.entries_dict, 'r') as f:
        entries_dict = json.load(f)
    print(f"Loaded crystal entries from {args.entries_dict}")

    model = Xmodel(embed_dim=3500, num_slots=args.num_slots, feature_dim=args.feature_dim, num_classes=args.num_classes)
    checkpoint = torch.load(args.load_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.to(device)

    # Use ASEDataset - note that it returns random mixtures on each call
    testset = ASEDataset(args.db_path, args.npz_dir, mode='test', encode_element=True, num_classes=args.num_classes)

    visualize(model, testset, device, entries_dict, cif_base_dir=args.cif_base_dir)
