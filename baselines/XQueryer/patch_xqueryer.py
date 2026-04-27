import os

# --- Patch train.py ---
train_file = '/data/home/zdhs0019/Projects/xrd_baselines/XQueryer/src/train.py'

if os.path.exists(train_file):
    with open(train_file, 'r') as f:
        content = f.read()

    # 1. Add calculate_quantitative_metrics function
    if 'def calculate_quantitative_metrics' not in content:
        old_str = "def calculate_sisdr(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:"
        new_str = """def calculate_quantitative_metrics(pred_patterns: torch.Tensor, target_patterns: torch.Tensor) -> float:
    \"\"\"Calculate Quantitative Mean Absolute Error (Quant MAE) in percentage points.\"\"\"
    if pred_patterns.dim() == 2:
        pred_patterns = pred_patterns.unsqueeze(0)
        target_patterns = target_patterns.unsqueeze(0)
    pred_intensities = torch.sum(torch.clamp(pred_patterns, min=0), dim=-1)
    target_intensities = torch.sum(target_patterns, dim=-1)
    
    pred_pct = pred_intensities / (pred_intensities.sum(dim=-1, keepdim=True) + 1e-8)
    target_pct = target_intensities / (target_intensities.sum(dim=-1, keepdim=True) + 1e-8)
    
    mae = torch.abs(pred_pct - target_pct).mean().item() * 100
    return mae

def calculate_sisdr(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:"""
        content = content.replace(old_str, new_str)

    # 2. Update metrics_sum initialization
    content = content.replace("'ratio_mae': 0,", "'quant_mae': 0,")

    # 3. Update evaluation logic in run_one_epoch
    old_eval_logic = """                        matched_logits = []
                        matched_targets = []
                        for r, c in zip(row_ind, col_ind):
                            gt_idx = torch.where(valid_gt_mask)[0][c]
                            total_loss += criterion_mse(pred_xrds[b, r], gt_xrds[b, gt_idx])
                            total_loss += criterion_mse(pred_ratios[b, r], gt_ratios[b, gt_idx])
                            total_loss += criterion_cls(feat_logits[b, r].unsqueeze(0), gt_ids[b, gt_idx].unsqueeze(0))
                            
                            metrics_sum['rwp'] += calculate_rwp(pred_xrds[b, r], gt_xrds[b, gt_idx])
                            metrics_sum['sisdr'] += calculate_sisdr(pred_xrds[b, r], gt_xrds[b, gt_idx])
                            metrics_sum['ratio_mae'] += torch.abs(pred_ratios[b, r] - gt_ratios[b, gt_idx]).item()
                            
                            matched_logits.append(feat_logits[b, r])
                            matched_targets.append(gt_ids[b, gt_idx])
                            metrics_sum['total_count'] += 1
                        
                        if matched_logits:
                            metrics_sum['top10_acc'] += get_id_acc_topk(torch.stack(matched_logits), torch.stack(matched_targets), k=10) * len(matched_logits)"""
                            
    new_eval_logic = """                        matched_logits = []
                        matched_targets = []
                        matched_pred_xrds = []
                        matched_gt_xrds = []
                        for r, c in zip(row_ind, col_ind):
                            gt_idx = torch.where(valid_gt_mask)[0][c]
                            total_loss += criterion_mse(pred_xrds[b, r], gt_xrds[b, gt_idx])
                            total_loss += criterion_mse(pred_ratios[b, r], gt_ratios[b, gt_idx])
                            total_loss += criterion_cls(feat_logits[b, r].unsqueeze(0), gt_ids[b, gt_idx].unsqueeze(0))
                            
                            metrics_sum['rwp'] += calculate_rwp(pred_xrds[b, r], gt_xrds[b, gt_idx])
                            metrics_sum['sisdr'] += calculate_sisdr(pred_xrds[b, r], gt_xrds[b, gt_idx])
                            
                            matched_pred_xrds.append(pred_xrds[b, r])
                            matched_gt_xrds.append(gt_xrds[b, gt_idx])
                            matched_logits.append(feat_logits[b, r])
                            matched_targets.append(gt_ids[b, gt_idx])
                            metrics_sum['total_count'] += 1
                            
                        if matched_pred_xrds:
                            q_mae = calculate_quantitative_metrics(torch.stack(matched_pred_xrds), torch.stack(matched_gt_xrds))
                            metrics_sum['quant_mae'] += q_mae * len(matched_pred_xrds)
                        
                        if matched_logits:
                            metrics_sum['top10_acc'] += get_id_acc_topk(torch.stack(matched_logits), torch.stack(matched_targets), k=10) * len(matched_logits)"""
    content = content.replace(old_eval_logic, new_eval_logic)

    # 4. Update log outputs
    content = content.replace("log.printlog(f\"Ratio MAE (Val):   {val_res['ratio_mae']*100:.2f}%\")", "log.printlog(f\"Quant MAE (Val):   {val_res['quant_mae']:.2f}%\")")
    content = content.replace("log.val_writer.add_scalar('ratio_mae', val_res['ratio_mae'], epoch)", "log.val_writer.add_scalar('quant_mae', val_res['quant_mae'], epoch)")

    with open(train_file, 'w') as f:
        f.write(content)
    print("Patched XQueryer/src/train.py successfully.")

# --- Patch infer.py ---
infer_file = '/data/home/zdhs0019/Projects/xrd_baselines/XQueryer/src/infer.py'

if os.path.exists(infer_file):
    with open(infer_file, 'r') as f:
        content = f.read()

    # Update quantitative metric calculation definition
    old_quant_def = """def calculate_quantitative_metrics(pred_patterns: torch.Tensor, target_patterns: torch.Tensor) -> Dict[str, Any]:
    \"\"\"Calculate Quantitative Mean Absolute Error (Quant MAE) in percentage points.\"\"\"
    pred_intensities = torch.sum(torch.clamp(pred_patterns, min=0), dim=-1)
    target_intensities = torch.sum(target_patterns, dim=-1)
    pred_pct = pred_intensities / (pred_intensities.sum(dim=-1, keepdim=True) + 1e-8)
    target_pct = target_intensities / (target_intensities.sum(dim=-1, keepdim=True) + 1e-8)
    mae = torch.abs(pred_pct - target_pct).mean().item() * 100
    return {'quant_mae': mae}"""
    
    new_quant_def = """def calculate_quantitative_metrics(pred_patterns: torch.Tensor, target_patterns: torch.Tensor) -> Dict[str, Any]:
    \"\"\"Calculate Quantitative Mean Absolute Error (Quant MAE) in percentage points.\"\"\"
    if pred_patterns.dim() == 2:
        pred_patterns = pred_patterns.unsqueeze(0)
        target_patterns = target_patterns.unsqueeze(0)
    pred_intensities = torch.sum(torch.clamp(pred_patterns, min=0), dim=-1)
    target_intensities = torch.sum(target_patterns, dim=-1)
    pred_pct = pred_intensities / (pred_intensities.sum(dim=-1, keepdim=True) + 1e-8)
    target_pct = target_intensities / (target_intensities.sum(dim=-1, keepdim=True) + 1e-8)
    mae = torch.abs(pred_pct - target_pct).mean().item() * 100
    return {'quant_mae': mae, 'pred_pct': pred_pct, 'target_pct': target_pct}"""
    content = content.replace(old_quant_def, new_quant_def)

    # 1. Update metrics dict
    content = content.replace("'ratio_mae_sum': 0,", "'quant_mae_sum': 0,")

    # 2. Fix the loop logic inside `run_one_epoch` where `ratio_err` is computed
    old_eval_loop = """                # Reconstruction Metrics
                rwp = calculate_rwp(pred_xrds[b, r], gt_xrds[b, gt_idx])
                pearson = calculate_pearson_correlation(pred_xrds[b, r], gt_xrds[b, gt_idx])
                sisdr = calculate_sisdr(pred_xrds[b, r], gt_xrds[b, gt_idx])
                sir, sar = calculate_sir_sar(pred_xrds[b, r], gt_xrds[b, gt_idx])
                delta_2theta = calculate_peak_shift_delta_2theta(pred_xrds[b, r], gt_xrds[b, gt_idx])
                fwhm_error = calculate_fwhm_error(pred_xrds[b, r], gt_xrds[b, gt_idx])
                intensity_consistency = calculate_intensity_ratio_consistency(pred_xrds[b, r], gt_xrds[b, gt_idx])
                ratio_err = torch.abs(pred_ratios[b, r] - gt_ratios[b, gt_idx]).item()
                
                metrics['rwp_sum'] += rwp
                metrics['pearson_sum'] += pearson
                metrics['sisdr_sum'] += sisdr
                metrics['sir_sum'] += sir
                metrics['sar_sum'] += sar
                metrics['delta_2theta_sum'] += delta_2theta
                metrics['fwhm_error_sum'] += fwhm_error
                metrics['intensity_consistency_sum'] += intensity_consistency
                metrics['ratio_mae_sum'] += ratio_err
                
                phase_data = {
                    "gt_id": target_id,
                    "gt_mpid": entries_dict.get(str(target_id), {}).get("mpid", "Unknown"),
                    "gt_ratio": float(gt_ratios[b, gt_idx].item()),
                    "pred_ratio": float(pred_ratios[b, r].item()),
                    "rwp": rwp,
                    "pearson": pearson,
                    "sisdr": sisdr,
                    "sir": sir,
                    "sar": sar,
                    "delta_2theta": delta_2theta,
                    "fwhm_error": fwhm_error,
                    "intensity_consistency": intensity_consistency,
                    "top10_preds": [
                        {
                            "id": int(idx),
                            "mpid": entries_dict.get(str(int(idx)), {}).get("mpid", "Unknown")
                        } for idx in top10_indices
                    ]
                }
                sample_results["phases"].append(phase_data)"""

    new_eval_loop = """                # Reconstruction Metrics
                rwp = calculate_rwp(pred_xrds[b, r], gt_xrds[b, gt_idx])
                pearson = calculate_pearson_correlation(pred_xrds[b, r], gt_xrds[b, gt_idx])
                sisdr = calculate_sisdr(pred_xrds[b, r], gt_xrds[b, gt_idx])
                sir, sar = calculate_sir_sar(pred_xrds[b, r], gt_xrds[b, gt_idx])
                delta_2theta = calculate_peak_shift_delta_2theta(pred_xrds[b, r], gt_xrds[b, gt_idx])
                fwhm_error = calculate_fwhm_error(pred_xrds[b, r], gt_xrds[b, gt_idx])
                intensity_consistency = calculate_intensity_ratio_consistency(pred_xrds[b, r], gt_xrds[b, gt_idx])
                
                metrics['rwp_sum'] += rwp
                metrics['pearson_sum'] += pearson
                metrics['sisdr_sum'] += sisdr
                metrics['sir_sum'] += sir
                metrics['sar_sum'] += sar
                metrics['delta_2theta_sum'] += delta_2theta
                metrics['fwhm_error_sum'] += fwhm_error
                metrics['intensity_consistency_sum'] += intensity_consistency
                
                phase_data = {
                    "gt_id": target_id,
                    "gt_mpid": entries_dict.get(str(target_id), {}).get("mpid", "Unknown"),
                    "gt_ratio": float(gt_ratios[b, gt_idx].item()),
                    "pred_ratio": float(pred_ratios[b, r].item()),
                    "rwp": rwp,
                    "pearson": pearson,
                    "sisdr": sisdr,
                    "sir": sir,
                    "sar": sar,
                    "delta_2theta": delta_2theta,
                    "fwhm_error": fwhm_error,
                    "intensity_consistency": intensity_consistency,
                    "top10_preds": [
                        {
                            "id": int(idx),
                            "mpid": entries_dict.get(str(int(idx)), {}).get("mpid", "Unknown")
                        } for idx in top10_indices
                    ]
                }
                sample_results["phases"].append(phase_data)
                
            # Compute Quant MAE for the sample
            matched_pred_xrds = []
            matched_gt_xrds = []
            for r, c in zip(row_ind, col_ind):
                gt_idx = torch.where(valid_gt_mask)[0][c]
                matched_pred_xrds.append(pred_xrds[b, r])
                matched_gt_xrds.append(gt_xrds[b, gt_idx])
            
            if matched_pred_xrds:
                quant_res = calculate_quantitative_metrics(torch.stack(matched_pred_xrds), torch.stack(matched_gt_xrds))
                q_mae = quant_res['quant_mae']
                metrics['quant_mae_sum'] += q_mae * len(matched_pred_xrds) # accumulate by number of phases
                
                # Update the JSON output to include the calculated quant_pct
                for i, phase_data in enumerate(sample_results["phases"]):
                    phase_data["pred_pct"] = float(quant_res['pred_pct'][0, i].item())
                    phase_data["target_pct"] = float(quant_res['target_pct'][0, i].item())"""
                    
    content = content.replace(old_eval_loop, new_eval_loop)

    # 3. Update prints
    content = content.replace("print(f\"Avg Ratio MAE:   {metrics['ratio_mae_sum']/metrics['total_phases']*100:.2f}%\")", "print(f\"Avg Quant MAE:   {metrics['quant_mae_sum']/metrics['total_samples']:.2f}%\")")

    with open(infer_file, 'w') as f:
        f.write(content)
    print("Patched XQueryer/src/infer.py successfully.")
