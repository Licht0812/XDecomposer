import numpy as np
import os
from tqdm import tqdm
import json

npz_dir = '/data/group/project1/Crystal/UniqCry'

def check_corruption():
    print(f"Scanning NPZ files in {npz_dir}...")
    files = [f for f in os.listdir(npz_dir) if f.endswith('.npz')]
    
    corrupted = []
    all_zero = []
    all_nan = []
    valid_count = 0
    
    key_priority = ['y', 'intensity', 'xrd_pattern', 'pattern', 'xrd']
    
    for f in tqdm(files):
        path = os.path.join(npz_dir, f)
        try:
            with np.load(path) as data:
                found_key = False
                for key in key_priority:
                    if key in data:
                        arr = data[key]
                        found_key = True
                        
                        # 检查全 NaN
                        if np.isnan(arr).all():
                            all_nan.append(f)
                        # 检查全 0
                        elif np.max(np.abs(arr)) < 1e-9:
                            all_zero.append(f)
                        else:
                            valid_count += 1
                        break
                
                if not found_key:
                    corrupted.append(f"No valid key in {f}")
                    
        except Exception as e:
            corrupted.append(f"{f}: {str(e)}")

    print("\n--- Scan Result ---")
    print(f"Total files scanned: {len(files)}")
    print(f"Valid samples: {valid_count}")
    print(f"All NaN samples: {len(all_nan)}")
    print(f"All Zero samples: {len(all_zero)}")
    print(f"Corrupted/Invalid files: {len(corrupted)}")
    
    if all_nan: print(f"Sample NaN file: {all_nan[0]}")
    if all_zero: print(f"Sample Zero file: {all_zero[0]}")
    if corrupted: print(f"Sample Corrupted file: {corrupted[0]}")

    # 将损坏的文件列表保存，供 dataset.py 过滤
    with open('corrupted_files.json', 'w') as f:
        json.dump({
            "all_nan": all_nan,
            "all_zero": all_zero,
            "corrupted": corrupted
        }, f)

if __name__ == '__main__':
    check_corruption()
