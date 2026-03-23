import os
import numpy as np
import argparse
from tqdm import tqdm
from ase.db import connect as ase_connect
from pymatgen.core import Structure
from pymatgen.io.cif import CifWriter

"""
1. 核心命名格式
不管是 CIF 文件还是 .xy 谱图文件，它们的文件名基础都是： {化学简式}_{空间群符号}.后缀
2. 详细组成部分
- 化学简式 ( formula ) ：使用 pymatgen 的 reduced_formula 。例如，如果原始结构是 Li4Mn2O8 ，它会被简化为 Li2MnO4 。
- 空间群符号 ( sg_symbol ) ：使用 pymatgen 自动分析得到的国际空间群符号。
  - 特殊处理 ：由于文件名中不能包含 / （某些空间群如 I4/mmm ），脚本会自动将 / 替换为 - ，变成 I4-mmm 。
  - 降级处理 ：如果对称性无法识别，则标记为 None 。
- 示例
  - IrC_F-43m.cif
  - BaWO4_I4_1.xy
3. 代码中的实现位置
- 空间群处理 ：见 第 42-45 行
- CIF 命名 ：见 第 54-56 行
- XY 命名 ：见 第 82-83 行
这种命名规则（ formula_spacegroup ）是 autoXRD 默认的标签解析方式。在运行 run_CNN.py 预测时，输出结果会直接显示这个文件名作为物相标签，让你一眼就能看出预测结果的成分和结构类型。
"""

def process_db(db_path, xrd_dir, output_dir, limit=None):
    """
    Process UniqCryLabeled.db and associated .npz files using pymatgen 
    to generate detailed CIF and .xy files.
    """
    # Create output directories
    cif_dir = os.path.join(output_dir, 'References')
    xy_dir = os.path.join(output_dir, 'Spectra')
    os.makedirs(cif_dir, exist_ok=True)
    os.makedirs(xy_dir, exist_ok=True)

    # Connect to ASE database to read the entries
    db = ase_connect(db_path)
    
    # Get total count for tqdm
    total = limit if limit else db.count()
    
    print(f"Starting processing {total} entries with pymatgen...")
    
    count = 0
    for row in tqdm(db.select(limit=limit), total=total):
        try:
            # 1. Get Atoms object and convert to pymatgen Structure
            atoms = row.toatoms()
            
            # Create pymatgen Structure
            struct = Structure(
                lattice=atoms.get_cell(),
                species=atoms.get_chemical_symbols(),
                coords=atoms.get_scaled_positions()
            )
            
            # Get space group information
            try:
                sg_symbol = struct.get_space_group_info()[0].replace('/', '-') # Avoid '/' in filenames
            except:
                sg_symbol = "None"

            # Metadata from DB
            kvp = row.key_value_pairs
            mpid = kvp.get('mpid', f"crystal_{row.id}").replace('.cif', '')
            label = kvp.get('Label', 'unknown')
            
            # 2. Write CIF using pymatgen (more detailed)
            # Filename format: formula_spacegroup.cif (matching autoXRD convention)
            formula = struct.composition.reduced_formula
            cif_filename = f"{formula}_{sg_symbol}.cif"
            cif_path = os.path.join(cif_dir, cif_filename)
            
            # Use CifWriter for detailed output
            writer = CifWriter(struct)
            writer.write_file(cif_path)
            
            # 3. Find and convert XRD data
            # Map mpid/id to correct npz file
            npz_candidates = [
                os.path.join(xrd_dir, f"{row.id}.npz"),
                os.path.join(xrd_dir, f"crystal_{row.id}.npz"),
                os.path.join(xrd_dir, f"{mpid}.npz")
            ]
            
            npz_path = None
            for p in npz_candidates:
                if os.path.exists(p):
                    npz_path = p
                    break
            
            if npz_path:
                data = np.load(npz_path, allow_pickle=True)
                if 'x' in data and 'y' in data:
                    x = data['x']
                    y = data['y']
                    # Spectra filename matches CIF name for easy identification
                    xy_filename = f"{formula}_{sg_symbol}.xy"
                    xy_path = os.path.join(xy_dir, xy_filename)
                    np.savetxt(xy_path, np.column_stack((x, y)), fmt='%.6f\t%.6f')
            
            count += 1
        except Exception as e:
            # print(f"Error processing row {row.id}: {e}")
            continue

    print(f"\nSuccessfully processed {count} entries.")
    print(f"Results saved in: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert DB and NPZ to detailed CIF and XY using pymatgen")
    parser.add_argument("--db", default="/data/group/project1/Crystal/UniqCryLabeled.db", help="Path to DB file")
    parser.add_argument("--xrd", default="/data/group/project1/Crystal/xrd_data", help="Path to XRD npz directory")
    parser.add_argument("--out", default="./converted_data_pymatgen", help="Output directory")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of entries to process")
    
    args = parser.parse_args()
    process_db(args.db, args.xrd, args.out, args.limit)
