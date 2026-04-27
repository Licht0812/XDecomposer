import os
import shutil
from collections import defaultdict
from pymatgen.core import Structure
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.io.cif import CifParser
import argparse

def fast_deduplicate(input_dir, output_dir):
    """
    Efficiently deduplicate CIF files by grouping by reduced formula first,
    then running StructureMatcher within each group.
    """
    if os.path.exists(output_dir):
        print(f"Error: Output directory {output_dir} already exists. Please remove it first.")
        return

    os.makedirs(output_dir)
    
    # 1. Group by composition (O(N))
    print(f"Scanning {input_dir} and grouping by composition...")
    groups = defaultdict(list)
    all_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.cif')]
    total = len(all_files)

    for i, f in enumerate(all_files):
        if i % 1000 == 0:
            print(f"  Processed {i}/{total} files...")
        try:
            # Use occupancy_tolerance to be consistent with autoXRD
            parser = CifParser(os.path.join(input_dir, f), occupancy_tolerance=1.25)
            struc = parser.parse_structures(primitive=False)[0]
            formula = struc.composition.reduced_formula
            groups[formula].append((f, struc))
        except Exception:
            continue

    print(f"Found {len(groups)} unique compositions among {total} files.")

    # 2. Match structures within groups (O(n_i^2) per group)
    print("Matching structures within groups...")
    matcher = StructureMatcher(scale=True, attempt_supercell=True, primitive_cell=False)
    unique_count = 0

    for i, (formula, members) in enumerate(groups.items()):
        if i % 100 == 0:
            print(f"  Processing group {i+1}/{len(groups)}: {formula}...")
        
        unique_in_group = []
        for filename, struc in members:
            is_unique = True
            for _, existing_struc in unique_in_group:
                if matcher.fit(struc, existing_struc):
                    is_unique = False
                    break
            if is_unique:
                unique_in_group.append((filename, struc))
        
        # Copy unique files to output directory
        for filename, _ in unique_in_group:
            shutil.copy(os.path.join(input_dir, filename), os.path.join(output_dir, filename))
            unique_count += 1

    print(f"Deduplication complete! Total unique files: {unique_count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fast CIF deduplication.')
    parser.add_argument('--input', default='All_CIFs', help='Input directory of CIFs')
    parser.add_argument('--output', default='Cleaned_CIFs', help='Output directory for unique CIFs')
    args = parser.parse_args()
    
    fast_deduplicate(args.input, args.output)
