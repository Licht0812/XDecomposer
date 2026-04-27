import os
path = "/data/group/project1/Crystal/UniqCry/data"
output = "/data/home/zdhs0019/Projects/xrd_baselines/dara/filenames.txt"
try:
    if os.path.exists(path):
        files = os.listdir(path)
        with open(output, "w") as f:
            f.write(f"Count: {len(files)}\n")
            for name in files[:20]:
                f.write(f"{name}\n")
    else:
        with open(output, "w") as f:
            f.write(f"Path not found: {path}")
except Exception as e:
    with open(output, "w") as f:
        f.write(f"Error: {str(e)}")
