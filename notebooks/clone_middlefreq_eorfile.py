import os
import sys
import glob
import shutil

if len(sys.argv) < 2:
    print("Usage: python clone_middle.py <folder_path>")
    sys.exit(1)

folder_path = sys.argv[1]
search_pattern = os.path.join(folder_path, "fch*.skyh5")

# 1. Get a sorted list of files
files = sorted(glob.glob(search_pattern))

if not files:
    print(f"No fch*.skyh5 files found in {folder_path}")
    sys.exit(1)

# 2. Find the middle file
mid_index = len(files) // 2
mid_file = files[mid_index]
mid_filename = os.path.basename(mid_file)

print(f"Total files: {len(files)}")
print(f"Middle file chosen as source: {mid_filename}")

# 3. Overwrite others
for target_file in files:
    if target_file != mid_file:
        # shutil.copy2 overwrites the destination file's data 
        # but keeps the target_file name intact
        shutil.copy2(mid_file, target_file)

print("Cloning complete!")

# python clone_middlefreq_eorfile.py "/lustre/aoc/projects/hera/rchandra/H6C_Validation_Stats/validation-sim/sky_models/eor-grf-256"