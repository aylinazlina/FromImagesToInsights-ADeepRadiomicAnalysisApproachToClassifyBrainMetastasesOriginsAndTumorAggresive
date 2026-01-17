# ============================================
# EXTRACT ALL PYRADIOMICS FEATURES (FAST)
# ============================================

import os
import csv
import SimpleITK as sitk
from radiomics import featureextractor
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# ---------------------------------------------------------
# PATHS (EDIT IF NEEDED)
# ---------------------------------------------------------

mask_dirs = [
    r"E:\nnUNet_results\Dataset002_BrainMets_Semi\nnUNetTrainer__nnUNetPlans__3d_fullres\fold_0\validation",
    r"E:\nnUNet_results\Dataset002_BrainMets_Semi\nnUNetTrainer__nnUNetPlans__3d_fullres\fold_1\validation",
    r"E:\nnUNet_results\Dataset002_BrainMets_Semi\nnUNetTrainer__nnUNetPlans__3d_fullres\fold_2\validation"
]

image_base_dir = r"E:\nnUNet_raw\Dataset002_BrainMets_Semi\imagesTr"

output_csv = "ALL_radiomics_features.csv"

# ---------------------------------------------------------
# PREPARE RADIOMICS EXTRACTOR (ALL FEATURES ENABLED)
# ---------------------------------------------------------

extractor = featureextractor.RadiomicsFeatureExtractor(
    binWidth=25,
    resampledPixelSpacing=None,
    geometryTolerance=1e-6
)

# Enable EVERYTHING
extractor.enableAllImageTypes()
extractor.enableAllFeatures()

# (diagnostics always included automatically)

# ---------------------------------------------------------
# FIND ALL IMAGE–MASK PAIRS
# ---------------------------------------------------------

def find_pairs():
    pairs = []

    for mask_dir in mask_dirs:
        for f in os.listdir(mask_dir):
            if f.endswith(".nii.gz"):

                case_id = f.replace(".nii.gz", "")   # case_0004

                mask_path = os.path.join(mask_dir, f)

                # nnUNet image name format: case_0004_0000.nii.gz
                image_name = case_id + "_0000.nii.gz"
                image_path = os.path.join(image_base_dir, image_name)

                if os.path.exists(image_path):
                    pairs.append((image_path, mask_path))

    return pairs


pairs = find_pairs()
print(f"Found {len(pairs)} valid image-mask pairs")


# ---------------------------------------------------------
# PROCESS ONE CASE
# ---------------------------------------------------------

def process_case(pair):
    img_path, mask_path = pair

    try:
        img = sitk.ReadImage(img_path)
        mask = sitk.ReadImage(mask_path)

        # Extract ALL features
        result = extractor.execute(img, mask)

        row = {"case": os.path.basename(img_path)}

        # Save every feature pyradiomics returns
        for key, value in result.items():
            row[key] = value

        return row

    except Exception as e:
        return {"case": os.path.basename(img_path), "error": str(e)}


# ---------------------------------------------------------
# PARALLEL PROCESSING
# ---------------------------------------------------------

rows = []

with ThreadPoolExecutor(max_workers=8) as ex:
    for r in tqdm(ex.map(process_case, pairs), total=len(pairs)):
        rows.append(r)

# ---------------------------------------------------------
# COLLECT ALL FEATURE NAMES (dynamic)
# ---------------------------------------------------------

all_keys = set()
for r in rows:
    all_keys.update(r.keys())

all_keys = ["case"] + sorted([k for k in all_keys if k != "case"])


# ---------------------------------------------------------
# SAVE CSV
# ---------------------------------------------------------

with open(output_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=all_keys)
    writer.writeheader()
    writer.writerows(rows)

print("✔ ALL radiomic features extracted!")
print("Saved to:", output_csv)
