import os
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor

# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------
IMAGES_DIR = r"E:\nnUNet_raw\Dataset002_BrainMets_Semi\imagesTs"
MASKS_DIR = r"E:\nnUNet_predictions\Kaggle_Submission_D002_Fold2"
OUTPUT_CSV = "radiomics_features_kaggle.csv"

# ---------------------------------------------------------
# RADIOMICS EXTRACTOR
# ---------------------------------------------------------
params = {
    "binWidth": 25,
    "interpolator": sitk.sitkBSpline,
    "normalize": False,
    "removeOutliers": 3,
    "correctMask": True,
}

extractor = featureextractor.RadiomicsFeatureExtractor(**params)
extractor.enableAllFeatures()

print("✔ Radiomics extractor ready.")

# ---------------------------------------------------------
# FIND IMAGE–MASK PAIRS
# ---------------------------------------------------------
def get_pairs():
    pairs = []

    for file in os.listdir(IMAGES_DIR):
        if not file.endswith("_0000.nii.gz"):
            continue

        # Extract case ID: "case_1000"
        case_id = file.replace("_0000.nii.gz", "")

        img_path = os.path.join(IMAGES_DIR, file)
        mask_path = os.path.join(MASKS_DIR, f"{case_id}.nii.gz")

        if os.path.exists(mask_path):
            pairs.append((case_id, img_path, mask_path))
        else:
            print(f"⚠ Mask missing for {case_id}")

    return pairs


pairs = get_pairs()
print(f"✔ Found {len(pairs)} valid image–mask pairs.")

if len(pairs) == 0:
    print("❌ ERROR: No pairs found. Check naming!")
    exit()

# ---------------------------------------------------------
# EXTRACT FEATURES
# ---------------------------------------------------------
rows = []

for case_id, img_path, mask_path in pairs:
    print(f"\n→ Extracting features for {case_id}")

    img = sitk.ReadImage(img_path)
    mask = sitk.ReadImage(mask_path)

    try:
        result = extractor.execute(img, mask)
    except Exception as e:
        print(f"❌ ERROR for {case_id}: {e}")
        continue

    row = {"case_id": case_id}

    for k, v in result.items():
        if isinstance(v, (int, float)):
            row[k] = v

    rows.append(row)

# ---------------------------------------------------------
# SAVE CSV
# ---------------------------------------------------------
df = pd.DataFrame(rows)
df.to_csv(OUTPUT_CSV, index=False)

print("\n✔ Radiomics extraction completed!")
print("✔ Saved to:", OUTPUT_CSV)
print("✔ Cases:", len(df))
print("✔ Features per case:", len(df.columns) - 1)
