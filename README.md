# From-Images-to-Insights-A-Deep-Radiomic-Analysis-Approach-to-Classify-Brain-Metastases-Origins-And-Tumor-Aggressiveness

# nnU-Net v2.6.2 Training & Prediction Commands
These are the commands used for training, validation, and testing with **nnU-Net v2.6.2** (Downloaded from the official repository: https://github.com/MIC-DKFZ/nnUNet).

---

## 1. Set Environment Paths (Adjust if your folders differ)

```powershell
# Paths
$env:nnUNet_raw = "D:\nnUNet_raw"
$env:nnUNet_preprocessed = "D:\nnUNet_preprocessed"
$env:nnUNet_results = "D:\nnUNet_results"

# Create folders if they don't exist
mkdir $env:nnUNet_preprocessed
mkdir $env:nnUNet_results

## 2. Preprocess
nnUNetv2_plan_and_preprocess -d 2 --verify_dataset_integrity


## 3. Training (Retrain) Commands

For Folder 0
nnUNetv2_train 2 3d_fullres 0 -tr nnUNetTrainer -p nnUNetPlans
For Folder 1
nnUNetv2_train 2 3d_fullres 1 -tr nnUNetTrainer -p nnUNetPlans
For Folder 2
nnUNetv2_train 2 3d_fullres 2 -tr nnUNetTrainer -p nnUNetPlans

##3. Testing on Kaggle Data
nnUNetv2_predict ^
-i "E:\nnUNet_raw\Dataset002_BrainMets_Semi\imagesTs" ^
-o "E:\nnUNet_predictions" ^
-d 1 -c 3d_fullres -f 2 ^
-tr nnUNetTrainer -p nnUNetPlans

#4.Restart Validation if Interrupted
nnUNetv2_train 2 3d_fullres 0 -tr nnUNetTrainer -p nnUNetPlans --val




