# CRAFTv2 Mini Runs — Windows PowerShell
# Runs 4 variants in separate windows for parallel execution.
# Each variant uses mini_train/val manifests (~5600 train frames).
# Expected time per run: 30-60 min on RTX 4070 Super.
#
# Usage: from RGBT_FUSION folder:
#   powershell -ExecutionPolicy Bypass -File minirun\run_minirun.ps1

$ROOT     = "C:\Users\IIITK\Documents\John-Elsa\RGBT_FUSION"
$DATASET  = "C:\Users\IIITK\Documents\John-Elsa\Anti-UAV-RGBT"
$FRAMES   = "C:\Users\IIITK\Documents\John-Elsa\frames"
$TRAIN_JSON = "$ROOT\manifests\mini_train_manifest.json"
$VAL_JSON   = "$ROOT\manifests\mini_val_manifest.json"

$COMMON = "--dataset-root `"$DATASET`" --frames-root `"$FRAMES`" " +
          "--train-json `"$TRAIN_JSON`" --val-json `"$VAL_JSON`" " +
          "--use-convnext --epochs 30 --patience 10 --batch-size 4 --num-workers 0"

# 1. Full pipeline (both modalities, all stages)
Start-Process powershell -ArgumentList "-NoExit", "-Command",
    "cd '$ROOT'; conda activate elsa; python -m craft_v2.train $COMMON --output-dir craft_v2/runs/minirun_full"

Start-Sleep -Seconds 5

# 2. RGB only
Start-Process powershell -ArgumentList "-NoExit", "-Command",
    "cd '$ROOT'; conda activate elsa; python -m craft_v2.train $COMMON --modality rgb --output-dir craft_v2/runs/minirun_rgb"

Start-Sleep -Seconds 5

# 3. IR only
Start-Process powershell -ArgumentList "-NoExit", "-Command",
    "cd '$ROOT'; conda activate elsa; python -m craft_v2.train $COMMON --modality ir --output-dir craft_v2/runs/minirun_ir"

Start-Sleep -Seconds 5

# 4. No alignment
Start-Process powershell -ArgumentList "-NoExit", "-Command",
    "cd '$ROOT'; conda activate elsa; python -m craft_v2.train $COMMON --no-align --output-dir craft_v2/runs/minirun_noalign"

Write-Host "4 minirun windows launched. Results in: craft_v2\runs\minirun_*"
