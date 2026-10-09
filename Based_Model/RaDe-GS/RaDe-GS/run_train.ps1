# Script tu dong huan luyen RaDe-GS (Rasterizing Depth in Gaussian Splatting) voi bo du lieu DTU (scan105)
$source_path = "../../../Dataset/DTU/scan105"
$model_path = "./output/dtu/scan105"

Write-Host "Bat dau huan luyen RaDe-GS..." -ForegroundColor Green
Write-Host "Dataset path: $source_path" -ForegroundColor Yellow
Write-Host "Output path: $model_path" -ForegroundColor Yellow

python train.py `
    -s $source_path `
    -m $model_path `
    -r 2 `
    --use_decoupled_appearance 3 `
    --eval
