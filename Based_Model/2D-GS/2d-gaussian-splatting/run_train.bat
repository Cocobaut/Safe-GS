@echo off
set SOURCE_PATH=../../../Dataset/DTU/scan105
set MODEL_PATH=./output/scan105

echo Bat dau huan luyen 2D Gaussian Splatting...
echo Dataset: %SOURCE_PATH%
echo Output: %MODEL_PATH%

python train.py -s %SOURCE_PATH% -m %MODEL_PATH% --eval
