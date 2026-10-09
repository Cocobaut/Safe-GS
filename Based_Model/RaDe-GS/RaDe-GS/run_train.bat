@echo off
set SOURCE_PATH=../../../Dataset/DTU/scan105
set MODEL_PATH=./output/dtu/scan105

echo Bat dau huan luyen RaDe-GS...
echo Dataset: %SOURCE_PATH%
echo Output: %MODEL_PATH%

python train.py -s %SOURCE_PATH% -m %MODEL_PATH% -r 2 --use_decoupled_appearance 3 --eval
