@echo off
set exp_dir=./exp
set config=configs/dtu/scan105.yaml
set gpu=0
set tag=release

python launch.py --exp_dir %exp_dir% --config %config% --gpu %gpu% --train --eval tag=%tag%
