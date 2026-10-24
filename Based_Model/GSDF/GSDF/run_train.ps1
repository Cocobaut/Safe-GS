$exp_dir = "./exp"
$config = "configs/dtu/scan105.yaml"
$gpu = "0"
$tag = "release"

python launch.py `
    --exp_dir $exp_dir `
    --config $config `
    --gpu $gpu `
    --train `
    --eval `
    tag=$tag