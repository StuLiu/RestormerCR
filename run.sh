

python CloudRemoval/Datasets/split_trainval_alltrain_docker.py

python CloudRemoval/Datasets/copy_var_docker.py

export CUDA_VISIBLE_DEVICES=0

name="1008_cr_restormer-s_128x128_1xb16_640k_alld_hybirdv1_docker"

# training
python CloudRemoval/train_docker.py \
  -opt CloudRemoval/Options/${name}.yml

## testing
#python CloudRemoval/test.py -opt CloudRemoval/Options/${name}.yml \
#  -ckpt experiments/${name}/models/net_g_latest.pth
#
#
#cd ./submits/${name}
#
#zip -r /output/results.zip results
#
#cd ../..
#rm -rf submits
