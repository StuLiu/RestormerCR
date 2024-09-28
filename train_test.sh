

#python CloudRemoval/Datasets/split_trainval_alltrain.py
#
#python CloudRemoval/Datasets/copy_var.py
#
## single GPU training restormer-s, 71.69, for 11h
#python CloudRemoval/train.py -opt CloudRemoval/Options/0908_cr_restormer-s_128x128_1xb16_160k_alld_hybirdv1_prog2.yml
#
### 8 GPUs training restormer-l, 75.04, for 48h
##CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=4321 \
##  CloudRemoval/train.py \
##  -opt CloudRemoval/Options/0902_cr_restormer-l_128x128_8xb2_320k_alld_hybirdv1.yml \
##  --launcher pytorch
export CUDA_VISIBLE_DEVICES=0

name="0928_cr_restormer-s_128x128_1xb16_320k_alld_hybirdv1_rcsv4"

# training
python CloudRemoval/train.py \
  -opt CloudRemoval/Options/${name}.yml

# testing
python CloudRemoval/test.py -opt CloudRemoval/Options/${name}.yml \
  -ckpt experiments/${name}/models/net_g_latest.pth


cd ./submits/${name}

zip -r ../../${name}.zip results