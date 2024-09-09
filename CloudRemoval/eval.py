'''
@Project : Restormer 
@File    : test2.py
@IDE     : PyCharm 
@Author  : Wang Liu
@Date    : 2024/9/2 上午11:24
@e-mail  : 1183862787@qq.com
'''
import logging
import os.path

import torch
import argparse
from os import path as osp

from basicsr.data import create_dataloader, create_dataset
from basicsr.models import create_model
# from basicsr.train import parse_options
from basicsr.utils import (MessageLogger, check_resume, get_env_info,
                           get_root_logger, get_time_str, init_tb_logger,
                           init_wandb_logger, make_exp_dirs, mkdir_and_rename,
                           set_random_seed)

from basicsr.utils.dist_util import get_dist_info, init_dist
from basicsr.utils.options import dict2str, parse


def parse_options(is_train=False):
    parser = argparse.ArgumentParser()
    parser.add_argument( '-opt', type=str,
                         default='CloudRemoval/Options/0901_cr_restormer-s_128x128_1xb16_160k_alld_hybirdv1.yml',
                         help='Path to option YAML file.')
    parser.add_argument( '-ckpt', type=str,
                         default='experiments/0901_cr_restormer-s_128x128_1xb16_160k_alld_hybirdv1/models/net_g_latest.pth')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    opt = parse(args.opt, is_train=is_train)

    # distributed settings
    if args.launcher == 'none':
        opt['dist'] = False
        print('Disable distributed.', flush=True)
    else:
        opt['dist'] = True
        if args.launcher == 'slurm' and 'dist_params' in opt:
            init_dist(args.launcher, **opt['dist_params'])
        else:
            init_dist(args.launcher)
            print('init dist .. ', args.launcher)

    opt['rank'], opt['world_size'] = get_dist_info()

    # random seed
    seed = opt.get('manual_seed')
    if seed is None:
        seed = random.randint(1, 10000)
        opt['manual_seed'] = seed
    set_random_seed(seed + opt['rank'])

    return opt, args

def main():
    # parse options, set distributed setting, set ramdom seed
    opt, args = parse_options(is_train=False)
    opt['num_gpu'] = 1
    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # # mkdir and initialize loggers
    # make_exp_dirs(opt)
    # log_file = osp.join(opt['path']['log'],
    #                     f"test_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(
        logger_name='basicsr', log_level=logging.INFO, log_file=None)
    logger.info(get_env_info())
    logger.info(dict2str(opt))

    # create vak dataset and dataloader
    val_loader = None
    for phase, dataset_opt in sorted(opt['datasets'].items()):
        if phase == 'val':
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(
                val_set,
                dataset_opt,
                num_gpu=opt['num_gpu'],
                dist=opt['dist'],
                sampler=None,
                seed=opt['manual_seed'])
            logger.info(
                f"Number of val images in {dataset_opt['name']}: {len(val_set)}")

    # create model
    model = create_model(opt)
    state_dict = torch.load(args.ckpt, map_location='cpu')['params']
    state_dict_new = {}
    for k in state_dict.keys():
        state_dict_new[k.replace('module.', '')] = state_dict[k]
    model.net_g.load_state_dict(state_dict_new)

    val_set_name = val_loader.dataset.opt['name']
    logger.info(f'Validating {val_set_name}...')
    rgb2bgr = opt['val'].get('rgb2bgr', False)
    use_image = opt['val'].get('use_image', False)
    # wheather use uint8 image to compute metrics
    model.nondist_validation(
        dataloader=val_loader,
        current_iter=0,
        tb_logger=None,
        save_img=False,
        rgb2bgr=rgb2bgr,
        use_image=use_image
    )


if __name__ == '__main__':
    main()
