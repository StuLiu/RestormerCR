import os

from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.data_util import (paired_paths_from_folder,
                                    paired_DP_paths_from_folder,
                                    paired_paths_from_lmdb,
                                    paired_paths_from_meta_info_file)
from basicsr.data.transforms import augment, paired_random_crop, paired_random_crop_DP, random_augmentation
from basicsr.utils import FileClient, imfrombytes, img2tensor, padding, padding_DP, imfrombytesDP
from basicsr.utils.perlin_cloud import cloud_gen_perlin

import random
import numpy as np
import torch
import cv2
import json


class Dataset_PairedImage(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and
    GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the
                template excludes the file extension. Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            geometric_augs (bool): Use geometric augmentations.

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(Dataset_PairedImage, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        
        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']
            self.paths = paired_paths_from_lmdb(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'])
        elif 'meta_info_file' in self.opt and self.opt[
                'meta_info_file'] is not None:
            self.paths = paired_paths_from_meta_info_file(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'],
                self.opt['meta_info_file'], self.filename_tmpl)
        else:
            self.paths = paired_paths_from_folder(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'],
                self.filename_tmpl)

        if self.opt['phase'] == 'train':
            self.geometric_augs = opt['geometric_augs']

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        index = index % len(self.paths)
        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        try:
            img_gt = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("gt path {} not working".format(gt_path))

        lq_path = self.paths[index]['lq_path']
        img_bytes = self.file_client.get(lq_path, 'lq')
        try:
            img_lq = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("lq path {} not working".format(lq_path))

        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # padding
            img_gt, img_lq = padding(img_gt, img_lq, gt_size)

            # random crop
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale,
                                                gt_path)

            # flip, rotation augmentations
            if self.geometric_augs:
                img_gt, img_lq = random_augmentation(img_gt, img_lq)
            
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq],
                                    bgr2rgb=True,
                                    float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)
        
        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': lq_path,
            'gt_path': gt_path
        }

    def __len__(self):
        return len(self.paths)


class Dataset_CloudRemoval(Dataset_PairedImage):

    def __init__(self, opt, debug=False):
        super().__init__(opt)
        self.debug = debug

    def show_img(self, img_gt, img_lq):
        cloudy = img_lq[:, :, :3]
        sarvh = img_lq[:, :, 3]
        sarvv = img_lq[:, :, 4]
        sarvh2 = np.stack([sarvh] * 3, axis=2)
        sarvv2 = np.stack([sarvv] * 3, axis=2)
        if img_gt is not None:
            img_all = np.concatenate([cloudy, sarvh2, sarvv2, img_gt], axis=1)
        else:
            img_all = np.concatenate([cloudy, sarvh2, sarvv2], axis=1)
        cv2.imshow('img_all', img_all)
        if ord('q') == cv2.waitKey(0):
            exit(0)

    def random_strong_augs(self, img_gt, img_lq, p=0.5):
        return img_gt, img_lq

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        index = index % len(self.paths)
        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.

        gt_path = self.paths[index]['gt_path']
        if self.opt['phase'] != 'test':
            img_bytes = self.file_client.get(gt_path, 'gt')
            try:
                img_gt = imfrombytes(img_bytes, float32=True)
            except:
                raise Exception("gt path {} not working".format(gt_path))
        else:
            img_gt = None

        lq_path = self.paths[index]['lq_path']
        img_bytes = self.file_client.get(lq_path, 'lq')
        try:
            img_lq = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("lq path {} not working".format(lq_path))

        # cv2.imshow('', (img_lq * 255).astype(np.uint8))
        # cv2.waitKey(0)

        sar_vh = self.paths[index]['lq_path'].replace('/opt_cloudy', '/SAR/VH').replace("_p_", "_VH_p_")
        sar_vh = cv2.imread(sar_vh, flags=cv2.IMREAD_UNCHANGED) / 255.0
        sar_vv = self.paths[index]['lq_path'].replace('/opt_cloudy', '/SAR/VV').replace("_p_", "_VV_p_")
        sar_vv = cv2.imread(sar_vv, flags=cv2.IMREAD_UNCHANGED) / 255.0
        img_lq = np.concatenate([img_lq, sar_vh[:, :, None], sar_vv[:, :, None]], axis=2)

        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # padding
            img_gt, img_lq = padding(img_gt, img_lq, gt_size)

            # random crop
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale, gt_path)

            # flip, rotation augmentations
            if self.geometric_augs:
                img_gt, img_lq = random_augmentation(img_gt, img_lq)

            # strong augmentations
            img_gt, img_lq = self.random_strong_augs(img_gt, img_lq)

        # show img_lg
        if self.debug:
            self.show_img(img_gt, img_lq)

        # BGR to RGB, HWC to CHW, numpy to tensor
        if self.opt['phase'] != 'test':
            img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=False, float32=True)
        else:
            img_lq = img2tensor(img_lq, bgr2rgb=False, float32=True)

        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            if self.opt['phase'] != 'test':
                normalize(img_gt, self.mean, self.std, inplace=True)

        if self.opt['phase'] != 'test':
            return {
                'lq': img_lq,
                'gt': img_gt,
                'lq_path': lq_path,
                'gt_path': gt_path
            }
        else:
            return {
                'lq': img_lq,
                'lq_path': lq_path
            }


class Dataset_CloudRemoval_RCS(Dataset_CloudRemoval):
    # less ocean, more cloud
    def __init__(self, opt):
        super().__init__(opt)

        with open(f'{opt["dataroot_gt"]}/../gtname2isocean_cloudyrate.json', 'r') as f:
            self.gtname2isocean_cloudyrate = json.load(f)

        self.name2index = {}
        self.names_isocean_true = []
        self.names_isocean_false = []
        index = 0
        for name, (isocean, cloudyrate) in self.gtname2isocean_cloudyrate.items():
            self.name2index[name] = index
            index += 1
            if isocean:
                self.names_isocean_true.append(name)
            else:
                # # result in bad performance for easy examples
                length = int(cloudyrate) + 1
                names_cp = [name] * length
                self.names_isocean_false.extend(names_cp)
        print(f'names_isocean_true-len={len(self.names_isocean_true)}, '
              f'names_isocean_false-len={len(self.names_isocean_false)}')

    def rcs(self):
        if np.random.rand() < 0.01:
            # selected from ocean
            name = random.choice(self.names_isocean_true)
        else:
            # selected from land
            name = random.choice(self.names_isocean_false)
        index = self.name2index[name]
        return index

    def __getitem__(self, index):
        index = self.rcs()
        return super().__getitem__(index)


class Dataset_CloudRemoval_RCSv2(Dataset_CloudRemoval):
    # less ocean
    def __init__(self, opt):
        super().__init__(opt)

        with open(f'{opt["dataroot_gt"]}/../gtname2isocean_cloudyrate.json', 'r') as f:
            self.gtname2isocean_cloudyrate = json.load(f)

        self.name2index = {}
        self.names_isocean_true = []
        self.names_isocean_false = []
        index = 0
        for name, (isocean, cloudyrate) in self.gtname2isocean_cloudyrate.items():
            self.name2index[name] = index
            index += 1
            if isocean:
                self.names_isocean_true.append(name)
            else:
                self.names_isocean_false.append(name)
        print(f'names_isocean_true-len={len(self.names_isocean_true)}, '
              f'names_isocean_false-len={len(self.names_isocean_false)}')

    def rcs(self):
        if np.random.rand() < 0.01:
            # selected from ocean
            name = random.choice(self.names_isocean_true)
        else:
            # selected from land
            name = random.choice(self.names_isocean_false)
        index = self.name2index[name]
        return index

    def __getitem__(self, index):
        index = self.rcs()
        return super().__getitem__(index)


class Dataset_CloudRemoval_RCSv3(Dataset_CloudRemoval):
    # more ocean
    def __init__(self, opt):
        super().__init__(opt)

        with open(f'{opt["dataroot_gt"]}/../gtname2isocean_cloudyrate.json', 'r') as f:
            self.gtname2isocean_cloudyrate = json.load(f)

        self.name2index = {}
        self.names_isocean_true = []
        self.names_isocean_false = []
        index = 0
        for name, (isocean, cloudyrate) in self.gtname2isocean_cloudyrate.items():
            self.name2index[name] = index
            index += 1
            if isocean:
                self.names_isocean_true.append(name)
            else:
                self.names_isocean_false.append(name)
        print(f'names_isocean_true-len={len(self.names_isocean_true)}, '
              f'names_isocean_false-len={len(self.names_isocean_false)}')

    def rcs(self):
        if np.random.rand() < 0.5:
            # selected from ocean
            name = random.choice(self.names_isocean_true)
        else:
            # selected from land
            name = random.choice(self.names_isocean_false)
        index = self.name2index[name]
        return index

    def __getitem__(self, index):
        index = self.rcs()
        return super().__getitem__(index)


class Dataset_CloudRemoval_RCSv4(Dataset_CloudRemoval):
    # random night
    def __init__(self, opt, debug=False):
        super().__init__(opt, debug)

    def random_strong_augs(self, img_gt, img_lq, p=0.5):
        # img_lq = np.concatenate([img_lq, sar_vh[:, :, None], sar_vv[:, :, None]], axis=2)
        if np.random.rand() < p:
            img_vis = img_lq[:, :, :3].copy()
            dark_rate = random.random() / 2
            img_lq[:, :, :3] = img_vis * dark_rate
        return img_gt, img_lq


class Dataset_CloudRemoval_RCSv5(Dataset_CloudRemoval):
    # random miss-SAR
    def __init__(self, opt, debug=False):
        super().__init__(opt, debug)

    @staticmethod
    def random_mask(img):
        img = img.copy()
        img_h, img_w = img.shape[:2]
        if np.random.rand() < 0.5:  # 裁剪高度
            crop_height = random.randint(1, img_h // 4)
            if np.random.rand() < 0.5:
                img[:crop_height, :, :] = 0
            else:
                img[-crop_height:, :, :] = 0
        else:
            crop_width = random.randint(1, img_w // 4)
            if np.random.rand() < 0.5:
                img[:, :crop_width, :] = 0
            else:
                img[:, -crop_width:, :] = 0
        return img

    def random_strong_augs(self, img_gt, img_lq, p=0.02):
        # img_lq = np.concatenate([img_lq, sar_vh[:, :, None], sar_vv[:, :, None]], axis=2)
        if np.random.rand() < p:
            img_sar = img_lq[:, :, 3:].copy()
            if np.random.rand() < 0.5:
                # part miss
                img_sar = self.random_mask(img_sar)
            else:
                # miss all
                img_sar = np.zeros_like(img_sar)
            img_lq[:, :, 3:] = img_sar
        return img_gt, img_lq


class Dataset_CloudRemoval_RCSv6(Dataset_CloudRemoval_RCSv4):
    # random night + more cloudy
    def __init__(self, opt, debug=False):
        super().__init__(opt, debug)

        with open(f'{opt["dataroot_gt"]}/../gtname2isocean_cloudyrate.json', 'r') as f:
            self.gtname2isocean_cloudyrate = json.load(f)

        self.indexes_more_cloudy = []
        index = 0
        for name, (_, cloudyrate) in self.gtname2isocean_cloudyrate.items():
            # # result in bad performance for easy examples
            length = int(cloudyrate / 10) + 1
            indexes_cp = [index] * length
            self.indexes_more_cloudy.extend(indexes_cp)
            index += 1
        print(f'indexes_more_cloudy-len={len(self.indexes_more_cloudy)}')

    def rcs(self):
        index = random.choice(self.indexes_more_cloudy)
        return index

    def __getitem__(self, index):
        if self.split == 'train':
            index = self.rcs()
        return super().__getitem__(index)


class Dataset_CloudRemoval_RCSv7(Dataset_CloudRemoval):
    # random night + berlin-cloud
    def __init__(self, opt, debug=False):
        super().__init__(opt, debug)

    @staticmethod
    def add_berlin_cloud(img_bgr, p=0.8):
        if random.random() < p:
            if np.sum(img_bgr > 160) < 1600:
                img_clody = cloud_gen_perlin(img_bgr * 255)
                return img_clody / 255
            else:
                return img_bgr
        else:
            return img_bgr

    def random_strong_augs(self, img_gt, img_lq, p=0.5):
        # img_lq = np.concatenate([img_lq, sar_vh[:, :, None], sar_vv[:, :, None]], axis=2)
        if np.random.rand() < p:
            img_vis = img_lq[:, :, :3].copy()
            dark_rate = random.random() / 2
            img_lq[:, :, :3] = img_vis * dark_rate

        if np.random.rand() < 0.8:
            img_lq[:, :, :3] = self.add_berlin_cloud(img_lq[:, :, :3])

        return img_gt, img_lq


class Dataset_GaussianDenoising(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and
    GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            gt_size (int): Cropped patched size for gt patches.
            use_flip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h
                and w for implementation).

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(Dataset_GaussianDenoising, self).__init__()
        self.opt = opt

        if self.opt['phase'] == 'train':
            self.sigma_type  = opt['sigma_type']
            self.sigma_range = opt['sigma_range']
            assert self.sigma_type in ['constant', 'random', 'choice']
        else:
            self.sigma_test = opt['sigma_test']
        self.in_ch = opt['in_ch']

        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None        

        self.gt_folder = opt['dataroot_gt']

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.gt_folder]
            self.io_backend_opt['client_keys'] = ['gt']
            self.paths = paths_from_lmdb(self.gt_folder)
        elif 'meta_info_file' in self.opt:
            with open(self.opt['meta_info_file'], 'r') as fin:
                self.paths = [
                    osp.join(self.gt_folder,
                             line.split(' ')[0]) for line in fin
                ]
        else:
            self.paths = sorted(list(scandir(self.gt_folder, full_path=True)))

        if self.opt['phase'] == 'train':
            self.geometric_augs = self.opt['geometric_augs']

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        index = index % len(self.paths)
        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')

        if self.in_ch == 3:
            try:
                img_gt = imfrombytes(img_bytes, float32=True)
            except:
                raise Exception("gt path {} not working".format(gt_path))

            img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB)
        else:
            try:
                img_gt = imfrombytes(img_bytes, flag='grayscale', float32=True)
            except:
                raise Exception("gt path {} not working".format(gt_path))

            img_gt = np.expand_dims(img_gt, axis=2)
        img_lq = img_gt.copy()


        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # padding
            img_gt, img_lq = padding(img_gt, img_lq, gt_size)

            # random crop
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale,
                                                gt_path)
            # flip, rotation
            if self.geometric_augs:
                img_gt, img_lq = random_augmentation(img_gt, img_lq)

            img_gt, img_lq = img2tensor([img_gt, img_lq],
                                        bgr2rgb=False,
                                        float32=True)


            if self.sigma_type == 'constant':
                sigma_value = self.sigma_range
            elif self.sigma_type == 'random':
                sigma_value = random.uniform(self.sigma_range[0], self.sigma_range[1])
            elif self.sigma_type == 'choice':
                sigma_value = random.choice(self.sigma_range)

            noise_level = torch.FloatTensor([sigma_value])/255.0
            # noise_level_map = torch.ones((1, img_lq.size(1), img_lq.size(2))).mul_(noise_level).float()
            noise = torch.randn(img_lq.size()).mul_(noise_level).float()
            img_lq.add_(noise)

        else:            
            np.random.seed(seed=0)
            img_lq += np.random.normal(0, self.sigma_test/255.0, img_lq.shape)
            # noise_level_map = torch.ones((1, img_lq.shape[0], img_lq.shape[1])).mul_(self.sigma_test/255.0).float()

            img_gt, img_lq = img2tensor([img_gt, img_lq],
                            bgr2rgb=False,
                            float32=True)

        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': gt_path,
            'gt_path': gt_path
        }

    def __len__(self):
        return len(self.paths)

class Dataset_DefocusDeblur_DualPixel_16bit(data.Dataset):
    def __init__(self, opt):
        super(Dataset_DefocusDeblur_DualPixel_16bit, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        
        self.gt_folder, self.lqL_folder, self.lqR_folder = opt['dataroot_gt'], opt['dataroot_lqL'], opt['dataroot_lqR']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        self.paths = paired_DP_paths_from_folder(
            [self.lqL_folder, self.lqR_folder, self.gt_folder], ['lqL', 'lqR', 'gt'],
            self.filename_tmpl)

        if self.opt['phase'] == 'train':
            self.geometric_augs = self.opt['geometric_augs']

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        index = index % len(self.paths)
        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        try:
            img_gt = imfrombytesDP(img_bytes, float32=True)
        except:
            raise Exception("gt path {} not working".format(gt_path))

        lqL_path = self.paths[index]['lqL_path']
        img_bytes = self.file_client.get(lqL_path, 'lqL')
        try:
            img_lqL = imfrombytesDP(img_bytes, float32=True)
        except:
            raise Exception("lqL path {} not working".format(lqL_path))

        lqR_path = self.paths[index]['lqR_path']
        img_bytes = self.file_client.get(lqR_path, 'lqR')
        try:
            img_lqR = imfrombytesDP(img_bytes, float32=True)
        except:
            raise Exception("lqR path {} not working".format(lqR_path))


        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # padding
            img_lqL, img_lqR, img_gt = padding_DP(img_lqL, img_lqR, img_gt, gt_size)

            # random crop
            img_lqL, img_lqR, img_gt = paired_random_crop_DP(img_lqL, img_lqR, img_gt, gt_size, scale, gt_path)
            
            # flip, rotation            
            if self.geometric_augs:
                img_lqL, img_lqR, img_gt = random_augmentation(img_lqL, img_lqR, img_gt)
        # TODO: color space transform
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_lqL, img_lqR, img_gt = img2tensor([img_lqL, img_lqR, img_gt],
                                    bgr2rgb=True,
                                    float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lqL, self.mean, self.std, inplace=True)
            normalize(img_lqR, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        img_lq = torch.cat([img_lqL, img_lqR], 0)
        
        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': lqL_path,
            'gt_path': gt_path
        }

    def __len__(self):
        return len(self.paths)
