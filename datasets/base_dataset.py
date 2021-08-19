from __future__ import division

import torch
from torch.utils.data import Dataset
from torchvision.transforms import Normalize
import numpy as np
import cv2
from os.path import join

import config
import constants
from utils.imutils import crop_v2, flip_img, flip_pose, flip_kp, transform, rot_aa, color_jitter

import pickle
import scipy.misc
import matplotlib.pyplot as plt


class BaseDataset(Dataset):
    """
    Base Dataset Class - Handles data loading and augmentation.
    Able to handle heterogeneous datasets (different annotations available for different datasets).
    You need to update the path to each dataset in utils/config.py.
    """

    def __init__(self, options, dataset, ignore_3d=False, use_augmentation=True, is_train=True, is_rotate=False):
        super(BaseDataset, self).__init__()
        self.dataset = dataset
        self.is_train = is_train
        self.options = options
        self.normalize_img = Normalize(mean=constants.IMG_NORM_MEAN, std=constants.IMG_NORM_STD)
        self.data = np.load(config.DATASET_FILES[is_train][dataset])
        self.imgname = self.data['imgname']

        self.img_pkl_path = config.DATASET_PKL_PATH
        if is_train:
            pkl_name = '{}_imgs.pkl'.format(self.dataset)
        else:
            pkl_name = '{}_imgs_test.pkl'.format(self.dataset)
        with open(join(self.img_pkl_path, pkl_name), 'rb') as f:
            self.img_pkl = pickle.load(f)
        self.is_rotate = is_rotate

        # Get paths to gt masks, if available
        try:
            self.maskname = self.data['maskname']
        except KeyError:
            pass
        try:
            self.partname = self.data['partname']
        except KeyError:
            pass

        # Bounding boxes are assumed to be in the center and scale format
        self.scale = self.data['scale']
        self.center = self.data['center']

        # If False, do not do augmentation
        self.use_augmentation = use_augmentation

        # Get gt SMPL parameters, if available
        try:
            self.pose = self.data['pose'].squeeze().astype(np.float)
            self.betas = self.data['shape'].squeeze().astype(np.float)
            if 'has_smpl' in self.data:
                self.has_smpl = self.data['has_smpl']
            else:
                self.has_smpl = np.ones(len(self.imgname))
        except KeyError:
            self.has_smpl = np.zeros(len(self.imgname))
        if ignore_3d:
            self.has_smpl = np.zeros(len(self.imgname))

        # Get gt 3D pose, if available
        try:
            self.pose_3d = self.data['S']
            self.has_pose_3d = 1
        except KeyError:
            self.has_pose_3d = 0
        if ignore_3d:
            self.has_pose_3d = 0

        # Get 2D keypoints
        try:
            keypoints_gt = self.data['part']
        except KeyError:
            keypoints_gt = np.zeros((len(self.imgname), 24, 3))
        try:
            keypoints_openpose = self.data['openpose']
        except KeyError:
            keypoints_openpose = np.zeros((len(self.imgname), 25, 3))

        self.keypoints = np.concatenate([keypoints_openpose, keypoints_gt], axis=1)

        # Get gender data, if available
        try:
            gender = self.data['gender']
            self.gender = np.array([0 if str(g) == 'm' else 1 for g in gender]).astype(np.int32)
        except KeyError:
            self.gender = -1 * np.ones(len(self.imgname)).astype(np.int32)

        self.length = self.scale.shape[0]

        # add by hao, for generating 2d keypoints heatmaps
        self.num_joints = 24
        self.heatmap_size = 14

        # add by hao, for generating and updateing image sizes
        self.all_img_sizes = {0: [[224, 128]], 1: [[224, 128], [128, 64]],
                              2: [[224, 128], [128, 64], [64, 40]], 3: [[224, 128], [128, 64], [64, 40], [40, 24]]}

        # add by hao, set evaluation flag
        # for evaluation, we want to evaluate the middle point of each interval
        self.eval = not is_train

    def set_eval(self, type=0):
        self.eval = True
        self.type = type

    def update_size_intervals(self, epoch):
        """ only used for first 6 epoch, after first 6, image sizes will always be [[224, 128], [128, 64], [64, 40], [40, 24]]"""
        if epoch <= 3:
            self.img_sizes = self.all_img_sizes[epoch]
        else:
            self.img_sizes = [[224, 128], [128, 64], [64, 40], [40, 24]]

    def augm_params(self):
        """Get augmentation parameters."""
        flip = 0  # flipping
        pn = np.ones(3)  # per channel pixel-noise
        rot = 0  # rotation
        sc = 1  # scaling
        if self.is_train:
            # We flip with probability 1/2
            if np.random.uniform() <= 0.5:
                flip = 1

            # Each channel is multiplied with a number
            # in the area [1-opt.noiseFactor,1+opt.noiseFactor]
            pn = np.random.uniform(1 - self.options.noise_factor, 1 + self.options.noise_factor, 3)

            # The rotation is a number in the area [-2*rotFactor, 2*rotFactor]
            rot = min(2 * self.options.rot_factor,
                      max(-2 * self.options.rot_factor, np.random.randn() * self.options.rot_factor))

            # The scale is multiplied with a number
            # in the area [1-scaleFactor,1+scaleFactor]
            sc = min(1 + self.options.scale_factor,
                     max(1 - self.options.scale_factor, np.random.randn() * self.options.scale_factor + 1))
            # but it is zero with probability 3/5
            if np.random.uniform() <= 0.6:
                rot = 0

        return flip, pn, rot, sc

    def rgb_processing(self, rgb_img, center, scale, rot, flip, pn, img_size):
        """Process rgb image and do augmentation."""
        if self.is_rotate:
            rgb_img = crop_v2(rgb_img, center, scale,
                              [constants.IMG_RES, constants.IMG_RES], rot=rot)

        # flip the image
        if flip:
            rgb_img = flip_img(rgb_img)
        # in the rgb image we add pixel noise in a channel-wise manner
        rgb_img[:, :, 0] = np.minimum(255.0, np.maximum(0.0, rgb_img[:, :, 0] * pn[0]))
        rgb_img[:, :, 1] = np.minimum(255.0, np.maximum(0.0, rgb_img[:, :, 1] * pn[1]))
        rgb_img[:, :, 2] = np.minimum(255.0, np.maximum(0.0, rgb_img[:, :, 2] * pn[2]))

        if img_size == 224:
            rgb_img_up = rgb_img.copy()
            # add color jitter
            if self.is_train:
                rgb_img_up = color_jitter(rgb_img_up, brightness=0.4, contrast=0.4, saturation=0.4, prob=0.5)
            rgb_img_up = rgb_img_up.clip(0, 255)
        else:
            shape = rgb_img.shape
            rgb_img_lr = scipy.misc.imresize(rgb_img, (img_size, img_size), interp='bicubic')
            rgb_img_lr = rgb_img_lr.clip(0, 255)
            rgb_img_up = scipy.misc.imresize(rgb_img_lr, (shape[0], shape[1]), interp='bicubic')  # naive upsampling
            # add color jitter
            if self.is_train:
                rgb_img_up = color_jitter(rgb_img_up, brightness=0.4, contrast=0.4, saturation=0.4, prob=0.5)
                rgb_img_up = rgb_img_up.clip(0, 255)

        rgb_img_up = np.transpose(rgb_img_up.astype('float32'), (2, 0, 1)) / 255.0
        return rgb_img_up

    def j2d_processing(self, kp, center, scale, r, f):
        """Process gt 2D keypoints and apply all augmentation transforms."""
        nparts = kp.shape[0]
        for i in range(nparts):
            kp[i, 0:2] = transform(kp[i, 0:2] + 1, center, scale,
                                   [constants.IMG_RES, constants.IMG_RES], rot=r)
        # convert to normalized coordinates
        kp[:, :-1] = 2. * kp[:, :-1] / constants.IMG_RES - 1.
        # flip the x coordinates
        if f:
            kp = flip_kp(kp)
        kp = kp.astype('float32')
        return kp

    def j2d_heatmap_location_processing(self, kp, center, scale, r, f):
        # first process the keypoints
        nparts = kp.shape[0]
        for i in range(nparts):
            kp[i, 0:2] = transform(kp[i, 0:2] + 1, center, scale,
                                   [constants.IMG_RES, constants.IMG_RES], rot=r)
        # flip the x coordinates
        if f:
            kp = flip_kp(kp)

        kp_heatmaps = kp.copy().astype(np.uint8)
        feat_stride = constants.IMG_RES / self.heatmap_size
        kp_heatmaps[:, :-1] = kp_heatmaps[:, :-1] / feat_stride
        kp_heatmaps = kp_heatmaps.astype('float32')
        return kp_heatmaps

    # add by haochen
    def j2d_heatmap_processing(self, kp, center, scale, r, f):
        # first process the keypoints
        nparts = kp.shape[0]
        for i in range(nparts):
            kp[i, 0:2] = transform(kp[i, 0:2] + 1, center, scale,
                                   [constants.IMG_RES, constants.IMG_RES], rot=r)
        # flip the x coordinates
        if f:
            kp = flip_kp(kp)

        # generate guaissan heatmap
        target_weight = np.ones((self.num_joints, 1), dtype=np.float32)
        target_weight[:, 0] = kp[:, -1]
        target = np.zeros((self.num_joints, self.heatmap_size[1], self.heatmap_size[0]), dtype=np.float32)
        tmp_size = self.sigma * 3

        for joint_id in range(self.num_joints):
            feat_stride = self.options.img_res / self.heatmap_size
            mu_x = int(kp[joint_id][0] / feat_stride[0] + 0.5)
            mu_y = int(kp[joint_id][1] / feat_stride[1] + 0.5)
            # Check that any part of the gaussian is in-bounds
            ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
            br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
            if ul[0] >= self.heatmap_size[0] or ul[1] >= self.heatmap_size[1] \
                    or br[0] < 0 or br[1] < 0:
                # If not, just return the image as is
                target_weight[joint_id] = 0
                continue

            # # Generate gaussian
            size = 2 * tmp_size + 1
            x = np.arange(0, size, 1, np.float32)
            y = x[:, np.newaxis]
            x0 = y0 = size // 2
            # The gaussian is not normalized, we want the center value to equal 1
            g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma ** 2))

            # Usable gaussian range
            g_x = max(0, -ul[0]), min(br[0], self.heatmap_size[0]) - ul[0]
            g_y = max(0, -ul[1]), min(br[1], self.heatmap_size[1]) - ul[1]
            # Image range
            img_x = max(0, ul[0]), min(br[0], self.heatmap_size[0])
            img_y = max(0, ul[1]), min(br[1], self.heatmap_size[1])

            v = target_weight[joint_id]
            if v > 0.5:
                target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                    g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

        return target, target_weight

    def j3d_processing(self, S, r, f):
        """Process gt 3D keypoints and apply all augmentation transforms."""
        # in-plane rotation
        rot_mat = np.eye(3)
        if not r == 0:
            rot_rad = -r * np.pi / 180
            sn, cs = np.sin(rot_rad), np.cos(rot_rad)
            rot_mat[0, :2] = [cs, -sn]
            rot_mat[1, :2] = [sn, cs]
        S[:, :-1] = np.einsum('ij,kj->ki', rot_mat, S[:, :-1])
        # flip the x coordinates
        if f:
            S[:, :-1] = flip_kp(S[:, :-1])
        S = S.astype('float32')
        return S

    def pose_processing(self, pose, r, f):
        """Process SMPL theta parameters  and apply all augmentation transforms."""
        # rotation or the pose parameters
        pose[:3] = rot_aa(pose[:3], r)
        # flip the pose parameters
        if f:
            pose = flip_pose(pose)
        # (72),float
        pose = pose.astype('float32')
        return pose

    def __getitem__(self, index):
        item = {}
        scale = self.scale[index].copy()
        center = self.center[index].copy()

        # Load image
        imgname = self.imgname[index]
        img_str = self.img_pkl[index]
        img_encode = np.asarray(bytearray(img_str), dtype=np.uint8)
        img = cv2.imdecode(img_encode, cv2.IMREAD_COLOR)
        img = img[:, :, ::-1].astype(np.float32)
        orig_shape = np.array(img.shape)[:2]

        # Get SMPL parameters, if available
        if self.has_smpl[index]:
            pose = self.pose[index].copy()
            betas = self.betas[index].copy()
        else:
            pose = np.zeros(72)
            betas = np.zeros(10)

        # Get augmentation parameters
        flip, pn, rot, sc = self.augm_params()
        sc = 1
        if not self.is_rotate:
            rot = 0

        # Process original HR image
        img_up = self.rgb_processing(img, center, sc * scale, rot, flip, pn, 224)
        img_up = torch.from_numpy(img_up).float()

        # Store image before normalization to use it in visualization
        item['img_hr'] = self.normalize_img(img_up)
        item['pose'] = torch.from_numpy(self.pose_processing(pose, rot, flip)).float()

        # Get 3D pose, if available
        if self.has_pose_3d:
            S = self.pose_3d[index].copy()
            item['pose_3d'] = torch.from_numpy(self.j3d_processing(S, rot, flip)).float()
        else:
            item['pose_3d'] = torch.zeros(24, 4, dtype=torch.float32)

        # Get 2D keypoints and apply augmentation transforms
        keypoints = self.keypoints[index].copy()
        item['keypoints'] = torch.from_numpy(self.j2d_processing(keypoints, center, sc * scale, rot, flip)).float()

        # Get 2D keypoints gaussian maps from only gt 2d keypoints
        keypoints_heatmaps = self.j2d_heatmap_location_processing(self.keypoints[index].copy()[25:], center, sc * scale,
                                                                  rot, flip)
        item['keypoints_heatmaps'] = torch.from_numpy(keypoints_heatmaps).float()

        img_lr = []
        # for each size of the img, conduct augmentataion and record 2d keypoint, 3d keypoint, pose
        for size_interval in self.img_sizes:
            if not self.eval:
                img_size = np.random.randint(size_interval[1], size_interval[0])
            else:
                if self.type == 0:
                    img_size = (size_interval[1] + size_interval[0]) // 2
                else:
                    img_size = size_interval[1]

            # Get augmentation parameters
            flip_lr, pn, rot_lr, sc = self.augm_params()
            sc = 1

            # do not flip and rotate the low res images
            flip_lr = flip
            rot_lr = rot

            # Process image
            img_up = self.rgb_processing(img, center, sc * scale, rot_lr, flip_lr, pn, img_size)
            img_up = torch.from_numpy(img_up).float()

            img_lr.append(self.normalize_img(img_up))

        item['img_lr'] = img_lr

        item['betas'] = torch.from_numpy(betas).float()
        item['imgname'] = imgname
        item['has_smpl'] = self.has_smpl[index]
        item['has_pose_3d'] = self.has_pose_3d
        item['scale'] = float(sc * scale)
        item['center'] = center.astype(np.float32)
        item['orig_shape'] = orig_shape
        item['is_flipped'] = flip
        item['rot_angle'] = np.float32(rot)
        item['gender'] = self.gender[index]
        item['sample_index'] = index
        item['dataset_name'] = self.dataset

        try:
            item['maskname'] = self.maskname[index]
        except AttributeError:
            item['maskname'] = ''
        try:
            item['partname'] = self.partname[index]
        except AttributeError:
            item['partname'] = ''

        return item

    def __len__(self):
        return len(self.imgname)


if __name__ == '__main__':
    from utils.train_options import TrainOptions

    args = TrainOptions()
    options = args.parse_args(['--name', 'hc'])

    ds = BaseDataset(options, '3dpw', is_train=False)

    print("done")



