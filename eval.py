import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import cv2
import os
import argparse
from tqdm import tqdm

import config
import constants
from models import hmr, SMPL
from utils.pose_utils import reconstruction_error
from torchvision.transforms import Normalize
import pickle
import scipy.misc

# Define command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', required=True, type=str, help='Path to network checkpoint')
parser.add_argument('--batch_size', default=16, type=int, help='Batch size for testing')
parser.add_argument('--num_workers', default=0, type=int, help='Number of processes for data loading')
parser.add_argument('--eval_size', default=32, type=int, help='Image resolution for evaluation')


class PW3D(Dataset):
    def __init__(self, data_name, pkl_name, img_size):
        self.normalize_img = Normalize(mean=constants.IMG_NORM_MEAN, std=constants.IMG_NORM_STD)
        self.data = np.load(data_name)
        self.imgname = self.data['imgname']
        self.pose = self.data['pose'].squeeze().astype(np.float)
        self.betas = self.data['shape'].squeeze().astype(np.float)
        # Get gender data, if available
        try:
            gender = self.data['gender']
            self.gender = np.array([0 if str(g) == 'm' else 1 for g in gender]).astype(np.int32)
        except KeyError:
            self.gender = -1 * np.ones(len(self.imgname)).astype(np.int32)

        with open(pkl_name, 'rb') as f:
            self.img_pkl = pickle.load(f)

        self.img_size = img_size

    def __len__(self):
        return len(self.imgname)

    def rgb_processing(self, rgb_img, img_size):
        """Process rgb image and do augmentation."""
        if img_size == 224:
            rgb_img_up = rgb_img.copy()
            rgb_img_up = rgb_img_up.clip(0, 255)
        else:
            shape = rgb_img.shape
            rgb_img_lr = scipy.misc.imresize(rgb_img, (img_size, img_size), interp='bicubic')
            rgb_img_lr = rgb_img_lr.clip(0, 255)
            rgb_img_up = scipy.misc.imresize(rgb_img_lr, (shape[0], shape[1]), interp='bicubic')  # naive upsampling

        rgb_img_up = np.transpose(rgb_img_up.astype('float32'), (2, 0, 1)) / 255.0
        return rgb_img_up

    def __getitem__(self, index):
        item = {}

        img_str = self.img_pkl[index]
        img_encode = np.asarray(bytearray(img_str), dtype=np.uint8)
        img = cv2.imdecode(img_encode, cv2.IMREAD_COLOR)
        img = img[:, :, ::-1].astype(np.float32)
        img_up = self.rgb_processing(img, self.img_size)
        img_up = self.normalize_img(torch.from_numpy(img_up).float())

        pose = self.pose[index].copy()
        betas = self.betas[index].copy()

        item['gender'] = self.gender[index]
        item['pose'] = torch.from_numpy(pose).float()
        item['betas'] = torch.from_numpy(betas).float()
        item['img_up'] = img_up

        return item


def size_to_scale(size):
    if size >= 224:
        scale = 0
    elif 128 <= size < 224:
        scale = 1
    elif 64 <= size < 128:
        scale = 2
    elif 40 <= size < 64:
        scale = 3
    else:
        scale = 4
    return scale


def run_evaluation(hmr_model, dataset, eval_size, batch_size=32, num_workers=32, log_freq=50):
    """Run evaluation on the datasets and metrics we report in the paper. """
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # focal length
    focal_length = constants.FOCAL_LENGTH

    # Transfer hmr_model to the GPU
    hmr_model.to(device)

    # Load SMPL hmr_model
    smpl_neutral = SMPL(config.SMPL_MODEL_DIR,
                        create_transl=False).to(device)
    smpl_male = SMPL(config.SMPL_MODEL_DIR,
                     gender='male',
                     create_transl=False).to(device)
    smpl_female = SMPL(config.SMPL_MODEL_DIR,
                       gender='female',
                       create_transl=False).to(device)

    # Regressor for H36m joints
    J_regressor = torch.from_numpy(np.load(config.JOINT_REGRESSOR_H36M)).float()

    # Create dataloader for the dataset
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Pose metrics
    # MPJPE and Reconstruction error for the non-parametric and parametric shapes
    mpjpe = np.zeros((len(dataset)))
    recon_err = np.zeros((len(dataset)))

    joint_mapper_h36m = constants.H36M_TO_J14

    # Iterate over the entire dataset
    for step, batch in enumerate(tqdm(data_loader, desc='Eval', total=len(data_loader))):
        # Get ground truth annotations from the batch
        gt_pose = batch['pose'].to(device)
        gt_betas = batch['betas'].to(device)

        gender = batch['gender'].to(device)
        images = batch['img_up']
        curr_batch_size = images.shape[0]

        with torch.no_grad():
            images = images.to(device)
            pred_rotmat, pred_betas, pred_camera, _ = hmr_model(images, scale=size_to_scale(eval_size))
            pred_output = smpl_neutral(betas=pred_betas, body_pose=pred_rotmat[:, 1:],
                                       global_orient=pred_rotmat[:, 0].unsqueeze(1), pose2rot=False)
            pred_vertices = pred_output.vertices

            J_regressor_batch = J_regressor[None, :].expand(pred_vertices.shape[0], -1, -1).to(device)

            gt_vertices = smpl_male(global_orient=gt_pose[:, :3], body_pose=gt_pose[:, 3:], betas=gt_betas).vertices
            gt_vertices_female = smpl_female(global_orient=gt_pose[:, :3], body_pose=gt_pose[:, 3:],
                                             betas=gt_betas).vertices
            gt_vertices[gender == 1, :, :] = gt_vertices_female[gender == 1, :, :]
            gt_keypoints_3d = torch.matmul(J_regressor_batch, gt_vertices)
            gt_pelvis = gt_keypoints_3d[:, [0], :].clone()
            gt_keypoints_3d = gt_keypoints_3d[:, joint_mapper_h36m, :]
            gt_keypoints_3d = gt_keypoints_3d - gt_pelvis

            pred_keypoints_3d = torch.matmul(J_regressor_batch, pred_vertices)
            pred_pelvis = pred_keypoints_3d[:, [0], :].clone()
            pred_keypoints_3d = pred_keypoints_3d[:, joint_mapper_h36m, :]
            pred_keypoints_3d = pred_keypoints_3d - pred_pelvis

            # Absolute error (MPJPE)
            error = torch.sqrt(((pred_keypoints_3d - gt_keypoints_3d) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
            mpjpe[step * batch_size:step * batch_size + curr_batch_size] = error

            # Reconstuction_error
            r_error = reconstruction_error(pred_keypoints_3d.cpu().numpy(), gt_keypoints_3d.cpu().numpy(), reduction=None)
            recon_err[step * batch_size:step * batch_size + curr_batch_size] = r_error

            # Print intermediate results during evaluation
            if step % log_freq == log_freq - 1:
                print('MPJPE: ' + str(1000 * mpjpe[:step * batch_size].mean()))
                print('Reconstruction Error: ' + str(1000 * recon_err[:step * batch_size].mean()))

    # Print final results during evaluation
    print('*** Final Results ***')
    print()
    print('MPJPE: {}'.format(1000 * mpjpe.mean()))
    print('Reconstruction Error: {}'.format(1000 * recon_err.mean()))
    print()


if __name__ == '__main__':
    args = parser.parse_args()

    pkl_path = os.path.join(config.DATASET_PKL_PATH, '3dpw_imgs_test.pkl')
    data_path = config.DATASET_FILES[0]['3dpw']
    ds = PW3D(data_path, pkl_path, args.eval_size)

    hmr_model = hmr(config.SMPL_MEAN_PARAMS)
    checkpoint = torch.load(args.checkpoint)
    hmr_model.load_state_dict(checkpoint, strict=False)
    hmr_model.eval()

    # Run evaluation
    run_evaluation(hmr_model, ds, eval_size=args.eval_size, batch_size=args.batch_size, num_workers=args.num_workers)
