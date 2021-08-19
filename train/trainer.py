import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets import MixedDataset
from models import hmr, SMPL, NTXent, FeatQueue
from utils.geometry import batch_rodrigues, perspective_projection, estimate_translation
from utils.renderer import Renderer
from utils import BaseTrainer

import config
import constants


class Trainer(BaseTrainer):
    
    def init_fn(self):
        self.train_ds = MixedDataset(self.options, ignore_3d=self.options.ignore_3d, is_train=True)

        self.model = hmr(config.SMPL_MEAN_PARAMS, pretrained=True).to(self.device)

        self.optimizer = torch.optim.Adam(params=self.model.parameters(),
                                          lr=self.options.lr)
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.95)


        self.smpl = SMPL(config.SMPL_MODEL_DIR,
                         batch_size=self.options.batch_size,
                         create_transl=False).to(self.device)

        # consistency loss
        self.criterion_consistency_contrastive = NTXent(tau=self.options.tau, kernel=self.options.kernel).to(self.device)
        self.criterion_consistency_mse = nn.MSELoss().to(self.device)
        # Per-vertex loss on the shape
        self.criterion_shape = nn.L1Loss().to(self.device)
        # Keypoint (2D and 3D) loss
        # No reduction because confidence weighting needs to be applied
        self.criterion_keypoints = nn.MSELoss(reduction='none').to(self.device)
        # Loss for SMPL parameter regression
        self.criterion_regr = nn.MSELoss().to(self.device)
        self.models_dict = {'model': self.model}
        self.optimizers_dict = {'optimizer': self.optimizer}
        self.focal_length = constants.FOCAL_LENGTH

        if self.options.pretrained_checkpoint is not None:
            self.load_pretrained(checkpoint_file=self.options.pretrained_checkpoint)

        # Create renderer
        self.renderer = Renderer(focal_length=self.focal_length, img_res=self.options.img_res, faces=self.smpl.faces)

        # Create input image flag
        self.input_img = self.options.input_img

        # initialize queue
        self.feat_queue = FeatQueue(max_queue_size=self.options.max_queue_size)


    def keypoint_loss(self, pred_keypoints_2d, gt_keypoints_2d, openpose_weight, gt_weight):
        """ Compute 2D reprojection loss on the keypoints.
        The loss is weighted by the confidence.
        The available keypoints are different for each dataset.
        """
        conf = gt_keypoints_2d[:, :, -1].unsqueeze(-1).clone()
        conf[:, :25] *= openpose_weight
        conf[:, 25:] *= gt_weight
        loss = (conf * self.criterion_keypoints(pred_keypoints_2d, gt_keypoints_2d[:, :, :-1]))
        return loss.mean()

    def keypoint_3d_loss(self, pred_keypoints_3d, gt_keypoints_3d, has_pose_3d):
        """Compute 3D keypoint loss for the examples that 3D keypoint annotations are available.
        The loss is weighted by the confidence.
        """
        pred_keypoints_3d = pred_keypoints_3d[:, 25:, :]
        conf = gt_keypoints_3d[:, :, -1].unsqueeze(-1).clone()
        gt_keypoints_3d = gt_keypoints_3d[:, :, :-1].clone()
        gt_keypoints_3d = gt_keypoints_3d[has_pose_3d == 1]
        conf = conf[has_pose_3d == 1]
        pred_keypoints_3d = pred_keypoints_3d[has_pose_3d == 1]
        if len(gt_keypoints_3d) > 0:
            gt_pelvis = (gt_keypoints_3d[:, 2,:] + gt_keypoints_3d[:, 3,:]) / 2
            gt_keypoints_3d = gt_keypoints_3d - gt_pelvis[:, None, :]
            pred_pelvis = (pred_keypoints_3d[:, 2,:] + pred_keypoints_3d[:, 3,:]) / 2
            pred_keypoints_3d = pred_keypoints_3d - pred_pelvis[:, None, :]
            return (conf * self.criterion_keypoints(pred_keypoints_3d, gt_keypoints_3d)).mean()
        else:
            return torch.FloatTensor(1).fill_(0.).to(self.device)

    def shape_loss(self, pred_vertices, gt_vertices, has_smpl):
        """Compute per-vertex loss on the shape for the examples that SMPL annotations are available."""
        pred_vertices_with_shape = pred_vertices[has_smpl == 1]
        gt_vertices_with_shape = gt_vertices[has_smpl == 1]
        if len(gt_vertices_with_shape) > 0:
            return self.criterion_shape(pred_vertices_with_shape, gt_vertices_with_shape)
        else:
            return torch.FloatTensor(1).fill_(0.).to(self.device)

    def smpl_losses(self, pred_rotmat, pred_betas, pred_cam_t, gt_pose, gt_betas, gt_cam_t, has_smpl):
        pred_rotmat_valid = pred_rotmat[has_smpl == 1]
        gt_rotmat_valid = batch_rodrigues(gt_pose.view(-1,3)).view(-1, 24, 3, 3)[has_smpl == 1]
        pred_betas_valid = pred_betas[has_smpl == 1]
        gt_betas_valid = gt_betas[has_smpl == 1]
        pred_cam_t_valid = pred_cam_t[has_smpl == 1]
        gt_cam_t_valid = gt_cam_t[has_smpl == 1]

        if len(pred_rotmat_valid) > 0:
            loss_regr_pose = self.criterion_regr(pred_rotmat_valid, gt_rotmat_valid)
            loss_regr_betas = self.criterion_regr(pred_betas_valid, gt_betas_valid)
            loss_regr_cam_t = self.criterion_regr(pred_cam_t_valid, gt_cam_t_valid)
        else:
            loss_regr_pose = torch.FloatTensor(1).fill_(0.).to(self.device)
            loss_regr_betas = torch.FloatTensor(1).fill_(0.).to(self.device)
            loss_regr_cam_t = torch.FloatTensor(1).fill_(0.).to(self.device)
        return loss_regr_pose, loss_regr_betas, loss_regr_cam_t

    def consistency_losses(self,
                           pred_rotmat, pred_betas, pred_cam_t, pred_feat_list,
                           gt_rotmat, gt_betas, gt_cam_t, gt_feat_list,
                           neg_feat_list):
        loss_pose = self.criterion_consistency_mse(pred_rotmat, gt_rotmat)
        loss_betas = self.criterion_consistency_mse(pred_betas, gt_betas)
        loss_cam_t = self.criterion_consistency_mse(pred_cam_t, gt_cam_t)
        loss_consistency_mse = self.options.pose_loss_weight * loss_pose +  self.options.beta_loss_weight * loss_betas + self.options.cam_loss_weight * loss_cam_t
        loss_feat = 0
        for pred_feat, gt_feat, neg_feat in zip(pred_feat_list, gt_feat_list, neg_feat_list):
            loss_feat += self.criterion_consistency_contrastive(pred_feat, gt_feat, neg_feat)
        loss_feat = self.options.feat_loss_weight * loss_feat
        loss =  loss_consistency_mse + loss_feat
        return loss, loss_consistency_mse, loss_feat

    def train_step(self, input_batch):
        self.model.train()

        images_hr = input_batch['img_hr']
        images_lr_list = input_batch['img_lr']
        images_list = [images_hr] + images_lr_list
        scale_names = ['224', '224_128', '128_64', '64_40', '40_24']
        scale_names = scale_names[:len(images_list)]
        feat_names = ['layer4']

        # Get data from the batch
        gt_keypoints_2d = input_batch['keypoints'] # 2D keypoints
        gt_pose = input_batch['pose'] # SMPL pose parameters
        gt_betas = input_batch['betas'] # SMPL beta parameters
        gt_joints = input_batch['pose_3d'] # 3D pose
        has_smpl = input_batch['has_smpl'].byte() # flag that indicates whether SMPL parameters are valid
        has_pose_3d = input_batch['has_pose_3d'].byte() # flag that indicates whether 3D pose is valid
        dataset_name = input_batch['dataset_name'] # name of the dataset the image comes from
        indices = input_batch['sample_index'].numpy() # index of example inside mixed dataset
        batch_size = images_hr.shape[0]


        # Get GT vertices and model joints
        # Note that gt_model_joints is different from gt_joints as it comes from SMPL
        gt_out = self.smpl(betas=gt_betas, body_pose=gt_pose[:,3:], global_orient=gt_pose[:,:3])
        gt_model_joints = gt_out.joints
        gt_vertices = gt_out.vertices

        # De-normalize 2D keypoints from [-1,1] to pixel space
        gt_keypoints_2d_orig = gt_keypoints_2d.clone()
        gt_keypoints_2d_orig[:, :, :-1] = 0.5 * self.options.img_res * (gt_keypoints_2d_orig[:, :, :-1] + 1)

        # Estimate camera translation given the model joints and 2D keypoints
        # by minimizing a weighted least squares loss
        gt_cam_t = estimate_translation(gt_model_joints, gt_keypoints_2d_orig, focal_length=self.focal_length, img_size=self.options.img_res)

        loss_shape = 0
        loss_keypoints = 0
        loss_keypoints_3d = 0
        loss_regr_pose = 0
        loss_regr_betas = 0
        loss_regr_cam_t = 0
        smpl_outputs = []
        for i, (images, scale_name) in enumerate(zip(images_list, scale_names[:len(images_list)])):
            images = images.to(self.device)
            # Feed images in the network to predict camera and SMPL parameters
            pred_rotmat, pred_betas, pred_camera, feat_list = self.model(images, scale=i)

            pred_output = self.smpl(betas=pred_betas, body_pose=pred_rotmat[:,1:], global_orient=pred_rotmat[:,0].unsqueeze(1), pose2rot=False)
            pred_vertices = pred_output.vertices
            pred_joints = pred_output.joints

            # Convert Weak Perspective Camera [s, tx, ty] to camera translation [tx, ty, tz] in 3D given the bounding box size
            # This camera translation can be used in a full perspective projection
            pred_cam_t = torch.stack([pred_camera[:,1],
                                      pred_camera[:,2],
                                      2*self.focal_length/(self.options.img_res * pred_camera[:,0] +1e-9)],dim=-1)

            camera_center = torch.zeros(batch_size, 2, device=self.device)
            pred_keypoints_2d = perspective_projection(pred_joints,
                                                       rotation=torch.eye(3, device=self.device).unsqueeze(0).expand(batch_size, -1, -1),
                                                       translation=pred_cam_t,
                                                       focal_length=self.focal_length,
                                                       camera_center=camera_center)
            # Normalize keypoints to [-1,1]
            pred_keypoints_2d = pred_keypoints_2d / (self.options.img_res / 2.)

            # Compute loss on SMPL parameters
            loss_pose, loss_betas, loss_cam_t = self.smpl_losses(pred_rotmat, pred_betas, pred_cam_t, gt_pose, gt_betas, gt_cam_t, has_smpl)
            loss_regr_pose = loss_regr_pose + (i + 1) * loss_pose
            loss_regr_betas = loss_regr_betas + (i + 1) * loss_betas
            loss_regr_cam_t = loss_regr_cam_t + (i + 1) * loss_cam_t

            # Compute 2D reprojection loss for the keypoints
            loss_keypoints = loss_keypoints + (i + 1) * self.keypoint_loss(pred_keypoints_2d, gt_keypoints_2d,
                                                        self.options.openpose_train_weight,
                                                        self.options.gt_train_weight)

            # Compute 3D keypoint loss
            loss_keypoints_3d = loss_keypoints_3d + (i + 1) * self.keypoint_3d_loss(pred_joints, gt_joints, has_pose_3d)

            # Per-vertex loss for the shape
            loss_shape = loss_shape + (i+1) * self.shape_loss(pred_vertices, gt_vertices, has_smpl)

            # save pred_rotmat, pred_betas, pred_cam_t for later, from large images to smaller images
            smpl_outputs.append([pred_rotmat, pred_betas, pred_cam_t, feat_list])

            # update queue size
            self.feat_queue.update_queue_size(batch_size)
            # update the queue
            self.feat_queue.update_all([feat.detach() for feat in feat_list], [name for name in feat_names])
            # update dataset name and index for each scale
            self.feat_queue.update('dataset_names', np.array(dataset_name))
            self.feat_queue.update('dataset_indices', indices)

        # Compute total loss except the consistency loss
        loss = self.options.shape_loss_weight * loss_shape +\
               self.options.keypoint_loss_weight * loss_keypoints + \
               self.options.keypoint_loss_weight * loss_keypoints_3d +\
               self.options.pose_loss_weight * loss_regr_pose + \
               self.options.beta_loss_weight * loss_regr_betas + \
               self.options.cam_loss_weight * loss_regr_cam_t
        loss = loss / len(images_list)

        # compute the consistency loss
        loss_consistency = 0
        for i in range(len(smpl_outputs)):
            gt_rotmat, gt_betas, gt_cam_t, gt_feat_list = smpl_outputs[i]
            gt_rotmat = gt_rotmat.detach()
            gt_betas = gt_betas.detach()
            gt_cam_t = gt_cam_t.detach()
            gt_feat_list = [feat.detach() for feat in gt_feat_list]
            # sample negative index
            indices_list = self.feat_queue.select_indices(dataset_name, indices, self.options.sample_size)
            neg_feat_list = self.feat_queue.batch_sample_all(indices_list, names=feat_names)
            for j in range(i+1, len(smpl_outputs)):
                # compute the consistency loss from high to low: 1:2, 1:3, 2:3 and weighted by 1/(j-i)
                pred_rotmat, pred_betas, pred_cam_t, pred_feat_list = smpl_outputs[j]
                loss_consistency_total, loss_consistency_smpl, loss_consistency_feat = self.consistency_losses(pred_rotmat, pred_betas, pred_cam_t, pred_feat_list, gt_rotmat, gt_betas, gt_cam_t, gt_feat_list, neg_feat_list)
                loss_consistency = loss_consistency + ((j - i) / len(smpl_outputs)) * loss_consistency_total
        loss_consistency = loss_consistency * self.consistency_loss_ramp * self.options.consistency_loss_weight

        loss += loss_consistency
        loss *= 60

        # Do backprop
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Pack output arguments
        output = {'pred_vertices': pred_vertices.detach(),
                  'pred_cam_t': pred_cam_t.detach()}
        losses = {'lr': self.optimizer.param_groups[0]['lr'],
                  'loss_ramp': self.consistency_loss_ramp,
                  'loss': loss.detach().item(),
                  'loss_consistency': loss_consistency.detach().item(),
                  'loss_consistency_smpl':loss_consistency_smpl.detach().item(),
                  'loss_consistency_feat':loss_consistency_feat.detach().item(),
                  'loss_keypoints': loss_keypoints.detach().item(),
                  'loss_keypoints_3d': loss_keypoints_3d.detach().item(),
                  'loss_regr_pose': loss_regr_pose.detach().item(),
                  'loss_regr_betas': loss_regr_betas.detach().item(),
                  'loss_shape': loss_shape.detach().item()}

        return output, losses


