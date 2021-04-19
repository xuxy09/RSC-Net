""" Demo using neural_renderer
"""

import argparse
import imageio
import scipy.misc
from models import hmr, SMPL
import config, constants
import torch
from torchvision.transforms import Normalize
import numpy as np
from utils.renderer_nr import Renderer


parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', required=True, type=str, help='Path to network checkpoint')
parser.add_argument('--img_path', required=True, type=str, help='Testing image path')


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


def get_render_results(vertices, cam_t, renderer):
    rendered_people_view_1 = renderer.visualize(vertices, cam_t, torch.ones((images.size(0), 3, 224, 224)))
    rendered_people_view_2 = renderer.visualize(vertices, cam_t, torch.ones((images.size(0), 3, 224, 224)),
                                                angle=(0, -90, 0))

    return rendered_people_view_1, rendered_people_view_2


if __name__ == '__main__':
    args = parser.parse_args()
    img_path = args.img_path
    checkpoint_path = args.checkpoint

    normalize_img = Normalize(mean=constants.IMG_NORM_MEAN, std=constants.IMG_NORM_STD)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    hmr_model = hmr(config.SMPL_MEAN_PARAMS)
    checkpoint = torch.load(checkpoint_path)
    hmr_model.load_state_dict(checkpoint, strict=False)
    hmr_model.eval()
    hmr_model.to(device)

    smpl_neutral = SMPL(config.SMPL_MODEL_DIR, create_transl=False).to(device)
    img_renderer = Renderer(focal_length=constants.FOCAL_LENGTH, img_res=constants.IMG_RES, faces=smpl_neutral.faces)

    img = imageio.imread(img_path)
    im_size = img.shape[0]
    im_scale = size_to_scale(im_size)
    img_up = scipy.misc.imresize(img, [224, 224])
    img_up = np.transpose(img_up.astype('float32'), (2, 0, 1)) / 255.0
    img_up = normalize_img(torch.from_numpy(img_up).float())
    images = img_up[None].to(device)

    with torch.no_grad():
        pred_rotmat, pred_betas, pred_camera, _ = hmr_model(images, scale=im_scale)
        pred_output = smpl_neutral(betas=pred_betas, body_pose=pred_rotmat[:, 1:],
                                   global_orient=pred_rotmat[:, 0].unsqueeze(1), pose2rot=False)
        pred_vertices = pred_output.vertices

        pred_cam_t = torch.stack([pred_camera[:, 1],
                                  pred_camera[:, 2],
                                  2 * constants.FOCAL_LENGTH / (constants.IMG_RES * pred_camera[:, 0] + 1e-9)],
                                 dim=-1)

    view_1, view_2 = get_render_results(pred_vertices, pred_cam_t, img_renderer)
    view_1 = view_1.cpu()[0].permute(1, 2, 0).numpy()
    view_2 = view_2.cpu()[0].permute(1, 2, 0).numpy()

    tmp = img_path.split('.')
    name_1 = '.'.join(tmp[:-2] + [tmp[-2] + '_view1'] + ['png'])
    name_2 = '.'.join(tmp[:-2] + [tmp[-2] + '_view2'] + ['png'])

    imageio.imwrite(name_1, (view_1 * 255).astype(np.uint8))
    imageio.imwrite(name_2, (view_2 * 255).astype(np.uint8))


