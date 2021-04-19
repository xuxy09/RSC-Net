import torch
import numpy as np
import neural_renderer as nr
from scipy.spatial.transform import Rotation as SciR

class Renderer():
    """Render with neural_renderer
    """
    def __init__(self, focal_length=5000., img_res=224, faces=None):
        # Parameters for rendering
        self.focal_length = focal_length
        self.render_res = img_res
        # We use Neural 3D mesh renderer for rendering masks and part segmentations
        self.neural_renderer = nr.Renderer(dist_coeffs=None, orig_size=self.render_res,
                                           image_size=img_res,
                                           light_intensity_ambient=0.5,
                                           light_intensity_directional=0.5,
                                           light_direction=(1, 0, 1),
                                           anti_aliasing=True)
        
        self.faces = torch.from_numpy(faces.astype(np.int32)).cuda()
        self.device = 'cuda'

    def get_textures(self, color):
        textures = torch.zeros(1, self.faces.shape[0], 1, 1, 1, 3)
        textures[..., 0] = color[0]
        textures[..., 1] = color[1]
        textures[..., 2] = color[2]
        return textures

    def visualize(self, vertices, camera, images, angle=None, color=(0.1,0.6,0.2)):
        """Wrapper function for rendering process."""
        cam_t = camera.to(self.device)
        vertices = vertices.to(self.device)
        batch_size = vertices.shape[0]

        textures = self.get_textures(color).expand(batch_size, -1, -1, -1, -1, -1).to(self.device)
        
        K = torch.eye(3, device=vertices.device)
        K[0,0] = self.focal_length 
        K[1,1] = self.focal_length 
        K[2,2] = 1
        K[0,2] = self.render_res / 2.
        K[1,2] = self.render_res / 2.
        K = K[None, :, :].expand(batch_size, -1, -1)

        if angle is None:
            R = torch.eye(3, device=self.device)[None, :, :].expand(batch_size, -1, -1)
        else:
            R = SciR.from_euler('zyx', angle, degrees=True).as_dcm()
            R = torch.from_numpy(R).type(torch.float32).to(self.device)[None, :, :].expand(batch_size, -1, -1)
        
        faces = self.faces[None, :, :].expand(batch_size, -1, -1)
        rendered, _, mask =  self.neural_renderer(vertices, faces, textures=textures, K=K, R=R, t=cam_t.unsqueeze(1))
        rendered = rendered * mask + images.to(self.device) * (1-mask)
        
        return rendered
