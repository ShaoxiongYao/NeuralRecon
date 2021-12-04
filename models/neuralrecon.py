import torch
import torch.nn as nn
from torch.nn.functional import interpolate
import torchvision
import cv2
import numpy as np

from .backbone import MnasMulti
from .neucon_network import NeuConNet
from .gru_fusion import GRUFusion
from utils import tocuda


class NeuralRecon(nn.Module):
    '''
    NeuralRecon main class.
    '''

    def __init__(self, cfg, crop_dynamic=None):
        super(NeuralRecon, self).__init__()
        self.cfg = cfg.MODEL
        alpha = float(self.cfg.BACKBONE2D.ARC.split('-')[-1])
        # other hparams
        self.pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1)
        self.pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1)
        self.n_scales = len(self.cfg.THRESHOLDS) - 1

        # networks
        self.backbone2d = MnasMulti(alpha)
        self.neucon_net = NeuConNet(cfg.MODEL)
        # for fusing to global volume
        self.fuse_to_global = GRUFusion(cfg.MODEL, direct_substitute=True)

        self.crop_dynamic = crop_dynamic
        self.seg_model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)


    def normalizer(self, x):
        """ Normalizes the RGB images to the input range"""
        return (x - self.pixel_mean.type_as(x)) / self.pixel_std.type_as(x)
    
    def get_person_mask(self, img, feat_shape_lst):
        img_tensor = img[0, :, :, :] / 255.0

        with torch.no_grad():
            outputs = self.seg_model([img_tensor])

        person_idx = (outputs[0]['labels'] == 1).nonzero()

        person_mask_lst = []
        if person_idx.shape[0] != 0:
            # takes the person with highest probability
            person_idx = person_idx.flatten()[0]
            person_mask = outputs[0]['masks'][person_idx, 0, :, :]

            for mask_h, mask_w in feat_shape_lst:
                resized_mask_tensor = interpolate(person_mask[None, None, :, :], 
                                                  size=(mask_h, mask_w), mode='bilinear')
                binary_mask = torch.round(1-resized_mask_tensor)
                person_mask_lst.append(binary_mask)

        return person_mask_lst
        

    def forward(self, inputs, save_mesh=False):
        '''

        :param inputs: dict: {
            'imgs':                    (Tensor), images,
                                    (batch size, number of views, C, H, W)
            'vol_origin':              (Tensor), origin of the full voxel volume (xyz position of voxel (0, 0, 0)),
                                    (batch size, 3)
            'vol_origin_partial':      (Tensor), origin of the partial voxel volume (xyz position of voxel (0, 0, 0)),
                                    (batch size, 3)
            'world_to_aligned_camera': (Tensor), matrices: transform from world coords to aligned camera coords,
                                    (batch size, number of views, 4, 4)
            'proj_matrices':           (Tensor), projection matrix,
                                    (batch size, number of views, number of scales, 4, 4)
            when we have ground truth:
            'tsdf_list':               (List), tsdf ground truth for each level,
                                    [(batch size, DIM_X, DIM_Y, DIM_Z)]
            'occ_list':                (List), occupancy ground truth for each level,
                                    [(batch size, DIM_X, DIM_Y, DIM_Z)]
            others: unused in network
        }
        :param save_mesh: a bool to indicate whether or not to save the reconstructed mesh of current sample
        :return: outputs: dict: {
            'coords':                  (Tensor), coordinates of voxels,
                                    (number of voxels, 4) (4 : batch ind, x, y, z)
            'tsdf':                    (Tensor), TSDF of voxels,
                                    (number of voxels, 1)
            When it comes to save results:
            'origin':                  (List), origin of the predicted partial volume,
                                    [3]
            'scene_tsdf':              (List), predicted tsdf volume,
                                    [(nx, ny, nz)]
        }
                 loss_dict: dict: {
            'tsdf_occ_loss_X':         (Tensor), multi level loss
            'total_loss':              (Tensor), total loss
        }
        '''
        inputs = tocuda(inputs)
        outputs = {}
        imgs = torch.unbind(inputs['imgs'], 1)

        # generate person mask

        if self.crop_dynamic == 'images':
            feat_shape_lst = [(480, 640)]
            person_mask_lst = [self.get_person_mask(img, feat_shape_lst) for img in imgs]

            masked_image = []
            for frame_idx, img in enumerate(imgs):
                if len(person_mask_lst[frame_idx]) != 0:
                    img *= person_mask_lst[frame_idx][0]
                masked_image.append(img)

                # visualize mask
                # vis_img = img.cpu().numpy()[0].transpose(1,2,0)
                # norm_image = cv2.normalize(vis_img, None, alpha = 0, beta = 255, 
                #                            norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
                # norm_image = norm_image.astype(np.uint8)
                # norm_image = cv2.cvtColor(norm_image, cv2.COLOR_BGR2RGB)
                # cv2.imshow("cropped image", norm_image)
                # cv2.waitKey(0)
            imgs = masked_image

        # image feature extraction
        # in: images; out: feature maps
        # feature dimensions: 120x160, 60x80, 30x40
        features = [self.backbone2d(self.normalizer(img)) for img in imgs]

        if self.crop_dynamic == 'features':
            feat_shape_lst = [(120,160), (60,80), (30,40)]
            person_mask_lst = [self.get_person_mask(img, feat_shape_lst) for img in imgs]
            for frame_idx in range(len(features)):
                if len(person_mask_lst[frame_idx]) == 0:
                    continue
                for feat_idx in range(3):
                    features[frame_idx][feat_idx] *= person_mask_lst[frame_idx][feat_idx]

        # coarse-to-fine decoder: SparseConv and GRU Fusion.
        # in: image feature; out: sparse coords and tsdf
        outputs, loss_dict = self.neucon_net(features, inputs, outputs)

        # fuse to global volume.
        if not self.training and 'coords' in outputs.keys():
            outputs = self.fuse_to_global(outputs['coords'], outputs['tsdf'], inputs, self.n_scales, outputs, save_mesh)

        # gather loss.
        print_loss = 'Loss: '
        for k, v in loss_dict.items():
            print_loss += f'{k}: {v} '

        weighted_loss = 0

        for i, (k, v) in enumerate(loss_dict.items()):
            weighted_loss += v * self.cfg.LW[i]

        loss_dict.update({'total_loss': weighted_loss})
        return outputs, loss_dict
