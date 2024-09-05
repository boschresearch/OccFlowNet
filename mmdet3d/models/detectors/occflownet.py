# Copyright (c) 2022 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

from .bevdet import BEVStereo4D

import torch
from mmdet.models import DETECTORS
from mmdet3d.models.builder import build_loss, build_head
from mmcv.cnn.bricks.conv_module import ConvModule
from mmcv.cnn.bricks.transformer import (build_feedforward_network)


@DETECTORS.register_module()
class OccFlowNet(BEVStereo4D):

    def __init__(self,
                 out_dim=64,
                 use_mask=False,
                 num_classes=18,
                 class_wise=False,
                 density_decoder=None,
                 semantic_decoder=None,
                 renderer=None,
                 tv_loss=False,
                 loss_occ_density=None,
                 loss_occ_semantics=None,
                 eval_threshold_range=[.05, .2, .5],
                 **kwargs):
        super(OccFlowNet, self).__init__(**kwargs)
        self.out_dim = out_dim
        self.final_conv = ConvModule(
                        self.img_view_transformer.out_channels,
                        out_dim,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=True,
                        conv_cfg=dict(type='Conv3d'))
        self.pts_bbox_head = None
        self.use_mask = use_mask
        self.num_classes = num_classes
        self.loss_occ_density = build_loss(loss_occ_density) if loss_occ_density is not None else None
        self.loss_occ_semantics = build_loss(loss_occ_semantics) if loss_occ_semantics is not None else None
        self.class_wise = class_wise
        self.align_after_view_transfromation = False
        self.eval_threshold_range = eval_threshold_range

        # self.voxel_decoder =  build_feedforward_network(voxel_decoder) if voxel_decoder is not None else None
        self.density_decoder = build_feedforward_network(density_decoder) if density_decoder is not None else None
        self.semantic_decoder = build_feedforward_network(semantic_decoder) if semantic_decoder is not None else None
        self.renderer = build_head(renderer) if renderer is not None else None

        # additional losses
        self.loss_tv =  build_loss(dict(type='TVLoss3D')) if tv_loss else None

    def loss_single(self, voxel_semantics, mask_camera, density_pred, semantic_pred):
        loss_ = dict()
        voxel_semantics=voxel_semantics.long().reshape(-1)
        density_pred = density_pred.reshape(-1)
        semantic_pred = semantic_pred.reshape(-1, self.num_classes-1)
        semantic_mask = voxel_semantics!=17
        if self.use_mask:
            mask_camera = mask_camera.reshape(-1)
            num_total_samples=mask_camera.sum()
        
            # compute loss
            combined_mask = (semantic_mask * mask_camera.to(torch.bool))
            loss_density = self.loss_occ_density(density_pred, semantic_mask.float(), mask_camera, num_total_samples)
            loss_semantic = self.loss_occ_semantics(semantic_pred[combined_mask], voxel_semantics[combined_mask])
        else:
            loss_density = self.loss_occ_density(density_pred, semantic_mask.float())
            loss_semantic = self.loss_occ_semantics(semantic_pred[semantic_mask], voxel_semantics[semantic_mask])

        loss_['loss_density'] = loss_density
        loss_['loss_semantic'] = loss_semantic

        return loss_

    def simple_test(self,
                    points,
                    img_metas,
                    img=None,
                    **kwargs):
        """Test function without augmentaiton."""
        img_feats, _, _ = self.extract_feat(
            points, img=img, img_metas=img_metas, **kwargs)
        voxel_feats = self.final_conv(img_feats[0]).permute(0, 4, 3, 2, 1) # bncdhw->bnwhdc
        
        out_dict = {}

        density_pred = self.density_decoder(voxel_feats)#.squeeze()
        semantics_pred = self.semantic_decoder(voxel_feats)#.squeeze()

        occupancy = torch.argmax(semantics_pred.squeeze(), dim=-1)
        free_space = torch.stack([density_pred.squeeze() < tr for tr in self.eval_threshold_range])
        out_dict['occupancy'] = occupancy.to(torch.uint8).cpu().numpy()
        out_dict['free_space'] =  free_space.cpu().numpy()

        return [out_dict]

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      img_inputs=None,
                      origins=None,
                      directions=None,
                      ray_dataset=None,
                      voxel_semantics=None,
                      mask_camera=None,
                      gt_depth=None,
                      voxel_flow=None,
                      flow_transforms=None,
                      box_indices=None,
                      **kwargs):
        
        # imgs, sensor2keyegos, ego2globals, intrins, post_rots, post_trans, bda  = self.prepare_inputs(img_inputs)
        img_feats, pts_feats, depth = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas, **kwargs)
        losses = dict()
        loss_depth = self.img_view_transformer.get_depth_loss(gt_depth, depth)
        losses['loss_depth'] = loss_depth

        voxel_feats = self.final_conv(img_feats[0]).permute(0, 4, 3, 2, 1) # bcdhw->bwhdc

        density_pred = self.density_decoder(voxel_feats)
        semantics_pred = self.semantic_decoder(voxel_feats)

        if self.loss_occ_density is not None and self.loss_occ_semantics is not None:
            assert voxel_semantics.min() >= 0 and voxel_semantics.max() <= 17
            loss_occ = self.loss_single(voxel_semantics, mask_camera, density_pred, semantics_pred)
            losses.update(loss_occ)

        if self.renderer is not None:
            voxel_outs = [density_pred.permute(0, 3, 2, 1, 4), semantics_pred.permute(0, 3, 2, 1, 4), None, voxel_flow, flow_transforms, box_indices]
            
            if self.loss_tv is not None:
                losses['loss_tv'] = self.loss_tv(voxel_outs)

            render_preds = self.renderer(voxel_outs, origins, directions, ray_dataset, img_inputs[-1])
            render_losses = self.renderer.calculate_losses(render_preds, ray_dataset)
            losses.update(render_losses)      

        return losses