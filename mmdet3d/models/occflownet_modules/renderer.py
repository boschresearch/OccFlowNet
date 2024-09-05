# Copyright (c) 2022 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

from abc import abstractmethod
import math
from typing import List

import nerfacc
from nerfacc.pdf import importance_sampling
from nerfacc.data_specs import RayIntervals
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import WeightedRandomSampler
from mmdet3d.models.builder import HEADS, build_loss, build_head
from matplotlib import cm
from .samplers import Sampler, UniformSampler, PDFSampler
from .utils import (interpolate_values, interpolate_values_tf,
                    interpolate_values_oob_unmasked, interpolate_values_flow)

color_map = np.array(
        [[0, 0, 0, 255],
        [255, 120, 50, 255],  # barrier orangey
        [255, 192, 203, 255],  # bicycle pink
        [255, 255, 0, 255],  # bus yellow
        [0, 150, 245, 255],  # car blue
        [0, 255, 255, 255],  # construction_vehicle cyan
        [200, 180, 0, 255],  # motorcycle dark orange
        [255, 0, 0, 255],  # pedestrian red
        [255, 240, 150, 255],  # traffic_cone light yellow
        [135, 60, 0, 255],  # trailer brown
        [160, 32, 240, 255],  # truck purple
        [255, 0, 255, 255],  # driveable_surface dark pink
        [139, 137, 137, 255],  # other_flat dark red
        [75, 0, 75, 255],  # sidewalk dark purple
        [150, 240, 80, 255],  # terrain light green
        [230, 230, 250, 255],  # manmade white
        [0, 175, 0, 255],  # vegetation green
        [0, 255, 127, 255],  # ego car dark cyan
        [255, 99, 71, 255],
        [0, 191, 255, 255]
    ])
        
class RenderModule(nn.Module):
    def __init__(self, output_size, name, loss=None):
        super().__init__()
        self.output_size = output_size
        self.name = name

        self.loss = build_loss(loss)

    @abstractmethod
    def get_value(self, samples, samples_start, samples_end, values, ray_indices, pc_range):
        """Get the value to integrate over"""
        
    @abstractmethod
    def get_value_oob(self, samples, samples_centers, values, ray_indices, oob_mask, pc_range):
        """Get the value to integrate over, with oob masking"""
    
    @abstractmethod
    def get_loss(self, pred, gt, ray_weights):
        """Compute the pixel-wise loss"""

    @abstractmethod
    def background_model(self, result, renderer):
        """Apply background model"""

    @abstractmethod
    def get_output(self, result, renderer):
        """Convert output to a plottable image"""

@HEADS.register_module()
class DepthRenderModule(RenderModule):
    def __init__(self, loss_cfg= dict(type='MSELoss', loss_weight=1.0), *args, **kwargs):
        super().__init__(1, 'depth', *args, loss=loss_cfg, **kwargs)
        self.color_map = cm.get_cmap('magma_r')
        
    def get_value(self, samples, samples_start, samples_end, values, ray_indices, pc_range):
        # value to integrate over = distance to ray origin
        return ((samples_start + samples_end) / 2.)[..., None]
    
    def get_value_oob(self, samples, samples_centers, values, ray_indices, oob_mask, pc_range):
        # value to integrate over = distance to ray origin
        return samples_centers[oob_mask].unsqueeze(-1)
    
    def get_loss(self, pred, gt, ray_weights):
        loss = self.loss(pred.squeeze(), gt, weight=ray_weights)
        return loss

    def get_output(self, result, renderer):
        """Normalize depth and apply jet color map"""
        result = result.squeeze(-1)
        normalized_depth = ((result - renderer.near) / (renderer.far - renderer.near)).cpu().numpy()
        emtpy_ray_mask = normalized_depth <= 1e-2
        colored_depth_image = (self.color_map(normalized_depth)[..., :3] * 255).astype(np.uint8)
        colored_depth_image[emtpy_ray_mask] = np.array([0,0,0], dtype=np.uint8)
        # Swap from RGB -> BGR
        colored_depth_image = colored_depth_image[:, [2, 1, 0]]
        return colored_depth_image

    def background_model(self, result, renderer):
        emtpy_ray_mask = result <= 1e-3
        result[emtpy_ray_mask] = renderer.far
        return result

@HEADS.register_module()
class SemanticRenderModule(RenderModule):
    def __init__(self, *args, loss_cfg=dict(type='CrossEntropyLoss', loss_weight=1.0), num_classes=17, **kwargs):
        super().__init__(num_classes, 'semantics', *args, loss=loss_cfg, **kwargs)
        self.num_classes = num_classes
        self.mask_tensor = torch.zeros(num_classes)
        self.mask_tensor[-1] = 1.
        self.color_map = color_map

    def get_loss(self, pred, gt, ray_weights):
        loss = self.loss(pred, gt, weight=ray_weights)
        return loss
    
    def get_value(self, samples, samples_start, samples_end, values, ray_indices, pc_range):
        # Value to integrate over = predicted class logits from semantic head
        # -> Interpolate logits at sample positions
        if values[1].dim() == 5:
            sampled_logits = interpolate_values(values[1], samples, ray_indices, pc_range, None)
        elif values[1].dim() == 6:
            sampled_logits = interpolate_values_flow(values[1], samples, ray_indices, pc_range, None)
        else:
            assert False, f"Semantics should be of dimension 5 or 6, got {values[1].dim()}"
        return sampled_logits
    
    def get_value_oob(self, samples, samples_centers, values, ray_indices, oob_mask, pc_range):
        # Value to integrate over = predicted class logits from semantic head
        # -> Interpolate logits at sample positions
        sampled_logits = interpolate_values_oob_unmasked(values[1], samples, ray_indices, oob_mask, pc_range)
        return sampled_logits
    
    def get_output(self, result, renderer):
        classes = result.argmax(-1)
        colored_image = self.color_map[classes.cpu().numpy()][..., :3]
        # Swap from RGB -> BGR
        colored_image = np.stack((colored_image[..., 2], colored_image[..., 1], colored_image[..., 0]), axis=-1).astype(np.uint8)
        return colored_image

@HEADS.register_module()
class Renderer(nn.Module):
    def __init__(self,
                coarse_sampler: Sampler = None,
                fine_sampler: Sampler = None,
                render_modules: List[dict] = None,
                render_range = [0.05, 40],
                grid_cfg=None,
                pc_range=None,
                rays_batch_size=32768,
                max_batches=1,
                prop_samples_per_ray=50,
                samples_per_ray=50,
                use_proposal=True,
                ignore_classes=None,
                temporal_filter=False,
                render_frame_ids=[0],
                dist_loss=False,
                class_balanced_loss=False,
                use_lidarseg_stats=False,
                adjust_weights_temporal=False,
                log_weighting=False,
                ens_weighting=None,
                weight_exponent=.5,
                global_weights=True,
                distance_scaled_loss=False,
                scale_ray_weights=False,
                wrs=False,
                use_flow=False,
                interpolate_flow=False):
        super().__init__()

        # Create render modules
        assert render_modules is not None, "Please provide at least one render module."
        self.render_modules = nn.ModuleList()
        for render_module in render_modules:
            self.render_modules.append(build_head(render_module))

        self.near = render_range[0]
        self.far = render_range[1]
        self.coarse_sampler = coarse_sampler if coarse_sampler is not None else UniformSampler()
        self.fine_sampler = fine_sampler if fine_sampler is not None else PDFSampler()
        self.rays_batch_size = rays_batch_size
        self.samples_per_ray = samples_per_ray
        self.prop_samples_per_ray = prop_samples_per_ray
        self.use_proposal = use_proposal
        self.ignore_classes = torch.tensor(ignore_classes) if ignore_classes else None
        self.max_batches = max_batches
        self.temporal_filter = temporal_filter
        self.use_flow = use_flow
        self.flow_function = self.apply_flow if interpolate_flow else self.apply_flow_clipped

        assert len(render_frame_ids)>0, "Please provide at least a single render frame"
        self.current_timestep = render_frame_ids.index(0)
        self.render_frame_ids = render_frame_ids
        if temporal_filter and self.ignore_classes is not None:
            self.interpolate_fn = interpolate_values_tf
        elif use_flow:
            self.interpolate_fn = interpolate_values_flow
        else:
             self.interpolate_fn = interpolate_values

        # Class balancing
        self.class_balanced_loss = class_balanced_loss
        self.distance_scaled_loss = distance_scaled_loss
        self.scale_ray_weights = scale_ray_weights
        wr_nusc_class_frequencies = torch.tensor([2854504, 7291443, 141614, 4239939, 32248552, 1583610, 364372, 2346381, 582961, 
                                                4829021, 14073691, 191019309, 6249651, 55095657, 58484771, 193834360, 131378779])
        
        nusc_class_frequencies = torch.tensor([347376232,   7805289,    126214,   4351628,  36046855,   1153917,
            411093,   2137060,    636812,   4397702,  14448748, 316958899,
            8559216,  70197461,  70289730, 178178063, 122581273]) if use_lidarseg_stats else torch.tensor([1163161, 2309034, 188743, 2997643, 20317180, 852476, 243808, 2457947, 
            497017, 2731022, 7224789, 214411435, 5565043, 63191967, 76098082, 128860031, 
            141625221])
        
        if adjust_weights_temporal: 
            nusc_class_frequencies[self.ignore_classes] = nusc_class_frequencies[self.ignore_classes] // (len(self.render_frame_ids)-1)
        
        if class_balanced_loss:
            if log_weighting:
                log_weights = torch.log(nusc_class_frequencies.sum() / nusc_class_frequencies)
                self.class_weight = (log_weights.numel() / log_weights.sum()) * log_weights # scale to 1 mean
            else:
                weights = (1 / torch.pow(nusc_class_frequencies, weight_exponent))
                self.class_weight =  weights / weights.sum() * len(nusc_class_frequencies)
            self.global_weights = global_weights
            self.log_weighting = log_weighting
            self.ens_weighting = ens_weighting
            self.weight_exponent = weight_exponent

        # WRS -> For comparing RenderOcc setting
        self.wrs = wrs
        if wrs:
            self.sample_weights = torch.exp(0.005 * (wr_nusc_class_frequencies.max() / wr_nusc_class_frequencies - 1))

        # Grid config
        self.upsample = int(grid_cfg[4])
        self.voxel_resolution = grid_cfg[3] / self.upsample
        self.bev_h = int(grid_cfg[0] * self.upsample)
        self.bev_w = int(grid_cfg[1] * self.upsample)
        self.bev_z = int(grid_cfg[2] * self.upsample)
        self.pc_range = pc_range
        self.dist_loss = build_loss(dist_loss) if dist_loss else None

        # generate pc_grid_centers if using flow
        if use_flow:
            self.pc_grid = self.generate_pc_grid()

    def generate_pc_grid(self):
        # [16,200,200]
        zs = torch.linspace(0.5 * self.voxel_resolution  + self.pc_range[2], self.pc_range[5] - 0.5 * self.voxel_resolution, self.bev_z).view(self.bev_z, 1, 1).expand(self.bev_z, self.bev_w, self.bev_h)
        xs = torch.linspace(0.5 * self.voxel_resolution  + self.pc_range[0], self.pc_range[3] - 0.5 * self.voxel_resolution, self.bev_w).view(1, self.bev_w, 1).expand(self.bev_z, self.bev_w, self.bev_h)
        ys = torch.linspace(0.5 * self.voxel_resolution  + self.pc_range[1], self.pc_range[4] - 0.5 * self.voxel_resolution, self.bev_h).view(1, 1, self.bev_h).expand(self.bev_z, self.bev_w, self.bev_h)
        
        ref_3d = torch.stack((ys, xs, zs), -1)
        return ref_3d

    def generate_samples(self, origins, directions, ray_indices, density):
        if self.use_proposal:
            # Initial uniform sampling
            init_samples_start, init_samples_end, init_samples_centers = self.coarse_sampler(origins, directions, self.prop_samples_per_ray, self.near, self.far)
            samples = self.create_3d_samples(origins, directions, init_samples_centers)
            init_density = self.interpolate_fn(density, samples, ray_indices, self.pc_range, self.current_timestep)
            _, init_transmittance, _ = nerfacc.render_weight_from_density(init_samples_start, init_samples_end, init_density.squeeze(-1))
            
            # pdf sampler using nerfacc functions
            cdfs = 1.0 - torch.cat([init_transmittance, torch.zeros_like(init_transmittance[:, :1])], dim=-1)
            intervals = RayIntervals(vals=torch.cat([init_samples_start, init_samples_end[:, -1:]], dim=-1))
            sampled_intervals, sampled_samples = importance_sampling(intervals, cdfs, self.samples_per_ray)
            samples_start, samples_end, samples_centers = sampled_intervals.vals[:, :-1], sampled_intervals.vals[:, 1:], sampled_samples.vals
        else:
            # only use coarse sampling (e.g., uniform)
            samples_start, samples_end, samples_centers = self.coarse_sampler(origins, directions, self.samples_per_ray, self.near, self.far)

        return samples_start, samples_end, samples_centers
    
    def create_static_occupancy(self, voxel_outs):
        static_density = voxel_outs[0].clone()
        dynamic_mask = ((voxel_outs[1].argmax(-1)[..., None] - self.ignore_classes.to(voxel_outs[1].device))==0).any(-1)
        static_density[dynamic_mask] = 0.
        return static_density
    
    def create_3d_samples(self, origins, directions, sample_centers):
        return origins[:, None, :] + sample_centers[..., None] * directions[:, None, :]
        
    def apply_flow(self, voxel_outs, bda):
        """"
            Move voxels according to flow by inverse sampling to target positions.
        """
        predicted_classes = torch.argmax(voxel_outs[1], dim=-1).byte()#.flatten(1, )
        
        temporal_density = voxel_outs[0][:, None, ...].repeat(1, len(self.render_frame_ids), 1, 1, 1, 1)
        temporal_semantics = voxel_outs[1][:, None, ...].repeat(1, len(self.render_frame_ids), 1, 1, 1, 1)

        # create batched source and target positions
        source_positions = torch.cat([b[:, :3]  for b in voxel_outs[3]], dim=0)
        target_positions_floor = torch.cat([b[:, 6:9]  for b in voxel_outs[3]], dim=0)
        source_labels = torch.cat([b[:, 9] for b in voxel_outs[3]], dim=0)
        batch_indices = torch.cat([source_positions.new_full((len(b), 1), i) for i, b in enumerate(voxel_outs[3])], dim=0)
        temporal_indices = torch.cat([b[:, 10] for b in voxel_outs[3]], dim=0).unsqueeze(-1)
        transforms = torch.cat(voxel_outs[4])
        box_indices = torch.cat(voxel_outs[5])
        bda = bda.to(transforms)

        # flip source positions according to bda
        source_positions_bda = source_positions.clone()
        for b, bda_b in enumerate(bda):
            if bda_b[0, 0] == -1:
                source_positions_bda[batch_indices.squeeze()==b, 2] = 199 -  source_positions_bda[batch_indices.squeeze()==b, 2]
            if bda_b[1, 1] == -1:
                source_positions_bda[batch_indices.squeeze()==b, 1] = 199 - source_positions_bda[batch_indices.squeeze()==b, 1]

        # compute batched transform box indices
        transform_offsets = box_indices.new_tensor([0] + [len(b) for b in voxel_outs[4]])
        box_indices = (box_indices + transform_offsets[batch_indices.flatten().long()]).long()

        # create class mask
        source_batch_indices = torch.cat((batch_indices, source_positions_bda), -1).long()
        selected_classes = predicted_classes[tuple(source_batch_indices.T)]
        class_mask = selected_classes == source_labels

        # if no box is valid, just skip
        if len(source_positions) < 1 or class_mask.sum()< 1:
            return temporal_density, temporal_semantics, None
        
        # mask everything
        source_positions = source_positions[class_mask].long()
        source_positions_bda = source_positions_bda[class_mask].long()
        target_positions_floor = target_positions_floor[class_mask].long()
        source_labels = source_labels[class_mask].long()
        batch_indices = batch_indices[class_mask].long()
        temporal_indices = temporal_indices[class_mask].long()
        box_indices = box_indices[class_mask]

        # get grid positions
        grid_original = self.pc_grid.to(transforms)

        # get moved positions
        target_transformations = transforms[box_indices]
        pc_positions = grid_original[tuple(source_positions.T)]
        pc_positions = torch.cat((pc_positions, pc_positions.new_ones((pc_positions.shape[0],1))),dim=-1)
        moved_pc_positions = (target_transformations @ pc_positions.unsqueeze(-1)).squeeze(-1)[:, :-1] 
        
        # 8 adjacent voxel centers to transformed positions
        nearest_centers_indices = torch.max(torch.min(
            target_positions_floor.unsqueeze(1) + torch.tensor(
                [[[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,1,0],[1,0,1],[1,1,1]]], device=target_positions_floor.device
            ), 
        torch.tensor([[[self.bev_z-1, self.bev_h-1, self.bev_w-1]]], device=target_positions_floor.device)), torch.tensor([[[0]]], device=target_positions_floor.device)).flatten(0, 1)

        # Compute weights by inverse distance -> Shepard's method
        nearest_centers_3d = grid_original[tuple(nearest_centers_indices.T)].view(-1, 8, 3)
        weights = 1 / torch.clamp(((moved_pc_positions.unsqueeze(1) - nearest_centers_3d).abs()).sum(-1), min= 1e-3)
        weights = weights / weights.sum(-1, keepdim=True)

        # get previous logits
        source_indices = torch.cat((batch_indices, temporal_indices, source_positions_bda), -1)
        source_logits_sem = (temporal_semantics[tuple(source_indices.T)].unsqueeze(1) * weights.unsqueeze(-1)).flatten(0,1)
        source_logits_den = (temporal_density[tuple(source_indices.T)].unsqueeze(1) * weights.unsqueeze(-1)).flatten(0,1)

        # create sum per index and normalize by weight
        nearest_centers_indices_bt = torch.cat((batch_indices[:, None, :].expand(-1, 8, -1), temporal_indices[:, None, :].expand(-1, 8, -1), nearest_centers_indices.view(-1, 8, 3)), -1).flatten(0, 1)
        
        # also flip the target indices if required
        for b, bda_b in enumerate(bda):
            if bda_b[0, 0] ==- 1:
                nearest_centers_indices_bt[nearest_centers_indices_bt[:, 0]==b, -1] = 199 - nearest_centers_indices_bt[nearest_centers_indices_bt[:, 0]==b, -1]
            if bda_b[1, 1] == -1:
                nearest_centers_indices_bt[nearest_centers_indices_bt[:, 0]==b, -2] = 199 - nearest_centers_indices_bt[nearest_centers_indices_bt[:, 0]==b, -2]

        indices_unique, labels = nearest_centers_indices_bt.unique(dim=0, return_inverse=True)
        indices_unique = indices_unique.view(-1, 5)

        # get logits at target positions for weighting
        target_logits_sem = voxel_outs[1][tuple(nearest_centers_indices_bt[:, [0, 2, 3, 4]].T)] * (1 - weights).flatten().unsqueeze(-1)
        target_logits_den = voxel_outs[0][tuple(nearest_centers_indices_bt[:, [0, 2, 3, 4]].T)] * (1 - weights).flatten().unsqueeze(-1) 

        # get final logits
        final_logits_sem = source_logits_sem + target_logits_sem
        final_logits_den = source_logits_den + target_logits_den

        # set old positions to 0
        temporal_semantics[tuple(source_indices.T)] = 0. 
        temporal_density[tuple(source_indices.T)] = 0. 

        sem_grouped = weights.new_zeros((indices_unique.shape[0], final_logits_sem.shape[-1])).scatter_add_(0, labels.view(labels.size(0), 1).expand(-1, final_logits_sem.size(-1)), final_logits_sem) 
        dens_grouped = weights.new_zeros((indices_unique.shape[0], final_logits_den.shape[-1])).scatter_add_(0, labels.view(labels.size(0), 1).expand(-1, final_logits_den.size(-1)), final_logits_den)


        temporal_semantics[tuple(indices_unique.T)] = sem_grouped
        temporal_density[tuple(indices_unique.T)] = dens_grouped

        voxel_outs[0] = temporal_density
        voxel_outs[1] = temporal_semantics

        return voxel_outs

    def apply_flow_clipped(self, voxel_outs, bda):
        """Easier/Faster version of the flow function"""
        predicted_classes = torch.argmax(voxel_outs[1], dim=-1).byte()#.flatten(1, )
        
        temporal_density = voxel_outs[0][:, None, ...].repeat(1, len(self.render_frame_ids), 1, 1, 1, 1)
        temporal_semantics = voxel_outs[1][:, None, ...].repeat(1, len(self.render_frame_ids), 1, 1, 1, 1)

        # create batched source and target positions
        source_positions = torch.cat([b[:, :3]  for b in voxel_outs[3]], dim=0)
        target_positions = torch.cat([b[:, 3:6]  for b in voxel_outs[3]], dim=0)
        source_labels = torch.cat([b[:, 9] for b in voxel_outs[3]], dim=0)
        batch_indices = torch.cat([source_positions.new_full((len(b), 1), i) for i, b in enumerate(voxel_outs[3])], dim=0)
        temporal_indices = torch.cat([b[:, 10] for b in voxel_outs[3]], dim=0).unsqueeze(-1)

        # Apply BDA augmentations
        for b, bda_b in enumerate(bda):
            if bda_b[0, 0] == -1:
                source_positions[batch_indices.squeeze()==b, 2] = 199 -  source_positions[batch_indices.squeeze()==b, 2]
                target_positions[batch_indices.squeeze()==b, 2] = 199 -  target_positions[batch_indices.squeeze()==b, 2]
            if bda_b[1, 1] == -1:
                source_positions[batch_indices.squeeze()==b, 1] = 199 - source_positions[batch_indices.squeeze()==b, 1]
                target_positions[batch_indices.squeeze()==b, 1] = 199 - target_positions[batch_indices.squeeze()==b, 1]

        # create class mask
        source_batch_indices = torch.cat((batch_indices, source_positions), -1).long()
        selected_classes = predicted_classes[tuple(source_batch_indices.T)]
        class_mask = selected_classes == source_labels

        # if no box is valid, just skip
        if len(source_positions) < 1 or class_mask.sum()< 1:
            return temporal_density, temporal_semantics, None

        logit_density = voxel_outs[0][tuple(source_batch_indices[class_mask].T)]
        logit_semantics = voxel_outs[1][tuple(source_batch_indices[class_mask].T)]

        # assign temporal and batch indices
        source_indices = torch.cat((batch_indices[class_mask], temporal_indices[class_mask], source_positions[class_mask]), -1).long()
        target_indices = torch.cat((batch_indices[class_mask], temporal_indices[class_mask], target_positions[class_mask]), -1).long()

        # assing box class to transformed positions
        temporal_density[tuple(source_indices.T)] = 0.
        temporal_semantics[tuple(source_indices.T)] = 0.

        # sum up same target indices and normalize
        unique_indices, labels, counts = target_indices.unique(dim=0, return_inverse=True, return_counts=True)
        unique_indices = unique_indices.view(-1, 5)
        dens_grouped = temporal_density.new_zeros((unique_indices.shape[0], logit_density.shape[-1])).scatter_add_(0, labels.view(labels.size(0), 1).expand(-1, logit_density.size(-1)), logit_density) / counts.unsqueeze(-1)
        sem_grouped = temporal_density.new_zeros((unique_indices.shape[0], logit_semantics.shape[-1])).scatter_add_(0, labels.view(labels.size(0), 1).expand(-1, logit_semantics.size(-1)), logit_semantics) / counts.unsqueeze(-1)

        temporal_density[tuple(unique_indices.T)] = dens_grouped
        temporal_semantics[tuple(unique_indices.T)] = sem_grouped

        voxel_outs[0] = temporal_density
        voxel_outs[1] = temporal_semantics

        return voxel_outs
   
    def sample_rays(self, gt_labels):
        if self.wrs:
            weights = self.sample_weights[gt_labels]
            indices = torch.tensor(list(WeightedRandomSampler(weights, num_samples=len(weights), replacement=False)))
        else:
            indices = torch.randperm(len(gt_labels))
        return indices

    def forward(self, voxel_outs, ray_origins, ray_directions, ray_dataset, bda, max_batch_override=None):
        if not isinstance(self.pc_range, torch.Tensor):
            self.pc_range = voxel_outs[0].new_tensor(self.pc_range)

        ray_indices = torch.cat([torch.cat((q[0].new_full( (q[0].shape[0], 1) , i), q[0]), axis=-1) for i, q in enumerate(ray_dataset)])
        ray_directions = torch.cat(ray_directions)
        gt_labels = torch.cat([r[2] for r in ray_dataset])
        num_rays = len(ray_indices)
        random_indices = self.sample_rays(gt_labels.long())

        # create filtered voxel grid (remove dynamic objects)
        if self.temporal_filter and self.ignore_classes is not None and voxel_outs[1] is not None:
            static_density = self.create_static_occupancy(voxel_outs)
            voxel_outs[0] = torch.stack((voxel_outs[0], static_density), dim=1)
        # or move dynamic objects if using flow
        elif self.use_flow:
            assert voxel_outs[3] is not None, "Attempting to use flow but did not provide it."
            voxel_outs = self.flow_function(voxel_outs, bda)

        # iterate over ray indices and render outputs
        num_batches = min(math.ceil(num_rays / self.rays_batch_size), self.max_batches if max_batch_override is None else max_batch_override)
        results = {f'{m.name}': ray_origins.new_zeros([num_rays, m.output_size]) for m in self.render_modules}
        results['indices'] = []
        if self.dist_loss is not None:
            results['loss_dist'] = 0
        for b in range(num_batches):
            indices = random_indices[b * self.rays_batch_size : (b+1) * self.rays_batch_size]
            selected_ray_indices = ray_indices[indices]
            directions = ray_directions[indices]
            origins = ray_origins[tuple(selected_ray_indices.T)]
            
            samples_start, samples_end, samples_centers = self.generate_samples(origins, directions, selected_ray_indices, voxel_outs[0])
            samples = self.create_3d_samples(origins, directions, samples_centers)
            densities = self.interpolate_fn(voxel_outs[0], samples, selected_ray_indices, self.pc_range, self.current_timestep).squeeze(-1)
            weights, _, _ = nerfacc.render_weight_from_density(samples_start, samples_end, densities)

            for module in self.render_modules:
                values = module.get_value(samples, samples_start, samples_end, voxel_outs, selected_ray_indices, self.pc_range)
                result = nerfacc.accumulate_along_rays(weights, values)
                results[f'{module.name}'][indices] = result
            results['indices'].extend(indices.tolist())

            if self.dist_loss is not None:
                dist_loss = self.dist_loss(weights, samples_centers, samples_end-samples_start)
                results['loss_dist'] += dist_loss / num_batches

        return results

    def forward_test(self, voxel_outs, origins, directions, ray_dataset, coors, bda, H, W):
        with torch.no_grad():
            # run inference
            _, T, nC, _ = origins.shape
            results = self.forward(voxel_outs, origins, directions, ray_dataset, bda, max_batch_override=1000)
            outputs = self.get_outputs(results)

            # format into images
            target_indices = torch.cat((ray_dataset[0][0], coors[0][:, [1,0]]), dim=-1)
            image_dict = {}
            for module in self.render_modules:
                module_image = torch.zeros((T, nC, H, W, 3), dtype=torch.uint8)
                module_image[tuple(target_indices.T)] = torch.tensor(outputs[f'{module.name}'])
                image_dict[module.name] = module_image.cpu().numpy()
        return image_dict
    
    def calculate_losses(self, pred, ray_dataset):
        indices = pred['indices']
        depth_gt = torch.cat([q[1] for q in ray_dataset])[indices]
        sem_gt = torch.cat([q[2] for q in ray_dataset])[indices]
        gt = {'depth':depth_gt, 'semantics':sem_gt}

        ray_weights = depth_gt.new_ones(depth_gt.shape)
        if self.class_balanced_loss:
            if self.global_weights:
                ray_weights *= self.class_weight[sem_gt.long()].to(sem_gt.device)
            elif self.log_weighting:
                batch_count = torch.clamp(sem_gt.bincount(), min=1)
                ray_weights_t = torch.log(batch_count.sum() / batch_count)
                ray_weights *= ray_weights_t[sem_gt.long()]
            elif self.ens_weighting:
                batch_count = torch.clamp(sem_gt.bincount(), min=1)
                ray_weights_t = (1 - self.ens_weighting) / (1 - torch.pow(self.ens_weighting, batch_count))
            else:
                batch_count = torch.clamp(sem_gt.bincount(), min=1)
                ray_weights_t = (1 / torch.pow(batch_count, self.weight_exponent)) / (1 / batch_count).sum() * len(batch_count)
                ray_weights *= ray_weights_t[sem_gt.long()]
        if self.distance_scaled_loss:
            distance_scale = 1 - (depth_gt / self.far)
            ray_weights *= distance_scale

        # rescale weights to 1 mean
        if self.scale_ray_weights:
            ray_weights = (ray_weights.numel() / ray_weights.sum()) * ray_weights

        losses_dict = {}
        for module in self.render_modules:
            losses_dict[f'loss_render_{module.name}'] = module.get_loss(pred[module.name][indices], gt[module.name], ray_weights)

        if self.dist_loss is not None:
            losses_dict['loss_dist'] = pred['loss_dist']
        return losses_dict
    
    def get_outputs(self, results):
        out = {}
        for module in self.render_modules:
            out_t = module.get_output(results[module.name].detach(), self)
            out[module.name] = out_t
        return out