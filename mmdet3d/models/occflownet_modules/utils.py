# Copyright (c) 2022 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import torch
import torch.nn.functional as F
from typing import List
from pyquaternion import Quaternion
import numpy as np
import copy 

class Boxes:
    """ Simple data class representing a set of 3d boxes, inspired by nuscenes-devkit."""

    def __init__(self,
                 bboxes: np.ndarray,
                 labels: np.ndarray,
                 names: List[str] = None,
                 instance_tokens: List[str] = None,
                 device=torch.device('cpu')):
        """
        :param center: Center of box given as x, y, z.
        :param size: Size of box in width, length, height.
        :param orientation: Box orientation.
        :param label: Integer label, optional.
        :param score: Classification score, optional.
        :param velocity: Box velocity in x, y, z direction.
        :param name: Box name, optional. Can be used e.g. for denote category name.
        :param token: Unique string identifier from DB.
        """

        if isinstance(bboxes, np.ndarray):
            bboxes = torch.tensor(bboxes, device=device)
            self.device = device
            self.labels = torch.tensor(labels, dtype=torch.uint8, device=self.device)
        else:
            self.labels = labels
            self.device = bboxes.device

        self.centers = bboxes[:, :3]
        self.wlhs = bboxes[:, 3:6]
        self.orientations = bboxes[:, 6:]
        self.names = names
        self.instance_tokens = instance_tokens
        self.box_padding = .4

    def _q_matrix(self):
        """Matrix representation of quaternion for multiplication purposes."""
        return torch.stack(
            [torch.stack(([self.orientations[:, 0], -self.orientations[:, 1], -self.orientations[:, 2], -self.orientations[:, 3]]),dim=1),
            torch.stack(([self.orientations[:, 1],  self.orientations[:, 0], -self.orientations[:, 3],  self.orientations[:, 2]]),dim=1),
            torch.stack(([self.orientations[:, 2],  self.orientations[:, 3],  self.orientations[:, 0], -self.orientations[:, 1]]),dim=1),
            torch.stack(([self.orientations[:, 3], -self.orientations[:, 2],  self.orientations[:, 1],  self.orientations[:, 0]]),dim=1)], 
            1)

    def _q_bar_matrix(self):
        """Matrix representation of quaternion for multiplication purposes.
        """
        return torch.stack(
            [
                torch.stack(([self.orientations[:, 0], -self.orientations[:, 1], -self.orientations[:, 2], -self.orientations[:, 3]]), dim=1),
                torch.stack(([self.orientations[:, 1], self.orientations[:, 0], self.orientations[:, 3], -self.orientations[:, 2]]), dim=1),
                torch.stack(([self.orientations[:, 2], -self.orientations[:, 3], self.orientations[:, 0], self.orientations[:, 1]]), dim=1),
                torch.stack(([self.orientations[:, 3], self.orientations[:, 2], -self.orientations[:, 1], self.orientations[:, 0]]), dim=1)
             ]
        , 1)

    def points_in_boxes(self, points, wlh_factor):
        """
        Checks whether points are inside the box.
        From: https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/utils/geometry_utils.py#L111
        """
        corners = self.corners(wlh_factor=wlh_factor)

        p1 = corners[:, 0, :]
        p_x = corners[:, 4, :]
        p_y = corners[:, 1, :]
        p_z = corners[:, 3, :]

        i = p_x - p1
        j = p_y - p1
        k = p_z - p1

        v = points.unsqueeze(1) - p1.unsqueeze(0)

        iv = torch.matmul(i.unsqueeze(0).unsqueeze(-2), v.unsqueeze(-1)).squeeze(-1).squeeze(-1)
        jv = torch.matmul(j.unsqueeze(0).unsqueeze(-2), v.unsqueeze(-1)).squeeze(-1).squeeze(-1)
        kv = torch.matmul(k.unsqueeze(0).unsqueeze(-2), v.unsqueeze(-1)).squeeze(-1).squeeze(-1)

        mask_x = torch.logical_and(0 <= iv, iv <= torch.matmul(i.unsqueeze(1), i.unsqueeze(-1)).squeeze(-1).squeeze(-1).unsqueeze(0))
        mask_y = torch.logical_and(0 <= jv, jv <= torch.matmul(j.unsqueeze(1), j.unsqueeze(-1)).squeeze(-1).squeeze(-1).unsqueeze(0))
        mask_z = torch.logical_and(0 <= kv, kv <= torch.matmul(k.unsqueeze(1), k.unsqueeze(-1)).squeeze(-1).squeeze(-1).unsqueeze(0))
        mask = torch.logical_and(torch.logical_and(mask_x, mask_y), mask_z)

        return mask.T
    
    def __len__(self):
        return self.centers.shape[0]
    
    def rotation_matrix(self):
        """Return the rotation matrices for all boxes"""
        return torch.matmul(self._q_matrix(), self._q_bar_matrix().conj().transpose(1,2))[:, 1:][:, :, 1:]

    def transform_by_matrix(self, transform_matrix):
        """Transform ALL boxes by a SINGLE transformation."""
        q = torch.tensor(Quaternion(matrix=transform_matrix)._q_matrix(), device=self.device)
        tm = torch.tensor(transform_matrix, device=self.device)
        self.orientations = torch.matmul(q, self.orientations.unsqueeze(-1)).squeeze(-1)
        self.centers = (tm[:3, :3] @ self.centers.unsqueeze(-1)).squeeze(-1) + tm[:3, 3]
    
    def mask_by_index(self, mask):
        self.centers = self.centers[mask]
        self.wlhs = self.wlhs[mask]
        self.orientations = self.orientations[mask]
        self.labels = self.labels[mask]
        self.names = self.names[mask]
        self.instance_tokens = self.instance_tokens[mask]

    def mask_by_range(self, pcr):
        mask = (self.centers[:, 0] > pcr[0]) & (self.centers[:, 0] < pcr[3]) & (
            self.centers[:, 1] > pcr[1]) & (self.centers[:, 1] < pcr[4]) & (
            self.centers[:, 2] > pcr[2]) & (self.centers[:, 2] < pcr[5])
        self.centers = self.centers[mask]
        self.wlhs = self.wlhs[mask]
        self.labels = self.labels[mask]
        self.orientations = self.orientations[mask]
        self.names = self.names[mask.cpu().numpy()]
        self.instance_tokens = self.instance_tokens[mask.cpu().numpy()]
    
    def to(self, device):
        self.centers = self.centers.to(device)
        self.wlhs = self.wlhs.to(device)
        self.orientations = self.orientations.to(device)
        self.labels = self.labels.to(device)
        self.device = device
        return self.copy()
        
    def yaw(self):
        """Return yaw of every box"""
        yaw = torch.arctan2(2 * (self.orientations[:, 0] * self.orientations[:, 3] - self.orientations[:, 1] * self.orientations[:, 2]),
            1 - 2 * (self.orientations[:, 2] ** 2 + self.orientations[:, 3] ** 2))
        return yaw

    def corners(self, wlh_factor = 1.0):
        """
        Returns the bounding box corners.
        :param wlh_factor: Multiply w, l, h by a factor to scale the box.
        :return: <np.float: 3, 8>. First four corners are the ones facing forward.
            The last four are the ones facing backwards.
        """
        w, l, h = self.wlhs.T * wlh_factor + self.box_padding

        # 3D bounding box corners. (Convention: x points forward, y to the left, z up.)
        x_corners = l.unsqueeze(-1) / 2 * torch.tensor([[1,  1,  1,  1, -1, -1, -1, -1]], device=self.device)
        y_corners = w.unsqueeze(-1) / 2 * torch.tensor([[1, -1, -1,  1,  1, -1, -1,  1]], device=self.device)
        z_corners = h.unsqueeze(-1) / 2 * torch.tensor([[1,  1, -1, -1,  1,  1, -1, -1]], device=self.device)
        corners = torch.stack((x_corners, y_corners, z_corners), -1)

        # Rotate
        corners = (self.rotation_matrix().unsqueeze(1) @ corners.unsqueeze(-1)).squeeze(-1)

        # Translate
        corners = corners + self.centers.unsqueeze(1)

        return corners
    
    def copy(self):
        """
        Create a copy of self.
        :return: A copy.
        """
        return copy.deepcopy(self)

def interpolate_values(values, samples, indices, pc_range, current_timestep):
    num_rays, num_samples, xyz = samples.shape
    r_min = pc_range[0:3]
    r_max = pc_range[3:6]
    
    # scale to [-1, 1] for torch grid_sample
    samples_scaled = ((samples - r_min)/(r_max - r_min)) * 2 - 1
    
    # reshape samples into correct batches
    B = values.shape[0]
    batch_indices = indices[:, 0]
    if B > 1:
        samples_rebatched = samples_scaled.new_zeros((B, num_rays, num_samples, 3))
        samples_rebatched[batch_indices, torch.arange(num_rays)] = samples_scaled
    else:
        samples_rebatched = samples_scaled.unsqueeze(0)

    sampled_values = F.grid_sample(
        values.permute(0, 4, 1, 2, 3),
        samples_rebatched.unsqueeze(1),
        mode='bilinear',
        padding_mode='zeros',
        align_corners=True
    ).squeeze(2).permute(0, 2, 3, 1)
    
    values_per_sample = sampled_values[batch_indices, torch.arange(num_rays)]

    return values_per_sample

def interpolate_values_tf(values, samples, indices, pc_range, current_timestep):

    num_rays, num_samples, _ = samples.shape
    r_min = pc_range[0:3]
    r_max = pc_range[3:6]
    
    # scale to [-1, 1] for torch grid_sample
    samples_scaled = ((samples - r_min)/(r_max - r_min)) * 2 - 1
    
    # reshape samples into correct batches
    B = values.shape[0] * 2
    samples_rebatched = samples_scaled.new_zeros((B, num_rays, num_samples, 3))
    batch_stride = indices[:, 0] * 2
    temporal_stride = (~(indices[:, 1] == current_timestep)).long() # current step = 0, temporal step = 1
    samples_rebatched[batch_stride + temporal_stride, torch.arange(num_rays)] = samples_scaled 

    sampled_values = F.grid_sample(
        values.permute(0, 1, 5, 2, 3, 4).flatten(0, 1),
        samples_rebatched.unsqueeze(1),
        mode='bilinear',
        padding_mode='zeros',
        align_corners=True
    ).squeeze(2).permute(0, 2, 3, 1)
    
    values_per_sample = sampled_values[batch_stride + temporal_stride, torch.arange(num_rays)]

    return values_per_sample

def interpolate_values_oob_unmasked(values, samples, indices, oob_mask, pc_range):
    """Interpolation function (without temporal filtering) when oob mask is already computed."""
    r_min = pc_range[0:3]
    r_max = pc_range[3:6]
    
    # scale to [-1, 1] for torch grid_sample
    samples = ((samples - r_min)/(r_max - r_min)) * 2 - 1

    # filter out OOB rays
    ray_indices = oob_mask.nonzero()[:, 0]
    num_valid_samples = len(ray_indices)
    samples_masked = samples[oob_mask]
    indices_masked = indices[ray_indices]

    B, _, _, _, C = values.shape
    sampled_values = samples_masked.new_zeros([num_valid_samples, C])
    for b in range(B):
        bt_indices = (indices_masked[:, 0] == b)
        sampled_values_bt = F.grid_sample(
            values[b].permute(3, 0, 1, 2).unsqueeze(0),
            samples_masked[None, None, None, bt_indices, ...],
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True
        )[0, :, 0, 0].permute(1, 0)
        sampled_values[bt_indices] = sampled_values_bt
    return sampled_values

def interpolate_values_flow(values, samples, indices, pc_range, current_timestep):
    num_rays, num_samples, _ = samples.shape
    r_min = pc_range[0:3]
    r_max = pc_range[3:6]
    
    # scale to [-1, 1] for torch grid_sample
    samples_scaled = ((samples - r_min)/(r_max - r_min)) * 2 - 1
    
    # reshape samples into correct batches
    B, T = values.shape[:2]
    samples_rebatched = samples_scaled.new_zeros((B, T, num_rays, num_samples, 3))
    samples_rebatched[indices[:, 0], indices[:, 1], torch.arange(num_rays)] = samples_scaled 

    sampled_values = F.grid_sample(
        values.permute(0, 1, 5, 2, 3, 4).flatten(0, 1),
        samples_rebatched.flatten(0, 1).unsqueeze(1),
        mode='bilinear',
        padding_mode='zeros',
        align_corners=True
    ).squeeze(2).permute(0, 2, 3, 1).view(B, T, num_rays, num_samples, -1)
    
    values_per_sample = sampled_values[indices[:, 0],  indices[:, 1], torch.arange(num_rays)]

    return values_per_sample