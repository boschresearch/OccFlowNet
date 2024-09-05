# Copyright (c) 2022 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

from argparse import ArgumentParser
import torch
import numpy as np
import os
from nuscenes.nuscenes import NuScenes
import mmcv
from tqdm import tqdm
from pyquaternion import Quaternion
from nuscenes.utils.geometry_utils import transform_matrix
import sys

class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
nuscenes_to_lidarseg = np.array([4, 10, 5, 3, 9, 1, 6, 2, 7, 8])
pc_range = np.array([-40., -40., -1.0, 40., 40., 5.4])
voxel_resolution = 0.4
bev_z, bev_w, bev_h = 16, 200, 200

use_classes = [2,3,4,5,6,7,9,10]

def load_annotations(ann_file):

    data = mmcv.load(ann_file, file_format='pkl')
    data_infos = list(sorted(data['infos'], key=lambda e: e['timestamp']))
    data_infos = data_infos
    metadata = data['metadata']
    return data_infos, metadata

def create_pc_grid():
    # [16,200,200]
    zs = torch.linspace(0.5 * voxel_resolution  + pc_range[2], pc_range[5] - 0.5 * voxel_resolution, bev_z).view(bev_z, 1, 1).expand(bev_z, bev_w, bev_h)
    xs = torch.linspace(0.5 * voxel_resolution  + pc_range[0], pc_range[3] - 0.5 * voxel_resolution, bev_w).view(1, bev_w, 1).expand(bev_z, bev_w, bev_h)
    ys = torch.linspace(0.5 * voxel_resolution  + pc_range[1], pc_range[4] - 0.5 * voxel_resolution, bev_h).view(1, 1, bev_h).expand(bev_z, bev_w, bev_h)
    
    # torch.Size([16, 200, 200, 3])
    ref_3d = torch.stack((ys, xs, zs), -1)
    ref_3d = ref_3d.flatten(0, 2)
    return ref_3d

def load_boxes(info, ego_l_t2global, device):
    # lidar2global (for lidar2ego_l_t transformation): ego2global @ lidar2ego
    lidar2ego_lidar = np.eye(4)
    lidar2ego_lidar[:3, :3] = Quaternion(info["lidar2ego_rotation"]).rotation_matrix
    lidar2ego_lidar[:3, 3] = info["lidar2ego_translation"]
    ego_lidar2global = np.eye(4)
    ego_lidar2global[:3, :3] = Quaternion(info["ego2global_rotation"]).rotation_matrix
    ego_lidar2global[:3, 3] = info["ego2global_translation"]
    lidar2global = ego_lidar2global @ lidar2ego_lidar
    lidar2ego_l_t = np.linalg.inv(ego_l_t2global) @ lidar2global

    # Load bboxes for flow GT
    mask = info['valid_flag']
    gt_bboxes_3d = info['gt_boxes'][mask]
    gt_names_3d = info['gt_names'][mask]
    gt_instance_tokens = np.array(info['gt_instance_tokens'])[mask]

    if len(gt_bboxes_3d)<1 or len(gt_names_3d)<1 or len(gt_instance_tokens)<1:
        return Boxes(np.empty(shape=(0, 10), dtype=np.float64), np.empty(shape=(0,), dtype=np.int64), 
                     names=np.empty(shape=(0,), dtype='<U32'), instance_tokens=np.empty(shape=(0,), dtype='<U32'), device=device)

    gt_labels_3d = []
    for cat in gt_names_3d:
        if cat in class_names:
            gt_labels_3d.append(class_names.index(cat))
        else:
            gt_labels_3d.append(-1)
    gt_labels_3d = nuscenes_to_lidarseg[gt_labels_3d]

    # Filter out boxes of static classes
    filter_mask = np.array([i in use_classes for i in gt_labels_3d])
    gt_bboxes_3d = gt_bboxes_3d[filter_mask]
    gt_labels_3d = gt_labels_3d[filter_mask]
    gt_names_3d = gt_names_3d[filter_mask]
    gt_instance_tokens = gt_instance_tokens[filter_mask]

    # With own custom Box class
    boxes = Boxes(gt_bboxes_3d, gt_labels_3d, names=gt_names_3d, instance_tokens=gt_instance_tokens, device=device)
    # transform from lidar to ego_lidar_t
    boxes.transform_by_matrix(lidar2ego_l_t)

    return boxes

def create_flow(data_infos, metadata, args, device):
    render_frame_ids = np.union1d(np.array(range(args.horizon+1)), -1*np.array(range(args.horizon+1)))
    current_timestep = np.where(render_frame_ids == 0)[0].item()
    temporal_frame_ids = render_frame_ids.tolist()
    temporal_frame_ids.pop(current_timestep)

    pc_grid = create_pc_grid().to(device=device)

    for index, info in enumerate(tqdm(data_infos)):
        scene_name = info['scene_name']
        sample_token = info['token']

        # check if this sample is already generated
        out_dir = os.path.join(args.out_dir, scene_name)
        target_file_path = os.path.join(out_dir, f'{sample_token}.npz')
        if os.path.exists(target_file_path) and not args.overwrite:
            continue

        ego_l_t2global = transform_matrix(translation=info['ego2global_translation'], rotation=Quaternion(info['ego2global_rotation']))
        boxes_c = load_boxes(info, ego_l_t2global, device) # current boxes
        boxes_c.mask_by_range(pc_range) # mask out boxes by range    

        # if no boxes are found, save empty:
        if len(boxes_c) < 1:
            mmcv.mkdir_or_exist(out_dir)
            np.savez_compressed(target_file_path, **{'flow': np.empty((0, 11), dtype=np.uint8), 'T': np.empty((0, 4, 4), dtype=np.float32), 
                                                     'indices': np.empty((0, ), dtype=np.uint16)})
            continue

        source_coordinates_t = []
        target_coordinates_t = []
        classes_t = []
        timesteps_t = []
        coords_floor_t = []
        transforms_t = []
        box_indices_t = []
        max_box_index = 0
        for i, t in enumerate(temporal_frame_ids):
            select_id = index + t
            # select_id = min(max(index + t, 0), len(data_infos)-1)
            if select_id < 0 or select_id >= len(data_infos):
                continue
            if not data_infos[select_id]['scene_token'] == info['scene_token']:
                # temp_info = info
                continue
            else:
                temp_info = data_infos[select_id]

            target_boxes = load_boxes(temp_info, ego_l_t2global, device)
            source_boxes = boxes_c.copy()

            # Iterate over temporal steps and compute transformations
            instance_indices = np.array([b.item() if len(b)>0 else -1 for b in [np.argwhere(it == target_boxes.instance_tokens) for it in source_boxes.instance_tokens]])
            instance_mask = instance_indices >= 0

            # if no correspondences between instances is found, skip this step
            if instance_mask.sum() < 1:
                continue

            # Get transformation matrices
            T_source = torch.eye(4, device=device)[None, :].repeat(instance_mask.sum(), 1, 1)
            T_source[:, :3, :3] = source_boxes.rotation_matrix()[instance_mask]
            T_source[:, :3, 3] = source_boxes.centers[instance_mask]

            T_target = torch.eye(4, device=device)[None, :].repeat(instance_mask.sum(), 1, 1)
            T_target[:, :3, :3] = target_boxes.rotation_matrix()[instance_indices][instance_mask]
            T_target[:, :3, 3] = target_boxes.centers[instance_indices][instance_mask]

            T = T_target @ torch.inverse(T_source)

            # Filter boxes
            source_boxes.mask_by_index(instance_mask) # Filter out boxes without correspondences

            # compute grid coordinates
            inside_mask = source_boxes.points_in_boxes(pc_grid, args.wlh_factor) # [n_boxes, n_voxels]
            inside_mask_indices = inside_mask.nonzero()
            selected_positions = pc_grid[inside_mask_indices[:, 1]]
            target_classes = source_boxes.labels[inside_mask_indices[:, 0]]

            selected_positions = torch.cat((selected_positions, selected_positions.new_ones((selected_positions.shape[0], 1))), dim=-1) # homogenous coordinates
            target_transformations = T[inside_mask_indices[:, 0]]
            moved_pc_positions = (target_transformations @ selected_positions.unsqueeze(-1)).squeeze(-1)[:, :-1]

            # clip positions to grid
            orig_coordinates = inside_mask.view(-1,bev_z,bev_h,bev_w).nonzero()[:, 1:]
            coords_z = ((moved_pc_positions[:, 2] - pc_range[2]) / .4).floor().long()
            coords_y = ((moved_pc_positions[:, 1] - pc_range[1]) / .4).floor().long()
            coords_x = ((moved_pc_positions[:, 0] - pc_range[0]) / .4).floor().long()
            target_coordinates = torch.stack((coords_z, coords_y, coords_x), -1)
            coords_floor = (((moved_pc_positions + 1e-5) - torch.tensor(pc_range[:3], device=device)[None, ...] - .2) / .4).floor().long()[:, [2,1,0]]

            # Clamp coordinates that are outside of the Grid
            target_coordinates = target_coordinates.clamp(min=torch.tensor([[0, 0, 0]], device=coords_x.device),
                                                          max=torch.tensor([[bev_z-1,bev_h-1,bev_w-1]], device=coords_x.device))
            # Also clamp floor coordinates
            coords_floor = coords_floor.clamp(min=torch.tensor([[0, 0, 0]], device=coords_x.device),
                                                          max=torch.tensor([[bev_z-1,bev_h-1,bev_w-1]], device=coords_x.device))
            # Store this sample 
            source_coordinates_t.append(orig_coordinates.to(torch.uint8).cpu().numpy())
            target_coordinates_t.append(target_coordinates.to(torch.uint8).cpu().numpy())
            coords_floor_t.append(coords_floor.to(torch.uint8).cpu().numpy())
            classes_t.append(target_classes.to(torch.uint8).cpu().numpy())
            timesteps_t.append(torch.full(target_classes.shape, i, dtype=torch.uint8).numpy())
            transforms_t.append(T.cpu().numpy())
            box_indices_t.append(inside_mask_indices[:, 0].cpu().numpy() + max_box_index)
            max_box_index += len(source_boxes)

        # if no corresponding boxes have been found in any time step
        if len(source_coordinates_t)<1:
            mmcv.mkdir_or_exist(out_dir)
            np.savez_compressed(target_file_path, **{'flow': np.empty((0, 11), dtype=np.uint8), 'T': np.empty((0, 4, 4), dtype=np.float32), 
                                                     'indices': np.empty((0, ), dtype=np.uint16)})
            continue

        # concatenate all boxes
        source_coordinates_t = np.concatenate(source_coordinates_t)
        target_coordinates_t = np.concatenate(target_coordinates_t)
        coords_floor_t = np.concatenate(coords_floor_t)
        classes_t = np.concatenate(classes_t)[:, None]
        timesteps_t = np.concatenate(timesteps_t)[:, None]

        # concatenate to a single np array (along dim=-1)
        flow_t = np.concatenate((source_coordinates_t, target_coordinates_t, coords_floor_t, classes_t, timesteps_t), axis=-1)
        transforms_t = np.concatenate(transforms_t)
        box_indices_t = np.concatenate(box_indices_t).astype(np.uint16)

        mmcv.mkdir_or_exist(out_dir)
        np.savez_compressed(target_file_path, **{'flow': flow_t, 'T': transforms_t, 'indices': box_indices_t})

if __name__ == '__main__':

    # parse args
    parser = ArgumentParser(description="Render occupancy (pred and/or gt)")

    parser.add_argument(
        "--data-root", type=str, help="Path to dataset", default='./data/nuscenes'
    )
    parser.add_argument(
        "--ann-root", type=str, help="Path to annotation files", default='./data/nuscenes'
    )
    parser.add_argument(
        "--out-dir", type=str, help="Path to output directory", default='./data/flow'
    )
    parser.add_argument(
        "--version", type=str, help="Version to load", default='v1.0-trainval'
    )
    parser.add_argument(
        "--overwrite", type=bool, help="Wether to overwrite existing generations", default=False
    )
    parser.add_argument(
        "--horizon", type=int, help="Time horizon for which to generate", default=3
    )
    parser.add_argument(
        "--wlh-factor", type=float, help="Padding factor for point in box filter", default=1.15
    )
    args = parser.parse_args()
    
    # cuda device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # enable imports of plugin
    sys.path.insert(0, os.path.split(os.path.split(os.path.abspath(__file__))[0])[0]) # Hacky
    from mmdet3d.models.occflownet_modules.utils import Boxes

    mmcv.mkdir_or_exist(args.out_dir)

    # Load annotation file
    version_prefix = '-mini' if args.version == 'v1.0-mini' else ''
    data_infos_train, metadata_train = load_annotations(os.path.join('data', f'bevdetv2-nuscenes{version_prefix}_infos_train.pkl'))
    data_infos_test, metadata_test = load_annotations(os.path.join('data', f'bevdetv2-nuscenes{version_prefix}_infos_val.pkl'))

    # Create flow data
    create_flow(data_infos_train, metadata_train, args, device)
    create_flow(data_infos_test, metadata_test, args, device)

