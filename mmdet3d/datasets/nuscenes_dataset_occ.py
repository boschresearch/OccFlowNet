# Copyright (c) 2022 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

# This source code is derived from BEVDet (https://github.com/HuangJunJie2017/BEVDet)
# Copyright (c) OpenMMLab. All rights reserved.
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.

import os
import os.path as osp
import mmcv
import torch
import cv2
import numpy as np

from .builder import DATASETS
from .nuscenes_dataset import NuScenesDataset
from .occ_metrics import Metric_mIoU

colors_map = np.array(
    [
        [0,   0,   0, 255],  # 0 undefined
        [255, 158, 0, 255],  # 1 car  orange
        [0, 0, 230, 255],    # 2 pedestrian  Blue
        [47, 79, 79, 255],   # 3 sign  Darkslategrey
        [220, 20, 60, 255],  # 4 CYCLIST  Crimson
        [255, 69, 0, 255],   # 5 traiffic_light  Orangered
        [255, 140, 0, 255],  # 6 pole  Darkorange
        [233, 150, 70, 255], # 7 construction_cone  Darksalmon
        [255, 61, 99, 255],  # 8 bycycle  Red
        [112, 128, 144, 255],# 9 motorcycle  Slategrey
        [222, 184, 135, 255],# 10 building Burlywood
        [0, 175, 0, 255],    # 11 vegetation  Green
        [165, 42, 42, 255],  # 12 trunk  nuTonomy green
        [0, 207, 191, 255],  # 13 curb, road, lane_marker, other_ground
        [75, 0, 75, 255], # 14 walkable, sidewalk
        [255, 0, 0, 255], # 15 unobsrvd
        [0, 0, 0, 0],  # 16 undefined
        [0, 0, 0, 0],  # 16 undefined
    ])



@DATASETS.register_module()
class NuScenesDatasetOccpancy(NuScenesDataset):

    def __init__(self, *args, eval_threshold_range=[.05, .2, .5], use_flow=False, threshold_save_indices=None, **kwargs):
        super().__init__(*args, **kwargs)

        self.set_eval_threshold_range(eval_threshold_range, threshold_save_indices)
        self.use_flow = use_flow
    
    def set_eval_threshold_range(self, eval_range, indices = None):
        self.eval_threshold_range = eval_range
        if indices is None:
            self.threshold_save_indices = list(range(len(eval_range)))
        else:
            self.threshold_save_indices = indices

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - sweeps (list[dict]): Infos of sweeps.
                - timestamp (float): Sample timestamp.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        input_dict = super(NuScenesDatasetOccpancy, self).get_data_info(index)
        # standard protocol modified from SECOND.Pytorch
        input_dict['occ_gt_path'] = self.data_infos[index]['occ_path']
        return input_dict

    def evaluate(self, occ_results, runner=None, show_dir=None, save_dir=None, **eval_kwargs):

        print('\nStarting Evaluation...')
        eval_dict = {}
        metrics = [Metric_mIoU(
                num_classes=18,
                use_lidar_mask=False,
                use_image_mask=True) for i in self.eval_threshold_range]
        
        for index, occ_pred in enumerate(occ_results):
            info = self.data_infos[index]
        
            occ_gt = np.load(os.path.join(info['occ_path'], 'labels.npz'))
            gt_semantics = occ_gt['semantics']
            mask_lidar = occ_gt['mask_lidar'].astype(bool)
            mask_camera = occ_gt['mask_camera'].astype(bool)

            if isinstance(occ_pred, np.ndarray):
                metrics[0].add_batch(occ_pred, gt_semantics, mask_lidar, mask_camera)
            else:
                preds = occ_pred['occupancy']
                for i, t in enumerate(self.eval_threshold_range):
                    preds_i = preds.copy()
                    preds_i[occ_pred['free_space'][i]] = 17
                    metrics[i].add_batch(preds_i, gt_semantics, mask_lidar, mask_camera)
        
        # for metric, t in zip(metrics, self.eval_threshold_range):
        eval_dict.update({
            f'mIoU_{t}': metric.count_miou()[2] for t, metric in zip(self.eval_threshold_range, metrics)
        })

        eval_dict.update({'top_mIoU': max(eval_dict.values())})

        if save_dir is not None:
            self.save_occupancy(occ_results, save_dir)

        return eval_dict

    def save_occupancy(self, results, out_path):
        mmcv.mkdir_or_exist(osp.join(out_path, 'occupancy'))
        mmcv.mkdir_or_exist(osp.join(out_path, 'free_space'))
        all_occs = {}
        all_fs = {}
        for index, output in enumerate(results):
            info = self.data_infos[index]
            scene_name, token = info['scene_name'], info['token']
            if scene_name not in all_occs.keys():
                all_occs[scene_name] = {}
                all_fs[scene_name] = {}
            all_occs[scene_name][token] = output['occupancy']
            all_fs[scene_name][token] = output['free_space'][self.threshold_save_indices]

        for scene, preds in all_occs.items():
            out_file_occ = osp.join(out_path, 'occupancy', f'{scene}.npz')
            np.savez_compressed(out_file_occ, **preds)
            
        for scene, preds in all_fs.items():
            out_file_fs = osp.join(out_path, 'free_space', f'{scene}.npz')
            np.savez_compressed(out_file_fs, **preds)
        
    def vis_occ(self, semantics):
        # simple visualization of result in BEV
        semantics_valid = np.logical_not(semantics == 17)
        d = np.arange(16).reshape(1, 1, 16)
        d = np.repeat(d, 200, axis=0)
        d = np.repeat(d, 200, axis=1).astype(np.float32)
        d = d * semantics_valid
        selected = np.argmax(d, axis=2)

        selected_torch = torch.from_numpy(selected)
        semantics_torch = torch.from_numpy(semantics)

        occ_bev_torch = torch.gather(semantics_torch, dim=2,
                                     index=selected_torch.unsqueeze(-1))
        occ_bev = occ_bev_torch.numpy()

        occ_bev = occ_bev.flatten().astype(np.int32)
        occ_bev_vis = colors_map[occ_bev].astype(np.uint8)
        occ_bev_vis = occ_bev_vis.reshape(200, 200, 4)[::-1, ::-1, :3]
        occ_bev_vis = cv2.resize(occ_bev_vis,(400,400))
        return occ_bev_vis