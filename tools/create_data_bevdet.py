# Copyright (c) 2022 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

# This source code is derived from BEVDet (https://github.com/HuangJunJie2017/BEVDet)
# Copyright (c) OpenMMLab. All rights reserved.
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.

import pickle
import argparse
import numpy as np
from nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion
from tqdm import tqdm

from tools.data_converter import nuscenes_converter as nuscenes_converter

map_name_from_general_to_detection = {
    'human.pedestrian.adult': 'pedestrian',
    'human.pedestrian.child': 'pedestrian',
    'human.pedestrian.wheelchair': 'ignore',
    'human.pedestrian.stroller': 'ignore',
    'human.pedestrian.personal_mobility': 'ignore',
    'human.pedestrian.police_officer': 'pedestrian',
    'human.pedestrian.construction_worker': 'pedestrian',
    'animal': 'ignore',
    'vehicle.car': 'car',
    'vehicle.motorcycle': 'motorcycle',
    'vehicle.bicycle': 'bicycle',
    'vehicle.bus.bendy': 'bus',
    'vehicle.bus.rigid': 'bus',
    'vehicle.truck': 'truck',
    'vehicle.construction': 'construction_vehicle',
    'vehicle.emergency.ambulance': 'ignore',
    'vehicle.emergency.police': 'ignore',
    'vehicle.trailer': 'trailer',
    'movable_object.barrier': 'barrier',
    'movable_object.trafficcone': 'traffic_cone',
    'movable_object.pushable_pullable': 'ignore',
    'movable_object.debris': 'ignore',
    'static_object.bicycle_rack': 'ignore',
}
classes = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]


def get_gt(info):
    """Generate gt labels from info.

    Args:
        info(dict): Infos needed to generate gt labels.

    Returns:
        Tensor: GT bboxes.
        Tensor: GT labels.
    """
    ego2global_rotation = info['cams']['CAM_FRONT']['ego2global_rotation']
    ego2global_translation = info['cams']['CAM_FRONT'][
        'ego2global_translation']
    trans = -np.array(ego2global_translation)
    rot = Quaternion(ego2global_rotation).inverse
    gt_boxes = list()
    gt_labels = list()
    for ann_info in info['ann_infos']:
        # Use ego coordinate.
        if (map_name_from_general_to_detection[ann_info['category_name']]
                not in classes
                or ann_info['num_lidar_pts'] + ann_info['num_radar_pts'] <= 0):
            continue
        box = Box(
            ann_info['translation'],
            ann_info['size'],
            Quaternion(ann_info['rotation']),
            velocity=ann_info['velocity'],
        )
        box.translate(trans)
        box.rotate(rot)
        box_xyz = np.array(box.center)
        box_dxdydz = np.array(box.wlh)[[1, 0, 2]]
        box_yaw = np.array([box.orientation.yaw_pitch_roll[0]])
        box_velo = np.array(box.velocity[:2])
        gt_box = np.concatenate([box_xyz, box_dxdydz, box_yaw, box_velo])
        gt_boxes.append(gt_box)
        gt_labels.append(
            classes.index(
                map_name_from_general_to_detection[ann_info['category_name']]))
    return gt_boxes, gt_labels


def nuscenes_data_prep(root_path, info_prefix, version, out_dir, max_sweeps=10):
    """Prepare data related to nuScenes dataset.

    Related data consists of '.pkl' files recording basic infos,
    2D annotations and groundtruth database.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        version (str): Dataset version.
        max_sweeps (int, optional): Number of input consecutive frames.
            Default: 10
    """
    nuscenes_converter.create_nuscenes_infos(
        root_path, out_dir, info_prefix, version=version, max_sweeps=max_sweeps)


def add_ann_adj_info(extra_tag, dataroot, out_path, nuscenes_version):
    # nuscenes_version = 'v1.0-trainval'
    # dataroot = './data/nuscenes/'
    nuscenes = NuScenes(nuscenes_version, dataroot)
    for set in ['train', 'val']:
        dataset = pickle.load(
            open('%s/%s_infos_%s.pkl' % (out_path, extra_tag, set), 'rb'))
        for id in tqdm(range(len(dataset['infos']))):
            # if id % 10 == 0:
            #     print('%d/%d' % (id, len(dataset['infos'])))
            info = dataset['infos'][id]
            # get sweep adjacent frame info
            sample = nuscenes.get('sample', info['token'])
            ann_infos = list()
            for ann in sample['anns']:
                ann_info = nuscenes.get('sample_annotation', ann)
                velocity = nuscenes.box_velocity(ann_info['token'])
                if np.any(np.isnan(velocity)):
                    velocity = np.zeros(3)
                ann_info['velocity'] = velocity
                ann_infos.append(ann_info)
            dataset['infos'][id]['ann_infos'] = ann_infos
            dataset['infos'][id]['ann_infos'] = get_gt(dataset['infos'][id])
            dataset['infos'][id]['scene_token'] = sample['scene_token']

            scene = nuscenes.get('scene', sample['scene_token'])
            dataset['infos'][id]['occ_path'] = \
                './data/gts/%s/%s'%(scene['name'], info['token'])
            # dataset['infos'][id]['occ_path'] = \
            #     './data/nuscenes/gts/%s/%s'%(scene['name'], info['token'])
        with open('%s/%s_infos_%s.pkl' % (out_path, extra_tag, set),
                  'wb') as fid:
        # with open('./data/nuscenes/%s_infos_%s.pkl' % (extra_tag, set),
        #           'wb') as fid:
            pickle.dump(dataset, fid)


parser = argparse.ArgumentParser(description='Data converter arg parser')
parser.add_argument(
    '--root-path',
    type=str,
    default='./data/nuscenes',
    help='specify the root path of dataset')
parser.add_argument(
    '--version',
    type=str,
    default='v1.0-trainval',
    required=False,
    help='specify the dataset version, no need for kitti')
parser.add_argument(
    '--out-dir',
    type=str,
    default='./data',
    required=False,
    help='output directory')
args = parser.parse_args()

if __name__ == '__main__':
    dataset = 'nuscenes'
    version = args.version
    root_path = args.root_path
    extra_tag = 'bevdetv2-nuscenes' if version != "v1.0-mini" else 'bevdetv2-nuscenes-mini'
    nuscenes_data_prep(
        root_path=root_path,
        info_prefix=extra_tag,
        version=version,
        out_dir=args.out_dir,
        max_sweeps=0)

    print('add_ann_infos')
    add_ann_adj_info(extra_tag, root_path, args.out_dir, version)
