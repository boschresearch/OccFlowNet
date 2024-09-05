# Copyright (c) Phigent Robotics. All rights reserved.

# Copyright (c) 2022 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

_base_ = ['../_base_/datasets/nus-3d.py', '../_base_/default_runtime.py']

class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

data_config = {
    'cams': [
        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT',
        'CAM_BACK', 'CAM_BACK_RIGHT'
    ],
    'Ncams':
    6,
    'input_size': (512, 1408),
    'src_size': (900, 1600),

    # Augmentation
    'resize': (-0.06, 0.11),
    'rot': (-5.4, 5.4),
    'flip': True,
    'crop_h': (0.0, 0.0),
    'resize_test': 0.00,
}

# Model
grid_config = {
    'x': [-40, 40, 0.4],
    'y': [-40, 40, 0.4],
    'z': [-1, 5.4, 0.4],
    'depth': [1.0, 45.0, 0.5],
}

############ CUSTOM ###############
bev_h_ = 200
bev_w_ = 200
bev_z_ = 16
voxel_resolution = 0.4
pc_range = [-40., -40., -1.0, 40., 40., 5.4]
voxel_size = [0.1, 0.1, 0.2]
ignore_classes = True
dynamic_classes = [0, 1, 3, 4, 5, 7, 9, 10] # this seems to be the best setting for classes to ignore!
eval_threshold_range=[.35]
###################################

voxel_size = [0.1, 0.1, 0.2]

numC_Trans = 32
hidden_dim = 256
voxel_feat_dim = 64

multi_adj_frame_id_cfg = (1, 1+1, 1)
render_frame_ids = [-3, -2, -1, 0, 1, 2, 3]

model = dict(
    type='OccFlowNet',
    align_after_view_transfromation=False,
    num_adj=len(range(*multi_adj_frame_id_cfg)),
    out_dim=voxel_feat_dim,
    eval_threshold_range=eval_threshold_range,
    img_backbone=dict(
        type='SwinTransformer',
        pretrain_img_size=224,
        patch_size=4,
        window_size=12,
        mlp_ratio=4,
        embed_dims=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        strides=(4, 2, 2, 2),
        out_indices=(2, 3),
        qkv_bias=True,
        qk_scale=None,
        patch_norm=True,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.1,
        use_abs_pos_embed=False,
        return_stereo_feat=True,
        act_cfg=dict(type='GELU'),
        norm_cfg=dict(type='LN', requires_grad=True),
        pretrain_style='official',
        output_missing_index_as_none=False),
    img_neck=dict(
        type='FPN_LSS',
        in_channels=512 + 1024,
        out_channels=512,
        # with_cp=False,
        extra_upsample=None,
        input_feature_index=(0, 1),
        scale_factor=2),
    img_view_transformer=dict(
        type='LSSViewTransformerBEVStereo',
        grid_config=grid_config,
        input_size=data_config['input_size'],
        in_channels=512,
        out_channels=numC_Trans,
        sid=False,
        collapse_z=False,
        loss_depth_weight=0.05,
        depthnet_cfg=dict(use_dcn=False,
                          aspp_mid_channels=96,
                          stereo=True,
                          bias=5.),
        downsample=16),
    img_bev_encoder_backbone=dict(
        type='CustomResNet3D',
        numC_input=numC_Trans * (len(range(*multi_adj_frame_id_cfg))+1),
        num_layer=[1, 2, 4],
        with_cp=False,
        num_channels=[numC_Trans,numC_Trans*2,numC_Trans*4],
        stride=[1,2,2],
        backbone_output_ids=[0,1,2]),
    img_bev_encoder_neck=dict(type='LSSFPN3D',
                              in_channels=numC_Trans*7,
                              out_channels=numC_Trans),
    pre_process=dict(
        type='CustomResNet3D',
        numC_input=numC_Trans,
        with_cp=False,
        num_layer=[1,],
        num_channels=[numC_Trans,],
        stride=[1,],
        backbone_output_ids=[0,]),
    density_decoder=dict(
        type='PointDecoder',
        in_channels=voxel_feat_dim,
        embed_dims=hidden_dim,
        num_hidden_layers=3,
        num_classes=1,
        final_act_cfg=dict(type='Sigmoid'),
    ),
    semantic_decoder=dict(
        type='PointDecoder',
        in_channels=voxel_feat_dim,
        num_hidden_layers=3,
        embed_dims=hidden_dim,
        num_classes=17,
        final_act_cfg=None
    ),
    renderer=dict(
        type='Renderer',
        render_modules=[
            dict(
                type='DepthRenderModule',
                loss_cfg= dict(type='MSELoss', loss_weight=0.05)
            ),
            dict(
                type='SemanticRenderModule',
                loss_cfg=dict(type='CrossEntropyLoss', loss_weight=1)
            )
        ],
            pc_range=pc_range,
            samples_per_ray=100,
            prop_samples_per_ray=50,
            grid_cfg=[bev_h_, bev_w_, bev_z_, voxel_resolution, 1],
            use_proposal=True,
            temporal_filter=True,
            ignore_classes=dynamic_classes if ignore_classes else None,
            render_frame_ids=render_frame_ids,
            class_balanced_loss=True,
            log_weighting=True,
    ),
)

# Data
dataset_type = 'NuScenesDatasetOccpancy'
data_root = 'data/'
nuscenes_root = 'data/nuscenes'
gt_root = 'data/gts'
file_client_args = dict(backend='disk')

bda_aug_conf = dict(
    rot_lim=(-0., 0.),
    scale_lim=(1., 1.),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5)

train_pipeline = [
    dict(
        type='PrepareImageInputs',
        is_train=True,
        data_config=data_config,
        sequential=True),
    # dict(type='LoadOccGTFromFile'),
    dict(
        type='LoadAnnotationsBEVDepth',
        bda_aug_conf=bda_aug_conf,
        classes=class_names,
        is_train=True),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(type='PointToMultiViewDepth', downsample=1, grid_config=grid_config),
    dict(type='GenerateRays', depth_range=[0.05, 39.0], ignore_classes = dynamic_classes if ignore_classes else None, file_client_args=file_client_args, render_frame_ids=render_frame_ids),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['img_inputs', 'gt_depth', 'origins', 'directions', 'ray_dataset', 'coor'])
]

val_pipeline = [
    dict(type='PrepareImageInputs', data_config=data_config, sequential=True),
    dict(
        type='LoadAnnotationsBEVDepth',
        bda_aug_conf=bda_aug_conf,
        classes=class_names,
        is_train=False),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points', 'img_inputs'])
        ])
]

ann_file_prefix = 'bevdetv2-nuscenes_infos'

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)

share_data_config = dict(
    type=dataset_type,
    classes=class_names,
    modality=input_modality,
    stereo=True,
    filter_empty_gt=False,
    img_info_prototype='bevdet4d',
    multi_adj_frame_id_cfg=multi_adj_frame_id_cfg,
    render_frame_ids=render_frame_ids,
    eval_threshold_range=eval_threshold_range
)

val_data_config = dict(
    pipeline=val_pipeline,
    ann_file=data_root + f'{ann_file_prefix}_val.pkl',
    gt_root=gt_root)

data = dict(
    samples_per_gpu=4,  # with 4 GPUs -> BS=16
    workers_per_gpu=4,
    train=dict(
        data_root=nuscenes_root,
        ann_file=data_root + f'{ann_file_prefix}_train.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        test_mode=False,
        use_valid_flag=True,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d='LiDAR'),
    val=val_data_config,
    test=val_data_config)

for key in ['val', 'train', 'test']:
    data[key].update(share_data_config)

# Optimizer
optimizer = dict(type='AdamW', lr=1e-4, weight_decay=1e-2)

optimizer_config = dict(grad_clip=dict(max_norm=5, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=200,
    warmup_ratio=0.001,
    step=[100,])
runner = dict(type='EpochBasedRunner', max_epochs=12)

custom_hooks = [
    dict(
        type='MEGVIIEMAHook',
        init_updates=10560,
        priority='NORMAL',
    ),
    dict(
        type='SyncbnControlHook',
        syncbn_start_epoch=0,
    ),
]
checkpoint_config = dict(interval=4)
evaluation = dict(interval=4, pipeline=val_pipeline)
load_from="ckpts/bevdet-stbase-4d-stereo-512x1408-cbgs.pth"
# fp16 = dict(loss_scale='dynamic')
