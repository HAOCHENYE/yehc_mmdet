_base_ = [
    # '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
model = dict(
    type='RepPointsDetector',
    # pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNetLite',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[128, 128, 192, 256],
        out_channels=32,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5),
    bbox_head=dict(
        type='RepPointsHead_NormalConv',
        num_classes=1,
        in_channels=32,
        feat_channels=96,
        point_feat_channels=96,
        stacked_convs=3,
        num_points=9,
        gradient_mul=0.1,
        point_strides=[8, 16, 32, 64, 128],
        point_base_scale=4,
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox_init=dict(type='SmoothL1Loss', beta=0.11, loss_weight=0.5),
        loss_bbox_refine=dict(type='SmoothL1Loss', beta=0.11, loss_weight=1.0),
        transform_method='moment'))
# training and testing settings

dataset_type = 'CocoDataset'
data_root = '/media/traindata_ro/coco/'
img_norm_cfg = dict(
    mean=[128, 128, 128], std=[128, 128, 128], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(640, 480), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

train_cfg = dict(
    init=dict(
        assigner=dict(type='PointAssigner', scale=4, pos_num=1),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    refine=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1),
        allowed_border=-1,
        pos_weight=-1,
        debug=False))
test_cfg = dict(
    nms_pre=1000,
    min_bbox_size=0,
    score_thr=0.05,
    nms=dict(type='nms', iou_threshold=0.5),
    max_per_img=100)

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/my_coco_train.json',
        img_prefix=data_root + 'train2017/',
        classes=["person"],
        pipeline=train_pipeline),

    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/my_coco_val.json',
        img_prefix=data_root + 'val2017/',
        classes=["person"],
        pipeline=test_pipeline),

    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/my_coco_val.json',
        img_prefix=data_root + 'val2017/',
        classes=["person"],
        pipeline=test_pipeline)

            )

optimizer = dict(lr=0.01)
work_dir = 'work_dirs/reppoints_resnet50'