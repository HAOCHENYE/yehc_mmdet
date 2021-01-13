# _base_ = [
#     '../_base_/datasets/coco_detection.py',
#     '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
# ]

img_norm_cfg = dict(
    mean=[128, 128, 128], std=[128, 128, 128], to_rgb=True)
dataset_type = 'CocoDataset'
data_root = '/usr/videodate/dataset/coco/'

model = dict(
    type='ATSS',
    backbone=dict(
        type='YL_Vovnet',
        stem_channels=32,
        stage_channels=(32, 32, 48, 64, 64, 72),
        concat_channels=(32, 32, 64, 128, 128, 128),
        block_per_stage=(1, 1, 1, 1, 2, 1),
        layer_per_block=(1, 1, 3, 3, 4, 4),
        ),
    neck=dict(
        type='PAFPN',
        in_channels=[32, 64, 128, 128, 128],
        out_channels=32,
        num_outs=5,
        start_level=0,
        add_extra_convs=False,
        norm_cfg=dict(type='BN', requires_grad=True)
        ),
    bbox_head=dict(
        type='ATSSPrivateHead',
        num_classes=1,
        in_channels=32,
        stacked_convs=2,
        feat_channels=96,
        scale=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=6,
            scales_per_octave=1,
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[0.1, 0.1, 0.2, 0.2]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)))
# training and testing settings
train_cfg = dict(
    assigner=dict(type='ATSSAssigner', topk=9),
    allowed_border=-1,
    pos_weight=-1,
    debug=False)
test_cfg = dict(
    nms_pre=1000,
    min_bbox_size=0,
    score_thr=0.05,
    nms=dict(type='nms', iou_threshold=0.3),
    max_per_img=100)
# optimizer




train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Resize',
        img_scale=[(1333, 480), (1333, 800)],
        multiscale_mode='range',
        keep_ratio=True),
    dict(type='PhotoMetricDistortion'),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='Normalize',
        mean=[127.5, 127.5, 127.5],
        std=[128, 128, 128],
        to_rgb=True),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
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
            dict(
                type='Normalize',
                mean=[127.5, 127.5, 127.5],
                std=[128, 128, 128],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=32,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/half_person_coco_train.json',
        img_prefix=data_root + 'train2017/',
        classes=["person"],
        pipeline=train_pipeline),

    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/half_person_coco_val.json',
        img_prefix=data_root + 'val2017/',
        classes=["person"],
        pipeline=test_pipeline),

    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/half_person_coco_val.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline)
            )

evaluation = dict(interval=10, metric='bbox')
# optimizer
# optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001,
#                  paramwise_cfg=dict(bias_lr_mult=2., bias_decay_mult=0.))
# optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
optimizer = dict(type='AdamW', lr=0.001)
optimizer_config = dict(grad_clip=None)

# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=2000,
    warmup_ratio=1.0 / 5,
    step=[90, 120, 130])
checkpoint_config = dict(interval=5)

log_config = dict(
    interval=20,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 140
device_ids = range(2)
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = 'work_dirs/atss_vovnet_pafpn_zxsConfig_private_head'
load_from = None
resume_from = None
workflow = [('train', 1)]
