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
        layer_per_block=(1, 1, 3, 3, 4, 4)),
    neck=dict(
        type='PAFPN',
        in_channels=[32, 64, 128, 128, 128],
        out_channels=32,
        num_outs=5,
        start_level=0,
        add_extra_convs=False,
        norm_cfg=dict(type='BN', requires_grad=True)),
    bbox_head=dict(
        type='PAA_ATSSHead',
        reg_decoded_bbox=True,
        score_voting=True,
        topk=9,
        num_classes=1,
        in_channels=32,
        stacked_convs=2,
        feat_channels=64,
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
        loss_bbox=dict(type='GIoULoss', loss_weight=1.3),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=0.5)))
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
        type='CocoDataset',
        ann_file=
        '/usr/videodate/dataset/coco/annotations/half_person_coco_train.json',
        img_prefix='/usr/videodate/dataset/coco/train2017/',
        classes=['person'],
        pipeline=[
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
        ]),
    val=dict(
        type='CocoDataset',
        ann_file=
        '/usr/videodate/dataset/coco/annotations/half_person_coco_val.json',
        img_prefix='/usr/videodate/dataset/coco/val2017/',
        classes=['person'],
        pipeline=[
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
        ]),
    test=dict(
        type='CocoDataset',
        ann_file=
        '/usr/videodate/dataset/coco/annotations/half_person_coco_val.json',
        img_prefix='/usr/videodate/dataset/coco/val2017/',
        pipeline=[
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
        ]))
evaluation = dict(interval=5, metric='bbox')
optimizer = dict(type='AdamW', lr=0.001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=2000,
    warmup_ratio=0.2,
    step=[90, 120, 130])

checkpoint_config = dict(interval=5)
log_config = dict(
    interval=20,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
total_epochs = 140
device_ids = range(0, 2)
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = 'work_dirs/PAA_atss_vovnet_pafpn_zxsConfig_private_head'
load_from = None
resume_from = None
workflow = [('train', 1)]
gpu_ids = range(0, 1)
