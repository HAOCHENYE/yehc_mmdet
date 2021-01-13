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
        norm_cfg=dict(type='SyncBN', requires_grad=True)),
    neck=dict(
        type='PAFPN',
        in_channels=[32, 64, 128, 128, 128],
        out_channels=32,
        num_outs=5,
        start_level=0,
        add_extra_convs=False,
        norm_cfg=dict(type='SyncBN', requires_grad=True)),
    bbox_head=dict(
        type='GFLPrivateHead',
        num_classes=1,
        in_channels=32,
        stacked_convs=2,
        feat_channels=96,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=6,
            scales_per_octave=1,
            strides=[8, 16, 32, 64, 128]),
        loss_cls=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0),
        loss_dfl=dict(type='DistributionFocalLoss', loss_weight=0.25),
        reg_max=16,
        loss_bbox=dict(type='CIoULoss', loss_weight=2.0)))
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
    samples_per_gpu=36,
    workers_per_gpu=8,
    train=dict(
        type='CocoDataset',
        ann_file=data_root + 'annotations/half_person_coco_train.json',
        img_prefix=data_root + 'train2017/',
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
        ann_file=data_root + 'annotations/half_person_coco_val.json',
        img_prefix=data_root + 'val2017/',
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
        ann_file=data_root + 'annotations/half_person_coco_val.json',
        img_prefix=data_root + 'val2017/',
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
    step=[110, 135, 150])
checkpoint_config = dict(interval=5)
log_config = dict(
    interval=20,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
total_epochs = 175
device_ids = range(0, 2)
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = 'work_dirs/gfl_vovnet_pafpn_private_head_coco_person'
load_from = None
resume_from = 'work_dirs/gfl_vovnet_pafpn_private_head_coco_person/latest.pth'
workflow = [('train', 1)]
gpu_ids = [1]
