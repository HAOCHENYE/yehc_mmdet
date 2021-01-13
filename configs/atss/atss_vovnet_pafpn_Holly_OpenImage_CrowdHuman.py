# '''In this config, I try to add more convs in stride4&stride8 to detect more small object'''

dataset_type = 'CocoDataset'
data_root = '/usr/videodate/yehc/'

model = dict(
    type='PAA',
    backbone=dict(
        type='YL_Vovnet',
        stem_channels=16,
        stage_channels=(16, 16, 16, 32, 32, 32),
        concat_channels=(16, 24, 32, 64, 128, 128),
        block_per_stage=(1, 1, 3, 4, 2, 2),
        layer_per_block=(1, 1, 2, 2, 4, 4),
        norm_cfg=dict(type='SyncBN', requires_grad=True)
        ),
    neck=dict(
        type='PAFPN',
        in_channels=[24, 32, 64, 128, 128],
        out_channels=32,
        num_outs=5,
        start_level=0,
        add_extra_convs=False,
        norm_cfg=dict(type='SyncBN', requires_grad=True)
        ),
    bbox_head=dict(
        type='ATSSPrivateHead',
        num_classes=1,
        in_channels=32,
        stacked_convs=2,
        feat_channels=96,
        scale=1,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=3,
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
# train_cfg = dict(
#     assigner=dict(
#         type='MaxIoUAssigner',
#         pos_iou_thr=0.1,
#         neg_iou_thr=0.1,
#         min_pos_iou=0,
#         ignore_iof_thr=-1),
#     allowed_border=-1,
#     pos_weight=-1,
#     debug=False)
test_cfg = dict(
    nms_pre=1000,
    min_bbox_size=0,
    score_thr=0.05,
    nms=dict(type='nms', iou_threshold=0.6),
    max_per_img=100)
# optimizer
train_pipeline = [

    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomRadiusBlur', prob=0.3, radius=11, std=0),
    dict(
        type='MinIoURandomCrop',
        min_ious=(0.1, 0.3, 0.5, 0.7, 0.9),
        min_crop_size=0.3),
    dict(
        type='Resize',
        img_scale=[(480, 320)],
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
        img_scale=(480, 320),
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
    samples_per_gpu=96,
    workers_per_gpu=8,
    train=[dict(
        type=dataset_type,
        ann_file=data_root + '/hollywoodheads/hollywoodhead_train.json',
        img_prefix=data_root + '/hollywoodheads/JPEGImages/',
        classes=["person"],
        pipeline=train_pipeline),
        dict(
        type=dataset_type,
        ann_file=data_root+'ImageDataSets/OpenImageV6_CrowdHuman/OpenImageCrowdHuman_train.json',
        img_prefix=data_root+'ImageDataSets/OpenImageV6_CrowdHuman/WIDER_train/images',
        classes=["person"],
        pipeline=train_pipeline)],

    val=dict(
        type=dataset_type,
        ann_file=data_root + '/hollywoodheads/hollywoodhead_val.json',
        img_prefix=data_root + '/hollywoodheads/JPEGImages/',
        classes=["person"],
        pipeline=test_pipeline),

    test=dict(
        type=dataset_type,
        ann_file=data_root + '/hollywoodheads/hollywoodhead_val.json',
        img_prefix=data_root + '/hollywoodheads/JPEGImages/',
        pipeline=test_pipeline)
            )
evaluation = dict(interval=5, metric='bbox')

# optimizer = dict(type='AdamW', lr=0.001)
# optimizer_config = dict(grad_clip=None)
optimizer = dict(type='SGD', lr=0.05, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)

# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=2000,
    warmup_ratio=1.0 / 5,
    step=[90, 100, 110])
checkpoint_config = dict(interval=5)

log_config = dict(
    interval=20,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])

# yapf:enable
# runtime settings

total_epochs = 120
device_ids = range(1)
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = 'work_dirs/atss_vovnet_pafpn_private_head_SGD_head_OpemImage_CrowdHuman_Holly_blur'
load_from = None
resume_from = None
workflow = [('train', 1)]
