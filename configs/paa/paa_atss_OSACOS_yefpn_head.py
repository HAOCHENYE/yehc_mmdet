# '''In this config, I try to add more convs in stride4&stride8 to detect more small object'''

dataset_type = 'CocoDataset'
data_root = '/usr/videodate/yehc/'

model = dict(
    type='PAA',
    backbone=dict(
        type='CSPOSANet',
        stem_channels=32,
        stage_channels=(32, 48, 48, 64, 64, 72),
        block_per_stage=(1, 2, 4, 8, 3, 3),
        kernel_size=[3, 3, 3, 5, 5, 5],
        conv_type=dict(type="NormalConv",
                       info=dict(norm_cfg=dict(type='SyncBN', requires_grad=True))),
        conv1x1=False
        ),
    neck=dict(
        type='YeFPN',
        in_channels=[48, 48, 64, 64, 72],
        out_channels=32,
        conv_cfg=dict(type="NormalConv",
                       info={"norm_cfg": None})),
    bbox_head=dict(
        type='PAA_ATSSHead',
        reg_decoded_bbox=True,
        score_voting=True,
        topk=9,
        num_classes=1,
        in_channels=32,
        stacked_convs=2,
        feat_channels=64,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=1,
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
        loss_bbox=dict(type='CIoULoss', loss_weight=1.3),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=0.5)))

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
    nms=dict(type='nms', iou_threshold=0.6),
    max_per_img=100)
# optimizer
train_pipeline = [

    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomRadiusBlur', prob=0.3, radius=5, std=0),
    # dict(
    #     type='MinIoURandomCrop',
    #     min_ious=(0.4, 0.5, 0.6, 0.7, 0.8),
    #     min_crop_size=0.3),
    dict(
        type='Resize',
        img_scale=[(80, 128), (128, 144), (144, 192), (192, 256), (256, 384)],
        multiscale_mode='value',
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
        img_scale=(128, 128),
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
    train=
        dict(
        type=dataset_type,
        ann_file=data_root+'ImageDataSets/OpenImageV6_CrowdHuman/OpenImageCrowdHuman_train.json',
        img_prefix=data_root+'ImageDataSets/OpenImageV6_CrowdHuman/WIDER_train/images',
        classes=["person"],
        pipeline=train_pipeline),

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
evaluation = dict(interval=2, metric='bbox')

optimizer = dict(type='AdamW', lr=0.001)
optimizer_config = dict(grad_clip=None)
# optimizer = dict(type='SGD', lr=0.05, momentum=0.9, weight_decay=0.0001)
# optimizer_config = dict(grad_clip=None)

# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=2000,
    warmup_ratio=1.0 / 5,
    step=[90, 110, 115])
checkpoint_config = dict(interval=2)

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
work_dir = 'work_dirs/paa_atss_vovnet_pafpn_private_head_head'
load_from = None
resume_from = 'work_dirs/paa_atss_vovnet_pafpn_private_head_head/epoch_80.pth'
workflow = [('train', 1)]
