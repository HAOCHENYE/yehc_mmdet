# _base_ = [
#     # '../_base_/datasets/coco_detection.py',
#     '../_base_/schedules/schedule_2x.py', '../_base_/default_runtime.py'
# ]

img_norm_cfg = dict(
    mean=[128, 128, 128], std=[128, 128, 128], to_rgb=True)
dataset_type = 'CocoDataset'
data_root = '/usr/videodate/dataset/coco/'

model = dict(
    type='PAA',
    backbone=dict(
        type='YL_Vovnet',
        stem_channels=32,
        stage_channels=(32, 32, 48, 64, 64, 72),
        concat_channels=(32, 32, 64, 256, 256, 384),
        block_per_stage=(1, 1, 1, 1, 1, 1),
        layer_per_block=(1, 1, 3, 3, 7, 4),
        ),
    neck=dict(
        type='FPN',
        in_channels=[32, 64, 256, 256, 384],
        out_channels=32,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=5),
    bbox_head=dict(
        type='PAAHead',
        reg_decoded_bbox=True,
        score_voting=True,
        topk=9,
        num_classes=1,
        in_channels=32,
        stacked_convs=4,
        feat_channels=96,
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=8,
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



# training and testing settings
train_cfg = dict(
    assigner=dict(
        type='MaxIoUAssigner',
        pos_iou_thr=0.1,
        neg_iou_thr=0.1,
        min_pos_iou=0,
        ignore_iof_thr=-1),
    allowed_border=-1,
    pos_weight=-1,
    debug=False)
test_cfg = dict(
    nms_pre=1000,
    min_bbox_size=0,
    score_thr=0.05,
    nms=dict(type='nms', iou_threshold=0.6),
    max_per_img=100)




train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(512, 512), keep_ratio=True),
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
        img_scale=(640, 480),
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
data = dict(
    samples_per_gpu=4,
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
        ann_file=data_root + 'annotations/my_coco_val.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline)
            )

evaluation = dict(interval=1, metric='bbox')
# optimizer
optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 5,
    step=[22, 26])
checkpoint_config = dict(interval=1)

log_config = dict(
    interval=20,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 30
device_ids = range(2)
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = 'work_dirs/paa_vovnet_try_fpn_30epoch_coco'
load_from = None
resume_from = None
workflow = [('train', 1)]
