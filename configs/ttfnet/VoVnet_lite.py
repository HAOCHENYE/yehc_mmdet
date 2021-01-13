# model settings
model = dict(
    type='TTFNet',
    backbone=dict(type='VoVNet_lite'),
    neck=None,
    bbox_head=dict(
        type='TTFHead_lite',
        inplanes=(32, 32, 32, 32),
        planes=(32, 32, 32),
        base_down_ratio=64,
        head_conv=128,     #默认头通道数
        wh_conv=64,        #wh头通道数
        hm_head_conv_num=2,
        wh_head_conv_num=2,
        num_classes=1,
        wh_offset_base=16, #？？？？？？？？？？？
        wh_agnostic=True,
        wh_gaussian=True,
        norm_cfg=dict(type='BN'),
        alpha=0.54,
        hm_weight=1.,
        wh_weight=5.,
        use_dla=True))
cudnn_benchmark = True
# training and testing settings
train_cfg = dict(
    vis_every_n_iters=100,
    debug=False)
test_cfg = dict(
    score_thr=0.01,
    max_per_img=100)
# dataset settings
dataset_type = 'CocoDataset'
data_root = '/media/traindata_ro/coco/'
img_norm_cfg = dict(
    mean=[128, 128, 128], std=[128, 128, 128], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(512, 512), keep_ratio=False),
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
        img_scale=(512, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=24,
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
        pipeline=test_pipeline)

            )
# optimizer
# optimizer = dict(type='SGD', lr=0.015, momentum=0.9, weight_decay=0.0004)
optimizer = dict(type='AdamW', lr=0.001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=6000,
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
device_ids = range(1)
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = 'work_dirs/VoVNet_lite_epch30_AdamW_OSA_lite_NEW_HEAD_stride8'
load_from = None
resume_from = None
workflow = [('train', 1)]
