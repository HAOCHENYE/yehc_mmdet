dataset_type = 'CocoDataset'
data_root = '/usr/videodate/dataset/coco/'
base_lr = 0.32
warmup_iters = 2000
model = dict(
    type='PAA',
    backbone=dict(
        type='CSPOSANet',
        stem_channels=32,
        stage_channels=(32, 48, 96, 128, 192, 256),
        block_per_stage=(2, 4, 8, 8, 8, 8),
        conv_type=dict(type="NormalConv",
                       info=dict(norm_cfg=dict(type='SyncBN', requires_grad=True))),
        ),
    neck=dict(
        type='CSP_PAFPN',
        in_channels=[48, 96, 128, 192, 256],
        out_channels=32,
        num_outs=5,
        start_level=0,
        add_extra_convs=False,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        conv_type=dict(type="ACBlock",
                       info=dict(norm_cfg=dict(type='SyncBN', requires_grad=True)
                                 ))),
    # neck=dict(
    #     type='PAFPN',
    #     in_channels=[48, 64, 80, 112, 180],
    #     out_channels=32,
    #     num_outs=5,
    #     start_level=0,
    #     add_extra_convs=False,
    #     norm_cfg=dict(type='BN', requires_grad=True)
    # ),
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
        loss_bbox=dict(type='CIoULoss', loss_weight=1.3),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=0.5)))

train_cfg = dict(
    assigner=dict(type='ATSSAssigner', topk=6),
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
        type='MinIoURandomCrop',
        min_ious=(0.6, 0.7, 0.8, 0.9),
        min_crop_size=0.6),
    dict(type="AutoAugment",
         policies=[dict(type='Shear', prob=0.2, level=2)]),
    dict(
        type='Resize',
        img_scale=[(1333, 480), (1333, 800)],
        multiscale_mode='range',
        keep_ratio=True),
    dict(type='PhotoMetricDistortion', brightness_delta=48),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type="RandomRadiusBlur",
         prob=0.2,
         radius=11),
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
    samples_per_gpu=24,
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
optimizer = dict(type='SGD', lr=base_lr, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=base_lr/1000,
    warmup='linear',
    warmup_iters=warmup_iters,
    warmup_ratio=1.0 / 5)

checkpoint_config = dict(interval=5)
log_config = dict(
    interval=20,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])


total_epochs = 180
device_ids = range(0, 2)
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = 'work_dirs/paa_atss_OSACSP_pafpn_private_SGD_lr0.32_cosine_ema'
# work_dir = 'work_dirs/paa_atss_OSACSP_pafpn_private_AdamW_lr0.001_step'
load_from = None
resume_from = 'work_dirs/paa_atss_OSACSP_pafpn_private_SGD_lr0.32_cosine_ema/epoch_150.pth'

custom_hooks = [dict(type="EMAHook", warm_up=warmup_iters, resume_from=resume_from, priority='HIGHEST')]

workflow = [('train', 1)]
gpu_ids = range(0, 1)
