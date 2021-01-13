dataset_type = 'CocoDataset'
data_root = '/usr/videodate/dataset/coco/'
base_lr = 0.32
warmup_iters = 2000

model = dict(
    type='GFL',
    backbone=dict(
        type='CSPOSANet',
        stem_channels=32,
        stage_channels=(32, 64, 72, 96, 128, 192),
        block_per_stage=(2, 3, 6, 8, 6, 6),
        kernel_size=[5, 5, 3, 3, 3, 5],
        conv_type=dict(type="NormalConv",
                       info=dict(norm_cfg=dict(type='BN', requires_grad=True))),
        conv1x1=False
        ),
    neck=dict(
        type='YeFPN',
        in_channels=[64, 72, 96, 128, 192],
        out_channels=32,
        conv_cfg=dict(type="NormalConv",
                       info={"norm_cfg": None})),
    bbox_head=dict(
        type='GFocalDeployNoSoftmax',
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
        loss_cls=dict(
            type='QualityFocalLoss',
            use_sigmoid=False,
            beta=2.0,
            loss_weight=1.0),
        loss_dfl=dict(type='DistributionFocalLoss', loss_weight=0.25),
        reg_max=12,
        reg_topk=4,
        reg_channels=64,
        add_mean=True,
        loss_bbox=dict(type='CIoULoss', loss_weight=2.0)))

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
# train_pipeline = [
#     dict(type='LoadImageFromFile', to_float32=True),
#     dict(type='LoadAnnotations', with_bbox=True),
#     # dict(
#     #     type='MinIoURandomCrop',
#     #     min_ious=(0.6, 0.7, 0.8, 0.9),
#     #     min_crop_size=0.6),
#     # dict(type="AutoAugment",
#     #      policies=[dict(type='Shear', prob=0.2, level=2)]),
#     dict(
#         type='Resize',
#         img_scale=[(1333, 480), (1333, 800)],
#         multiscale_mode='range',
#         keep_ratio=True),
#     dict(type='PhotoMetricDistortion', brightness_delta=48),
#     # dict(type='PhotoMetricDistortion', brightness_delta=48),
#     dict(type='RandomFlip', flip_ratio=0.5),
#     # dict(type="RandomRadiusBlur",
#     #      prob=0.2,
#     #      radius=11),
#     dict(
#         type='Normalize',
#         mean=[127.5, 127.5, 127.5],
#         std=[128, 128, 128],
#         to_rgb=True),
#     dict(type='Pad', size_divisor=32),
#     dict(type='DefaultFormatBundle'),
#     dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
# ]
# test_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(
#         type='MultiScaleFlipAug',
#         img_scale=(1333, 800),
#         flip=False,
#         transforms=[
#             dict(type='Resize', keep_ratio=True),
#             dict(type='RandomFlip'),
#             dict(
#                 type='Normalize',
#                 mean=[127.5, 127.5, 127.5],
#                 std=[128, 128, 128],
#                 to_rgb=True),
#             dict(type='Pad', size_divisor=32),
#             dict(type='ImageToTensor', keys=['img']),
#             dict(type='Collect', keys=['img'])
#         ])
# ]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
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
            dict(type='PhotoMetricDistortion', brightness_delta=48),
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
evaluation = dict(interval=1, metric='bbox')
# optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
optimizer = dict(type='AdamW', lr=0.001)
# optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=2000,
    warmup_ratio=0.01,
    step=[48, 66, 70])



find_unused_parameters=True
checkpoint_config = dict(interval=1)
log_config = dict(
    interval=20,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])

# custom_hooks = [dict(type="EMAHook", momentum=0.1, interval=2, warm_up=warmup_iters, resume_from=None, priority='HIGHEST')]
total_epochs = 60
device_ids = range(0, 2)
dist_params = dict(backend='nccl')
log_level = 'INFO'
# work_dir = 'work_dirs/paa_atss_OSACSP_pafpn_private_SGD_lr0.32_cosine_ema'
work_dir = 'work_dirs/gflv2_OSACSO_yefpn_private_without_softmax/'
load_from = None
resume_from = 'work_dirs/gflv2_OSACSO_yefpn_private_without_softmax/epoch_36.pth'
# resume_from = None
workflow = [('train', 1)]
gpu_ids = range(0, 1)
