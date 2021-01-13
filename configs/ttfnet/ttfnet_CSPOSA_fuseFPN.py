# model settings
dataset_type = 'CocoDataset'
data_root = '/usr/videodate/yehc/'

model = dict(
    type='TTFNet',
    backbone=dict(
        type='CSPOSANet',
        stem_channels=32,
        stage_channels=(64, 128, 192, 256),
        block_per_stage=(2, 3, 8, 6),
        kernel_size=[3, 3, 3, 3],
        num_out=4,
        conv_type=dict(type="NormalConv",
                       info=dict(norm_cfg=dict(type='SyncBN', requires_grad=True))),
        conv1x1=True
        ),
    neck=dict(
        type='FuseFPN',
        in_channels=[64, 128, 192, 256],
        out_channels=96,
        conv_cfg=dict(type="NormalConv",
                      info={"norm_cfg": None})),
    bbox_head=dict(
        type='TTFHead',
        planes=(96, 96, 96),
        base_down_ratio=32,
        head_conv=96,     #默认头通道数
        wh_conv=96,        #wh头通道数
        hm_head_conv_num=2,
        wh_head_conv_num=2,
        num_classes=1,
        wh_offset_base=16, #？？？？？？？？？？？
        wh_agnostic=True,
        wh_gaussian=True,
        norm_cfg=dict(type='SyncBN'),
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

img_norm_cfg = dict(
    mean=[128, 128, 128], std=[128, 128, 128], to_rgb=True)

train_pipeline = [

    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomRadiusBlur', prob=0.3, radius=5, std=0),
    dict(
        type='MinIoURandomCrop',
        min_ious=(0.6, 0.7, 0.8, 0.9),
        min_crop_size=0.3),
    dict(
        type='Resize',
        img_scale=[(320, 320)],
        multiscale_mode='value',
        keep_ratio=False),
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
        img_scale=(320, 320),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
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
    samples_per_gpu=72,
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
# data = dict(
#     samples_per_gpu=48,
#     workers_per_gpu=10,
#     train=dict(
#         type='CocoDataset',
#         ann_file=
#         '/usr/videodate/dataset/coco/annotations/half_person_coco_train.json',
#         img_prefix='/usr/videodate/dataset/coco/train2017/',
#         classes=['person'],
#         pipeline=[
#             dict(type='LoadImageFromFile', to_float32=True),
#             dict(type='LoadAnnotations', with_bbox=True),
#             dict(
#                 type='Resize',
#                 img_scale=(512, 512),
#                 multiscale_mode='range',
#                 keep_ratio=False),
#             dict(type='PhotoMetricDistortion'),
#             dict(type='RandomFlip', flip_ratio=0.5),
#             dict(
#                 type='Normalize',
#                 mean=[127.5, 127.5, 127.5],
#                 std=[128, 128, 128],
#                 to_rgb=True),
#             dict(type='Pad', size_divisor=32),
#             dict(type='DefaultFormatBundle'),
#             dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
#         ]),
#     val=dict(
#         type='CocoDataset',
#         ann_file=
#         '/usr/videodate/dataset/coco/annotations/half_person_coco_val.json',
#         img_prefix='/usr/videodate/dataset/coco/val2017/',
#         classes=['person'],
#         pipeline=[
#             dict(type='LoadImageFromFile'),
#             dict(
#                 type='MultiScaleFlipAug',
#                 img_scale=(512, 512),
#                 flip=False,
#                 transforms=[
#                     dict(type='Resize', keep_ratio=False),
#                     dict(type='RandomFlip'),
#                     dict(
#                         type='Normalize',
#                         mean=[127.5, 127.5, 127.5],
#                         std=[128, 128, 128],
#                         to_rgb=True),
#                     dict(type='Pad', size_divisor=32),
#                     dict(type='ImageToTensor', keys=['img']),
#                     dict(type='Collect', keys=['img'])
#                 ])
#         ]),
#     test=dict(
#         type='CocoDataset',
#         ann_file=
#         '/usr/videodate/dataset/coco/annotations/half_person_coco_val.json',
#         img_prefix='/usr/videodate/dataset/coco/val2017/',
#         pipeline=[
#             dict(type='LoadImageFromFile'),
#             dict(
#                 type='MultiScaleFlipAug',
#                 img_scale=(512, 512),
#                 flip=False,
#                 transforms=[
#                     dict(type='Resize', keep_ratio=False),
#                     dict(type='RandomFlip'),
#                     dict(
#                         type='Normalize',
#                         mean=[127.5, 127.5, 127.5],
#                         std=[128, 128, 128],
#                         to_rgb=True),
#                     dict(type='Pad', size_divisor=32),
#                     dict(type='ImageToTensor', keys=['img']),
#                     dict(type='Collect', keys=['img'])
#                 ])
#         ]))
evaluation = dict(interval=2, metric='bbox')

# optimizer = dict(type='AdamW', lr=0.001)
# optimizer_config = dict(grad_clip=None)

optimizer = dict(type='SGD', lr=0.015, momentum=0.9, weight_decay=0.0004)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=600,
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
work_dir = 'work_dirs/ttfnet_CSPOSA_fuseFPN'
load_from = None
resume_from = None
workflow = [('train', 1)]