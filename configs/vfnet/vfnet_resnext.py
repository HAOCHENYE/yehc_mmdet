dataset_type = 'CocoDataset'
data_root = '/media/traindata/coco/'
base_lr = 0.32
warmup_iters = 2000

model = dict(
    type='GFL',
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNeXtDy',
        depth=101,
        groups=32,
        base_width=4,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs=True,
        extra_convs_on_inputs=False,  # use P5
        num_outs=5,
        relu_before_extra_convs=True),
    bbox_head=dict(
        type='VFNetHead',
        num_classes=4,
        in_channels=256,
        stacked_convs=3,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        center_sampling=False,
        dcn_on_last_conv=True,
        use_atss=True,
        use_vfl=True,
        loss_cls=dict(
            type='VarifocalLoss',
            use_sigmoid=True,
            alpha=0.75,
            gamma=2.0,
            iou_weighted=True,
            loss_weight=1.0),
        loss_bbox=dict(type='CIoULoss', loss_weight=1.5),
        loss_bbox_refine=dict(type='CIoULoss', loss_weight=2.0)))

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


train_pipline = [
            dict(type='LoadImageFromFile', to_float32=True),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                type='Resize',
                img_scale=[(256, 256)],
                multiscale_mode='range',
                keep_ratio=True),
            dict(type='RandomRadiusBlur', prob=0.3, radius=5, std=0),
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
        ]

val_pipline = [
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
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        type='CocoDataset',
        ann_file=data_root + "coco_half_person_81_train.json",
        img_prefix=data_root + 'train2017/new_images',
        classes=['person', 'bottle', 'chair', 'potted plant', 'camera'],
        pipeline=train_pipline),

    val=dict(
        type='CocoDataset',
        ann_file=data_root + "coco_half_person_81_val.json",
        img_prefix=data_root + 'val2017/new_images',
        classes=['person', 'bottle', 'chair', 'potted plant', 'camera'],
        pipeline=val_pipline),
    test=dict(
        type='CocoDataset',
        ann_file=data_root + "coco_half_person_81_val.json",
        img_prefix=data_root + 'val2017/new_images',
        classes=['person', 'bottle', 'chair', 'potted plant', 'camera'],
        pipeline=val_pipline))
evaluation = dict(interval=1, metric='bbox', classwise=True)
# optimizer = dict(type='AdamW', lr=0.001)
# optimizer_config = dict(grad_clip=None)
# lr_config = dict(
#     policy='step',
#     warmup='linear',
#     warmup_iters=2000,
#     warmup_ratio=0.01,
#     step=[90, 110, 115])

optimizer = dict(type='SGD',
                 lr=0.01,
                 momentum=0.9,
                 weight_decay=0.0001,
                 paramwise_cfg=dict(bias_lr_mult=2., bias_decay_mult=0.))
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[16, 22])
total_epochs = 24

checkpoint_config = dict(interval=1)
log_config = dict(
    interval=20,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])

# custom_hooks = [dict(type="EMAHook", momentum=0.1, interval=2, warm_up=warmup_iters, resume_from=None, priority='HIGHEST')]

device_ids = range(0, 2)
dist_params = dict(backend='nccl')
log_level = 'INFO'
# work_dir = 'work_dirs/paa_atss_OSACSP_pafpn_private_SGD_lr0.32_cosine_ema'
work_dir = 'work_dirs/vfnet_resnext/'
load_from = None
resume_from = None
# resume_from = None
workflow = [('train', 1)]
gpu_ids = range(0, 2)
