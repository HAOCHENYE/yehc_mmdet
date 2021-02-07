dataset_type = 'CocoDataset'
data_root = '/media/traindata/coco/'
base_lr = 0.32
warmup_iters = 2000

model = dict(
    type='GFL',
    backbone=dict(
        type='RepVGGNet',
        stem_channels=64,
        stage_channels=(64, 64, 72, 96, 128, 192),
        block_per_stage=(1, 3, 6, 8, 6, 6),
        kernel_size=[5, 5, 5, 5, 5, 5]
        ),
    neck=dict(
        type='YeFPN',
        in_channels=[64, 72, 96, 128, 192],
        out_channels=64,
        conv_cfg=dict(type="NormalConv",
                       info={"norm_cfg": None})),
    bbox_head=dict(
        type='VFNetDeployPrivateHead',
        norm_cfg=dict(type='BN', requires_grad=True),
        num_classes=6,
        in_channels=64,
        stacked_convs=2,
        feat_channels=64,
        strides=[8, 16, 32, 64, 128],
        center_sampling=False,
        dcn_on_last_conv=False,
        sample_cfg=dict(
            _delete_=True,
            type='CombinedSampler',
            num=128,
            pos_fraction=0.25,
            add_gt_as_proposals=True,
            pos_sampler=dict(type='InstanceBalancedPosSampler'),
            neg_sampler=dict(
                type='IoUBalancedNegSampler',
                floor_thr=-1,
                floor_fraction=0,
                num_bins=3)),
        use_atss=True,
        use_vfl=True,
        # bbox_coder=dict(_delete_=True, type='TBLRBBoxCoder', normalizer=4.0),
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
            dict(type='LoadPasetImages',
                 class_names=["fake_person", "camera"],
                 base_cls_num=3,
                 image_root="/home/ubuntu/yehc/detection/yehc_mmdet/data_paste",
                 to_float32=True,
                 ),

            dict(
                type='Resize',
                img_scale=[(1333, 480), (1333, 960)],
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
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type='CocoDataset',
        ann_file=data_root + "coco_half_person_80_train.json",
        img_prefix=data_root + 'train2017/images',
        classes=['person', 'bottle', 'chair', 'potted plant'],
        pipeline=train_pipline),

    val=dict(
        type='CocoDataset',
        ann_file=data_root + "coco_half_person_80_val.json",
        img_prefix=data_root + 'val2017/images',
        classes=['person', 'bottle', 'chair', 'potted plant'],
        pipeline=val_pipline),
    test=dict(
        type='CocoDataset',
        ann_file=data_root + "coco_half_person_80_val.json",
        img_prefix=data_root + 'val2017/images',
        classes=['person', 'bottle', 'chair', 'potted plant'],
        pipeline=val_pipline))

evaluation = dict(interval=2, metric='bbox', classwise=True)



optimizer = dict(type='AdamW', lr=0.001)
optimizer_config = dict(update_iter=3, grad_clip=None)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=2000,
    warmup_ratio=0.01,
    step=[90, 110])
# learning policy

total_epochs = 120

checkpoint_config = dict(interval=2)
log_config = dict(
    interval=20,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])

# custom_hooks = [dict(type="EMAHook", momentum=0.1, interval=2, warm_up=warmup_iters, resume_from=None, priority='HIGHEST')]

device_ids = range(0, 2)
dist_params = dict(backend='nccl')
log_level = 'INFO'
# work_dir = 'work_dirs/paa_atss_OSACSP_pafpn_private_SGD_lr0.32_cosine_ema'
work_dir = 'work_dirs/vfnet_test/'
load_from = None
resume_from = None
# resume_from = None
workflow = [('train', 1)]
gpu_ids = range(0, 2)
