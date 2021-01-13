dataset_type = 'CocoDataset'
data_root = '/usr/videodate/yehc/ImageDataSets/WIDERFACE/'

img_norm_cfg = dict(
    mean=[127.5, 127.5, 127.5], std=[128, 128, 128], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomRadiusBlur', prob=0.3, radius=5, std=0),
    dict(
        type='MinIoURandomCrop',
        min_ious=(0.1, 0.3, 0.5, 0.7, 0.9),
        min_crop_size=0.3),
    dict(
        type='Resize',
        img_scale=(320, 320),
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
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=64,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        classes=['person'],
        ann_file=data_root + 'wider_face_train_annot_coco_style.json',
        img_prefix=data_root + 'WIDER_train/images',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        classes=['person'],
        ann_file=data_root + 'wider_face_val_annot_coco_style.json',
        img_prefix=data_root + 'WIDER_val/images',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        classes=['person'],
        ann_file=data_root + 'wider_face_val_annot_coco_style.json',
        img_prefix=data_root + 'WIDER_val/images',
        pipeline=test_pipeline))

evaluation = dict(metric=['bbox'], interval=2)
