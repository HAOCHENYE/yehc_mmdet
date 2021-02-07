# model settings
_base_ = [
    '../_base_/datasets/widerface.py'
]
base_lr = 0.01
warmup_iters = 500

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
                       info=dict(norm_cfg=dict(type='BN', requires_grad=True))),
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

# optimizer = dict(type='AdamW', lr=0.001)
# optimizer_config = dict(grad_clip=None)

evaluation = dict(interval=5, metric='bbox', classwise=True)

optimizer = dict(type='SGD', lr=base_lr, momentum=0.937, weight_decay=0.0005)
optimizer_config = dict(grad_clip=None)


lr_config = dict(
    policy='CosineAnnealing',
    min_lr=base_lr/1000,
    warmup='linear',
    warmup_iters=warmup_iters,
    warmup_ratio=0.001)
total_epochs = 300

checkpoint_config = dict(interval=5)
log_config = dict(
    interval=20,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])

# custom_hooks = [dict(type="EMAHook", warm_up=warmup_iters, resume_from=None, priority='HIGHEST')]



device_ids = range(2)
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = 'work_dirs/ttfnet_CSPOSA_widerface_adamw'
load_from = None
resume_from = None
workflow = [('train', 1)]