_base_ = [
    'mmseg::_base_/schedules/schedule_15k.py',
    'mmseg::_base_/default_runtime.py',
    '../_base_/datasets/ddhr_xian_opt_sar_128x128.py',
]
norm_cfg = dict(type='SyncBN', requires_grad=True)
crop_size = (128, 128)
data_preprocessor = dict(
    type='RGBXDataPreProcessor',
    size=crop_size,
    mean=[70.832, 71.778, 67.988, 81.981, 81.981, 81.981],
    std=[49.720, 49.708, 50.006, 48.360, 48.360, 48.360],
    bgr_to_rgb=False,
    pad_val=0,
    seg_pad_val=255)

checkpoint_file = '/workspace/mmseg/checkpoints/convnext-v2-tiny_fcmae-in21k-pre_3rdparty_in1k-384px_20230104-d8579f84.pth'  # noqa
model = dict(
    type='EarlyFusionSegmentor',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='WHFNet',
        backbone2=dict(
            type='mmpretrain.SwinTransformerV2',
            arch='tiny',
            img_size=128,
            window_size=4,
            out_indices=[0, 1, 2, 3],
            init_cfg=dict(type='Pretrained',
                          checkpoint='checkpoints/swin_tiny_patch4_window7_224-160bb0a5.pth',
                          prefix='backbone.')
        ),
        backbone=dict(
            type='mmpretrain.ConvNeXt',
            arch='tiny',
            out_indices=[0, 1, 2, 3],
            drop_path_rate=0.4,
            layer_scale_init_value=0., 
            gap_before_final_norm=False,
            use_grn=True, 
            init_cfg=dict(type='Pretrained',
                          checkpoint=checkpoint_file,
                          prefix='backbone.')),
        
            
    ),
    decode_head=dict(
        type='UPerHead',
        in_channels=[96, 192, 384, 768],
        in_index=[0, 1, 2, 3],
        channels=256,
        pool_scales=(1, 2, 3, 6),
        dropout_ratio=0.1,
        num_classes=5,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=[
            dict(type='CrossEntropyLoss',
                 use_sigmoid=False,
                 avg_non_ignore=True,
                 loss_weight=1.0),
        ],
    ),
    auxiliary_head=[
        dict(
            type='FCNHead',
            in_channels=384,
            in_index=2,
            channels=128,
            num_convs=1,
            concat_input=False,
            dropout_ratio=0.1,
            num_classes=5,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(type='CrossEntropyLoss',
                            use_sigmoid=False,
                            avg_non_ignore=True,
                            loss_weight=0.4)),
            dict(type='FCNHead',
                in_channels=768,
                in_index=4,
                channels=128,
                num_convs=1,
                concat_input=False,
                dropout_ratio=0.1,
                num_classes=5,
                norm_cfg=norm_cfg,
                align_corners=False,
                loss_decode=dict(type='CrossEntropyLoss',
                                use_sigmoid=False,
                                avg_non_ignore=True,
                                loss_weight=0.3)),
            dict(type='FCNHead',
                in_channels=768,
                in_index=5,
                channels=128,
                num_convs=1,
                concat_input=False,
                dropout_ratio=0.1,
                num_classes=5,
                norm_cfg=norm_cfg,
                align_corners=False,
                loss_decode=dict(type='CrossEntropyLoss',
                                use_sigmoid=False,
                                avg_non_ignore=True,
                                loss_weight=0.3)),
    ],
    train_cfg=dict(),
    test_cfg=dict(mode='whole'),
)
optim_wrapper = dict(_delete_=True,
                     type='AmpOptimWrapper',
                     optimizer=dict(type='AdamW',
                                    lr=0.0001,
                                    betas=(0.9, 0.999),
                                    weight_decay=0.05),
                     paramwise_cfg={
                         'decay_rate': 0.9,
                         'decay_type': 'stage_wise',
                         'num_layers': 6
                     },
                     constructor='CustomLearningRateDecayOptimizerConstructor',
                     loss_scale='dynamic')
param_scheduler = [
    dict(type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0,
         end=1500),
    dict(type='PolyLR',
         power=1.0,
         begin=1500,
         end=15000,
         eta_min=0.0,
         by_epoch=False)
]
default_hooks = dict(
    checkpoint=dict(save_best='mIoU', save_last=False, max_keep_ckpts=1))

randomness = dict(seed=42)  # 42

vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(vis_backends=vis_backends)
