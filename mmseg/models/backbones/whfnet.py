"""
@author: 420xincheng
@description: WHFNet: A Wavelet-Driven Heterogeneous Fusion Network for High-Frequency Enhanced Optical-SAR RSS
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from mmseg.registry import MODELS
from ..utils import wavelet_transform_chan
from timm.models.layers import DropPath
import einops


# extract frequency features
class FreqEncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels=[96, 192,384,768], out_indices=[0,1]):
        super(FreqEncoderBlock, self).__init__()

        self.channels = out_channels
        self.out_indices = out_indices

        self.stages = nn.ModuleList()
        for i in range(4):
            if i == 0:
                self.stages.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, self.channels[0], kernel_size=4, stride=4),
                        nn.BatchNorm2d(self.channels[0]),
                        nn.ReLU(inplace=True)
                    )
                )
            else:
                layer = nn.Sequential(
                    nn.Conv2d(self.channels[i-1], self.channels[i], kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(self.channels[i]),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2)
                )
                self.stages.append(layer)
    def forward(self, x):
        feat_list = []
        for idx, stage in enumerate(self.stages):
            x = stage(x)
            if idx in self.out_indices:
                feat_list.append(x)
        return feat_list


class SpatialFrequencyFusionModule(nn.Module):
    """spatial-frequency fusion module (SFFM)"""
    def __init__(self, in_channels, emb_dim, attn_dropout=0.1):
        super().__init__()
        self.emb_dim = emb_dim
        self.proj_in = nn.Conv2d(in_channels, emb_dim, kernel_size=3, stride=1, padding=1)
        self.proj_freq = nn.Conv2d(in_channels, emb_dim, kernel_size=3, stride=1, padding=1)

        self.sa = nn.Sequential(
            nn.Conv2d(emb_dim, emb_dim // 2, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(0.1),
            nn.Conv2d(emb_dim // 2, emb_dim, kernel_size=3, padding=1, bias=True),
            nn.Sigmoid()
        )

        self.conv = nn.Sequential(
            nn.Conv2d(emb_dim * 2, emb_dim * 2, kernel_size=5, stride=1, padding=2, dilation=1, groups=emb_dim * 2),
            nn.BatchNorm2d(emb_dim * 2),
            nn.Conv2d(emb_dim * 2, emb_dim, kernel_size=3, stride=1, padding=1),
        )


    def forward(self, spatial_feat, freq_feat):
        B, C, H, W = spatial_feat.shape
        residual = spatial_feat
        spatial_feat = self.proj_in(spatial_feat)
        freq_feat = self.proj_freq(freq_feat)
        freq_sa_map = self.sa(spatial_feat - freq_feat)
        spatial_feat = spatial_feat + freq_sa_map * freq_feat
        feat_concat = torch.cat([spatial_feat, freq_feat], dim=1)

        out = self.conv(feat_concat)

        out = out + residual
        return out


class CrossModalInteractor(nn.Module):
    def __init__(self, in_channels, emb_dim):
        super().__init__()
        self.opt_freq_spatial_fusion = SpatialFrequencyFusionModule(in_channels, emb_dim=emb_dim)
        self.sar_freq_spatial_fusion = SpatialFrequencyFusionModule(in_channels, emb_dim=emb_dim)
        
    def forward(self, opt_high_feat, opt_feat, sar_high_feat, sar_feat):
        opt_feat = self.opt_freq_spatial_fusion(opt_feat, opt_high_feat)
        sar_feat = self.sar_freq_spatial_fusion(sar_feat, sar_high_feat)
        x_fused = opt_feat + sar_feat
        return opt_feat, sar_feat, x_fused


@MODELS.register_module()
class WHFNet_baseline(BaseModule):
    def __init__(self,
                 backbone,
                 backbone2,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        self.backbone = MODELS.build(backbone)
        self.backbone2 = MODELS.build(backbone2)

        # ConvNext
        self.depths = self.backbone.depths
        self.channels = self.backbone.channels
        self.num_stages = self.backbone.num_stages
        self.out_indices = self.backbone.out_indices

    def forward(self, x_opt, x_sar):
        x_opt, hw_shape = self.backbone2.patch_embed(x_opt)
        x_opt = self.backbone2.drop_after_pos(x_opt)

        outs = []

        for i in range(self.num_stages):
            x_sar = self.backbone.downsample_layers[i](x_sar)
            x_sar = self.backbone.stages[i](x_sar)

            x_opt, hw_shape = self.backbone2.stages[i](x_opt, hw_shape)


            if i in self.out_indices:
                sar_norm_layer = getattr(self.backbone, f'norm{i}')
                opt_norm_layer = getattr(self.backbone2, f'norm{i}')

                sar_out = sar_norm_layer(x_sar)
                opt_out = opt_norm_layer(x_opt)

                opt_out = opt_out.view(-1, *hw_shape,
                               self.backbone2.stages[i].out_channels).permute(0, 3, 1, 2).contiguous()

                x_fused = opt_out + sar_out  
                outs.append(x_fused)

        return tuple(outs)
         

@MODELS.register_module()
class WHFNet(BaseModule):
    """
    WHFNet: A Wavelet-Driven Heterogeneous Fusion Network for High-Frequency Enhanced Optical-SAR RSS
    """
    def __init__(self,
                 backbone,
                 backbone2,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        self.backbone = MODELS.build(backbone)
        self.backbone2 = MODELS.build(backbone2)

        self.depths = self.backbone.depths
        self.channels = self.backbone.channels
        self.num_stages = self.backbone.num_stages
        self.out_indices = self.backbone.out_indices
        
        self.cmi_modules = nn.ModuleList([
            CrossModalInteractor(in_channels, emb_dim=in_channels)
            for in_channels in self.channels
        ])

        self.freq_encoders = nn.ModuleList([
            FreqEncoderBlock(3, out_channels=self.channels, out_indices=[0,1,2,3]), #  
            FreqEncoderBlock(3, out_channels=self.channels, out_indices=[0,1,2,3]), #  
    
        ]) 

    def forward(self, x_opt, x_sar):
        # wavelet_transform
        _, opt_high_freq = wavelet_transform_chan(x_opt, log_transform=False)
        _, sar_high_freq = wavelet_transform_chan(x_sar, log_transform=True)

        opt_high_feats = self.freq_encoders[0](opt_high_freq)
        sar_high_feats = self.freq_encoders[1](sar_high_freq)
        
        x_opt, hw_shape = self.backbone2.patch_embed(x_opt)
        x_opt = self.backbone2.drop_after_pos(x_opt)

        outs = []
        opt_feat_outs, sar_feat_outs = [], []

        for i in range(self.num_stages):
            x_sar = self.backbone.downsample_layers[i](x_sar)
            x_sar = self.backbone.stages[i](x_sar)  # B,C,H,W

            x_opt, hw_shape = self.backbone2.stages[i](x_opt, hw_shape)  # B, seq, C

            if i in self.out_indices:
                
                sar_norm_layer = getattr(self.backbone, f'norm{i}')
                opt_norm_layer = getattr(self.backbone2, f'norm{i}')

                sar_feat = sar_norm_layer(x_sar)
                opt_feat = opt_norm_layer(x_opt)
                opt_feat = opt_feat.view(-1, *hw_shape,
                               self.backbone2.stages[i].out_channels).permute(0, 3, 1, 2).contiguous()
                
                
                opt_freq = opt_high_feats[i]
                sar_freq = sar_high_feats[i]  

                opt_feat, sar_feat, x_fused = self.cmi_modules[i](opt_freq, opt_feat, sar_freq, sar_feat)

                if i == 3: # TODO：ablation
                    opt_feat_outs.append(opt_feat)
                    sar_feat_outs.append(sar_feat)

                outs.append(x_fused)  # [0 1 2 3]
        
        outs.append(opt_feat_outs[0])  # TODO: stage4 feature output
        outs.append(sar_feat_outs[0])

        return tuple(outs)           
