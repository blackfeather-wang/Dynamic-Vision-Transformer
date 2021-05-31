# Copyright (c) [2012]-[2021] Shanghai Yitu Technology Co., Ltd.
#
# This source code is licensed under the Clear BSD License
# LICENSE file in the root directory of this file
# All rights reserved.
"""
T2T-ViT
"""
import torch
import torch.nn as nn

from timm.models.helpers import load_pretrained
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_
import numpy as np

from .token_transformer import Token_transformer
from .token_performer import Token_performer
from .transformer_block import Block, get_sinusoid_encoding, Mlp

from torch import Tensor


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225),
        'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    'T2t_vit_7': _cfg(),
    'T2t_vit_10': _cfg(),
    'T2t_vit_12': _cfg(),
    'T2t_vit_14': _cfg(),
    'T2t_vit_19': _cfg(),
    'T2t_vit_24': _cfg(),
    'T2t_vit_t_14': _cfg(),
    'T2t_vit_t_19': _cfg(),
    'T2t_vit_t_24': _cfg(),
    'T2t_vit_14_resnext': _cfg(),
    'T2t_vit_14_wide': _cfg(),
}


class GELU(nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        return torch.nn.functional.gelu(input)


class T2T_module(nn.Module):
    """
    Tokens-to-Token encoding module
    """

    def __init__(self, img_size=224, tokens_type='performer', in_chans=3, embed_dim=768, token_dim=64):
        super().__init__()

        if tokens_type == 'transformer_less_less_token':
            print('adopt transformer_less_less_token encoder for tokens-to-token')
            self.soft_split0 = nn.Unfold(kernel_size=(14, 14), stride=(8, 8), padding=(3, 3))
            self.soft_split1 = nn.Unfold(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
            self.soft_split2 = nn.Unfold(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))

            self.attention1 = Token_transformer(dim=in_chans * 14 * 14, in_dim=token_dim, num_heads=1, mlp_ratio=1.0)
            self.attention2 = Token_transformer(dim=token_dim * 3 * 3, in_dim=token_dim, num_heads=1, mlp_ratio=1.0)
            self.project = nn.Linear(token_dim * 3 * 3, embed_dim)

            self.num_patches = (img_size // (8 * 2 * 2)) * (img_size // (8 * 2 * 2))

        if tokens_type == 'transformer_less_token':
            print('adopt transformer_less_token encoder for tokens-to-token')
            self.soft_split0 = nn.Unfold(kernel_size=(11, 11), stride=(6, 6), padding=(5, 5))
            self.soft_split1 = nn.Unfold(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
            self.soft_split2 = nn.Unfold(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))

            self.attention1 = Token_transformer(dim=in_chans * 11 * 11, in_dim=token_dim, num_heads=1, mlp_ratio=1.0)
            self.attention2 = Token_transformer(dim=token_dim * 3 * 3, in_dim=token_dim, num_heads=1, mlp_ratio=1.0)
            self.project = nn.Linear(token_dim * 3 * 3, embed_dim)

            self.num_patches = 100

        if tokens_type == 'transformer':
            print('adopt transformer encoder for tokens-to-token')
            self.soft_split0 = nn.Unfold(kernel_size=(7, 7), stride=(4, 4), padding=(2, 2))
            self.soft_split1 = nn.Unfold(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
            self.soft_split2 = nn.Unfold(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))

            self.attention1 = Token_transformer(dim=in_chans * 7 * 7, in_dim=token_dim, num_heads=1, mlp_ratio=1.0)
            self.attention2 = Token_transformer(dim=token_dim * 3 * 3, in_dim=token_dim, num_heads=1, mlp_ratio=1.0)
            self.project = nn.Linear(token_dim * 3 * 3, embed_dim)

            self.num_patches = (img_size // (4 * 2 * 2)) * (img_size // (4 * 2 * 2))

        elif tokens_type == 'performer_less_less_token':
            print('adopt performer_less_less_token encoder for tokens-to-token')
            self.soft_split0 = nn.Unfold(kernel_size=(14, 14), stride=(8, 8), padding=(3, 3))
            self.soft_split1 = nn.Unfold(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
            self.soft_split2 = nn.Unfold(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))

            self.attention1 = Token_performer(dim=in_chans * 14 * 14, in_dim=token_dim, kernel_ratio=0.5)
            self.attention2 = Token_performer(dim=token_dim * 3 * 3, in_dim=token_dim, kernel_ratio=0.5)
            self.project = nn.Linear(token_dim * 3 * 3, embed_dim)

            self.num_patches = (img_size // (8 * 2 * 2)) * (img_size // (8 * 2 * 2))

        elif tokens_type == 'performer_less_token':
            print('adopt performer_less_token encoder for tokens-to-token')
            self.soft_split0 = nn.Unfold(kernel_size=(11, 11), stride=(6, 6), padding=(5, 5))
            self.soft_split1 = nn.Unfold(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
            self.soft_split2 = nn.Unfold(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))

            self.attention1 = Token_performer(dim=in_chans * 11 * 11, in_dim=token_dim, kernel_ratio=0.5)
            self.attention2 = Token_performer(dim=token_dim * 3 * 3, in_dim=token_dim, kernel_ratio=0.5)
            self.project = nn.Linear(token_dim * 3 * 3, embed_dim)

            self.num_patches = 100

        elif tokens_type == 'performer':
            print('adopt performer encoder for tokens-to-token')
            self.soft_split0 = nn.Unfold(kernel_size=(7, 7), stride=(4, 4), padding=(2, 2))
            self.soft_split1 = nn.Unfold(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
            self.soft_split2 = nn.Unfold(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))

            self.attention1 = Token_performer(dim=in_chans * 7 * 7, in_dim=token_dim, kernel_ratio=0.5)
            self.attention2 = Token_performer(dim=token_dim * 3 * 3, in_dim=token_dim, kernel_ratio=0.5)
            self.project = nn.Linear(token_dim * 3 * 3, embed_dim)

            self.num_patches = (img_size // (4 * 2 * 2)) * (img_size // (4 * 2 * 2))

        elif tokens_type == 'convolution':  # just for comparison with conolution, not our model
            # for this tokens type, you need change forward as three convolution operation
            print('adopt convolution layers for tokens-to-token')
            self.soft_split0 = nn.Conv2d(3, token_dim, kernel_size=(7, 7), stride=(4, 4),
                                         padding=(2, 2))  # the 1st convolution
            self.soft_split1 = nn.Conv2d(token_dim, token_dim, kernel_size=(3, 3), stride=(2, 2),
                                         padding=(1, 1))  # the 2nd convolution
            self.project = nn.Conv2d(token_dim, embed_dim, kernel_size=(3, 3), stride=(2, 2),
                                     padding=(1, 1))  # the 3rd convolution

            self.num_patches = (img_size // (4 * 2 * 2)) * (
            img_size // (4 * 2 * 2))  # there are 3 sfot split, stride are 4,2,2 seperately

    def forward(self, x):

        # step0: soft split
        x = self.soft_split0(x).transpose(1, 2)

        # iteration1: re-structurization/reconstruction
        x = self.attention1(x)
        B, new_HW, C = x.shape
        x = x.transpose(1, 2).reshape(B, C, int(np.sqrt(new_HW)), int(np.sqrt(new_HW)))
        # iteration1: soft split
        x = self.soft_split1(x).transpose(1, 2)

        # iteration2: re-structurization/reconstruction
        x = self.attention2(x)
        B, new_HW, C = x.shape
        x = x.transpose(1, 2).reshape(B, C, int(np.sqrt(new_HW)), int(np.sqrt(new_HW)))
        # iteration2: soft split
        x = self.soft_split2(x).transpose(1, 2)

        # final tokens
        x = self.project(x)
        return x


class T2T_ViT(nn.Module):
    def __init__(self, img_size=224, tokens_type='performer', in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, token_dim=64,
                 feature_reuse=False, relation_reuse=False):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.feature_reuse = feature_reuse
        self.relation_reuse = relation_reuse

        self.tokens_to_token = T2T_module(
            img_size=img_size, tokens_type=tokens_type, in_chans=in_chans, embed_dim=embed_dim, token_dim=token_dim)
        num_patches = self.tokens_to_token.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(data=get_sinusoid_encoding(n_position=num_patches + 1, d_hid=embed_dim),
                                      requires_grad=False)
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
                                        Block(
                                            dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                                            qk_scale=qk_scale,
                                            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i],
                                            norm_layer=norm_layer,
                                            feature_reuse=feature_reuse)
                                        for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        self.depth = depth
        self.num_heads = num_heads
        
        if relation_reuse:
            # an efficient upsampling implementation
            if self.tokens_to_token.num_patches == 100:
                self.relation_reuse_upsample = torch.nn.Upsample(size=(14, 14), mode='nearest')
                self.adaptive_avgpool = torch.nn.AdaptiveAvgPool2d((10, 10))
            else:
                self.relation_reuse_upsample = torch.nn.Upsample(size=(20, 20), mode='nearest')
                self.adaptive_avgpool = torch.nn.AdaptiveAvgPool2d((14, 14))

            self.relation_reuse_conv = nn.Sequential(
                nn.Conv2d(self.num_heads * self.depth, self.num_heads * self.depth * 3, kernel_size=1, stride=1, padding=0, bias=True),
                GELU(),
                nn.Conv2d(self.num_heads * self.depth * 3, self.num_heads * self.depth, kernel_size=1, stride=1, padding=0, bias=True)
            )
            
        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    @torch.jit.ignore
    def no_weight_decay(self):
        return {'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, features_to_be_reused_list, relations_to_be_reused_list):
        B = x.shape[0]
        x = self.tokens_to_token(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        if relations_to_be_reused_list is not None:

            new_hw = x.size(1) - 1
            relations_to_be_reused = torch.cat(relations_to_be_reused_list,1)
            relations_to_be_reused = self.relation_reuse_conv(relations_to_be_reused)

            relation_temp = relations_to_be_reused[:, :, :, 1:]
            B, h, n, hw  = relation_temp.shape
            relation_temp = relation_temp.reshape(-1, n, int(np.sqrt(hw)), int(np.sqrt(hw)))

            split_index = int(relation_temp.size(0) / 2)
            relation_temp = torch.cat(
                (
                    self.relation_reuse_upsample(relation_temp[:split_index * 1]),
                    self.relation_reuse_upsample(relation_temp[split_index * 1:]),
                ), 0
            )
            relation_temp = self.adaptive_avgpool(relation_temp)
            relation_temp = relation_temp.reshape(-1, n, new_hw)

            relation_temp = torch.cat((relations_to_be_reused[:, :, :, 0].reshape(-1,n,1), relation_temp), dim=2).transpose(1,2)
            relation_cls_token_temp = relation_temp[:,:,0:1]
            relation_temp = relation_temp[:,:,1:]
            relation_temp = relation_temp.reshape(-1, (new_hw+1), int(np.sqrt(hw)), int(np.sqrt(hw)))

            split_index = int(relation_temp.size(0) / 2)
            relation_temp = torch.cat(
                (
                    self.relation_reuse_upsample(relation_temp[:split_index * 1]),
                    self.relation_reuse_upsample(relation_temp[split_index * 1:]),
                ), 0
            )
            relation_temp = self.adaptive_avgpool(relation_temp)
            relation_temp = relation_temp.reshape(-1, (new_hw + 1), new_hw)

            relation_temp = torch.cat((relation_cls_token_temp, relation_temp), dim=2).transpose(1,2).reshape(B,-1,(new_hw+1),(new_hw+1))

            relations_to_be_reused_list = relation_temp.chunk(self.depth, 1)

        feature_list = []
        relation_list = []

        for blk_index in range(len(self.blocks)):

            if features_to_be_reused_list is not None:
                features_to_be_reused = features_to_be_reused_list[0]
            else:
                features_to_be_reused = None

            if relations_to_be_reused_list is not None:
                relations_to_be_reused = relations_to_be_reused_list[blk_index]
            else:
                relations_to_be_reused = None

            x, relation = self.blocks[blk_index](x, features_to_be_reused=features_to_be_reused, relations_to_be_reused=relations_to_be_reused)
            relation_list.append(relation)

        feature_list.append(x)

        x = self.norm(x)
        return x[:, 0], feature_list, relation_list

    def forward(self, x, features_to_be_reused_list=None, relations_to_be_reused_list=None):
        x, feature_list, relation_list = self.forward_features(x, features_to_be_reused_list, relations_to_be_reused_list)
        x = self.head(x)
        return x, feature_list, relation_list



class DVT_T2t_vit_14_model(nn.Module):
    def __init__(self, feature_reuse, relation_reuse, **kwargs):
        super().__init__()
        self.feature_reuse = feature_reuse
        self.relation_reuse = relation_reuse
        self.less_less_token = T2t_vit_14_less_less_token(feature_reuse=False,
                                                          relation_reuse=False,
                                                          **kwargs)

        self.less_token = T2t_vit_14_less_token(feature_reuse=feature_reuse,
                                                relation_reuse=relation_reuse,
                                                **kwargs)

        self.normal_token = T2t_vit_14(feature_reuse=feature_reuse,
                                       relation_reuse=relation_reuse,
                                       **kwargs)

    def forward(self, x):
        
        if self.feature_reuse == True and self.relation_reuse == True:
            less_less_token_output, features_to_be_reused_list, relations_to_be_reused_list = self.less_less_token(x, features_to_be_reused_list=None, relations_to_be_reused_list=None)
            less_token_output, features_to_be_reused_list, relations_to_be_reused_list = self.less_token(x, features_to_be_reused_list=features_to_be_reused_list, relations_to_be_reused_list=relations_to_be_reused_list)
            normal_output, _, _ = self.normal_token(x, features_to_be_reused_list=features_to_be_reused_list, relations_to_be_reused_list=relations_to_be_reused_list)
            
        elif self.feature_reuse == False and self.relation_reuse == True:
            less_less_token_output, features_to_be_reused_list, relations_to_be_reused_list = self.less_less_token(x, features_to_be_reused_list=None, relations_to_be_reused_list=None)
            less_token_output, features_to_be_reused_list, relations_to_be_reused_list = self.less_token(x, features_to_be_reused_list=None, relations_to_be_reused_list=relations_to_be_reused_list)
            normal_output, _, _ = self.normal_token(x, features_to_be_reused_list=None, relations_to_be_reused_list=relations_to_be_reused_list)
            
        elif self.feature_reuse == True and self.relation_reuse == False:
            less_less_token_output, features_to_be_reused_list, relations_to_be_reused_list = self.less_less_token(x, features_to_be_reused_list=None, relations_to_be_reused_list=None)
            less_token_output, features_to_be_reused_list, relations_to_be_reused_list = self.less_token(x, features_to_be_reused_list=features_to_be_reused_list, relations_to_be_reused_list=None)
            normal_output, _, _ = self.normal_token(x, features_to_be_reused_list=features_to_be_reused_list, relations_to_be_reused_list=None)
            
        else:
            less_less_token_output, features_to_be_reused_list, relations_to_be_reused_list = self.less_less_token(x, features_to_be_reused_list=None, relations_to_be_reused_list=None)
            less_token_output, features_to_be_reused_list, relations_to_be_reused_list = self.less_token(x, features_to_be_reused_list=None, relations_to_be_reused_list=None)
            normal_output, _, _ = self.normal_token(x, features_to_be_reused_list=None, relations_to_be_reused_list=None)
            
        return less_less_token_output, less_token_output, normal_output


class DVT_T2t_vit_12_model(nn.Module):
    def __init__(self, feature_reuse, relation_reuse, **kwargs):
        super().__init__()
        self.feature_reuse = feature_reuse
        self.relation_reuse = relation_reuse
        self.less_less_token = T2t_vit_12_less_less_token(feature_reuse=False,
                                                          relation_reuse=False,
                                                          **kwargs)

        self.less_token = T2t_vit_12_less_token(feature_reuse=feature_reuse,
                                                relation_reuse=relation_reuse,
                                                **kwargs)

        self.normal_token = T2t_vit_12(feature_reuse=feature_reuse,
                                       relation_reuse=relation_reuse,
                                       **kwargs)

    def forward(self, x):
        
        if self.feature_reuse == True and self.relation_reuse == True:
            less_less_token_output, features_to_be_reused_list, relations_to_be_reused_list = self.less_less_token(x, features_to_be_reused_list=None, relations_to_be_reused_list=None)
            less_token_output, features_to_be_reused_list, relations_to_be_reused_list = self.less_token(x, features_to_be_reused_list=features_to_be_reused_list, relations_to_be_reused_list=relations_to_be_reused_list)
            normal_output, _, _ = self.normal_token(x, features_to_be_reused_list=features_to_be_reused_list, relations_to_be_reused_list=relations_to_be_reused_list)
            
        elif self.feature_reuse == False and self.relation_reuse == True:
            less_less_token_output, features_to_be_reused_list, relations_to_be_reused_list = self.less_less_token(x, features_to_be_reused_list=None, relations_to_be_reused_list=None)
            less_token_output, features_to_be_reused_list, relations_to_be_reused_list = self.less_token(x, features_to_be_reused_list=None, relations_to_be_reused_list=relations_to_be_reused_list)
            normal_output, _, _ = self.normal_token(x, features_to_be_reused_list=None, relations_to_be_reused_list=relations_to_be_reused_list)
            
        elif self.feature_reuse == True and self.relation_reuse == False:
            less_less_token_output, features_to_be_reused_list, relations_to_be_reused_list = self.less_less_token(x, features_to_be_reused_list=None, relations_to_be_reused_list=None)
            less_token_output, features_to_be_reused_list, relations_to_be_reused_list = self.less_token(x, features_to_be_reused_list=features_to_be_reused_list, relations_to_be_reused_list=None)
            normal_output, _, _ = self.normal_token(x, features_to_be_reused_list=features_to_be_reused_list, relations_to_be_reused_list=None)
        
        else:
            less_less_token_output, features_to_be_reused_list, relations_to_be_reused_list = self.less_less_token(x, features_to_be_reused_list=None, relations_to_be_reused_list=None)
            less_token_output, features_to_be_reused_list, relations_to_be_reused_list = self.less_token(x, features_to_be_reused_list=None, relations_to_be_reused_list=None)
            normal_output, _, _ = self.normal_token(x, features_to_be_reused_list=None, relations_to_be_reused_list=None)
            
        return less_less_token_output, less_token_output, normal_output



@register_model
def DVT_T2t_vit_14(**kwargs):
    return DVT_T2t_vit_14_model(feature_reuse=True, relation_reuse=True, **kwargs)

@register_model
def DVT_T2t_vit_12(**kwargs):
    return DVT_T2t_vit_12_model(feature_reuse=True, relation_reuse=True, **kwargs)



@register_model
def T2t_vit_14(pretrained=False, **kwargs):  # adopt performer for tokens to token
    if pretrained:
        kwargs.setdefault('qk_scale', 384 ** -0.5)
    model = T2T_ViT(tokens_type='performer', embed_dim=384, depth=14, num_heads=6, mlp_ratio=3., **kwargs)
    model.default_cfg = default_cfgs['T2t_vit_14']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model


@register_model
def T2t_vit_14_less_token(pretrained=False, **kwargs):  # adopt performer for tokens to token
    if pretrained:
        kwargs.setdefault('qk_scale', 384 ** -0.5)
    model = T2T_ViT(tokens_type='performer_less_token', embed_dim=384, depth=14, num_heads=6, mlp_ratio=3., **kwargs)
    model.default_cfg = default_cfgs['T2t_vit_14']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model


@register_model
def T2t_vit_14_less_less_token(pretrained=False, **kwargs):  # adopt performer for tokens to token
    if pretrained:
        kwargs.setdefault('qk_scale', 384 ** -0.5)
    model = T2T_ViT(tokens_type='performer_less_less_token', embed_dim=384, depth=14, num_heads=6, mlp_ratio=3., **kwargs)
    model.default_cfg = default_cfgs['T2t_vit_14']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model


@register_model
def T2t_vit_12(pretrained=False, **kwargs):  # adopt performer for tokens to token
    if pretrained:
        kwargs.setdefault('qk_scale', 256 ** -0.5)
    model = T2T_ViT(tokens_type='performer', embed_dim=256, depth=12, num_heads=4, mlp_ratio=2., **kwargs)
    model.default_cfg = default_cfgs['T2t_vit_12']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model


@register_model
def T2t_vit_12_less_token(pretrained=False, **kwargs):  # adopt performer for tokens to token
    if pretrained:
        kwargs.setdefault('qk_scale', 256 ** -0.5)
    model = T2T_ViT(tokens_type='performer_less_token', embed_dim=256, depth=12, num_heads=4, mlp_ratio=2., **kwargs)
    model.default_cfg = default_cfgs['T2t_vit_12']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model


@register_model
def T2t_vit_12_less_less_token(pretrained=False, **kwargs):  # adopt performer for tokens to token
    if pretrained:
        kwargs.setdefault('qk_scale', 256 ** -0.5)
    model = T2T_ViT(tokens_type='performer_less_less_token', embed_dim=256, depth=12, num_heads=4, mlp_ratio=2., **kwargs)
    model.default_cfg = default_cfgs['T2t_vit_12']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model




@register_model
def T2t_vit_7(pretrained=False, **kwargs):  # adopt performer for tokens to token
    if pretrained:
        kwargs.setdefault('qk_scale', 256 ** -0.5)
    model = T2T_ViT(tokens_type='performer', embed_dim=256, depth=7, num_heads=4, mlp_ratio=2., **kwargs)
    model.default_cfg = default_cfgs['T2t_vit_7']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model


@register_model
def T2t_vit_10(pretrained=False, **kwargs):  # adopt performer for tokens to token
    if pretrained:
        kwargs.setdefault('qk_scale', 256 ** -0.5)
    model = T2T_ViT(tokens_type='performer', embed_dim=256, depth=10, num_heads=4, mlp_ratio=2., **kwargs)
    model.default_cfg = default_cfgs['T2t_vit_10']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model


@register_model
def T2t_vit_19(pretrained=False, **kwargs):  # adopt performer for tokens to token
    if pretrained:
        kwargs.setdefault('qk_scale', 448 ** -0.5)
    model = T2T_ViT(tokens_type='performer', embed_dim=448, depth=19, num_heads=7, mlp_ratio=3., **kwargs)
    model.default_cfg = default_cfgs['T2t_vit_19']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model


@register_model
def T2t_vit_24(pretrained=False, **kwargs):  # adopt performer for tokens to token
    if pretrained:
        kwargs.setdefault('qk_scale', 512 ** -0.5)
    model = T2T_ViT(tokens_type='performer', embed_dim=512, depth=24, num_heads=8, mlp_ratio=3., **kwargs)
    model.default_cfg = default_cfgs['T2t_vit_24']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model



@register_model
def T2t_vit_t_14(pretrained=False, **kwargs):  # adopt transformers for tokens to token
    if pretrained:
        kwargs.setdefault('qk_scale', 384 ** -0.5)
    model = T2T_ViT(tokens_type='transformer', embed_dim=384, depth=14, num_heads=6, mlp_ratio=3., **kwargs)
    model.default_cfg = default_cfgs['T2t_vit_t_14']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model


@register_model
def T2t_vit_t_19(pretrained=False, **kwargs):  # adopt transformers for tokens to token
    if pretrained:
        kwargs.setdefault('qk_scale', 448 ** -0.5)
    model = T2T_ViT(tokens_type='transformer', embed_dim=448, depth=19, num_heads=7, mlp_ratio=3., **kwargs)
    model.default_cfg = default_cfgs['T2t_vit_t_19']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model


@register_model
def T2t_vit_t_24(pretrained=False, **kwargs):  # adopt transformers for tokens to token
    if pretrained:
        kwargs.setdefault('qk_scale', 512 ** -0.5)
    model = T2T_ViT(tokens_type='transformer', embed_dim=512, depth=24, num_heads=8, mlp_ratio=3., **kwargs)
    model.default_cfg = default_cfgs['T2t_vit_t_24']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model

# rexnext and wide structure
@register_model
def T2t_vit_14_resnext(pretrained=False, **kwargs):
    if pretrained:
        kwargs.setdefault('qk_scale', 384 ** -0.5)
    model = T2T_ViT(tokens_type='performer', embed_dim=384, depth=14, num_heads=32, mlp_ratio=3., **kwargs)
    model.default_cfg = default_cfgs['T2t_vit_14_resnext']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model


@register_model
def T2t_vit_14_wide(pretrained=False, **kwargs):
    if pretrained:
        kwargs.setdefault('qk_scale', 512 ** -0.5)
    model = T2T_ViT(tokens_type='performer', embed_dim=768, depth=4, num_heads=12, mlp_ratio=3., **kwargs)
    model.default_cfg = default_cfgs['T2t_vit_14_wide']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model

