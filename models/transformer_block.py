# Copyright (c) [2012]-[2021] Shanghai Yitu Technology Co., Ltd.
#
# This source code is licensed under the Clear BSD License
# LICENSE file in the root directory of this file
# All rights reserved.
"""
Borrow from timm(https://github.com/rwightman/pytorch-image-models)
"""
import torch
import torch.nn as nn
import numpy as np
from timm.models.layers import DropPath
from torch import Tensor
import pdb

class GELU(nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        return torch.nn.functional.gelu(input)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

            
    def forward(self, x, relations_to_be_reused):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if relations_to_be_reused is not None:

            attn = attn + relations_to_be_reused

        relation = attn
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x, relation

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=GELU, norm_layer=nn.LayerNorm,
                 feature_reuse=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        
        self.feature_reuse = feature_reuse

        if self.feature_reuse:
            self.norm2 = norm_layer(dim + 48)
            self.mlp = Mlp(in_features=dim + 48, hidden_features=mlp_hidden_dim, out_features=dim,
                           act_layer=act_layer, drop=drop)
        else:
            self.norm2 = norm_layer(dim)
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim,
                           act_layer=act_layer, drop=drop)

        if self.feature_reuse:
            self.feature_resuse_mlp = nn.Sequential(
                norm_layer(dim),
                Mlp(in_features=dim, hidden_features=128, out_features=48, act_layer=act_layer, drop=drop)
            )


    def forward(self, x, features_to_be_reused=None, relations_to_be_reused=None):

        identity = x
        x, relation = self.attn(self.norm1(x),relations_to_be_reused)
        x = identity + self.drop_path(x)

        identity = x

        if features_to_be_reused is not None:

            features_to_be_reused = self.feature_resuse_mlp(features_to_be_reused)
            feature_temp = features_to_be_reused[:, 1:, :]
            B, new_HW, C = feature_temp.shape
            feature_temp = feature_temp.transpose(1, 2).reshape(B, C, int(np.sqrt(new_HW)), int(np.sqrt(new_HW)))

            # an efficient upsampling implementation
            if (identity.size(1) - 1) == 100:
                feature_temp = torch.nn.functional.interpolate(feature_temp, (14, 14), mode='nearest')
                feature_temp = torch.nn.functional.adaptive_avg_pool2d(feature_temp, (10, 10)).view(B, C, identity.size(1) - 1).transpose(1, 2)
            else:
                feature_temp = torch.nn.functional.interpolate(feature_temp, (20, 20), mode='nearest')
                feature_temp = torch.nn.functional.adaptive_avg_pool2d(feature_temp, (14, 14)).view(B, C, identity.size(1) - 1).transpose(1, 2)

            feature_temp = torch.cat((torch.zeros(B, 1, 48).cuda(), feature_temp), dim=1)
            x = torch.cat((x, feature_temp), dim=2)

        x = identity + self.drop_path(self.mlp(self.norm2(x)))
        return x, relation


def get_sinusoid_encoding(n_position, d_hid):
    ''' Sinusoid position encoding table '''

    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)
