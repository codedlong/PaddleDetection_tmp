# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr
from paddle.nn.initializer import XavierUniform

from ppdet.core.workspace import register, serializable
from ppdet.modeling.layers import ConvNormLayer
from ..shape_spec import ShapeSpec

__all__ = ['MyFPN']


@register
@serializable
class MyFPN(nn.Layer):
    """
    BackwardFusion: backward fusion networks

    Args:
        in_channels (list[int]): input channels of each level which can be
            derived from the output shape of backbone by from_config
        out_channel (list[int]): output channel of each level
        spatial_scales (list[float]): the spatial scales between input feature
            maps and original input image which can be derived from the output
            shape of backbone by from_config
        has_extra_convs (bool): whether to add extra conv to the last level.
            default False
        extra_stage (int): the number of extra stages added to the last level.
            default 1
        use_c5 (bool): Whether to use c5 as the input of extra stage,
            otherwise p5 is used. default True
        norm_type (string|None): The normalization type in FPN module. If
            norm_type is None, norm will not be used after conv and if
            norm_type is string, bn, gn, sync_bn are available. default None
        norm_decay (float): weight decay for normalization layer weights.
            default 0.
        freeze_norm (bool): whether to freeze normalization layer.
            default False
        relu_before_extra_convs (bool): whether to add relu before extra convs.
            default False

    """

    def __init__(self,
                 in_channels=[672, 960, 512, 256, 256, 128],
                 out_channel=256,
                 spatial_scales=[0.25, 0.125, 0.0625, 0.03125],
                 norm_type=None,
                 norm_decay=0.,
                 extra_stage=1,
                 freeze_norm=False):
        super(MyFPN, self).__init__()
        for s in range(extra_stage):
            spatial_scales = spatial_scales + [spatial_scales[-1] / 2.]
        self.in_channels = in_channels
        self.spatial_scales = spatial_scales
        self.out_channel = out_channel
        self.norm_type = norm_type
        self.norm_decay = norm_decay
        self.freeze_norm = freeze_norm

        self.lateral_convs = []
        self.fpn_convs = []
        fan = self.out_channel * 3 * 3
        fpn_out_channel = 256

        for i in range(self.in_channels):
            lateral_name = 'fpn_inner_' + str(i) + '_lateral'
            if self.norm_type is not None:
                lateral = self.add_sublayer(
                    lateral_name,
                    ConvNormLayer(
                        ch_in=self.in_channels[i],
                        ch_out=self.out_channel,
                        filter_size=1,
                        stride=1,
                        norm_type=self.norm_type,
                        norm_decay=self.norm_decay,
                        freeze_norm=self.freeze_norm,
                        initializer=XavierUniform(fan_out=self.in_channels[i])))
            self.lateral_convs.append(lateral)

            fpn_name = 'fpn_sum_' + str(i)
            if self.norm_type is not None:
                fpn_conv = self.add_sublayer(
                    fpn_name,
                    ConvNormLayer(
                        ch_in=fpn_out_channel,
                        ch_out=fpn_out_channel,
                        filter_size=3,
                        stride=1,
                        norm_type=self.norm_type,
                        norm_decay=self.norm_decay,
                        freeze_norm=self.freeze_norm,
                        initializer=XavierUniform(fan_out=fan)))
            self.fpn_convs.append(fpn_conv)

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {
            'in_channels': [i.channels for i in input_shape],
            'spatial_scales': [1.0 / i.stride for i in input_shape],
        }

    def forward(self, body_feats):
        laterals = []
        num_levels = len(body_feats) # len(body_feats) = 6
        for i in range(num_levels):
            laterals.append(self.lateral_convs[i](body_feats[i]))

        # upside + element add
        for i in range(1, num_levels):
            lvl = num_levels - i
            upsample = F.interpolate(
                laterals[lvl],
                scale_factor=2.,
                mode='nearest', )
            laterals[lvl - 1] += upsample

        # 特征融合之后通过3×3/256卷积，防止混叠效应
        fpn_output = []
        for lvl in range(num_levels): # num_levels = 6
            fpn_output.append(self.fpn_convs[lvl](laterals[lvl]))

        return fpn_output

    @property
    def out_shape(self):
        return [
            ShapeSpec(
                channels=self.out_channel, stride=1. / s)
            for s in self.spatial_scales
        ]
