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


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
from ppdet.core.workspace import register, create
from .meta_arch import BaseArch

__all__ = ['SSD']


@register
class SSD(BaseArch):
    """
    Single Shot MultiBox Detector, see https://arxiv.org/abs/1512.02325

    Args:
        backbone (nn.Layer): backbone instance
        ssd_head (nn.Layer): `SSDHead` instance
        post_process (object): `BBoxPostProcess` instance
    """

    __category__ = 'architecture'
    __inject__ = ['post_process']

    def __init__(self, backbone, ssd_head, post_process, neck=None, r34_backbone=False):
        super(SSD, self).__init__()
        self.backbone = backbone
        self.ssd_head = ssd_head
        self.post_process = post_process
        self.neck = neck
        self.r34_backbone = r34_backbone
        # params_info = paddle.summary(self.backbone, (1, 3, 320, 320))
        # print(params_info)
        if self.r34_backbone:
            from ppdet.modeling.backbones.resnet import ResNet
            assert isinstance(self.backbone, ResNet) and \
                   self.backbone.depth == 34, \
                "If you set r34_backbone=True, please use ResNet-34 as backbone."
            self.backbone.res_layers[2].blocks[0].branch2a.conv._stride = [1, 1]
            self.backbone.res_layers[2].blocks[0].short.conv._stride = [1, 1]

    @classmethod
    def from_config(cls, cfg, *args, **kwargs):
        # backbone
        backbone = create(cfg['backbone'])

        # neck
        kwargs = {'input_shape': backbone.out_shape}
        neck = create(cfg['neck'], **kwargs)

        # head
        kwargs = {'input_shape': neck.out_shape}
        # kwargs = {'input_shape': backbone.out_shape}
        ssd_head = create(cfg['ssd_head'], **kwargs)

        return {
            'backbone': backbone,
            # 'neck': neck,
            "ssd_head": ssd_head,
        }

    def _forward(self):
        # Backbone
        body_feats = self.backbone(self.inputs)

        # if self.neck is not None:
        body_feats = self.neck(body_feats)

        # SSD Head
        if self.training:
            return self.ssd_head(body_feats, self.inputs['image'],
                                 self.inputs['gt_bbox'],
                                 self.inputs['gt_class'])
        else:
            preds, anchors = self.ssd_head(body_feats, self.inputs['image'])
            bbox, bbox_num = self.post_process(preds, anchors,
                                               self.inputs['im_shape'],
                                               self.inputs['scale_factor'])
            return bbox, bbox_num

    def get_loss(self, ):
        return {"loss": self._forward()}

    def get_pred(self):
        bbox_pred, bbox_num = self._forward()
        output = {
            "bbox": bbox_pred,
            "bbox_num": bbox_num,
        }
        return output
