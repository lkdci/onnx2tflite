from typing import Tuple

import torch
from torch import Tensor
import copy

from super_gradients.training.models.detection_models.yolo_nas.yolo_nas_variants import YoloNAS
from super_gradients.common.registry import register_detection_module
from super_gradients.training.models.detection_models.yolo_nas.dfl_heads import NDFLHeads
from super_gradients.training.utils.bbox_utils import batch_distance2bbox
from super_gradients.training.models.detection_models.pp_yolo_e.pp_yolo_head import generate_anchors_for_grid_cell
from super_gradients.common.registry import register_model
from super_gradients.training.models import get_arch_params
from super_gradients.training.utils import HpmStruct, get_param


@register_model()
class YoloNAS_S_TFLite(YoloNAS):
    def __init__(self, arch_params):
        default_arch_params = get_arch_params("yolo_nas_s_arch_params")
        default_arch_params["heads"]["NDFLHeadsTFlite"] = default_arch_params["heads"].pop("NDFLHeads")
        merged_arch_params = HpmStruct(**copy.deepcopy(default_arch_params))
        merged_arch_params.override(**arch_params.to_dict())
        super().__init__(
            backbone=merged_arch_params.backbone,
            neck=merged_arch_params.neck,
            heads=merged_arch_params.heads,
            num_classes=get_param(merged_arch_params, "num_classes", None),
            in_channels=get_param(merged_arch_params, "in_channels", 3),
            bn_momentum=get_param(merged_arch_params, "bn_momentum", None),
            bn_eps=get_param(merged_arch_params, "bn_eps", None),
            inplace_act=get_param(merged_arch_params, "inplace_act", None),
        )


@register_detection_module()
class NDFLHeadsTFlite(NDFLHeads):
    def forward_eval(self, feats: Tuple[Tensor, ...]) -> Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, ...]]:
        cls_score_list, reg_distri_list, reg_dist_reduced_list = [], [], []

        for i, feat in enumerate(feats):
            b, _, h, w = feat.shape
            height_mul_width = h * w
            reg_distri, cls_logit = getattr(self, f"head{i + 1}")(feat)
            reg_distri_list.append(torch.permute(reg_distri.flatten(2), [0, 2, 1]))

            reg_dist_reduced = torch.permute(reg_distri.reshape([-1, 4, self.reg_max + 1, height_mul_width]), [0, 2, 3, 1])
            reg_dist_reduced = torch.nn.functional.conv2d(torch.nn.functional.softmax(reg_dist_reduced, dim=1), weight=self.proj_conv)#.squeeze(1)

            # cls and reg
            cls_score_list.append(cls_logit.reshape([b, self.num_classes, height_mul_width]))
            reg_dist_reduced_list.append(reg_dist_reduced)

        cls_score_list = torch.cat(cls_score_list, dim=-1)  # [B, C, Anchors]
        # cls_score_list = torch.permute(cls_score_list, [0, 2, 1])  # # [B, Anchors, C]

        reg_distri_list = torch.cat(reg_distri_list, dim=1)  # [B, Anchors, 4 * (self.reg_max + 1)]
        reg_dist_reduced_list = torch.cat(reg_dist_reduced_list, dim=2)  # [B, 1, Anchors, 4]

        # Decode bboxes
        # Note in eval mode, anchor_points_inference is different from anchor_points computed on train
        if self.eval_size:
            anchor_points_inference, stride_tensor = self.anchor_points, self.stride_tensor
        else:
            anchor_points_inference, stride_tensor = self._generate_anchors(feats)

        pred_scores = cls_score_list.sigmoid()
        pred_bboxes = batch_distance2bbox(anchor_points_inference, reg_dist_reduced_list) * stride_tensor  # [B, Anchors, 4]

        decoded_predictions = pred_bboxes, pred_scores

        if torch.jit.is_tracing():
            return decoded_predictions

        anchors, anchor_points, num_anchors_list, _ = generate_anchors_for_grid_cell(feats, self.fpn_strides, self.grid_cell_scale, self.grid_cell_offset)

        raw_predictions = cls_score_list, reg_distri_list, anchors, anchor_points, num_anchors_list, stride_tensor
        return decoded_predictions, raw_predictions

    def _generate_anchors(self, feats=None, dtype=None, device=None):
        anchor_points, stride_tensor = super()._generate_anchors(feats, dtype, device)
        return anchor_points.unsqueeze(0).unsqueeze(0), stride_tensor.unsqueeze(0).unsqueeze(0)
