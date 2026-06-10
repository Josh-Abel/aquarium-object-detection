
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import MultiScaleRoIAlign


from .anchor_utils import AnchorGenerator
from .backbone_utils import _mobilenet_extractor, _validate_trainable_layers
from .rpn import RegionProposalNetwork, RPNHead
from .roi_heads import RoIHeads

import torchvision.models as models 


class FasterRCNN(nn.Module):

    def __init__(self, backbone, num_classes=None, rpn_anchor_generator=None, \
                 rpn_head=None, rpn_pre_nms_top_n_train=2000, rpn_pre_nms_top_n_test=1000, rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=500, \
                 rpn_nms_thresh=0.7, rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3, rpn_batch_size_per_image=256, rpn_positive_fraction=0.5, rpn_score_thresh=0.05, \
                 box_roi_pool=None, box_head=None, box_predictor=None, box_score_thresh=0.05, box_nms_thresh=0.5, box_detections_per_img=100, box_fg_iou_thresh=0.5, \
                 box_bg_iou_thresh=0.5, box_batch_size_per_image=512, box_positive_fraction=0.25, bbox_reg_weights=None):
        super().__init__()


        out_channels = backbone.out_channels

        anchor_sizes = ((32, 64, 128, 256, 512,),) * 3
    
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
        rpn_anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)

        rpn_head = RPNHead(out_channels, rpn_anchor_generator.num_anchors_per_location()[0])

        rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
        rpn_post_nms_top_n = dict(training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)

        rpn = RegionProposalNetwork(
            rpn_anchor_generator,
            rpn_head,
            rpn_fg_iou_thresh,
            rpn_bg_iou_thresh,
            rpn_batch_size_per_image,
            rpn_positive_fraction,
            rpn_pre_nms_top_n,
            rpn_post_nms_top_n,
            rpn_nms_thresh,
            score_thresh=rpn_score_thresh,
        )

        if box_roi_pool is None:
            box_roi_pool = MultiScaleRoIAlign(featmap_names=["0", "1", "2", "3"], output_size=7, sampling_ratio=2)

        if box_head is None:
            resolution = box_roi_pool.output_size[0]
            representation_size = 1024
            box_head = TwoMLPHead(out_channels * resolution**2, representation_size)

        if box_predictor is None:
            representation_size = 1024
            box_predictor = FastRCNNPredictor(representation_size, num_classes)

        roi_heads = RoIHeads(
            # Box
            box_roi_pool,
            box_head,
            box_predictor,
            box_fg_iou_thresh,
            box_bg_iou_thresh,
            box_batch_size_per_image,
            box_positive_fraction,
            bbox_reg_weights,
            box_score_thresh,
            box_nms_thresh,
            box_detections_per_img,
        )

        self.backbone = backbone
        self.rpn = rpn
        self.roi_heads = roi_heads

    def forward(self, images, targets=None):

        original_image_sizes = []
        for img in images:
            val = img.shape[-2:]

            original_image_sizes.append((val[0], val[1]))

        images = torch.stack(images)
        features = self.backbone(images)
               
        proposals, proposal_losses = self.rpn(images, features, targets)
        detections, detector_losses = self.roi_heads(features, proposals, images, targets)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        
        if self.training:
            return losses
        
        return detections, {**detector_losses,**proposal_losses}


class TwoMLPHead(nn.Module):

    def __init__(self, in_channels, representation_size):
        super().__init__()

        self.fc6 = nn.Linear(in_channels, representation_size)
        self.fc7 = nn.Linear(representation_size, representation_size)

    def forward(self, x):
        x = x.flatten(start_dim=1)

        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))

        return x


class FastRCNNPredictor(nn.Module):
  
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.cls_score = nn.Linear(in_channels, num_classes)
        self.bbox_pred = nn.Linear(in_channels, num_classes * 4)

    def forward(self, x):
        x = x.flatten(start_dim=1)
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)

        return scores, bbox_deltas






def fasterrcnn_mobilenet_v3(num_classes, trainable_backbone_layers=None):

    
    trainable_backbone_layers = _validate_trainable_layers(True, trainable_backbone_layers, 6, 3)
    
    backbone = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
    backbone = _mobilenet_extractor(backbone, True, trainable_backbone_layers)
    
    model = FasterRCNN(
        backbone, num_classes, 
    )


    return model
