import warnings
from collections import OrderedDict

from torch import nn
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelMaxPool

class IntermediateLayerGetter(nn.ModuleDict):
   
    def __init__(self, model, return_layers):
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")
        orig_return_layers = return_layers
        return_layers = {str(k): str(v) for k, v in return_layers.items()}
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super().__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x):
        out = OrderedDict()
        for name, module in self.items():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out


class BackboneWithFPN(nn.Module):

    def __init__(
        self,
        backbone,
        return_layers,
        in_channels_list,
        out_channels,
        extra_blocks=None,
        norm_layer=None,
    ):
        super().__init__()

        if extra_blocks is None:
            extra_blocks = LastLevelMaxPool()

        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            extra_blocks=extra_blocks,
            norm_layer=norm_layer,
        )
        self.out_channels = out_channels

    def forward(self, x):
        x = self.body(x)
        x = self.fpn(x)
        return x


def _validate_trainable_layers(
    is_trained,
    trainable_backbone_layers,
    max_value,
    default_value,
):
    # don't freeze any layers if pretrained model or backbone is not used
    if not is_trained:
        if trainable_backbone_layers is not None:
            warnings.warn(
                "Changing trainable_backbone_layers has no effect if "
                "neither pretrained nor pretrained_backbone have been set to True, "
                f"falling back to trainable_backbone_layers={max_value} so that all layers are trainable"
            )
        trainable_backbone_layers = max_value

    # by default freeze first blocks
    if trainable_backbone_layers is None:
        trainable_backbone_layers = default_value
    if trainable_backbone_layers < 0 or trainable_backbone_layers > max_value:
        raise ValueError(
            f"Trainable backbone layers should be in the range [0,{max_value}], got {trainable_backbone_layers} "
        )
    return trainable_backbone_layers



def _mobilenet_extractor(
    backbone,
    fpn,
    trainable_layers,
    returned_layers=None,
    extra_blocks=None,
    norm_layer=None,
):
    backbone = backbone.features
    # Gather the indices of blocks which are strided. These are the locations of C1, ..., Cn-1 blocks.
    # The first and last blocks are always included because they are the C0 (conv1) and Cn.
    stage_indices = [0] + [i for i, b in enumerate(backbone) if getattr(b, "_is_cn", False)] + [len(backbone) - 1]
    num_stages = len(stage_indices)

    # find the index of the layer from which we won't freeze
    if trainable_layers < 0 or trainable_layers > num_stages:
        raise ValueError(f"Trainable layers should be in the range [0,{num_stages}], got {trainable_layers} ")
    freeze_before = len(backbone) if trainable_layers == 0 else stage_indices[num_stages - trainable_layers]

    for b in backbone[:freeze_before]:
        for parameter in b.parameters():
            parameter.requires_grad_(False)

    out_channels = 256
    if fpn:
        if extra_blocks is None:
            extra_blocks = LastLevelMaxPool()

        if returned_layers is None:
            returned_layers = [num_stages - 2, num_stages - 1]
        if min(returned_layers) < 0 or max(returned_layers) >= num_stages:
            raise ValueError(f"Each returned layer should be in the range [0,{num_stages - 1}], got {returned_layers} ")
        return_layers = {f"{stage_indices[k]}": str(v) for v, k in enumerate(returned_layers)}

        in_channels_list = [backbone[stage_indices[i]].out_channels for i in returned_layers]
        return BackboneWithFPN(
            backbone, return_layers, in_channels_list, out_channels, extra_blocks=extra_blocks, norm_layer=norm_layer
        )
    else:
        m = nn.Sequential(
            backbone,
            nn.Conv2d(backbone[-1].out_channels, out_channels, 1),
        )
        m.out_channels = out_channels
        return m