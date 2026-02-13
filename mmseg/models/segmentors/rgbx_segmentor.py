from typing import List, Optional

from mmseg.registry import MODELS
from mmseg.models.segmentors import EncoderDecoder

import torch.nn as nn
from torch import Tensor


@MODELS.register_module()
class EarlyFusionSegmentor(EncoderDecoder):

    def extract_feat(self, inputs: Tensor) -> List[Tensor]:
        img = inputs[:, :3]
        img2 = inputs[:, 3:]
        x = self.backbone(img, img2)
        return x
