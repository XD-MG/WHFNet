from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import mmcv
from mmcv.transforms import BaseTransform
from mmcv.transforms import LoadImageFromFile as MMCV_LoadImageFromFile
from mmcv.transforms import Resize as MMCV_Resize
from mmcv.transforms import RandomResize as MMCV_RandomResize
from mmcv.transforms import RandomFlip as MMCV_RandomFlip
from mmseg.datasets.transforms import RandomCrop
import mmengine.fileio as fileio
from mmseg.registry import TRANSFORMS


@TRANSFORMS.register_module()
class MultiRandomCrop(RandomCrop):

    def transform(self, results: dict) -> dict:
        img = results['img']
        img2 = results['img2']
        crop_bbox = self.crop_bbox(results)

        img = self.crop(img, crop_bbox)
        img2 = self.crop(img2, crop_bbox)

        # crop semantic seg
        for key in results.get('seg_fields', []):
            results[key] = self.crop(results[key], crop_bbox)

        results['img'] = img
        results['img2'] = img2
        results['img_shape'] = img.shape[:2]
        return results


@TRANSFORMS.register_module()
class MultiRandomResize(MMCV_RandomResize):

    def __init__(self, **kwargs) -> None:
        super().__init__(resize_type='MultiResize', **kwargs)

    def transform(self, results: dict) -> dict:
        results['scale'] = self._random_scale()
        self.resize.scale = results['scale']
        results = self.resize(results)
        return results


@TRANSFORMS.register_module()
class MultiResize(MMCV_Resize):

    def _resize_img(self, results: dict) -> None:
        if results.get('img', None) is not None:
            if self.keep_ratio:
                img, scale_factor = mmcv.imrescale(
                    results['img'],
                    results['scale'],
                    interpolation=self.interpolation,
                    return_scale=True,
                    backend=self.backend)
                img2, _ = mmcv.imrescale(results['img2'],
                                         results['scale'],
                                         interpolation=self.interpolation,
                                         return_scale=True,
                                         backend=self.backend)
                # the w_scale and h_scale has minor difference
                # a real fix should be done in the mmcv.imrescale in the future
                new_h, new_w = img.shape[:2]
                h, w = results['img'].shape[:2]
                w_scale = new_w / w
                h_scale = new_h / h
            else:
                img, w_scale, h_scale = mmcv.imresize(
                    results['img'],
                    results['scale'],
                    interpolation=self.interpolation,
                    return_scale=True,
                    backend=self.backend)
                img2, w_scale, h_scale = mmcv.imresize(
                    results['img2'],
                    results['scale'],
                    interpolation=self.interpolation,
                    return_scale=True,
                    backend=self.backend)

            results['img'] = img
            results['img2'] = img2
            results['img_shape'] = img.shape[:2]
            results['scale_factor'] = (w_scale, h_scale)
            results['keep_ratio'] = self.keep_ratio


@TRANSFORMS.register_module()
class MultiRandomFlip(MMCV_RandomFlip):

    def _flip(self, results: dict) -> None:
        """Flip images, bounding boxes, semantic segmentation map and
        keypoints."""
        # flip image
        results['img'] = mmcv.imflip(results['img'],
                                     direction=results['flip_direction'])
        results['img2'] = mmcv.imflip(results['img2'],
                                      direction=results['flip_direction'])

        img_shape = results['img'].shape[:2]

        # flip bboxes
        if results.get('gt_bboxes', None) is not None:
            results['gt_bboxes'] = self._flip_bbox(results['gt_bboxes'],
                                                   img_shape,
                                                   results['flip_direction'])

        # flip keypoints
        if results.get('gt_keypoints', None) is not None:
            results['gt_keypoints'] = self._flip_keypoints(
                results['gt_keypoints'], img_shape, results['flip_direction'])

        # flip seg map
        if results.get('gt_seg_map', None) is not None:
            results['gt_seg_map'] = self._flip_seg_map(
                results['gt_seg_map'], direction=results['flip_direction'])
            results['swap_seg_labels'] = self.swap_seg_labels
