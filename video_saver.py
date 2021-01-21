import math
import pickle
from pathlib import Path

import numpy as np
import pycocotools.mask as mask_util
from detectron2.utils.visualizer import _PanopticPrediction


class _DetectedInstance:
    __slots__ = ["label", "bbox", "mask_rle", "idd", "ttl"]

    def __init__(self, label, bbox, mask_rle, ttl, idd=None):
        self.label = label
        self.bbox = bbox
        self.mask_rle = mask_rle
        self.ttl = ttl
        self.idd = idd


class VideoSaver:
    def __init__(self, metadata):
        self.metadata = metadata
        self._old_instances = []

    def mask2bbox(self, mask):
        a = np.where(mask != 0)
        bbox = (np.min(a[1]), np.min(a[0]), np.max(a[1]), np.max(a[0]))
        return bbox

    def draw_panoptic_seg_predictions(
        self, frame_shape, panoptic_seg, segments_info, save_path
    ):
        pred = _PanopticPrediction(panoptic_seg, segments_info, self.metadata)

        # save semantic mask
        semantic_mask = np.zeros(frame_shape)
        for mask, sinfo in pred.semantic_masks():
            semantic_mask[mask] = sinfo["category_id"]
            # print(sinfo)
        np.save(f"{save_path}_sem_seg.npy", semantic_mask)

        # save instances
        all_instances = list(pred.instance_masks())
        if len(all_instances) == 0:
            return
        masks, sinfo = list(zip(*all_instances))
        masks_rles = mask_util.encode(
            np.asarray(np.asarray(masks).transpose(1, 2, 0), dtype=np.uint8, order="F")
        )

        category_ids = [x["category_id"] for x in sinfo]
        detected = []
        for cat, rle in zip(category_ids, masks_rles):
            detected.append(_DetectedInstance(cat, bbox=None, mask_rle=rle, ttl=8))
        idds = self._assign_idd(detected)
        pickle.dump(idds, open(f"{save_path}_ids.p", "wb"))

        labels = [self.metadata.thing_classes[k] for k in category_ids]
        pickle.dump(labels, open(f"{save_path}_labels.p", "wb"))

        boxes = [self.mask2bbox(m) for m in masks]
        np.save(f"{save_path}_boxes.npy", boxes)

        for k, (mask, bb) in enumerate(zip(masks, boxes)):
            crop = mask[
                math.floor(bb[1]) : math.ceil(bb[3]),
                math.floor(bb[0]) : math.ceil(bb[2]),
            ].copy()
            np.save(f"{save_path}_mask{k}.npy", crop)

    def _assign_idd(self, instances):
        """
        Naive tracking heuristics to assign same idd to the same instance,
        will update the internal state of tracked instances.
        Returns:
            list[tuple[float]]: list of colors.
        """

        # Compute iou with either boxes or masks:
        is_crowd = np.zeros((len(instances),), dtype=np.bool)
        if instances[0].bbox is None:
            assert instances[0].mask_rle is not None
            # use mask iou only when box iou is None
            # because box seems good enough
            rles_old = [x.mask_rle for x in self._old_instances]
            rles_new = [x.mask_rle for x in instances]
            ious = mask_util.iou(rles_old, rles_new, is_crowd)
            threshold = 0.5
        else:
            boxes_old = [x.bbox for x in self._old_instances]
            boxes_new = [x.bbox for x in instances]
            ious = mask_util.iou(boxes_old, boxes_new, is_crowd)
            threshold = 0.6
        if len(ious) == 0:
            ious = np.zeros((len(self._old_instances), len(instances)), dtype="float32")

        # Only allow matching instances of the same label:
        for old_idx, old in enumerate(self._old_instances):
            for new_idx, new in enumerate(instances):
                if old.label != new.label:
                    ious[old_idx, new_idx] = 0

        matched_new_per_old = np.asarray(ious).argmax(axis=1)
        max_iou_per_old = np.asarray(ious).max(axis=1)

        # Try to find match for each old instance:
        extra_instances = []
        for idx, inst in enumerate(self._old_instances):
            if max_iou_per_old[idx] > threshold:
                newidx = matched_new_per_old[idx]
                if instances[newidx].idd is None:
                    instances[newidx].idd = inst.idd
                    continue
            # If an old instance does not match any new instances,
            # keep it for the next frame in case it is just missed by the detector
            inst.ttl -= 1
            if inst.ttl > 0:
                extra_instances.append(inst)

        # Assign new idd to newly-detected instances:
        idd_offset = max([-1] + [d.idd for d in instances if d.idd is not None])
        for inst in instances:
            if inst.idd is None:
                inst.idd = idd_offset + 1
                idd_offset += 1
        self._old_instances = instances[:] + extra_instances
        return [inst.idd for inst in instances]
