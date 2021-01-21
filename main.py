import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import tqdm
from detectron2.config import get_cfg, set_global_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.projects.panoptic_deeplab import add_panoptic_deeplab_config
from video_saver import VideoSaver
from detectron2.engine import default_argument_parser, default_setup


def setup(args):
    cfg = get_cfg()
    add_panoptic_deeplab_config(cfg)
    cfg.merge_from_file(
        "detectron2/projects/Panoptic-DeepLab/configs/Cityscapes-PanopticSegmentation/panoptic_deeplab_R_52_os16_mg124_poly_90k_bs32_crop_512_1024.yaml"  # TODO
    )
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def grouper(inputs, n):
    num_groups = len(inputs) // n
    return [tuple(inputs[i * n : (i + 1) * n]) for i in range(num_groups)]


def run_on_one_sequence(predictor, metadata, video_frames, savedir):
    video_saver = VideoSaver(metadata)
    for path in video_frames:
        head_tail = os.path.split(path)
        save_path = savedir / path.name
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        predictions = predictor(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        panoptic_seg, segments_info = predictions["panoptic_seg"]
        video_saver.draw_panoptic_seg_predictions(
            frame.shape[:2], panoptic_seg.to("cpu"), segments_info, save_path
        )


if __name__ == "__main__":
    input_path = Path("carla")  # TODO
    args = default_argument_parser().parse_args()
    cfg = setup(args)
    predictor = DefaultPredictor(cfg)
    metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])
    for iteration in input_path.iterdir():  # TODO
        for world in iteration.iterdir():  # TODO
            frame_list = list(world.iterdir())  # TODO
            video_list = grouper(sorted(frame_list), 30)
            for video in video_list:
                savedir = Path("output_dir")  # TODO
                savedir.mkdir(parents=True, exist_ok=True)
                run_on_one_sequence(predictor, metadata, video, savedir)
