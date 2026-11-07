"""Train seg cascade_mask_rcnn on KITTI-MOTS from the MOT-stage cascade_rcnn weights (vseg02), then evaluate.

    UNIFIED_SEG_DATASET=kittimots UNIFIED_IMAGE_H=384 python train_seg_cascade_mask_rcnn_km.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.runner import main
from common.seg.init_from_det import with_init
from detectors.cascade_mask_rcnn.build import build_model

if __name__ == "__main__":
    main("train", task="seg", version="cascade_mask_rcnn_km", build_model=with_init(build_model, "cascade_mask_rcnn", "cascade_rcnn"))
