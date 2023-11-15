from collections import OrderedDict
from typing import List, Tuple, Dict, Any, Set, Sequence
import detectron2
from detectron2.utils.logger import setup_logger
import itertools
from mask2former.data.dataset_mappers.mask_former_instance_dataset_mapper import MaskFormerInstanceDatasetMapper
from mask2former.data.dataset_mappers.mask_former_panoptic_dataset_mapper import MaskFormerPanopticDatasetMapper
from mask2former.data.dataset_mappers.mask_former_semantic_dataset_mapper import MaskFormerSemanticDatasetMapper
setup_logger()

# import some common libraries
import numpy as np
import cv2
import torch
import copy
import random
import logging
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    SemSegEvaluator,
    verify_results,
)

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog, build_detection_train_loader
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.data.catalog import DatasetCatalog
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.data.datasets import register_coco_instances
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.logger import setup_logger
from mask2former import (
    COCOInstanceNewBaselineDatasetMapper,
    COCOPanopticNewBaselineDatasetMapper,
    InstanceSegEvaluator,
    MaskFormerInstanceDatasetMapper,
    MaskFormerPanopticDatasetMapper,
    MaskFormerSemanticDatasetMapper,
    SemanticSegmentorWithTTA,
    add_maskformer2_config,
)
register_coco_instances("minnie_train", {
  'stuff_classes': ['apple'],
  'stuff_colors': [(255, 0, 0)]
}, "/lgdata/agathon/Mask2Former/minnie_coco_train.json", "")
register_coco_instances("minnie_val", {
  'stuff_classes': ['apple'],
  'stuff_colors': [(255, 0, 0)]
}, "/lgdata/agathon/Mask2Former/minnie_coco_val.json", "")

#visualize training data
minnie_train_metadata = MetadataCatalog.get("minnie_train")
dataset_dicts = DatasetCatalog.get("minnie_train")

import random
from detectron2.utils.visualizer import Visualizer

SAMPLE = False
if SAMPLE:
  for d in random.sample(dataset_dicts, 3):
    print(d['file_name'])
    img = cv2.imread(d["file_name"])

    visualizer = Visualizer(img[:, :, ::-1], metadata=minnie_train_metadata, scale=0.5)
    vis = visualizer.draw_dataset_dict(d)
    cv2.imwrite('test-{}.png'.format(d['image_id']), vis.get_image()[:, :, ::-1])


#EVALUATION

from detectron2.engine import DefaultTrainer
# trainer = Trainer(cfg) 
# from detectron2.evaluation import COCOEvaluator

class CocoTrainer(DefaultTrainer):
  @classmethod
  def build_evaluator(cls, cfg, dataset_name, output_folder=None):
    if output_folder is None:
      os.makedirs("coco_eval", exist_ok=True)
      output_folder = "coco_eval"

    return COCOEvaluator(dataset_name, cfg, False, output_folder)


  @classmethod
  def build_evaluator(cls, cfg, dataset_name, output_folder=None):
    """
    Create evaluator(s) for a given dataset.
    This uses the special metadata "evaluator_type" associated with each
    builtin dataset. For your own dataset, you can simply create an
    evaluator manually in your script and do not have to worry about the
    hacky if-else logic here.
    """
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    evaluator_list = []
    evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
    # # semantic segmentation
    # if evaluator_type in ["sem_seg", "ade20k_panoptic_seg"]:
    #     evaluator_list.append(
    #         SemSegEvaluator(
    #             dataset_name,
    #             distributed=True,
    #             output_dir=output_folder,
    #         )
    #     )
    # instance segmentation
    if evaluator_type == "coco":
        evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
    
    # panoptic segmentation
    # if evaluator_type in [
    #     "coco_panoptic_seg",
    #     "ade20k_panoptic_seg",
    #     "cityscapes_panoptic_seg",
    #     "mapillary_vistas_panoptic_seg",
    # ]:
    #     if cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON:
    #         evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
    
    # COCO
    # if evaluator_type == "coco_panoptic_seg" and cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON:
    #     evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
    # if evaluator_type == "coco_panoptic_seg" and cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON:
    #     evaluator_list.append(SemSegEvaluator(dataset_name, distributed=True, output_dir=output_folder))
    # # Mapillary Vistas
    # if evaluator_type == "mapillary_vistas_panoptic_seg" and cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON:
    #     evaluator_list.append(InstanceSegEvaluator(dataset_name, output_dir=output_folder))
    # if evaluator_type == "mapillary_vistas_panoptic_seg" and cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON:
    #     evaluator_list.append(SemSegEvaluator(dataset_name, distributed=True, output_dir=output_folder))
    # # Cityscapes
    # if evaluator_type == "cityscapes_instance":
    #     assert (
    #         torch.cuda.device_count() > comm.get_rank()
    #     ), "CityscapesEvaluator currently do not work with multiple machines."
    #     return CityscapesInstanceEvaluator(dataset_name)
    # if evaluator_type == "cityscapes_sem_seg":
    #     assert (
    #         torch.cuda.device_count() > comm.get_rank()
    #     ), "CityscapesEvaluator currently do not work with multiple machines."
    #     return CityscapesSemSegEvaluator(dataset_name)
    # if evaluator_type == "cityscapes_panoptic_seg":
    #     if cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON:
    #         assert (
    #             torch.cuda.device_count() > comm.get_rank()
    #         ), "CityscapesEvaluator currently do not work with multiple machines."
    #         evaluator_list.append(CityscapesSemSegEvaluator(dataset_name))
    #     if cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON:
    #         assert (
    #             torch.cuda.device_count() > comm.get_rank()
    #         ), "CityscapesEvaluator currently do not work with multiple machines."
    #         evaluator_list.append(CityscapesInstanceEvaluator(dataset_name))
    # # ADE20K
    # if evaluator_type == "ade20k_panoptic_seg" and cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON:
    #     evaluator_list.append(InstanceSegEvaluator(dataset_name, output_dir=output_folder))
    # # LVIS
    # if evaluator_type == "lvis":
    #     return LVISEvaluator(dataset_name, output_dir=output_folder)
    # if len(evaluator_list) == 0:
    #     raise NotImplementedError(
    #         "no Evaluator for the dataset {} with the type {}".format(
    #             dataset_name, evaluator_type
    #         )
    #     )
    # elif len(evaluator_list) == 1:
    #     return evaluator_list[0]
    return DatasetEvaluators(evaluator_list)

  @classmethod
  def build_optimizer(cls, cfg, model):
    weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM
    weight_decay_embed = cfg.SOLVER.WEIGHT_DECAY_EMBED

    defaults = {}
    defaults["lr"] = cfg.SOLVER.BASE_LR
    defaults["weight_decay"] = cfg.SOLVER.WEIGHT_DECAY

    norm_module_types = (
      torch.nn.BatchNorm1d,
      torch.nn.BatchNorm2d,
      torch.nn.BatchNorm3d,
      torch.nn.SyncBatchNorm,
      # NaiveSyncBatchNorm inherits from BatchNorm2d
      torch.nn.GroupNorm,
      torch.nn.InstanceNorm1d,
      torch.nn.InstanceNorm2d,
      torch.nn.InstanceNorm3d,
      torch.nn.LayerNorm,
      torch.nn.LocalResponseNorm,
    )

    params: List[Dict[str, Any]] = []
    memo: Set[torch.nn.parameter.Parameter] = set()
    for module_name, module in model.named_modules():
      for module_param_name, value in module.named_parameters(recurse=False):
        if not value.requires_grad:
          continue
        # Avoid duplicating parameters
        if value in memo:
          continue
        memo.add(value)

        hyperparams = copy.copy(defaults)
        if "backbone" in module_name:
          hyperparams["lr"] = hyperparams["lr"] * cfg.SOLVER.BACKBONE_MULTIPLIER
        if (
          "relative_position_bias_table" in module_param_name
          or "absolute_pos_embed" in module_param_name
        ):
          print(module_param_name)
          hyperparams["weight_decay"] = 0.0
        if isinstance(module, norm_module_types):
          hyperparams["weight_decay"] = weight_decay_norm
        if isinstance(module, torch.nn.Embedding):
          hyperparams["weight_decay"] = weight_decay_embed
        params.append({"params": [value], **hyperparams})

    def maybe_add_full_model_gradient_clipping(optim):
      # detectron2 doesn't have full model gradient clipping now
      clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
      enable = (
        cfg.SOLVER.CLIP_GRADIENTS.ENABLED
        and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
        and clip_norm_val > 0.0
      )

      class FullModelGradientClippingOptimizer(optim):
        def step(self, closure=None):
          all_params = itertools.chain(*[x["params"] for x in self.param_groups])
          torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
          super().step(closure=closure)

      return FullModelGradientClippingOptimizer if enable else optim

    optimizer_type = cfg.SOLVER.OPTIMIZER
    if optimizer_type == "SGD":
      optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
          params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
      )
    elif optimizer_type == "ADAMW":
      optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
        params, cfg.SOLVER.BASE_LR
      )
    else:
      raise NotImplementedError(f"no optimizer type {optimizer_type}")
    if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
      optimizer = maybe_add_gradient_clipping(cfg, optimizer)
    return optimizer

  @classmethod
  def test_with_TTA(cls, cfg, model):
    logger = logging.getLogger("detectron2.trainer")
    # In the end of training, run an evaluation with TTA.
    logger.info("Running inference with test-time augmentation ...")
    model = SemanticSegmentorWithTTA(cfg, model)
    evaluators = [
      cls.build_evaluator(
        cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
      )
      for name in cfg.DATASETS.TEST
    ]
    res = cls.test(cfg, model, evaluators)
    res = OrderedDict({k + "_TTA": v for k, v in res.items()})
    return res

  @classmethod
  def build_train_loader(cls, cfg):
    # if cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_semantic":
    #   mapper = MaskFormerSemanticDatasetMapper(cfg, True)
    #   return build_detection_train_loader(cfg, mapper=mapper)
    # elif cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_panoptic":
    #   mapper = MaskFormerPanopticDatasetMapper(cfg, True)
    #   return build_detection_train_loader(cfg, mapper=mapper)
    # # Instance segmentation dataset mapper
    # elif cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_instance":
    mapper = MaskFormerInstanceDatasetMapper(cfg, True)
    return build_detection_train_loader(cfg, mapper=mapper)
    # raise Exception('NOTHING')

  @classmethod
  def build_lr_scheduler(cls, cfg, optimizer):
    """
    It now calls :func:`detectron2.solver.build_lr_scheduler`.
    Overwrite it if you'd like a different scheduler.
    """
    return build_lr_scheduler(cfg, optimizer)


from detectron2.config import get_cfg
#from detectron2.evaluation.coco_evaluation import COCOEvaluator
import os

cfg = get_cfg()
# cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
add_deeplab_config(cfg)
add_maskformer2_config(cfg)
cfg.INPUT.DATASET_MAPPER_NAME = 'mask_former_instance'
cfg.merge_from_file('configs/coco/instance-segmentation/swin/maskformer2_swin_base_384_bs16_50ep.yaml')

# cfg.merge_from_file("configs/coco/panoptic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_100ep.yaml")
cfg.MODEL.WEIGHTS = None  # 'https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/panoptic/maskformer2_swin_large_IN21k_384_bs16_100ep/model_final_f07440.pkl'
cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = True
cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = True
cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = False


cfg.DATASETS.TRAIN = ("minnie_train",)
cfg.DATASETS.TEST = ("minnie_val",)
cfg.OUTPUT_DIR = '/lgdata/agathon/Mask2Former/output'

cfg.DATALOADER.NUM_WORKERS = 5
cfg.MODEL.WEIGHTS = 'model.pk1'
# cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.0001   # 0.000025

cfg.SOLVER.WARMUP_ITERS = 700
cfg.SOLVER.MAX_ITER = 1500 #adjust up if val mAP is still rising, adjust down if overfit
cfg.SOLVER.STEPS = (1000, 1400)
cfg.SOLVER.GAMMA = 0.05

cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2 #your number of classes + 1

cfg.TEST.EVAL_PERIOD = 500


os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = CocoTrainer(cfg)
trainer.resume_or_load(resume=True)
trainer.train()


#METRICS
#test evaluation
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.85
predictor = DefaultPredictor(cfg)
evaluator = COCOEvaluator("minnie_val", cfg, False, output_dir="./output/")
# evaluator = LVISEvaluator('minnie_val', output_dir='./output/')
val_loader = build_detection_test_loader(cfg, "minnie_val")
inference_on_dataset(trainer.model, val_loader, evaluator)

#saving the weights
# %ls ./output/

# cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
# cfg.DATASETS.TEST = ("my_dataset_test", )
# cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set the testing threshold for this model
# predictor = DefaultPredictor(cfg)
# test_metadata = MetadataCatalog.get("my_dataset_test")

# from detectron2.utils.visualizer import ColorMode
# import glob

# for imageName in glob.glob('/detection/test_img/*jpg'):
#   im = cv2.imread(imageName)
#   outputs = predictor(im)
#   v = Visualizer(im[:, :, ::-1],
#                 metadata=test_metadata, 
#                 scale=0.8
#                  )
#   out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
#   cv2_imshow(out.get_image()[:, :, ::-1])