import torch
import torchvision
import torch.nn as nn
import traceback
import os
import cv2
import random
import copy
import tqdm
import math
import gc
import numpy as np
import matplotlib.pyplot as plot
from resnet import resnet_backbone
import utility.utils as utils
import utility.transforms as T
from torchvision.models.detection.mask_rcnn import MaskRCNN
from functools import partial
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_preprocessing_fn
from training import get_maskrcnn_model_instance, MinnieAugDataset, MINNIE_IMLIST, MINNIE_MSLIST

# import detectron2
# from detectron2.utils.logger import setup_logger

from utility.engine import train_one_epoch, evaluate

checkpoint = torch.load('model_7.pth', map_location='cpu')

num_classes = 2
model = get_maskrcnn_model_instance(num_classes)
model.load_state_dict(checkpoint['model'])
model.to('cpu')

# sample some images
SAMPLE = 7
images = []
dev_len = int(len(MINNIE_IMLIST) * 0.15) #test - train split --> holdout 15% as validation set
dev_images = MINNIE_IMLIST[:dev_len] #get 85% for training
dev_masks = MINNIE_MSLIST[:dev_len]

dset = MinnieAugDataset(dev_images, dev_masks, train=False)
indexs = random.sample(list(range(len(dset))), SAMPLE)

from detectron2.utils.visualizer import Visualizer
from detectron2.structures.instances import Instances

with torch.no_grad():
  model.eval()
  for index in indexs:
    img, mask = dset[index]
    mask_cand_path = dset._masks[index].replace('/masks/', '/segs/')
    candidates = cv2.imread(mask_cand_path, 0)

    inp = torch.zeros((4, 1280, 720))
    inp[:3, :, :] = torch.as_tensor(img).to(torch.float) / 255.0
    inp[3, :, :] = torch.as_tensor(candidates).to(torch.float) / 255.0

    outputs = model([inp])[0]

    visualizer = Visualizer(img.transpose(1, 2, 0), scale=0.5)


    instances = Instances((1280, 720))
    instances.set('pred_boxes', outputs['boxes'])
    instances.set('scores', outputs['scores'])
    print(outputs['masks'].view(-1, 1280, 720))
    instances.set('pred_masks', outputs['masks'].view(-1, 1280, 720))

    vis = visualizer.draw_instance_predictions(
      instances
    )
    cv2.imwrite('pred-{}.png'.format(index), vis.get_image()[:, :, ::-1])
