from utility.coco_utils import get_coco_api_from_dataset
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
from PIL import Image
import numpy as np
import matplotlib.pyplot as plot
from functools import partial
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_preprocessing_fn

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
setup_logger(name="mask2former")

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
from detectron2.projects.deeplab import add_deeplab_config
# coco_metadata = MetadataCatalog.get("coco_2017_val_panoptic")
from mask2former import add_maskformer2_config

# specify a seed to keep things consistent
random.seed(1000)
torch.manual_seed(1000)
torch.cuda.manual_seed_all(1000)

DEVICE = torch.device('cuda')
TRAIN_FOLDER = os.path.join('detection', 'train')
MINNIE_IMAGES = os.path.join(TRAIN_FOLDER, 'images')
MINNIE_MASKS = os.path.join(TRAIN_FOLDER, 'masks')
MINNIE_IMLIST = os.listdir(MINNIE_IMAGES)
MINNIE_MSLIST = os.listdir(MINNIE_MASKS)

# assert they're the same names and have similar number of images
assert all([im == ms for im, ms in zip(MINNIE_IMLIST, MINNIE_MSLIST)]
           ), 'Mismatching datasets!'

# now prepend the file paths
MINNIE_IMLIST = list(map(partial(os.path.join, MINNIE_IMAGES), MINNIE_IMLIST))
MINNIE_MSLIST = list(map(partial(os.path.join, MINNIE_MASKS), MINNIE_MSLIST))

# shuffle the image lists (random shuffle on each elem we zip the image pairs)
MINNIE_JOINED = list(zip(MINNIE_IMLIST, MINNIE_MSLIST))
random.shuffle(MINNIE_JOINED)
MINNIE_IMLIST_O, MINNIE_MSLIST_O = copy.copy(
    MINNIE_IMLIST), copy.copy(MINNIE_MSLIST)
MINNIE_IMLIST, MINNIE_MSLIST = zip(*MINNIE_JOINED)

# ensure the list is properly shuffled
assert any([(im != im_o and ms != ms_o and os.path.basename(im) == os.path.basename(ms)) for im, ms,
           im_o, ms_o in zip(MINNIE_IMLIST, MINNIE_MSLIST, MINNIE_IMLIST_O, MINNIE_MSLIST_O)]), 'Not shuffled!'


class MinnieDataset(datasets.VisionDataset):
  def __init__(self, images, masks):
    self._images = images
    self._masks = masks
    # self._transforms = transforms


    # define augmentations
    self.transform = transforms.Compose([
      # transforms.ToTensor(),
      transforms.Pad((8, 0, 8, 0))  # 1280x720 (UNET needs something divisible by 32) --> 1280x736
      # transforms.Normalize(
      #   mean=(MNIST_MEAN,),
      #   std=(MNIST_STD,)
      # )
    ])

  # overrides len function for this class
  def __len__(self):
      return len(self._images)  # gets number of images -- lenght of dataset

  # overrides [] operation for this class
  def __getitem__(self, index):
    # index - index of image/mask pair to load
    im_path = self._images[index]
    mask_path = self._masks[index] #load the path of the image

    # read the RGB image
    image = cv2.imread(im_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # opencv by default reads color images as BGR, switch to RGB

    # read the mask image as binary
    mask = cv2.imread(mask_path, 0)  # read unchanged

    # Convert the PIL image to np array
    mask = np.array(mask)
    obj_ids = np.unique(mask)
    obj_ids = np.unique(mask)
    
    # Remove background id
    obj_ids = obj_ids[1:]

    # Split the color-encoded masks into a set of binary masks
    masks = mask == obj_ids[:, None, None]

    # Get bbox coordinates for each mask
    num_objs = len(obj_ids)
    boxes = []
    h, w = mask.shape
    for ii in range(num_objs):
        pos = np.where(masks[ii])
        xmin = np.min(pos[1])
        xmax = np.max(pos[1])
        ymin = np.min(pos[0])
        ymax = np.max(pos[0])

        if xmin == xmax or ymin == ymax:
            continue

        xmin = np.clip(xmin, a_min=0, a_max=w)
        xmax = np.clip(xmax, a_min=0, a_max=w)
        ymin = np.clip(ymin, a_min=0, a_max=h)
        ymax = np.clip(ymax, a_min=0, a_max=h)
        boxes.append([xmin, ymin, xmax, ymax])

    # Convert everything into a torch.Tensor
    boxes = torch.as_tensor(boxes, dtype=torch.float32)

    # There is only one class (apples)
    labels = torch.ones((num_objs,), dtype=torch.int64)
    masks = torch.as_tensor(masks, dtype=torch.uint8)

    image_id = torch.tensor([index])
    area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

    # All instances are not crowd
    iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

    target = {}
    target['file_name'] = im_path
    target['mask_name'] = mask_path
    target["boxes"] = boxes
    target["labels"] = labels
    target["masks"] = masks
    target["image_id"] = image_id
    target["area"] = area
    target["iscrowd"] = iscrowd

    # [1280x720] -> 0 (background) 1 (apple)  (Segmentation, binary)
    # [1280x720] -> 0 (background 1 (apple 1) 2 (apple 2) ... 4 (apple 4) (Instance Segmentation, group pixels according to apple instances)
    # instance segmentation requires recursive neural network
    #image = TF.to_tensor(image)
    #mask = TF.to_tensor(mask)
    return (image, target) #numpy array


dev_split = 0.15
dev_len = int(len(MINNIE_IMLIST) * dev_split) #test - train split --> holdout 15% as validation set
dev_images = MINNIE_IMLIST[:dev_len] #get 85% for training
dev_masks = MINNIE_MSLIST[:dev_len]
train_images = MINNIE_IMLIST[dev_len:]
train_masks = MINNIE_MSLIST[dev_len:]


# make validation set
dataset = MinnieDataset(dev_images, dev_masks)
coco_dev = get_coco_api_from_dataset(dataset)

import json
with open('minnie_coco_val.json', 'w') as jw:
  json.dump(coco_dev, jw)

# make training set
dataset = MinnieDataset(train_images, train_masks)
coco_train = get_coco_api_from_dataset(dataset)

with open('minnie_coco_train.json', 'w') as jw:
  json.dump(coco_train, jw)