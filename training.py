# %%


# %%
# TEAM 22 -- Labor Challenge

# The plan: try a bunch of things

# tasks we want to try first
# image segmentation models: UNET, Deeplabv3+
# ensembling segmentation masks
# data augmentation: rotate, flip, zoom in/out (scaling), row shifting, gaussian noise,
#                    artifacts (masking, occlusion, special effects), img aug, torch vision transforms
# metrics and error analysis: dice score

# what do people want to work on?
# Emily: research ensembling
# Maha: data augmentation
# David:
# Anita: try out a variety of baseline models --> https://github.com/qubvel/segmentation_models.pytorch


# Notes
# https://competitions.codalab.org/competitions/21694 shows score to beat/reach is about 0.8
# how will we handle detecting apples that are on ground vs on tree?
# should not 100% flip image upside down, only tilt a little to each side
# how to handle (probably) dirty dataset? masks are not consistent

# ---------------------------------------------------------------------
# ENSEMBLE POSIBILITIES:

# OPTION 1

# 	Sum probabilities of every class -- pass to softmax function(so it is normalized) -- simplest approach

# OPTION 2

# 	Vote Using Mean Probabilities
# 	Vote Using Sum Probabilities
# 	Vote Using Weighted Sum Probabilities

# OPTION 3
# 	Average

# 	Model 1: 99.00
# 	Model 2: 101.00
# 	Model 3: 98.00
# 	The mean predicted would be calculated as:

# 	Mean Prediction = (99.00 + 101.00 + 98.00) / 3
# 	Mean Prediction = 298.00 / 3
# 	Mean Prediction = 99.33

# ---------------------------------------------------------------------

# %% [markdown]
# ## Import libaries

# %%
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

# import detectron2
# from detectron2.utils.logger import setup_logger

from utility.engine import train_one_epoch, evaluate
# setup_logger()
# setup_logger(name="mask2former")

# from detectron2 import model_zoo
# from detectron2.engine import DefaultPredictor
# from detectron2.config import get_cfg
# from detectron2.utils.visualizer import Visualizer, ColorMode
# from detectron2.data import MetadataCatalog
# from detectron2.projects.deeplab import add_deeplab_config
# coco_metadata = MetadataCatalog.get("coco_2017_val_panoptic")
# from mask2former import add_maskformer2_config


# %% [markdown]
# ## Create custom dataloader to load the segmentation masks and apply augmentations

# %%
# specify a seed to keep things consistent
random.seed(1000)
torch.manual_seed(1000)
torch.cuda.manual_seed_all(1000)

# define global params
MINNIE_LOADER_ARGS = {
    'num_workers': 4,
    'pin_memory': True
}
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

# PyTorch requirement: create dataset class for loading data from a source

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
        # transforms.append(T.)
    return T.Compose(transforms)


class MinnieAugDataset(datasets.VisionDataset):
    def __init__(self, images, masks, transforms=None, train=False):
        self._images = images
        self._masks = masks
        self._train = train

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
        image = image.transpose(2, 0, 1)
        # print("fresh image shape:", image.shape)

        # read the mask image as binary
        mask = cv2.imread(mask_path, 0)  # read unchanged

        # print("image_tensor shape:", image_tensor.shape)
        if self._train:
            # apple shaped cutout works on torch tensors, not PIL images
            image_tensor = torch.as_tensor(image)
            mask_tensor = torch.as_tensor(mask)
            # print('AFTER')

            num_apples = random.randint(0, 5)
            for _ in range(num_apples):
                x = random.randint(100, 1100)
                y = random.randint(100, 600)

                # basic square for apple body
                i = 100 + x
                j = 100 + y
                h = 110
                w = 110
                image_tensor = TF.erase(image_tensor, i, j, h, w, 0)
                mask_tensor = TF.erase(mask_tensor, i, j, h, w, 0)

                # basic square for apple stem
                i = 40 + x
                j = 150 + y
                h = 60
                w = 10
                image_tensor = TF.erase(image_tensor, i, j, h, w, 0)
                mask_tensor = TF.erase(mask_tensor, i, j, h, w, 0)

                # basic square for apple stem leaf
                i = 60 + x
                j = 150 + y
                h = 20
                w = 50
                image_tensor = TF.erase(image_tensor, i, j, h, w, 0)
                mask_tensor = TF.erase(mask_tensor, i, j, h, w, 0)

            # must apply remaining transformations to PIL images
            pil_image = TF.to_pil_image(image_tensor)
            pil_mask = TF.to_pil_image(mask_tensor)
            
            # padding
            # if random.random() > 0.5:
            pil_image=TF.pad(pil_image, (8,0,8,0))
            pil_mask=TF.pad(pil_mask,(8,0,8,0))

            # mirror the image at random
            if random.random() > 0.5:
                pil_image = TF.hflip(pil_image)
                pil_mask = TF.hflip(pil_mask)

            # rotation
            if random.random() > 0.5:
                angle = random.randint(-45, 45)
                pil_image = TF.rotate(pil_image, angle) 
                pil_mask = TF.rotate(pil_mask, angle)

            # GaussianBlur for images
            if random.random() < 0.25:
                pil_image = TF.gaussian_blur(pil_image, 3)

            #change it back to numpy
            image = np.asarray(pil_image)
            mask = np.asarray(pil_mask)

        # print("numpy image shape:", image.shape)

        # @TODO each color represents a different instance
        # for instance segmentation, but for now we'll just do a binary is apple or not
        mask = (mask > 0).astype(np.float16)  # convert from instance into binary  
        # mask = (mask > 0).to(torch.float)

        # [1280x720] -> 0 (background) 1 (apple)  (Segmentation, binary)
        # [1280x720] -> 0 (background 1 (apple 1) 2 (apple 2) ... 4 (apple 4) (Instance Segmentation, group pixels according to apple instances)
        # instance segmentation requires recursive neural network

        if image.shape[-1] == 3:
            image = image.transpose(2, 0, 1)
        # print('OK', image.shape, mask.shape)

        return (image.copy(), mask.copy()) #numpy array


from PIL import Image
class MinnieDataset(datasets.VisionDataset):
    def __init__(self, images, masks, train=False):
        self._images = images
        self._masks = masks
        self.train = train
        # self._transforms = transforms


        # define augmentations
        # self.transform = transforms.Compose([
        #     transforms.ToTensor(),
        #     # transforms.Pad((8, 0, 8, 0))  # 1280x720 (UNET needs something divisible by 32) --> 1280x736
        #     # transforms.Normalize(
        #     #   mean=(MNIST_MEAN,),
        #     #   std=(MNIST_STD,)
        #     # )
        # ])
        self.transform = get_transform(self.train)

    # overrides len function for this class
    def __len__(self):
        return len(self._images)  # gets number of images -- lenght of dataset

    # overrides [] operation for this class
    def __getitem__(self, index):
        # index - index of image/mask pair to load
        im_path = self._images[index]
        mask_path = self._masks[index] #load the path of the image
        mask_cand_path = self._masks[index].replace('/masks/', '/segs/')

        # read the RGB image
        # image = cv2.imread(im_path)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # opencv by default reads color images as BGR, switch to RGB
        
        image = np.array(Image.open(im_path).convert("RGB"))
        mask = Image.open(mask_path)     # Each color of mask corresponds to a different instance with 0 being the background

        # Convert the PIL image to np array
        mask = np.array(mask)


        # read the mask image as binary
        # mask = cv2.imread(mask_path, 0)  # read unchanged
        candidates = cv2.imread(mask_cand_path, 0)

        # add candidates as channel
        nimage = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.float32)
        nimage[:, :, :3] = image.astype(np.float32) / 255.0  # - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        nimage[:, :, 3] = candidates.astype(np.float32) / 255.0  #  - 127.0) / 127.0
        # nimage = image.astype(np.float32)
        # mult = np.clip((candidates.astype(np.float32) / 255.0), 0.18, 1.0)
        # nimage[:, :, 0] *= mult
        # nimage[:, :, 1] *= mult
        # nimage[:, :, 2] *= mult

        # print(np.max(nimage[:, :, 3]))
        # nimage = nimage.astype(np.float16) / 255.0
        image = nimage

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
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["segmentation"] = torch.as_tensor((mask > 0.0), dtype=torch.float32)
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        # [1280x720] -> 0 (background) 1 (apple)  (Segmentation, binary)
        # [1280x720] -> 0 (background 1 (apple 1) 2 (apple 2) ... 4 (apple 4) (Instance Segmentation, group pixels according to apple instances)
        # instance segmentation requires recursive neural network
        #image = TF.to_tensor(image)
        #mask = TF.to_tensor(mask)
        # if self.train:
        image, target = self.transform(image, target)
        # print(image.dtype, target)
        return (image, target) #numpy array


def load_minnie(bs: int = 100, drop_last=False, dev_split=0.15, first=False, train=False):
    """ Handles loading the apple dataset train/test (if specified) """
    # do dev/test split on shuffled dataset
    dev_len = int(len(MINNIE_IMLIST) * dev_split) #test - train split --> holdout 15% as validation set
    dev_images = MINNIE_IMLIST[:dev_len] #get 85% for training
    dev_masks = MINNIE_MSLIST[:dev_len]
    train_images = MINNIE_IMLIST[dev_len:]
    train_masks = MINNIE_MSLIST[dev_len:]

    # PyTorch DataLoader takes a Dataset and does fancy stuff in the background (batching, shuffling, etc)
    train_loader = DataLoader(
        dataset=MinnieAugDataset(train_images, train_masks, train=True) if first else MinnieDataset(train_images, train_masks, train=True),
        batch_size=bs,
        shuffle=True,
        drop_last=drop_last,
        **MINNIE_LOADER_ARGS # pin memory for GPU optimizations
    )

    test_loader = DataLoader(
        dataset=MinnieAugDataset(dev_images, dev_masks, train=False) if first else MinnieDataset(dev_images, dev_masks, train=False),
        batch_size=bs,
        drop_last=drop_last,
        shuffle=False,
        **MINNIE_LOADER_ARGS
    )

    return train_loader, test_loader

# %%
# model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
#     in_channels=3, out_channels=1, init_features=32, pretrained=True)

# %% [markdown]
# ## Test loading some data from custom dataloader


# %%
RUN_TESTS = False
if RUN_TESTS:  # set
    d = MinnieDataset(MINNIE_IMLIST, MINNIE_MSLIST)
    im, ms = d[0]
    f, axarr = plot.subplots(1, 2)
    print('Max range image', np.min(im), np.max(im), im.dtype, im.shape)
    print('Max range mask', np.min(ms), np.max(ms), ms.dtype, im.shape)
    axarr[0].imshow(im)
    axarr[1].imshow(ms)
    plot.show()

# %% [markdown]
# ## Now test dataloader with augmentations to make sure the images/masks make sense

# %%
if RUN_TESTS:
    SAMPLE_IMAGES = 9
    minnie_train, _ = load_minnie(bs=1)  # @TODO add augmentations

    pairs = []
    for X, Y in minnie_train:
        if len(pairs) == SAMPLE_IMAGES:
            break

        pairs.append((X.numpy()[0], Y.numpy()[0]))

    f, axarr = plot.subplots(3, 3*2, figsize=(10, 10))
    for ind, (X, Y) in enumerate(pairs):
        row = int(ind / 3)
        col = int(2*ind % 3*2)
        axarr[row, col].imshow(X)
        axarr[row, col+1].imshow(Y)
    plot.show()

# %%
# just the quickstart code for now, trying to build the pytorch model class
# reference: https://github.com/qubvel/segmentation_models.pytorch/blob/master/examples/binary_segmentation_intro.ipynb
# reference notebook: https://github.com/qubvel/segmentation_models.pytorch/blob/master/tests/test_preprocessing.py

# torchvision also has other models I guess: https://pytorch.org/vision/0.8/models.html#semantic-segmentation

# Basic idea: multiple convolutional filters created to detect different image features (e.g. horizontal vs vertical lines)
# Example: 32 filters --> 32 images 
# Each layer of filters and pooling (choose pixel with max value) will produce smaller spatial resolution
# Decoder uses some of past filters (created by Encoder) to construct the image mask

# Encoder Network: input an image, output bottleneck, aka useful information (image features)
# Decoder Network: reverses the Encoder process, output segmentation mask with only apples labeled (removed all other info)
# UNET = encoder + decoder

# This class defines the UNET model to work with the PyTorch framework
class FirstModel(nn.Module):
    def __init__(self):
        super().__init__()

        # self.model = smp.Unet(
        #     encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        #     # use `imagenet` pre-trained weights for encoder initialization
        #     encoder_weights="imagenet",
        #     # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        #     in_channels=3,
        #     # model output channels (number of classes in your dataset--just apples or no apples here)
        #     classes=1,
        # )
        self.model = torchvision.models.segmentation.deeplabv3.deeplabv3_resnet50(
            num_classes=1,
            weights=None
        )

        # preprocessing parameters for image -- not sure what this is doing yet
        params = smp.encoders.get_preprocessing_params("resnet34")
        self.register_buffer("std", torch.tensor(
            params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(
            params["mean"]).view(1, 3, 1, 1))

        # for image segmentation dice loss could be the best first choice
        self.loss_fn = smp.losses.DiceLoss(
            smp.losses.BINARY_MODE, from_logits=True)

    # defines forward computation of the model, is this currently correct for unet?
    # not sure yet where mean and std come from but this was one of their unit tests:
    # params = smp.encoders.get_preprocessing_params("resnet18")
    # assert params["mean"] == [0.485, 0.456, 0.406]
    # assert params["std"] == [0.229, 0.224, 0.225]
    # assert params["input_range"] == [0, 1]
    # assert params["input_space"] == "RGB"
    def forward(self, image):
        # print(image.shape, self.mean.shape)

        # normalize image here using resnet34 mean and std for now
        image = (image - self.mean) / self.std
        mask = self.model(image)
        return mask['out']


# cfg = get_cfg()
# add_deeplab_config(cfg)
# add_maskformer2_config(cfg)
# cfg.merge_from_file("configs/coco/panoptic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_100ep.yaml")
# cfg.MODEL.WEIGHTS = 'https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/panoptic/maskformer2_swin_large_IN21k_384_bs16_100ep/model_final_f07440.pkl'
# cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = False
# cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = True
# cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = False
# predictor = DefaultPredictor(cfg)


# %% [markdown]
# # Creating the model
# %%


# %% [markdown]
# # Training the model

# %%
# hyperparams/definitions
BATCH_SIZE = 4
EPOCHS = 1
LEARNING_RATE = 8e-5
WEIGHT_DECAY = 1e-6   # think of this as L2 regularization

# load the dataloaders
minnie_train, minnie_test = load_minnie(bs=BATCH_SIZE, first=True, train=True)

def get_maskrcnn_model_instance(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    backbone = resnet_backbone()
    # model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)  # , image_mean=[0.0, 0.0, 0.0], image_std=[1.0, 1.0, 1.0])# , image_mean=[0.485, 0.456, 0.406, 0.5], image_std=[0.229, 0.224, 0.225, 1.0])
    model = MaskRCNN(backbone, num_classes=num_classes, image_mean=[0.485, 0.456, 0.406, 0.0], image_std=[0.229, 0.224, 0.225, 1.0])

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # print(in_features)
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)  # , image_mean=[0.485, 0.456, 0.406, 0.5], image_std=[0.229, 0.224, 0.225, 1.0])

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)  # , image_mean=[0.485, 0.456, 0.406, 0.5], image_std=[0.229, 0.224, 0.225, 1.0])
    
    return model


def get_frcnn_model_instance(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def main(args):
    print(args)
    device = torch.device(DEVICE) # args.device

    # Data loading code
    import time
    print("Loading data")
    num_classes = 2

    dev_split = 0.15
    dev_len = int(len(MINNIE_IMLIST) * dev_split) #test - train split --> holdout 15% as validation set
    dev_images = MINNIE_IMLIST[:dev_len] #get 85% for training
    dev_masks = MINNIE_MSLIST[:dev_len]
    train_images = MINNIE_IMLIST[dev_len:]
    train_masks = MINNIE_MSLIST[dev_len:]

    dataset = MinnieDataset(train_images, train_masks, train=True)
    dataset_test = MinnieDataset(dev_images, dev_masks, train=False)
    # from data.apple_dataset import AppleDataset
    # dataset = AppleDataset(os.path.join(args.data_path, 'train'), get_transform(train=True))
    # dataset_test = AppleDataset(os.path.join(args.data_path, 'train'), get_transform(train=False))

    print("Creating data loaders")
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                                                num_workers=args.workers, collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1,
                                                    shuffle=False, num_workers=args.workers,
                                                    collate_fn=utils.collate_fn)

    print("Creating model")
    # Create the correct model type
    # if args.model == 'maskrcnn':
    preproc_model = torch.nn.Sequential(
        nn.Conv2d(4, 32, (3, 3)),
        nn.LeakyReLU(inplace=True),
        nn.Conv2d(32, 3, (5, 5))
    )
    model = get_maskrcnn_model_instance(num_classes)
    # else:
    #     model = get_frcnn_model_instance(num_classes)

    # Move model to the right device
    model.to(device)
    # preproc_model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]  # + [p for p in preproc_model.parameters() if p.requires_grad]
    # optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)

    #  lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_steps, gamma=args.lr_gamma)

    if args.resume:
        print('RESUMING!')
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        # optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

    print("Start training")
    start_time = time.time()
    for epoch in range(args.epochs):
        train_one_epoch(model, preproc_model, optimizer, data_loader, device, epoch, args.print_freq)
        lr_scheduler.step()

        if args.output_dir:
            torch.save({
            'epoch': epoch,
            # 'pre_model': preproc_model.state_dict(),
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            },  os.path.join(args.output_dir, 'model_{}.pth'.format(epoch)))

        # evaluate after every epoch
        evaluate(model, preproc_model, data_loader_test, device=device)

    total_time = time.time() - start_time
    import datetime
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


# eventually run this on validation set after each epoch

def test(model):
    global minnie_test
    model.model.eval()  # put model in evaluation mode

    with torch.no_grad():
        loss_total = 0.0
        total_batches = len(minnie_test)
        tqdm_train = tqdm.tqdm(minnie_test, total=total_batches) #progress bar
        for batch, (X, Y) in enumerate(tqdm_train): #X is the image of the tree, Y is the human labeled mask to locate apples
            # move batch to specified device
            # channels first instead of channels last
            X = X.to(DEVICE).to(torch.float)
            Y = Y.to(DEVICE)

            # run predictions through model (run through U-Net AKA prediction mask)
            Y_pred = model(X)

            # calculate loss and get gradients
            # print(Y_pred.shape, Y.shape)
            # Tell us how much overlap between predicted and ground truth masks
            loss = model.loss_fn(Y_pred.view(Y.shape), Y)  # DICE score which is area of intersections of pixels divided by the union/ie area of union
            loss_total += float(loss.item())

            # report current progress
            last_dice = 1.0 - (loss_total / (batch + 1))
            tqdm_train.set_description('Test Dice: {}'.format(last_dice))
    
    print('Testing dice ', last_dice)
    return last_dice

def training_single_ensemble(number: int):
    global minnie_train

    # Create instance of model and move it to the GPU -- to cuda
    # model = UnetModel().to(DEVICE)
    model = FirstModel().to(DEVICE)
    
    # tells the model to be in training mode -- getting ready to train
    model.model.train()

    # optimizer (does gradient descent for us)
    # model.parameters() is a list of all parameters for all layers (the filters)
    optimizer = torch.optim.Adam(model.model.parameters(), LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    total_batches = len(minnie_train)
    print('Total batches', len(minnie_train))

    best_dice = 0.0
    try:
        # 1 epoch = 1 pass over entire training set
        for epoch in range(EPOCHS):
            print('Epoch', epoch)
            # train_one_epoch()

            loss_total = 0.0
            tqdm_train = tqdm.tqdm(minnie_train, total=total_batches) #progress bar
            for batch, (X, Y) in enumerate(tqdm_train): #X is the image of the tree, Y is the human labeled mask to locate apples
                # move batch to specified device
                # channels first instead of channels last
                X = X.to(DEVICE).to(torch.float)
                Y = Y.to(DEVICE)

                # run predictions through model (run through U-Net AKA prediction mask)
                Y_pred = model(X)

                # reset gradients to zero. Default function of pytorch is to accumulate gradients
                optimizer.zero_grad()

                # calculate loss and get gradients
                # print(Y_pred.shape, Y.shape)
                # Tell us how much overlap between predicted and ground truth masks
                # print('hello', Y_pred.shape, Y.shape)
                loss = model.loss_fn(Y_pred.view(Y.shape), Y)  # DICE score which is area of intersections of pixels divided by the union/ie area of union
                loss.backward()  # computes the gradients for our model
                loss_total += float(loss.item())

                # run optimizer
                optimizer.step()

                # report current progress
                tqdm_train.set_description('Dice: {}'.format(
                    1.0 - (loss_total / (batch + 1))))
            
            test_dice = test(model=model)
            if test_dice >= best_dice:
                print('New best (old)', best_dice, '(new)', test_dice)
                best_dice = test_dice
                torch.save({
                    'epoch': epoch,
                    'model': model.model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                },  'first_ensemble_lowest_{}.pth'.format(number))
    except Exception as err:
        print('Error!', str(err))
        traceback.print_exception(err)

    # finally clean up GPU mem
    model = None
    X = None
    Y = None
    del model
    del X, Y
    with torch.no_grad():
        torch.cuda.empty_cache()
    gc.collect()


# # %%
# training()


def train_ensemble():
    ENSEMBLE_SIZE = 8
    print('Training an ensemble of ', ENSEMBLE_SIZE)
    for ens in range(ENSEMBLE_SIZE):
        print('Training model ', ens + 1)
        training_single_ensemble(ens)
    print('Done!')


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch Detection Training')
    parser.add_argument('--data_path', default='/lgdata/agathon/Mask2Former/detection', help='dataset')
    parser.add_argument('--dataset', default='AppleDataset', help='dataset')
    parser.add_argument('--model', default='maskrcnn', help='model')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('-b', '--batch-size', default=5, type=int)
    parser.add_argument('--epochs', default=20, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 16)')
    parser.add_argument('--lr', default=1e-4, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)', dest='weight_decay')
    parser.add_argument('--lr-step-size', default=8, type=int, help='decrease lr every step-size epochs')
    parser.add_argument('--lr-steps', default=[8, 11], nargs='+', type=int, help='decrease lr every step-size epochs')
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--print-freq', default=20, type=int, help='print frequency')
    parser.add_argument('--output-dir', default='.', help='path where to save')
    parser.add_argument('--resume', default='', help='resume from checkpoint')

    args = parser.parse_args()
    print(args.model)
    # assert(args.model in ['mrcnn', 'frcnn'])

    if args.output_dir:
        utils.mkdir(args.output_dir)

    main(args)
    # train_ensemble()

# %%
