import torch
from training import FirstModel, load_minnie, transforms, MINNIE_IMLIST, MINNIE_MSLIST, MinnieAugDataset
from copy import deepcopy
import math
import matplotlib.pyplot as plot
import numpy as np
import cv2
import os
import tqdm

padding = transforms.transforms.Pad((8, 0, 8, 0))
ENSEMBLE_SIZE = 6
DEVICE = 'cuda'


# sample some data
minnie_train, minnie_test = load_minnie(bs=5, first=True, train=False)
model = FirstModel().to(DEVICE)
model.model.eval()
avg = None
models_high = []  # these were trained to about 0.82 dice score
models_low = []   # these were trained to about 0.47 dice score (roughly half)

for i in range(ENSEMBLE_SIZE):
  with torch.no_grad():
    model_data = torch.load('first_ensemble_{}.pth'.format(i))
    model.model.load_state_dict(model_data['model'])
    models_high.append(deepcopy(model))

    model_data = torch.load('first_ensemble_lowest_{}.pth'.format(i))
    model.model.load_state_dict(model_data['model'])
    models_low.append(deepcopy(model))


minnie_train = MinnieAugDataset(MINNIE_IMLIST, MINNIE_MSLIST, train=False)
base_folder = '/lgdata/agathon/Mask2Former/detection/train/segs'

with torch.no_grad():
  for i in tqdm.tqdm(range(len(minnie_train)), total=len(minnie_train)):
    X, Y = minnie_train[i]
    im_name = os.path.basename(minnie_train._images[i])
    X = torch.as_tensor(X).to(DEVICE)
    Y = torch.as_tensor(Y).to(DEVICE)

    # make the candidate list
    num_models = float(len(models_high))
    candidates_high = torch.zeros_like(Y)
    for m in models_high:
      Y_pred = m(X)
      Y_pred[Y_pred <= 0.2] = 0.0
      candidates_high = torch.clamp(candidates_high + Y_pred, 0.0, 1.0)

    # now save the candidates
    candidates_high = candidates_high.cpu().numpy().reshape((1280, 720))
    
    candidates_low = torch.zeros_like(Y)
    for m in models_low:
      Y_pred = m(X)
      Y_pred[Y_pred <= 0.2] = 0.0
      candidates_low = torch.clamp(candidates_low + Y_pred, 0.0, 1.0)

    # now save the candidates
    candidates_low = candidates_low.cpu().numpy().reshape((1280, 720))

    # get the "corrected" candidates list
    candidates = np.clip(255.0 * (candidates_high + (0.5 * candidates_low)), a_min=0, a_max=255).astype(np.uint8)
    cv2.imwrite(os.path.join(base_folder, im_name), candidates)
