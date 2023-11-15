import torch
from training import FirstModel, load_minnie, transforms
import math
import matplotlib.pyplot as plot


padding = transforms.transforms.Pad((8, 0, 8, 0))
ENSEMBLE_SIZE = 7
DEVICE = 'cuda'


# sample some data
_, minnie_test = load_minnie(bs=1, first=True, train=False)
for (X, Y) in minnie_test:
  X = X.to(DEVICE)
  Y = Y.to(DEVICE)

  print(X.shape, Y.shape, X[0].shape, X[0].permute(1, 2, 0).shape, Y[0].shape)
  break


images = [('Image', X[0].permute(1, 2, 0).cpu().numpy()), ('GT', Y[0].cpu().numpy())]


model = FirstModel().to(DEVICE)
model.model.eval()
avg = None
for i in range(ENSEMBLE_SIZE):
  with torch.no_grad():
    model_data = torch.load('first_ensemble_low_{}.pth'.format(i))
    model.model.load_state_dict(model_data['model'])

    # make predictions
    Y_pred = model(X)

    if avg is None:
      avg = Y_pred.clone()
    else:
      avg += Y_pred.clone()

    print('{} Dice: {}'.format(i, 1.0 - model.loss_fn(Y_pred.view(Y.shape), Y)))

    images.append(('Ensemble {}'.format(i), Y_pred.cpu().numpy()))

candidates = (avg > 0.2)[0].cpu().numpy()
avg /= float(ENSEMBLE_SIZE)

images.append(('Candidates', candidates))
images.append(('Average', avg[0].cpu().numpy()))

square = math.ceil(math.sqrt(float(len(images))))
f, axarr = plot.subplots(square - 1, square, figsize=(10, 14))
for ind, (title, Y) in enumerate(images):
    row = int(ind / square)
    col = int(ind % square)
    Y = Y.reshape(1280, 720, -1)
    Y[Y < 0.2] = 0.0
    axarr[row, col].imshow(Y)
    axarr[row, col].set_title(title)
# plot.title('Ensemble predictions')
plot.tight_layout()
plot.savefig('ensemble-pred-1.png')