"""Sanity check for vnfood-prep features: kNN accuracy (train -> test) per feature + thumbnail montage."""
import csv
import os
import sys

import numpy as np
import torch
from PIL import Image

# default: <data repo>/data/processed/vnfood-prep locally (VNFOOD_DATA or ../VNDishClassification-data),
# or ./vnfood-prep on urmom; outputs are written next to the data
_data = os.environ.get('VNFOOD_DATA', os.path.join('..', 'VNDishClassification-data'))
_default = os.path.join(_data, 'data', 'processed', 'vnfood-prep')
prep = sys.argv[1] if len(sys.argv) > 1 else (_default if os.path.isdir(_default) else 'vnfood-prep')
rows = list(csv.DictReader(open(os.path.join(prep, 'splits.csv'), encoding='utf-8')))
labels = sorted({r['label'] for r in rows})
y = np.array([labels.index(r['label']) for r in rows])
kept = np.array([r['reason'] == '' for r in rows])
tr = np.where(kept & np.array([r['split'] == 'train' for r in rows]))[0]
te = np.where(kept & np.array([r['split'] == 'test' for r in rows]))[0]
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
print('train', len(tr), 'test', len(te), 'classes', len(labels))


def knn_acc(X, k=10, metric='cos'):
    A = torch.tensor(X[tr], device=dev, dtype=torch.float32)
    B = torch.tensor(X[te], device=dev, dtype=torch.float32)
    if metric == 'cos':
        A = torch.nn.functional.normalize(A, dim=1)
        B = torch.nn.functional.normalize(B, dim=1)
    ytr = torch.tensor(y[tr], device=dev)
    correct1 = correctk = 0
    for s in range(0, len(B), 1024):
        b = B[s:s + 1024]
        sim = b @ A.T if metric == 'cos' else -torch.cdist(b, A)
        idx = sim.topk(k, dim=1).indices
        nb = ytr[idx]
        correct1 += (nb[:, 0].cpu().numpy() == y[te][s:s + 1024]).sum()
        vote = torch.mode(nb, dim=1).values.cpu().numpy()
        correctk += (vote == y[te][s:s + 1024]).sum()
    return correct1 / len(te), correctk / len(te)


for name in ['feat_color', 'feat_hog', 'feat_clip', 'feat_siglip']:
    X = np.load(os.path.join(prep, name + '.npy'))
    if name == 'feat_color':
        X = np.sqrt(X)                      # Hellinger for histograms
    a1, ak = knn_acc(X)
    print('%-12s dim %5d  1-NN %.3f  10-NN %.3f' % (name, X.shape[1], a1, ak))

imgs = np.load(os.path.join(prep, 'images.npy'), mmap_mode='r')
rng = np.random.RandomState(0)
pick = rng.choice(len(rows), 48, replace=False)
grid = np.zeros((6 * 224, 8 * 224, 3), np.uint8)
for k, i in enumerate(pick):
    r, c = divmod(k, 8)
    grid[r * 224:(r + 1) * 224, c * 224:(c + 1) * 224] = imgs[i]
Image.fromarray(grid).resize((8 * 112, 6 * 112)).save(os.path.join(prep, 'montage.jpg'), quality=85)
print('montage labels:', [rows[i]['label'] for i in pick])
