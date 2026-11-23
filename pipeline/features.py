"""Step 2b (N1.md): thumbnails + colour / HOG / CLIP / SigLIP features.

Reads  <root>/vnfood-prep/splits.csv and images from <root>/vnfood103-clean/.
Writes <root>/vnfood-prep/{images.npy, feat_color.npy, feat_hog.npy, feat_clip.npy,
feat_siglip.npy, prep_manifest.json}. Row i of every .npy == row i of splits.csv.

Usage:
    python pipeline/features.py --root ../VNDishClassification-data/data/processed [--stage cpu|clip|siglip|all] [--workers N]
    (on urmom: --root C:\\Users\\khoanguyen\\vnfood)

Works on Linux and Windows (spawn-safe). GPU stages need torch + open_clip.
"""
import argparse
import csv
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
from PIL import Image

THUMB = 224                 # images.npy: resize short side -> 224, centre crop 224x224
HSV_BINS = (16, 4, 4)       # feat_color: HSV histogram, L1-normalised -> 256 dims
HOG_SIZE = 128              # feat_hog: grayscale 128x128 ...
HOG_PPC = 16                # ... 16x16 cells, 2x2 blocks, 9 orientations -> 1764 dims
CLIP_MODEL = ('ViT-L-14-quickgelu', 'openai')
SIGLIP_MODEL = ('ViT-SO400M-14-SigLIP2', 'webli')

CLEAN_DIR = None            # set in worker initialiser


def _init(clean_dir):
    global CLEAN_DIR
    CLEAN_DIR = clean_dir


def resize_crop(im, size):
    w, h = im.size
    s = size / min(w, h)
    nw, nh = max(size, round(w * s)), max(size, round(h * s))
    im = im.resize((nw, nh), Image.BICUBIC)
    left, top = (nw - size) // 2, (nh - size) // 2
    return im.crop((left, top, left + size, top + size))


def cpu_one(args):
    from skimage.feature import hog
    i, rel = args
    im = Image.open(os.path.join(CLEAN_DIR, rel)).convert('RGB')
    thumb = resize_crop(im, THUMB)
    hsv = np.asarray(thumb.convert('HSV')).reshape(-1, 3)
    hist, _ = np.histogramdd(hsv, bins=HSV_BINS, range=((0, 256),) * 3)
    color = (hist.ravel() / hist.sum()).astype(np.float32)
    gray = np.asarray(thumb.convert('L').resize((HOG_SIZE, HOG_SIZE), Image.BICUBIC), dtype=np.float32) / 255.0
    h = hog(gray, orientations=9, pixels_per_cell=(HOG_PPC, HOG_PPC), cells_per_block=(2, 2),
            block_norm='L2-Hys', feature_vector=True).astype(np.float32)
    return i, np.asarray(thumb, dtype=np.uint8), color, h


def stage_cpu(paths, clean_dir, prep, workers):
    n = len(paths)
    images = np.lib.format.open_memmap(os.path.join(prep, 'images.npy'), mode='w+',
                                       dtype=np.uint8, shape=(n, THUMB, THUMB, 3))
    color = np.zeros((n, int(np.prod(HSV_BINS))), np.float32)
    hogs = None
    t0 = time.time()
    with Pool(workers, initializer=_init, initargs=(clean_dir,)) as pool:
        for k, (i, th, c, h) in enumerate(pool.imap_unordered(cpu_one, list(enumerate(paths)), chunksize=64)):
            if hogs is None:
                hogs = np.zeros((n, h.shape[0]), np.float32)
            images[i] = th
            color[i] = c
            hogs[i] = h
            if (k + 1) % 5000 == 0:
                print('  cpu %d/%d  %.0fs' % (k + 1, n, time.time() - t0), flush=True)
    images.flush()
    del images
    np.save(os.path.join(prep, 'feat_color.npy'), color)
    np.save(os.path.join(prep, 'feat_hog.npy'), hogs)
    print('cpu stage done in %.0fs: color %s, hog %s' % (time.time() - t0, color.shape, hogs.shape), flush=True)
    return dict(images=dict(shape=[n, THUMB, THUMB, 3], dtype='uint8',
                            method='resize short side to %d (bicubic) + centre crop' % THUMB),
                feat_color=dict(shape=list(color.shape), method='HSV histogram %s bins, L1-normalised, on thumbnail' % (HSV_BINS,)),
                feat_hog=dict(shape=list(hogs.shape), method='HOG on gray %dx%d, %dpx cells, 2x2 blocks, 9 orient, L2-Hys'
                              % (HOG_SIZE, HOG_SIZE, HOG_PPC)))


class ImgDS:
    def __init__(self, paths, clean_dir, preprocess):
        self.paths, self.clean_dir, self.preprocess = paths, clean_dir, preprocess

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.preprocess(Image.open(os.path.join(self.clean_dir, self.paths[i])).convert('RGB'))


def stage_model(paths, clean_dir, prep, workers, name, pretrained, out_name, batch):
    import torch
    import open_clip
    from torch.utils.data import DataLoader

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    t0 = time.time()
    model, _, preprocess = open_clip.create_model_and_transforms(name, pretrained=pretrained)
    model = model.eval().to(dev)
    print('%s/%s loaded on %s in %.0fs' % (name, pretrained, dev, time.time() - t0), flush=True)

    dl = DataLoader(ImgDS(paths, clean_dir, preprocess), batch_size=batch, shuffle=False,
                    num_workers=workers, pin_memory=(dev == 'cuda'), persistent_workers=workers > 0)
    feats = []
    t0 = time.time()
    with torch.no_grad(), torch.autocast(dev, dtype=torch.float16, enabled=(dev == 'cuda')):
        for b, x in enumerate(dl):
            f = model.encode_image(x.to(dev, non_blocking=True))
            f = torch.nn.functional.normalize(f.float(), dim=-1)
            feats.append(f.cpu().numpy())
            if (b + 1) % 50 == 0:
                print('  %s %d/%d  %.0fs' % (out_name, (b + 1) * batch, len(paths), time.time() - t0), flush=True)
    feats = np.concatenate(feats).astype(np.float32)
    np.save(os.path.join(prep, out_name + '.npy'), feats)
    print('%s done in %.0fs: %s' % (out_name, time.time() - t0, feats.shape), flush=True)
    size = getattr(model.visual, 'image_size', None)
    return {out_name: dict(shape=list(feats.shape), dtype='float32', model=name, pretrained=pretrained,
                           input_size=list(size) if isinstance(size, (tuple, list)) else size,
                           source_images='vnfood103-clean (short side <= 512)', l2_normalised=True,
                           open_clip=open_clip.__version__, torch=torch.__version__)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True, help='folder containing vnfood103-clean/ and vnfood-prep/')
    ap.add_argument('--stage', default='all', choices=['cpu', 'clip', 'siglip', 'all'])
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument('--batch', type=int, default=128)
    ap.add_argument('--limit', type=int, default=0, help='smoke test: only the first N rows')
    ap.add_argument('--out', default='', help='output folder (default: <root>/vnfood-prep)')
    a = ap.parse_args()

    clean_dir = os.path.join(a.root, 'vnfood103-clean')
    with open(os.path.join(a.root, 'vnfood-prep', 'splits.csv'), encoding='utf-8') as fh:
        paths = [r['path'] for r in csv.DictReader(fh)]
    if a.limit:
        paths = paths[:a.limit]
    prep = a.out or os.path.join(a.root, 'vnfood-prep')
    os.makedirs(prep, exist_ok=True)
    print('rows in splits.csv:', len(paths), '| workers', a.workers, flush=True)

    man_path = os.path.join(prep, 'prep_manifest.json')
    manifest = {}
    if os.path.exists(man_path):
        with open(man_path, encoding='utf-8') as fh:
            manifest = json.load(fh)
    manifest.setdefault('files', {})
    manifest['rows'] = len(paths)
    manifest['row_order'] = 'same as splits.csv (all rows, including dup/conflict)'

    stages = ['cpu', 'clip', 'siglip'] if a.stage == 'all' else [a.stage]
    for st in stages:
        if st == 'cpu':
            info = stage_cpu(paths, clean_dir, prep, a.workers)
        elif st == 'clip':
            info = stage_model(paths, clean_dir, prep, min(a.workers, 8), *CLIP_MODEL, 'feat_clip', a.batch)
        else:
            info = stage_model(paths, clean_dir, prep, min(a.workers, 8), *SIGLIP_MODEL, 'feat_siglip', a.batch)
        manifest['files'].update(info)
        manifest['updated'] = time.strftime('%Y-%m-%d %H:%M:%S')
        with open(man_path, 'w', encoding='utf-8') as fh:
            json.dump(manifest, fh, ensure_ascii=False, indent=2)

    # checks: every .npy has len(splits) rows, finite values
    for f in sorted(os.listdir(prep)):
        if f.endswith('.npy'):
            arr = np.load(os.path.join(prep, f), mmap_mode='r')
            ok = arr.shape[0] == len(paths)
            finite = True if arr.dtype == np.uint8 else bool(np.isfinite(arr[:2000]).all())
            print('check %-16s %-22s rows_ok=%s finite=%s' % (f, arr.shape, ok, finite), flush=True)
            if not (ok and finite):
                sys.exit('check failed for ' + f)
    print('all done', flush=True)


if __name__ == '__main__':
    main()
