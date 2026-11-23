# %% [markdown]
# # VN Dish Classification — Người 1: data preprocessing
#
# Source of truth for both the local run and the Kaggle notebook `vndish-classification`.
# Edit this file, then rebuild the notebook with `python pipeline/build_notebook.py`.
# Run locally from the code repo root: `python pipeline/prep.py`. Data lives in the sibling repo
# `../VNDishClassification-data` (reads data/raw/, writes data/processed/); override with VNFOOD_DATA.
#
# **Input:** Kaggle dataset `meowluvmatcha/vnfood-30-100` (VNFood103, CC BY-NC-SA 4.0).
#
# **Steps (per N1.md)**
# 1. Clean images → `vnfood103-clean/` (`<label>/<hash16>.jpg` + `merge_manifest.csv`)
# 2. pHash dedup (Hamming ≤ 4, transitive), drop `conflict` groups, keep 1 per `dup` group,
#    split 70/15/15 per dish (seed 42) → `vnfood-prep/` (`splits.csv`, summaries)
#
# Thumbnails and colour / HOG / CLIP / SigLIP features come in a later section.

# %%
import os, csv, json, hashlib, time, shutil
from collections import Counter, defaultdict
import multiprocessing as mp

import numpy as np
from PIL import Image, ImageOps

ON_KAGGLE = os.path.isdir('/kaggle/input')
# Data lives in the separate repo ../VNDishClassification-data (override with VNFOOD_DATA).
DATA_REPO = os.environ.get('VNFOOD_DATA', os.path.abspath(os.path.join(os.getcwd(), '..', 'VNDishClassification-data')))


KAGGLE_DATASET = 'meowluvmatcha/vnfood-30-100'


def _search(base):
    for dp, dns, _ in os.walk(base):
        if 'vnfood_combined_dataset' in dns:
            return os.path.join(dp, 'vnfood_combined_dataset')
        if {'train', 'val', 'test'} <= set(dns):
            return dp
        depth = dp[len(base):].count(os.sep)
        dns[:] = [] if depth >= 6 else [d for d in dns if d not in ('processed', 'outputs', '.git')]
    return None


def find_src():
    # Locate the vnfood_combined_dataset folder: env var, attached Kaggle input,
    # kagglehub download (Kaggle, internet on), or the local project folder.
    env = os.environ.get('VNFOOD_SRC')
    if env:
        return env
    bases = ['/kaggle/input'] if ON_KAGGLE else [os.path.join(DATA_REPO, 'data', 'raw'), os.getcwd()]
    for base in bases:
        hit = _search(base)
        if hit:
            return hit
    if ON_KAGGLE:
        import kagglehub
        print('input not attached, downloading', KAGGLE_DATASET, 'with kagglehub ...')
        hit = _search(kagglehub.dataset_download(KAGGLE_DATASET))
        if hit:
            return hit
    for base in bases:   # not found: show what is mounted
        print('contents of', base)
        for dp, dns, fns in os.walk(base):
            depth = dp[len(base):].count(os.sep)
            print('  ' * depth + dp, len(fns), 'files')
            if depth >= 3:
                dns[:] = []
    raise FileNotFoundError('vnfood_combined_dataset not found; attach meowluvmatcha/vnfood-30-100 or set VNFOOD_SRC')


SRC = find_src()
WORK = os.environ.get('VNFOOD_WORK', '/kaggle/working' if ON_KAGGLE else os.path.join(DATA_REPO, 'data', 'processed'))
CLEAN = os.path.join(WORK, 'vnfood103-clean')
PREP = os.path.join(WORK, 'vnfood-prep')
N_WORKERS = int(os.environ.get('VNFOOD_WORKERS', os.cpu_count() or 4))

MAX_SHORT = 512          # resize so the short side is at most this (never upscale)
JPEG_Q = 90
GRAY_THR = 3.0           # mean |R-G|+|G-B| on a 64x64 thumbnail; <= this counts as gray
HAMMING = 4              # pHash near-duplicate threshold
SEED = 42
SPLIT_FRAC = (0.70, 0.15, 0.15)
EXPECTED_N = 49223

print('ON_KAGGLE', ON_KAGGLE, '| SRC', SRC, '| WORK', WORK, '| workers', N_WORKERS)


def pool():
    # 'fork' so functions defined in the notebook are visible to workers (Linux only).
    return mp.get_context('fork').Pool(N_WORKERS)

# %% [markdown]
# ## Step 1 — Clean images → `vnfood103-clean`
#
# - ignore the original train/val/test (kept as `orig_split` in the manifest), flatten by dish
# - drop unreadable images (`error: …`) and gray images (`gray`: gray mode, or RGB with ~equal channels)
# - normalise: EXIF rotation, RGB (alpha composited on white), short side ≤ 512 px, JPEG q90, no metadata
#   (some `.jpg` files are actually PNG / MPO / BMP — always re-encoded)
# - file name = first 16 hex of SHA-1 of the **original relative path** (unique, so identical
#   duplicate images never overwrite each other)

# %%
def source_of(fname):
    if fname.startswith('VNFoods_Original'):
        return '30vnfoods'
    if fname.startswith('100_VNFoods_Aug_'):
        return '100vnfoods'
    if fname.startswith('VinaFood21_Extended'):
        return 'vinafood21'
    return 'unknown'


def to_rgb(im):
    if im.mode in ('RGBA', 'LA', 'PA') or (im.mode == 'P' and 'transparency' in im.info):
        im = im.convert('RGBA')
        bg = Image.new('RGB', im.size, (255, 255, 255))
        bg.paste(im, mask=im.getchannel('A'))
        return bg
    return im.convert('RGB')


def is_gray(im_orig, rgb):
    if im_orig.mode in ('1', 'L', 'LA', 'I', 'I;16', 'F'):
        return True
    a = np.asarray(rgb.resize((64, 64)), dtype=np.int16)
    d = np.abs(a[..., 0] - a[..., 1]) + np.abs(a[..., 1] - a[..., 2])
    return float(d.mean()) <= GRAY_THR


def clean_one(orig_path):
    rel = os.path.relpath(orig_path, SRC).replace(os.sep, '/')   # train/pho/xxx.jpg
    split, label, fname = rel.split('/')
    row = dict(path='', label=label, orig_split=split, source=source_of(fname),
               orig_path=rel, status='ok', orig_w='', orig_h='', w='', h='')
    try:
        im = Image.open(orig_path)
        im.load()
        row['orig_w'], row['orig_h'] = im.size
        im = ImageOps.exif_transpose(im)
        rgb = to_rgb(im)
        if is_gray(im, rgb):
            row['status'] = 'gray'
            return row
        w, h = rgb.size
        s = min(w, h)
        if s > MAX_SHORT:
            r = MAX_SHORT / s
            rgb = rgb.resize((max(1, round(w * r)), max(1, round(h * r))), Image.LANCZOS)
        name = hashlib.sha1(rel.encode('utf-8')).hexdigest()[:16] + '.jpg'
        os.makedirs(os.path.join(CLEAN, label), exist_ok=True)
        rgb.save(os.path.join(CLEAN, label, name), 'JPEG', quality=JPEG_Q)   # no exif= -> metadata dropped
        row['path'] = label + '/' + name
        row['w'], row['h'] = rgb.size
    except Exception as e:
        row['status'] = ('error: %s: %s' % (type(e).__name__, e))[:200]
    return row


MANIFEST_COLS = ['path', 'label', 'orig_split', 'source', 'orig_path', 'status', 'orig_w', 'orig_h', 'w', 'h']

t0 = time.time()
orig_paths = sorted(os.path.join(dp, f) for dp, _, fs in os.walk(SRC) for f in fs
                    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')))
print('found', len(orig_paths), 'images')
assert len(orig_paths) == EXPECTED_N, len(orig_paths)

os.makedirs(CLEAN, exist_ok=True)
with pool() as p:
    manifest = p.map(clean_one, orig_paths, chunksize=128)

with open(os.path.join(CLEAN, 'merge_manifest.csv'), 'w', newline='', encoding='utf-8') as fh:
    wr = csv.DictWriter(fh, MANIFEST_COLS)
    wr.writeheader()
    wr.writerows(manifest)
print('step 1 done in %.0fs' % (time.time() - t0))

# %%
# Acceptance checks for vnfood103-clean (N1.md, section Điều kiện)
status = Counter('error' if r['status'].startswith('error') else r['status'] for r in manifest)
label_dirs = sorted(d for d in os.listdir(CLEAN) if os.path.isdir(os.path.join(CLEAN, d)))
n_files = sum(len(os.listdir(os.path.join(CLEAN, d))) for d in label_dirs)
orig_labels = sorted({r['label'] for r in manifest})

print('status:', dict(status))
print('gray by source:', dict(Counter(r['source'] for r in manifest if r['status'] == 'gray')))
print('errors:', [r['orig_path'] + ' ' + r['status'] for r in manifest if r['status'].startswith('error')][:10])
print('label dirs:', len(label_dirs), '| files:', n_files)

assert len(manifest) == EXPECTED_N
assert label_dirs == orig_labels and len(label_dirs) == 103, 'missing dish folders'
assert status['ok'] == n_files, (status['ok'], n_files)
assert all(r['path'] == '' for r in manifest if r['status'] != 'ok')
assert abs(status['gray'] - 571) <= 30, 'gray count far from 571 — review gray detection'
ok_rows = [r for r in manifest if r['status'] == 'ok']
assert max(min(int(r['w']), int(r['h'])) for r in ok_rows) <= MAX_SHORT
print('vnfood103-clean checks passed')

# %% [markdown]
# ## Step 2 — pHash dedup + split → `vnfood-prep`
#
# - pHash (DCT 32×32 → 8×8, median) on the cleaned images
# - near-duplicate groups: Hamming ≤ 4, transitive (union-find)
# - group with > 1 label → whole group `conflict`
# - same-label group → keep the image with most original pixels, rest `dup`
# - split 70/15/15 per dish, seed 42, by group (dups follow their group's split)

# %%
_N = 32
_k = np.arange(_N)[:, None]
_n = np.arange(_N)[None, :]
_DCT = 2 * np.cos(np.pi * _k * (2 * _n + 1) / (2 * _N))
_BITS = (1 << np.arange(63, -1, -1, dtype=np.uint64)).astype(np.uint64)


def phash_one(rel):
    try:
        im = Image.open(os.path.join(CLEAN, rel))
        g = np.asarray(im.convert('L').resize((_N, _N), Image.LANCZOS), dtype=np.float64)
        low = (_DCT @ g @ _DCT.T)[:8, :8].ravel()
        bits = (low > np.median(low)).astype(np.uint64)
        return int((bits * _BITS).sum())
    except Exception:
        return None


_LUT = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint8)


def popcount64(x):
    if hasattr(np, 'bitwise_count'):
        return np.bitwise_count(x)
    return _LUT[x.view(np.uint8)].reshape(x.shape + (8,)).sum(-1)


t0 = time.time()
rows = [dict(r) for r in manifest if r['status'] == 'ok']
with pool() as p:
    hashes = p.map(phash_one, [r['path'] for r in rows], chunksize=256)
for r, h in zip(rows, hashes):
    r['phash'] = '' if h is None else '%016x' % h
good = [i for i, h in enumerate(hashes) if h is not None]
print('pHash done in %.0fs, unreadable:' % (time.time() - t0), len(rows) - len(good))

# union-find over pairs with Hamming <= HAMMING
H = np.array([hashes[i] for i in good], dtype=np.uint64)
parent = np.arange(len(H))


def find(x):
    root = x
    while parent[root] != root:
        root = parent[root]
    while parent[x] != root:
        parent[x], x = root, parent[x]
    return root


t0 = time.time()
CH = 1024
n_pairs = 0
for s in range(0, len(H), CH):
    d = popcount64(H[s:s + CH, None] ^ H[None, :])
    ii, jj = np.nonzero(d <= HAMMING)
    for i, j in zip(ii + s, jj):
        if i < j:
            n_pairs += 1
            a, b = find(i), find(j)
            if a != b:
                parent[a] = b
print('pairs <= %d: %d (%.0fs)' % (HAMMING, n_pairs, time.time() - t0))

for r in rows:
    r['group'], r['reason'], r['split'] = '', '', ''
for gi, i in enumerate(good):
    rows[i]['group'] = int(find(gi))
for i, r in enumerate(rows):
    if r['phash'] == '':
        r['reason'] = 'bad'

members = defaultdict(list)
for i in good:
    members[rows[i]['group']].append(i)
# renumber groups 0..G-1 in a stable order (by first path)
order = sorted(members, key=lambda g: min(rows[i]['path'] for i in members[g]))
remap = {g: k for k, g in enumerate(order)}
members = {remap[g]: v for g, v in members.items()}
for i in good:
    rows[i]['group'] = remap[rows[i]['group']]

keepers = {}
for g, idx in members.items():
    if len({rows[i]['label'] for i in idx}) > 1:
        for i in idx:
            rows[i]['reason'] = 'conflict'
        continue
    best = max(idx, key=lambda i: (int(rows[i]['orig_w']) * int(rows[i]['orig_h']), rows[i]['path']))
    keepers[g] = best
    for i in idx:
        if i != best:
            rows[i]['reason'] = 'dup'

# split per dish by group, seed 42
rng = np.random.RandomState(SEED)
by_label = defaultdict(list)
for g, i in keepers.items():
    by_label[rows[i]['label']].append(g)
group_split = {}
for label in sorted(by_label):
    gs = sorted(by_label[label], key=lambda g: rows[keepers[g]]['path'])
    perm = rng.permutation(len(gs))
    n = len(gs)
    n_tr = int(round(SPLIT_FRAC[0] * n))
    n_va = int(round(SPLIT_FRAC[1] * n))
    for rank, pi in enumerate(perm):
        group_split[gs[pi]] = 'train' if rank < n_tr else ('val' if rank < n_tr + n_va else 'test')
for i in good:
    g = rows[i]['group']
    if g in group_split:
        rows[i]['split'] = group_split[g]

SPLIT_COLS = ['path', 'label', 'split', 'reason', 'group', 'phash', 'source', 'orig_split', 'orig_path', 'w', 'h']
os.makedirs(PREP, exist_ok=True)
with open(os.path.join(PREP, 'splits.csv'), 'w', newline='', encoding='utf-8') as fh:
    wr = csv.DictWriter(fh, SPLIT_COLS, extrasaction='ignore')
    wr.writeheader()
    wr.writerows(rows)
print('splits.csv:', len(rows), 'rows')

# %%
# Acceptance checks for vnfood-prep
kept = [r for r in rows if r['reason'] == '']
all_labels = sorted({r['label'] for r in manifest})
per = defaultdict(Counter)
for r in kept:
    per[r['label']][r['split']] += 1
missing = [l for l in all_labels if min(per[l][s] for s in ('train', 'val', 'test')) == 0]
assert not missing, ('dish missing a split', missing)

g_splits = defaultdict(set)
g_kept = Counter()
for r in rows:
    if r['split']:
        g_splits[r['group']].add(r['split'])
    if r['reason'] == '':
        g_kept[r['group']] += 1
assert all(len(v) == 1 for v in g_splits.values()), 'a pHash group spans two splits'
assert all(v == 1 for v in g_kept.values()), 'a group keeps more than one image'
print('vnfood-prep checks passed:', dict(Counter(r['split'] for r in kept)))

# %% [markdown]
# ## Summary tables

# %%
reasons = ['gray', 'error', 'bad', 'conflict', 'dup']


def reason_of(r):
    return 'error' if r['status'].startswith('error') else r['status']


full = []   # one row per original image with its final fate
prep_by_orig = {r['orig_path']: r for r in rows}
for m in manifest:
    p = prep_by_orig.get(m['orig_path'])
    fate = reason_of(m) if m['status'] != 'ok' else (p['reason'] or 'kept')
    full.append(dict(label=m['label'], source=m['source'], fate=fate, split=p['split'] if p else ''))


def table(key):
    t = defaultdict(Counter)
    for r in full:
        t[r[key]][r['fate']] += 1
        t[r[key]]['raw'] += 1
        if r['fate'] == 'kept':
            t[r[key]][r['split']] += 1
    out = []
    for k in sorted(t):
        c = t[k]
        lost = c['raw'] - c['kept']
        d = {key: k, 'raw': c['raw']}
        d.update({x: c[x] for x in reasons})
        d.update(kept=c['kept'], train=c['train'], val=c['val'], test=c['test'],
                 lost_pct=round(100.0 * lost / c['raw'], 1))
        if d['lost_pct'] > 30:
            main = max(reasons, key=lambda x: c[x])
            d['note'] = 'lost >30%%, mostly %s (%d)' % (main, c[main])
        else:
            d['note'] = ''
        out.append(d)
    return out


by_label_t = table('label')
by_source_t = table('source')
for name, t in [('summary_by_label.csv', by_label_t), ('summary_by_source.csv', by_source_t)]:
    with open(os.path.join(PREP, name), 'w', newline='', encoding='utf-8') as fh:
        wr = csv.DictWriter(fh, list(t[0].keys()))
        wr.writeheader()
        wr.writerows(t)

fate = Counter(r['fate'] for r in full)
steps = {
    'raw': len(manifest),
    'after_clean (ok)': status['ok'],
    'after_bad': status['ok'] - fate['bad'],
    'after_conflict': status['ok'] - fate['bad'] - fate['conflict'],
    'after_dedup (kept)': fate['kept'],
}
summary = dict(
    steps=steps,
    removed=dict((x, fate[x]) for x in reasons),
    splits=dict(Counter(r['split'] for r in full if r['fate'] == 'kept')),
    n_groups=len(members),
    n_multi_groups=sum(1 for v in members.values() if len(v) > 1),
    n_conflict_groups=sum(1 for g, v in members.items() if g not in keepers),
    conflict_label_sets=Counter(
        ' | '.join(sorted({rows[i]['label'] for i in v})) for g, v in members.items() if g not in keepers
    ).most_common(),
    dishes_lost_over_30pct=[(d['label'], d['lost_pct'], d['note']) for d in by_label_t if d['lost_pct'] > 30],
    params=dict(MAX_SHORT=MAX_SHORT, JPEG_Q=JPEG_Q, GRAY_THR=GRAY_THR, HAMMING=HAMMING,
                SEED=SEED, SPLIT_FRAC=SPLIT_FRAC),
)
with open(os.path.join(PREP, 'prep_summary.json'), 'w', encoding='utf-8') as fh:
    json.dump(summary, fh, ensure_ascii=False, indent=2)
shutil.copyfile(os.path.join(CLEAN, 'merge_manifest.csv'), os.path.join(PREP, 'merge_manifest.csv'))

print(json.dumps({k: summary[k] for k in ['steps', 'removed', 'splits', 'n_groups',
                                          'n_multi_groups', 'n_conflict_groups']}, indent=2))
print('top conflict label sets:', summary['conflict_label_sets'][:10])
print('by source:')
for d in by_source_t:
    print(' ', d)
print('dishes lost >30%%: %d' % len(summary['dishes_lost_over_30pct']))
for x in summary['dishes_lost_over_30pct']:
    print(' ', x)
