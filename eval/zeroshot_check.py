"""Zero-shot check of classes.json text fields against image features (test split, kept rows)."""
import csv
import json
import os
import sys

import numpy as np
import open_clip
import torch

# default: <data repo>/data/processed/vnfood-prep locally (VNFOOD_DATA or ../VNDishClassification-data),
# or ./vnfood-prep on urmom; outputs are written next to the data
_data = os.environ.get('VNFOOD_DATA', os.path.join('..', 'VNDishClassification-data'))
_default = os.path.join(_data, 'data', 'processed', 'vnfood-prep')
prep = sys.argv[1] if len(sys.argv) > 1 else (_default if os.path.isdir(_default) else 'vnfood-prep')
classes = json.load(open(os.path.join(prep, 'classes.json'), encoding='utf-8'))
rows = list(csv.DictReader(open(os.path.join(prep, 'splits.csv'), encoding='utf-8')))
lab2id = {c['label']: c['id'] for c in classes}
te = np.array([i for i, r in enumerate(rows) if r['reason'] == '' and r['split'] == 'test'])
y = np.array([lab2id[rows[i]['label']] for i in te])
dev = 'cuda'

MODELS = {'feat_siglip': ('ViT-SO400M-14-SigLIP2', 'webli'), 'feat_clip': ('ViT-L-14-quickgelu', 'openai')}
PROMPTS = {
    'name_vi': lambda c: c['name_vi'],
    'name_en': lambda c: 'a photo of %s, a Vietnamese dish' % c['name_en'],
    'desc': lambda c: 'a photo of food: %s' % c['desc'],
    'desc_vi': lambda c: c['desc_vi'],
    'name_en+desc': lambda c: 'a photo of %s: %s' % (c['name_en'], c['desc']),
}
per_class = {}
for feat, (name, pre) in MODELS.items():
    model, _, _ = open_clip.create_model_and_transforms(name, pretrained=pre)
    model = model.eval().to(dev)
    tok = open_clip.get_tokenizer(name)
    X = torch.tensor(np.load(os.path.join(prep, feat + '.npy'))[te], device=dev)
    for pname, f in PROMPTS.items():
        with torch.no_grad(), torch.autocast(dev, dtype=torch.float16):
            T = model.encode_text(tok([f(c) for c in classes]).to(dev)).float()
        T = torch.nn.functional.normalize(T, dim=-1)
        pred = (X @ T.T).argmax(1).cpu().numpy()
        acc = (pred == y).mean()
        print('%-12s %-13s top1 %.3f' % (feat, pname, acc), flush=True)
        if feat == 'feat_siglip' and pname == 'desc':
            for c in classes:
                m = y == c['id']
                per_class[c['label']] = round(float((pred[m] == c['id']).mean()), 3)
    del model
    torch.cuda.empty_cache()
worst = sorted(per_class.items(), key=lambda kv: kv[1])[:15]
print('worst classes (siglip, desc):', worst)
json.dump(per_class, open(os.path.join(prep, 'zeroshot_desc_per_class.json'), 'w'), indent=1)
