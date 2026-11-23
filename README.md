# VNDishClassification

Data preparation for a Vietnamese dish classification / text-to-image search project
(103 dishes, Kaggle dataset [`meowluvmatcha/vnfood-30-100`](https://www.kaggle.com/datasets/meowluvmatcha/vnfood-30-100), CC BY-NC-SA 4.0).
The task description is in [`docs/N1.md`](docs/N1.md).

This repo (`VNDishClassification-src`) holds the **code**. Data and the summary report live in the separate repo
**`VNDishClassification-data`**, checked out next to this one:

```
VNDishClassification/
├── VNDishClassification-src/    code (this repo)
└── VNDishClassification-data/   data/ (raw + processed, not in git) and report/
```

Scripts find it at `../VNDishClassification-data`; set `VNFOOD_DATA=/path/to/VNDishClassification-data` to use another location.

## Layout

```
docs/        task description (N1.md) and the EDA question list
pipeline/    prep.py (clean + dedup + split), features.py (thumbnails, colour, HOG, CLIP, SigLIP2), build_notebook.py
notebooks/   vndishclassification.ipynb, generated from pipeline/prep.py for Kaggle
eval/        check_features.py (kNN sanity check), zeroshot_check.py (classes.json text vs images)
captions/    412 search captions: template, draft, checker, Google Sheet builder
```

## Run

All commands run from the root of this repo.

```bash
# 1–2. clean images, pHash dedup, 70/15/15 split  (~4 min on 14 cores; needs pillow, numpy)
python pipeline/prep.py

# 2b. thumbnails + features (CLIP/SigLIP need a GPU; we run this on the "urmom" PC)
python pipeline/features.py --root ../VNDishClassification-data/data/processed

# captions: check a filled sheet export, then write the final file
python captions/check_captions.py <export>.csv --export captions/captions.csv
```

Rebuild the Kaggle notebook after editing `pipeline/prep.py`: `python pipeline/build_notebook.py`.

## Results (current run)

- 49,223 raw images → 48,655 after cleaning (568 gray removed) → **40,573 kept** after dedup
- split: train 28,400 · val 6,081 · test 6,092; every dish in every split, no duplicate group across splits
- features: `images.npy` (224×224), colour 256-d, HOG 1764-d, CLIP ViT-L/14 768-d, SigLIP2 SO400M 1152-d
- kNN (train→test) accuracy: SigLIP2 90.1%, CLIP 83.7%

Full numbers: `report/DATA_SUMMARY.md` in `VNDishClassification-data`.

## Status

- [x] cleaned images, dedup, split, features, `classes.json` (draft, needs native review)
- [x] 412 captions drafted by a model (`author = claude-opus-5-5`); team replaces them in the Google Sheet
- [ ] publish datasets `vnfood103-clean`, `vnfood-prep`, `vnfood-captions`
- [ ] Kaggle notebook `vndish-classification`: re-attach the input dataset and run

## License

Code: [MIT](LICENSE). Data: not included in this repository; the images and everything derived from them are
CC BY-NC-SA 4.0 (non-commercial, share-alike) — see `data/DISCLAIMER.md` in `VNDishClassification-data`.
