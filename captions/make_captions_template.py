"""Build captions_template.csv: 103 dishes x 4 captions (2 short + 2 long), split val/test per dish (seed 42).

Required columns (N1.md): id, label, author, split, text_vi, text_en.
Helper columns start with '_' (dropped by check_captions.py --export):
  _length      short (5-10 words) or long (15-30 words)
  _name_vi     dish name, for the writer's reference only
  _look        visible description from classes.json (inspiration, do not copy verbatim)
  _avoid       words that must NOT appear (dish name words, main ingredient in name, place names)
  _confusable  look-alike dishes: the caption should mention a visible difference
Usage (from project root): python captions/make_captions_template.py
  WARNING: overwrites captions/captions_template.csv (empty template, no captions).
"""
import csv
import json
import os
import sys

import numpy as np

GENERIC = {'bánh', 'bún', 'cơm', 'mì', 'xôi', 'canh', 'cháo', 'nước', 'thịt', 'món', 'với', 'và', 'trứng', 'nấu', 'lá'}
PLACES = ['Huế', 'Quảng', 'Nha Trang', 'Hồ Tây', 'Lã Vọng', 'Hà Nội', 'Hải Phòng', 'Sài Gòn', 'Phú Quốc',
          'Sóc Trăng', 'Hội An', 'Tây Bắc', 'Sapa', 'Sa Pa', 'Lạng Sơn', 'Nam Định', 'Bắc', 'Nam', 'Trung']
# main ingredients that are part of the dish name (N1.md: do not mention, e.g. "lá lốt" for bò lá lốt)
EXTRA_AVOID = {
    'bo-la-lot': ['lá lốt'], 'bo-nuong-la-lot': ['lá lốt'], 'banh-mi-chao': ['bánh mì', 'chảo'],
    'bun-dau-mam-tom': ['mắm tôm'], 'bun-mam': ['mắm'], 'bun-rieu': ['riêu'], 'cha-ca-la-vong': ['chả cá'],
    'banh-da-cua': ['bánh đa'], 'com-chay-cha-bong': ['cơm cháy', 'chà bông', 'ruốc'], 'goi-cuon': ['gỏi cuốn', 'cuốn'],
    'luon-om-chuoi-dau': ['lươn'], 'luon-xao-xa-ot': ['lươn'], 'baba-nau-chuoi-dau': ['ba ba', 'baba'],
    'trung-vit-lon': ['hột vịt lộn', 'lộn'], 'cut-lon-xao-me': ['cút lộn'], 'xoi-gac': ['gấc'],
    'tau-hu-non': ['tàu hũ', 'đậu hũ', 'tào phớ'], 'tau-hu-nhoi-thit': ['đậu hũ', 'tàu hũ', 'nhồi'],
    'muc-nhoi-thit': ['mực', 'nhồi'], 'canh-kho-hoa': ['khổ qua', 'mướp đắng', 'nhồi'],
    'ca-muoi-xoi': ['cà pháo', 'muối xổi'], 'chao-vit': ['chao', 'vịt'], 'goi-ca-chich': ['cá trích', 'gỏi'],
}

# visually similar pairs not (or barely) caught by shared duplicates; from zero-shot errors and dish knowledge
_PAIRS = [('cha-lui', 'nem-nuong-nha-trang'), ('banh-mi-chao', 'bo-ne'), ('cao-lau', 'mi-quang'),
          ('banh-can', 'banh-khot'), ('banh-chung', 'banh-tet'), ('hu-tieu', 'banh-canh'), ('pho', 'hu-tieu'),
          ('banh-bo', 'banh-da-lon'), ('xoi-gac', 'xoi-xeo'), ('oc-huong-xao', 'oc-len-xao-dua')]
EXTRA_CONF = {}
for _a, _b in _PAIRS:
    EXTRA_CONF.setdefault(_a, []).append(_b)
    EXTRA_CONF.setdefault(_b, []).append(_a)


def main(classes_path, out_path):
    classes = json.load(open(classes_path, encoding='utf-8'))
    rng = np.random.RandomState(42)
    rows = []
    for c in classes:
        words = [w for w in c['name_vi'].replace('(', ' ').replace(')', ' ').replace('/', ' ').split()
                 if w.lower() not in GENERIC and len(w) > 1]
        avoid = [c['name_vi']] + sorted(set(w.lower() for w in words)) + EXTRA_AVOID.get(c['label'], [])
        avoid += [p for p in PLACES if p.lower() in c['name_vi'].lower()]
        # look-alikes: dishes sharing >= 2 duplicate images (1 shared image is usually noise) + manual pairs
        conf = [k for k, v in c.get('shares_images_with', {}).items() if v >= 2]
        for other in EXTRA_CONF.get(c['label'], []):
            if other not in conf:
                conf.append(other)
        note = c.get('note', '')
        if 'near-duplicate of' in note:
            other = note.split('near-duplicate of ')[1].split(';')[0].split(' ')[0]
            if other not in conf:
                conf.insert(0, other)
        for length in ('short', 'long'):
            splits = ['val', 'test']
            if rng.rand() < 0.5:
                splits.reverse()
            for k, sp in enumerate(splits, 1):
                rows.append(dict(
                    id='c%03d_%s%d' % (c['id'], length[0], k), label=c['label'], author='', split=sp,
                    text_vi='', text_en='', _length=length, _name_vi=c['name_vi'], _look=c['desc_vi'],
                    _avoid='; '.join(dict.fromkeys(avoid)), _confusable=', '.join(conf),
                ))
    cols = ['id', 'label', 'author', 'split', 'text_vi', 'text_en', '_length', '_name_vi', '_look', '_avoid', '_confusable']
    with open(out_path, 'w', newline='', encoding='utf-8-sig') as fh:   # utf-8-sig: opens correctly in Excel
        w = csv.DictWriter(fh, cols)
        w.writeheader()
        w.writerows(rows)
    print('wrote', out_path, len(rows), 'rows')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.environ.get('VNFOOD_DATA', os.path.join('..', 'VNDishClassification-data')),
                                                     'data', 'processed', 'vnfood-prep', 'classes.json'),
         sys.argv[2] if len(sys.argv) > 2 else 'captions/captions_template.csv')
