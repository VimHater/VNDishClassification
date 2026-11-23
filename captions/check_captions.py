"""Check a filled captions file against N1.md rules; optionally export the final captions.csv.

Usage:
  python captions/check_captions.py captions/captions_filled.csv                          # report problems
  python captions/check_captions.py <sheet-export>.csv --export captions/captions.csv   # final file
Errors (must fix): missing text/author, wrong counts or split balance, duplicate id, word count out of range.
Warnings (review): a word from _avoid appears, duplicate texts.
"""
import argparse
import csv
import re
from collections import Counter, defaultdict

REQUIRED = ['id', 'label', 'author', 'split', 'text_vi', 'text_en']
RANGE = {'short': (5, 10), 'long': (15, 30)}


def n_words(s):
    return len(re.findall(r'\w+', s))


def has_phrase(text, phrase):
    return re.search(r'(?<!\w)' + re.escape(phrase.lower()) + r'(?!\w)', text.lower()) is not None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('path')
    ap.add_argument('--export', default='')
    a = ap.parse_args()
    rows = [r for r in csv.DictReader(open(a.path, encoding='utf-8-sig')) if (r.get('id') or '').strip()]
    errors, warns = [], []

    if len(rows) != 412:
        errors.append('expected 412 rows, got %d' % len(rows))
    for k, v in Counter(r['id'] for r in rows).items():
        if v > 1:
            errors.append('duplicate id %s' % k)
    per = defaultdict(Counter)
    for r in rows:
        per[r['label']][(r['split'], r['_length'])] += 1
    if len(per) != 103:
        errors.append('expected 103 dishes, got %d' % len(per))
    for lab, c in per.items():
        for key in [('val', 'short'), ('val', 'long'), ('test', 'short'), ('test', 'long')]:
            if c[key] != 1:
                errors.append('%s: needs exactly 1 %s %s caption, has %d' % (lab, key[1], key[0], c[key]))

    filled = 0
    for r in rows:
        vi, en = r['text_vi'].strip(), r['text_en'].strip()
        if not vi and not en:
            continue
        filled += 1
        tag = '%s (%s)' % (r['id'], r['label'])
        if not vi or not en:
            errors.append('%s: text_vi and text_en must both be filled' % tag)
        if not r['author'].strip():
            errors.append('%s: author missing' % tag)
        lo, hi = RANGE[r['_length']]
        nw = n_words(vi)
        if vi and not lo <= nw <= hi:
            errors.append('%s: %s caption has %d words in text_vi (need %d-%d)' % (tag, r['_length'], nw, lo, hi))
        for p in [x.strip() for x in r['_avoid'].split(';') if x.strip()]:
            if has_phrase(vi, p):
                warns.append('%s: text_vi contains avoided word "%s"' % (tag, p))
    for t, v in Counter(r['text_vi'].strip().lower() for r in rows if r['text_vi'].strip()).items():
        if v > 1:
            warns.append('duplicate text_vi used %d times: %s' % (v, t[:60]))

    print('filled %d / %d captions' % (filled, len(rows)))
    print('authors:', dict(Counter(r['author'].strip() for r in rows if r['author'].strip())))
    for e in errors:
        print('ERROR  ', e)
    for w in warns:
        print('WARN   ', w)
    print('%d errors, %d warnings' % (len(errors), len(warns)))

    if a.export:
        if errors or filled < len(rows):
            raise SystemExit('not exporting: fix errors and fill all captions first')
        with open(a.export, 'w', newline='', encoding='utf-8') as fh:
            w = csv.DictWriter(fh, REQUIRED, extrasaction='ignore')
            w.writeheader()
            w.writerows(rows)
        print('exported', a.export)


if __name__ == '__main__':
    main()
