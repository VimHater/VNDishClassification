"""Build notebooks/vndishclassification.ipynb from pipeline/prep.py (`# %%` cell markers). Run from project root."""
import json, re, sys

src = sys.argv[1] if len(sys.argv) > 1 else 'pipeline/prep.py'
dst = sys.argv[2] if len(sys.argv) > 2 else 'notebooks/vndishclassification.ipynb'

text = open(src, encoding='utf-8').read()
cells = []
for chunk in re.split(r'^# %%', text, flags=re.M)[1:]:
    header, _, body = chunk.partition('\n')
    body = body.strip('\n')
    if header.strip() == '[markdown]':
        md = '\n'.join(l[2:] if l.startswith('# ') else l.lstrip('#') for l in body.splitlines())
        cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': md})
    else:
        cells.append({'cell_type': 'code', 'metadata': {}, 'execution_count': None,
                      'outputs': [], 'source': body})

nb = {
    'nbformat': 4, 'nbformat_minor': 4,
    'metadata': {
        'kernelspec': {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'},
        'language_info': {'name': 'python'},
    },
    'cells': cells,
}
json.dump(nb, open(dst, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('wrote', dst, len(cells), 'cells')
