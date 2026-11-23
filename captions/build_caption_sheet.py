"""Build captions_sheet.xlsx (for Google Sheets) with reference photos from the TRAIN split only.

Tabs:
  Huong dan   - rules for writers (N1.md)
  Captions    - 412 rows (same columns as captions_template.csv) + a photo strip per dish
  Dishes      - 103 rows: names, visible description, photo strip, look-alike dish strip

Photos: per dish, 5 train images (kept rows) that are typical (close to the dish's mean SigLIP
embedding) but diverse (greedy farthest-point among the 40 most typical). Never val/test images.

Usage (from project root, needs openpyxl): python captions/build_caption_sheet.py captions/captions_filled.csv
"""
import csv
import io
import json
import os

import numpy as np
from openpyxl import Workbook
from openpyxl.drawing.image import Image as XLImage
from openpyxl.drawing.spreadsheet_drawing import AnchorMarker, OneCellAnchor
from openpyxl.drawing.xdr import XDRPositiveSize2D
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side

THIN = Side(style='thin', color='BFBFBF')          # light grey between rows/columns
THICK = Side(style='thick', color='305496')        # dark blue line under each dish block
from PIL import Image

# Data lives in the separate repo ../VNDishClassification-data (override with VNFOOD_DATA).
DATA_REPO = os.environ.get('VNFOOD_DATA', os.path.abspath(os.path.join(os.getcwd(), '..', 'VNDishClassification-data')))
PREP = os.path.join(DATA_REPO, 'data', 'processed', 'vnfood-prep')
CLEAN = os.path.join(DATA_REPO, 'data', 'processed', 'vnfood103-clean')
OUT = 'captions/captions_sheet.xlsx'
N_PHOTOS, POOL = 5, 40

classes = json.load(open(os.path.join(PREP, 'classes.json'), encoding='utf-8'))
by_label = {c['label']: c for c in classes}
rows = list(csv.DictReader(open(os.path.join(PREP, 'splits.csv'), encoding='utf-8')))
F = np.load(os.path.join(PREP, 'feat_siglip.npy'), mmap_mode='r')
import sys
CAPS_IN = sys.argv[1] if len(sys.argv) > 1 else 'captions/captions_template.csv'   # e.g. captions/captions_filled.csv
MACHINE_AUTHOR = 'claude-opus-5-5'
caps = list(csv.DictReader(open(CAPS_IN, encoding='utf-8-sig')))


def pick(label):
    idx = np.array([i for i, r in enumerate(rows) if r['label'] == label and r['split'] == 'train' and r['reason'] == ''])
    X = np.asarray(F[idx], dtype=np.float32)
    mu = X.mean(0)
    mu /= np.linalg.norm(mu)
    order = np.argsort(-(X @ mu))[:POOL]
    chosen = [order[0]]
    while len(chosen) < N_PHOTOS:
        d = 1 - X[order] @ X[chosen].T            # cosine distance to chosen set
        chosen.append(order[int(np.argmax(d.min(1)))])
    return [rows[idx[k]]['path'] for k in chosen]


def strip(paths, size):
    out = Image.new('RGB', (size * len(paths) + 2 * (len(paths) - 1), size), 'white')
    for k, p in enumerate(paths):
        im = Image.open(os.path.join(CLEAN, p)).convert('RGB')
        w, h = im.size
        s = min(w, h)
        im = im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s)).resize((size, size), Image.BICUBIC)
        out.paste(im, (k * (size + 2), 0))
    buf = io.BytesIO()
    out.save(buf, 'JPEG', quality=80)
    buf.seek(0)
    return buf, out.size


def grid(paths, size, per_row=3):
    """Photos in a grid (3 on top, 2 below) so they fill a block of rows next to the text cells."""
    n_rows = -(-len(paths) // per_row)
    out = Image.new('RGB', (per_row * size + 2 * (per_row - 1), n_rows * size + 2 * (n_rows - 1)), 'white')
    for k, p in enumerate(paths):
        im = Image.open(os.path.join(CLEAN, p)).convert('RGB')
        w, h = im.size
        s = min(w, h)
        im = im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s)).resize((size, size), Image.BICUBIC)
        out.paste(im, ((k % per_row) * (size + 2), (k // per_row) * (size + 2)))
    buf = io.BytesIO()
    out.save(buf, 'JPEG', quality=78)
    buf.seek(0)
    return buf, out.size


# Google Sheets geometry (images float over the grid, so they must fit inside their cells):
#   column width W (chars) renders as ~7 px per char (default 14.43 -> 100 px); use 6.8 to be safe
#   row height in points -> px * 4/3
EMU = 9525
PAD = 4


def col_px(width):
    return int(width * 6.8)


def row_px(points):
    return int(points * 4 / 3)


def place(ws, buf, size, row, col, box_w, box_h):
    """Anchor an image at (row, col) inset by PAD px; refuse if it would spill out of the box."""
    w, h = size
    if w + 2 * PAD > box_w or h + 2 * PAD > box_h:
        raise SystemExit('image %dx%d does not fit box %dx%d at row %d col %d' % (w, h, box_w, box_h, row, col))
    img = XLImage(buf)
    img.width, img.height = w, h
    img.anchor = OneCellAnchor(_from=AnchorMarker(col=col - 1, colOff=PAD * EMU, row=row - 1, rowOff=PAD * EMU),
                               ext=XDRPositiveSize2D(w * EMU, h * EMU))
    ws.add_image(img)


photos = {c['label']: pick(c['label']) for c in classes}
json.dump(photos, open(os.path.join(PREP, 'caption_reference_photos.json'), 'w'), indent=1)
print('picked photos for', len(photos), 'dishes')

wb = Workbook()
HEAD = Font(bold=True, color='FFFFFF')
FILL = PatternFill('solid', fgColor='305496')
WRAP = Alignment(wrap_text=True, vertical='top')
EDIT = PatternFill('solid', fgColor='FFF2CC')

# --- Tab 1: instructions -------------------------------------------------------------
ws = wb.active
ws.title = 'Huong dan'
guide = [
    'HƯỚNG DẪN VIẾT CÂU MÔ TẢ (N1.md)',
    '',
    'Mỗi món 4 câu: 2 câu ngắn (5–10 từ) + 2 câu dài (15–30 từ). Cột split đã chia sẵn: mỗi món 1 ngắn + 1 dài cho val, 1 ngắn + 1 dài cho test.',
    'Chỉ điền 3 cột tô vàng: author (tên viết tắt), text_vi (tiếng Việt có dấu), text_en (tự dịch, không dùng công cụ dịch).',
    'Tả CÁI NHÌN THẤY: loại sợi/bánh, màu nước dùng/nước sốt, topping, rau, cách bày, đồ đựng. Viết như người đang tả món mình ăn mà quên tên.',
    'KHÔNG nêu tên món, tên gọi khác, nguyên liệu chính có trong tên món, địa danh gắn với món. Xem cột _avoid.',
    'Với món dễ nhầm (cột _confusable), câu phải nêu điểm nhìn thấy khác với món kia nếu có. Xem tab Dishes để so ảnh.',
    '4 câu của cùng một món phải KHÁC NHAU thật sự (tả khía cạnh khác), không viết lại câu kia bằng từ khác.',
    'Không chép nguyên cột _look (mô tả tham khảo) — chỉ dùng để gợi ý.',
    'Ảnh tham khảo lấy từ tập train, tả món nói chung chứ không tả riêng một bức ảnh (không tả màu bát, mặt bàn của 1 ảnh).',
    '',
    'Tốt: tô bún sợi to tròn, nước đỏ cam, có chả và giò heo',
    'Không được: bún bò Huế nước đỏ (lộ tên món)',
    'Tốt: cuốn thịt nhỏ gói lá xanh, xếp đĩa, rắc đậu phộng',
    'Không được: thịt bò cuốn lá lốt (lộ tên món)',
    '',
    'Câu chữ XÁM NGHIÊNG có author = %s là bản nháp do máy viết. Muốn thay: viết đè câu của bạn vào text_vi/text_en và đổi author thành tên viết tắt của bạn (chữ sẽ tự chuyển sang màu đen).' % MACHINE_AUTHOR,
    'Câu do máy viết được giữ nguyên author = %s để khi đánh giá có thể tách riêng câu người viết và câu máy viết.' % MACHINE_AUTHOR,
    'Xong: File > Download > CSV (tab Captions), rồi chạy: python captions/check_captions.py <file>.csv --export captions/captions.csv',
]
for k, t in enumerate(guide, 1):
    ws.cell(k, 1, t).alignment = Alignment(wrap_text=True)
ws['A1'].font = Font(bold=True, size=14)
ws.column_dimensions['A'].width = 140

# --- Tab 2: captions -----------------------------------------------------------------
ws = wb.create_sheet('Captions')
# photos sit directly left of the cells the writer types in
PHOTO_COL = 'photos (train)'
cols = ['id', 'label', '_name_vi', 'split', '_length', PHOTO_COL, 'text_vi', 'text_en', 'author',
        '_avoid', '_confusable', '_look']
GRID_PHOTO = 105                                   # px per photo -> grid 319 x 212
GRID_W, GRID_H = 3 * GRID_PHOTO + 4, 2 * GRID_PHOTO + 2
ROW_PT = 44                                        # 4 rows x 44 pt = 234 px per dish block
PHOTO_W = -(-(GRID_W + 2 * PAD + 8) // 6.8)        # column width (chars) that holds the grid
widths = [9, 16, 20, 6, 7, PHOTO_W, 55, 55, 8, 28, 24, 45]
PH = cols.index(PHOTO_COL) + 1
for j, (c, w) in enumerate(zip(cols, widths), 1):
    cell = ws.cell(1, j, c)
    cell.font, cell.fill = HEAD, FILL
    ws.column_dimensions[cell.column_letter].width = w
ws.freeze_panes = 'D2'          # header row + id/label/name stay visible
first_row = {}
for k, r in enumerate(caps, 2):
    last_of_dish = k == len(caps) + 1 or caps[k - 1]['label'] != r['label']   # caps[k-1] is the next row
    for j, c in enumerate(cols, 1):
        cell = ws.cell(k, j)
        if c != PHOTO_COL:
            cell.value = r.get(c, '')
            cell.alignment = WRAP
        if c in ('author', 'text_vi', 'text_en'):
            cell.fill = EDIT
        cell.border = Border(left=THIN, right=THIN, top=THIN, bottom=THICK if last_of_dish else THIN)
    ws.row_dimensions[k].height = ROW_PT
    first_row.setdefault(r['label'], k)
# machine-written rows: grey italic text while author is still the model name (rule follows edits in Sheets)
from openpyxl.formatting.rule import FormulaRule
_au = ws.cell(1, cols.index('author') + 1).column_letter
_first = ws.cell(1, cols.index('text_vi') + 1).column_letter
ws.conditional_formatting.add('%s2:%s%d' % (_first, _au, len(caps) + 1),
                              FormulaRule(formula=['$%s2="%s"' % (_au, MACHINE_AUTHOR)],
                                          font=Font(italic=True, color='808080')))
for label, k in first_row.items():
    # merge_cells copies the top-left cell's edges to the whole merged block
    ws.cell(k, PH).border = Border(left=THIN, right=THIN, top=THIN, bottom=THICK)
    ws.merge_cells(start_row=k, start_column=PH, end_row=k + 3, end_column=PH)
    buf, size = grid(photos[label], GRID_PHOTO)
    place(ws, buf, size, k, PH, col_px(PHOTO_W), 4 * row_px(ROW_PT))

# --- Tab 3: dishes -------------------------------------------------------------------
ws = wb.create_sheet('Dishes')
cols = ['id', 'label', 'name_vi', 'name_en', 'desc_vi', 'look-alike', 'photos (train)',
        'first look-alike', 'look-alike photos (train)']
STRIP_PHOTO = 96                                   # strip 5 x 96 = 488 x 96 px
STRIP_W = -(-(5 * STRIP_PHOTO + 8 + 2 * PAD + 8) // 6.8)
DISH_ROW_PT = 82                                   # 109 px
widths = [5, 18, 22, 24, 45, 18, STRIP_W, 16, STRIP_W]
for j, (c, w) in enumerate(zip(cols, widths), 1):
    cell = ws.cell(1, j, c)
    cell.font, cell.fill = HEAD, FILL
    ws.column_dimensions[cell.column_letter].width = w
ws.freeze_panes = 'C2'
for k, c in enumerate(classes, 2):
    conf = next((r['_confusable'] for r in caps if r['label'] == c['label']), '')
    first = conf.split(',')[0].strip() if conf else ''
    for j, v in enumerate([c['id'], c['label'], c['name_vi'], c['name_en'], c['desc_vi'], conf], 1):
        ws.cell(k, j, v).alignment = WRAP
    for j in range(1, len(cols) + 1):
        ws.cell(k, j).border = Border(left=THIN, right=THIN, top=THIN, bottom=THICK)
    ws.row_dimensions[k].height = DISH_ROW_PT
    buf, size = strip(photos[c['label']], STRIP_PHOTO)
    place(ws, buf, size, k, 7, col_px(STRIP_W), row_px(DISH_ROW_PT))
    if first in photos:
        ws.cell(k, 8, first).alignment = WRAP
        buf, size = strip(photos[first], STRIP_PHOTO)
        place(ws, buf, size, k, 9, col_px(STRIP_W), row_px(DISH_ROW_PT))

wb.save(OUT)
print('wrote', OUT, '%.1f MB' % (os.path.getsize(OUT) / 1e6))
