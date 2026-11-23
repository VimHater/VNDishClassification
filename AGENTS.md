# AGENTS.md

Guidance for coding agents working in this repository. Read `README.md` for the layout and `docs/N1.md` for the task.

## Conventions

- Run every script from the root of this repo (`python pipeline/prep.py`, not `cd pipeline`).
- Data and the report are in the separate sibling repo `../VNDishClassification-data` (`data/raw/`, `data/processed/`, `report/`).
  Scripts locate it through `VNFOOD_DATA` (default `../VNDishClassification-data` relative to the current directory).
  Report scripts (`report/*.py`) live in that repo and run from its root.
- `pipeline/prep.py` is the single source of truth for steps 1–2. The notebook `notebooks/vndishclassification.ipynb` is generated
  from it with `python pipeline/build_notebook.py`; never edit the notebook by hand.
- `prep.py` uses `multiprocessing` with the `fork` context (Linux / Kaggle). `features.py` is spawn-safe and also runs on Windows.
- Scripts are deterministic (seed 42). Rerunning `prep.py` overwrites the data repo's `data/processed/`; it takes ~4 minutes.
- The data folder is large (≈ 25 GB locally). `data/processed/` is published on Hugging Face as
  `VimHater/VNfood103processed` (upload with `hf upload`, not git push); `data/raw/` is never uploaded.
  Do not move, delete, re-download or re-upload it without asking.
- Python deps: `pillow`, `numpy` everywhere; `scikit-image`, `torch`, `open_clip_torch` for `features.py`; `openpyxl` for
  `captions/build_caption_sheet.py` (not installed system-wide on this machine; use a throwaway venv).

## Data rules (from N1.md, verified on the data)

- Never use the original train/val/test split (it leaks). Splits come from `data/processed/vnfood-prep/splits.csv` (data repo).
- `splits.csv` and every `.npy` have 48,655 rows in the same order, including `dup`/`conflict` rows. Training uses rows with empty `reason`.
- Folder names with misleading meaning: `chao-vit` = vịt nấu chao, `ca-muoi-xoi` = cà muối xổi (eggplant), `goi-ca-chich` = gỏi cá trích.
- Some original `.jpg` files are PNG/MPO/BMP; always open by content and re-encode.

## Captions

- 412 rows (103 dishes × 2 short + 2 long, val/test split fixed). Rules and checks: `captions/check_captions.py`.
- Machine-written captions must keep `author = claude-opus-5-5` so evaluation can separate them from human ones.
  Do not put a person's initials on text an agent wrote.
- `captions/make_captions_template.py` overwrites the template; `captions/draft_captions.py` regenerates the draft file.

## External systems

- **Kaggle**: input dataset `meowluvmatcha/vnfood-30-100`; notebook `nguyenduyanhkhoa/vndish-classification`.
  Do not upload the raw dataset. Do not push the notebook with the Kaggle MCP `save_notebook`: it removes the attached
  dataset input, and scheduled runs cannot re-attach it. Rebuild the `.ipynb` locally and let the user import it.
- **urmom** (`ssh urmom`): Windows PC with an RTX 3070 Ti, PowerShell shell. Project copy and venv at
  (run `.venv\Scripts\python.exe`). Ask before installing anything there.
- **Google Drive**: caption sheet "VNFood103 – captions (N1) v5". Only the Drive connector is available, so changes mean a new
  file; keep the team on one sheet and trash old versions only when the user asks.
- **Summary page**: https://claude.ai/artifact/JPUtuWRwGXCEQiXpK4GgjX — plain HTML, no theming (user's request).
  Built by the data repo's `report/` scripts; update it by publishing `report/vnfood_summary.html` (data repo) with that URL.
