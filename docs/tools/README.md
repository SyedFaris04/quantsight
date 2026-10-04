# Illustrated PDF guide

The finished guide is `output/pdf/QuantSight_Project_Guide_2026-10-05.pdf`.
Its editable feature explanations are in `docs/ILLUSTRATED_PROJECT_GUIDE.md`.
The PDF contains 70 landscape pages, including 58 screenshot-based feature pages.

## Rebuild on this Windows workspace

- Start the local frontend on port 3000 and API on port 8001.
- The capture tool needs Python Playwright and an installed Chromium executable.
- The builder needs ReportLab and Pillow. Keep these documentation dependencies
  separate from the application's Render requirements.
- On this workstation, use `backend/data/research/documentation_tools/Scripts/python.exe`
  for the builder and `backend/.venv/Scripts/python.exe` for the browser capture.
- Run the following capture commands from `backend`. Override `--browser` if
  Chromium is installed at a different path.

```powershell
.venv/Scripts/python.exe -m research.capture_project_guide
.venv/Scripts/python.exe -m research.capture_project_guide --features
.venv/Scripts/python.exe -m research.capture_project_guide --extras
```

Run the builder from the repository root:

```powershell
backend/data/research/documentation_tools/Scripts/python.exe docs/tools/build_project_guide.py
```

- Captures use a new guest browser and never sign into an account.
- The feature capture enters one illustrative holding, answers one stateless game
  question and requests a real chatbot reply. These guest records do not sync to
  Supabase. Live mode reads current Yahoo inputs; no admin forecast job is called.
- Raw captures and separate documentation dependencies stay in the ignored
  `backend/data/research/` directory.
- The narrow model-summary column is split into two unaltered crops for readability.
  Other screenshots are section excerpts; wide app tables may show only their
  initial columns. Their full measurements remain available in the app exports.
- The builder uses local Segoe UI fonts, embeds them and checks source paths and
  text-panel overflow before writing the PDF.
- For a later release, review dates, counts, results, code hash, screenshot crop
  coordinates and explanations before rebuilding. This guide describes a dated
  system snapshot; changing screenshots alone does not update the written facts.

## Visual verification

Render every PDF page with Poppler `pdftoppm`, inspect all rendered pages, and use
PyMuPDF or pypdf to check page count, text, bookmarks and links. Raw renders belong
under `tmp/pdfs/`, not in the deployed application. This release was checked with
Poppler at 110 dpi, with closer review of diagrams and adjusted screenshot crops.

The PDF explains existing system features. It does not change model weights,
training data, serving code, database schema or the untouched-test status.
