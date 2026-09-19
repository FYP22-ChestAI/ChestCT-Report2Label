# ChestCT-Report2Label

Extracts CT-RATE-compatible multi-abnormality labels from free-text chest CT
radiology reports, so a local report archive can be turned into a labeled
dataset without manual annotation from scratch.

It reuses the same 18-abnormality vocabulary and the same fine-tuned RadBERT
text classifier introduced by the [CT-RATE](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE)
dataset, so labels produced here line up with CT-RATE labels one-for-one.
The pipeline itself is generic: it works on plain report text and makes no
assumption about which institution, reporting system, or template a report
came from.

## Pipeline

```
raw report (PDF/DOCX/TXT)
  -> ingestion        extract plain text
  -> parsing          find the CT/accession id, split into sections
  -> preprocessing    clean whitespace, strip identifying info, normalize text
  -> extraction       RadBERT classifier -> per-label probability + evidence sentence
  -> labels.csv       one row per CT id, one 0/1 column per abnormality
```

The output is keyed by CT id only (like CT-RATE's own label files) — scans
live separately and are joined on that id by whoever consumes the labels.

`validation/` compares predictions against a small manually annotated CSV and
ranks the most confidently-wrong predictions for review — useful once you
have a handful of hand-labeled reports to sanity-check against.

## Setup (run on the machine that will actually do inference — e.g. your server, not necessarily where you edit code)

1. Get the code onto that machine (clone/pull the repo, or copy it over), then
   create an environment and install dependencies:

   ```bash
   cd ChestCT-Report2Label
   python -m venv .venv
   source .venv/bin/activate        # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. Install the Hugging Face CLI, if it isn't already there:

   ```bash
   curl -LsSf https://hf.co/cli/install.sh | bash -s
   ```

3. Authenticate. You must have accepted the
   [CT-RATE dataset's terms](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE)
   on huggingface.co first — the checkpoint is gated behind that.

   ```bash
   hf auth login
   ```

4. Download the fine-tuned classifier checkpoint straight into `models/` (run
   from the repo root so `--local-dir .` lines up with the project layout):

   ```bash
   hf download ibrahimhamamci/CT-RATE models/RadBertClassifier.pth \
       --repo-type dataset --local-dir .
   ```

   This lands the file at `models/RadBertClassifier.pth` (~500 MB, gitignored
   — never commit it). The base encoder (`zzxslp/RadBERT-RoBERTa-4m`)
   downloads separately and automatically the first time you run anything,
   via `transformers`.

5. Sanity-check the setup before touching any real reports:

   ```bash
   python scripts/test_model.py
   ```

   `configs/model.yaml` has `device: auto`, so this picks up a GPU
   automatically if the machine has one — no config change needed.

## Usage

Drop report files into `data/raw/` (PDF, DOCX, or TXT), then run the scripts
in order:

```bash
python scripts/extract_reports.py       # data/raw       -> data/processed
python scripts/predict_labels.py        # data/processed -> data/outputs/predictions
```

`extract_reports.py` writes one JSON per report plus `data/processed/reports.csv`
— the extracted report table, one row per report: `report_id`, `ct_id`, one
column per configured section (`indication`, `technique`, `comparison`,
`findings`, `impression`; empty if a report doesn't have it), and
`report_text` (the exact text fed to the classifier). Columns follow
`sections.headers` in `configs/pipeline.yaml`, so adding a section there adds
a column.

Each prediction is saved as JSON with the probability, the binary label, and
the top evidence sentence(s) the classifier scored highest for that label.
`predict_labels.py` also writes two CSVs next to them:

- `labels.csv` — the dataset: `ct_id` plus one 0/1 column per label (reports
  with no detectable CT id are skipped with a warning).
- `_summary.csv` — per-label probabilities, for a quick spreadsheet-style scan.

To check predictions against a small hand-labeled set:

```bash
python scripts/validate_predictions.py --annotations data/annotations/manual_labels.csv
```

That CSV needs a `ct_id` column plus one 0/1 column per label (see
`configs/labels.yaml` for the exact 18 label names, in the order the
classifier was trained on).

## Notebooks

`notebooks/` walks through the same pipeline interactively — useful for
demoing results rather than reading raw JSON. They reuse the same venv as everything else:

```bash
pip install -e ".[notebooks]"     # or: pip install jupyterlab ipykernel nbconvert
python -m ipykernel install --user --name report2label --display-name "Python (report2label)"
jupyter lab notebooks/
```

Pick the "Python (report2label)" kernel when a notebook opens. Run them in
order — `01` just sanity-checks the model, `02` runs the full pipeline over
whatever is in `data/raw`/`data/outputs/predictions` and prints
probabilities + evidence per report, `03`/`04` need
`data/annotations/manual_labels.csv` to exist first.

To hand someone a static, already-executed copy (no Jupyter install needed
on their end) — re-run it headless and export to HTML:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/02_test_reports.ipynb
jupyter nbconvert --to html notebooks/02_test_reports.ipynb
```

That produces `notebooks/02_test_reports.html`, which opens in any browser.

## Configuration

- `configs/labels.yaml` — the 18-label vocabulary, in classifier output order.
- `configs/model.yaml` — base encoder, checkpoint path, max length, threshold.
- `configs/pipeline.yaml` — I/O paths, section headers, PHI-removal toggle,
  evidence settings, per-label threshold overrides.

## Tests

```bash
pytest
```

Tests cover the pure-logic parts of the pipeline (parsing, preprocessing,
thresholding, evidence selection) and don't require the classifier
checkpoint or a GPU.

## Attribution

- Label vocabulary and classifier architecture: [CT-RATE](https://arxiv.org/abs/2403.17834) (Hamamci et al.).
- Base language model: [RadBERT-RoBERTa-4m](https://huggingface.co/zzxslp/RadBERT-RoBERTa-4m) (Yan et al.).

See [LICENSE](LICENSE) for this repository's license. The CT-RATE dataset and
its pretrained checkpoints are separately licensed (CC-BY-NC-SA, gated) — you
must accept those terms yourself to download `RadBertClassifier.pth`.
