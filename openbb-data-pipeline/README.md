# OpenBB GitHub Data Pipeline

This folder is the minimal bridge for:

`OpenBB -> GitHub raw files -> Google Sheets History sync`

## Structure

- `config/universe.csv`
  - editable ticker universe
- `build_openbb_dataset.py`
  - builds the data artifacts with OpenBB yfinance fetchers
- `data/adj_close_matrix.csv`
  - price matrix consumed by Apps Script
- `data/universe.csv`
  - import audit table
- `data/manifest.json`
  - last refresh metadata

## Local Run

```powershell
cd C:\Users\kyu65\OneDrive\Documentos\Playground\openbb-data-pipeline
..\.venv-openbb\Scripts\python.exe .\build_openbb_dataset.py --start-date 2020-01-02
```

## Typical Flow

1. Add tickers to `config/universe.csv`
2. Run `build_openbb_dataset.py`
3. Push the `openbb-data-pipeline/` folder to GitHub
4. In Google Sheets, fill the Apps Script data settings with the GitHub repo/path
5. Run the GitHub sync menu item in the sheet

## Universe Columns

- `active`
- `ticker`
- `provider_symbol`
- `display_name`
- `asset_type`
- `benchmark`
- `notes`

## GitHub Actions

The included workflow `.github/workflows/openbb-data-refresh.yml` can refresh the data on a schedule or on demand.
