# WhoScored Events Data

Data pipeline for importing, processing, validating, and archiving Bundesliga WhoScored data.

## Includes

- Bundesliga schedules and historical results
- Raw and processed match events
- Lineups, formations, player and team statistics
- Shot and pass sequences
- Normalized key events
- MongoDB backup and archive utilities
- Automated data validation tests

## Requirements

- Python and the project virtual environment
- MongoDB
- The configuration in `config/config.yaml`

## Run the pipeline

```powershell
python pipeline.py
```

## Run tests

```powershell
pytest
```
