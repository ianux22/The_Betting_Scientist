# Football Modeling

A Streamlit app and utilities to merge fixture calendars with historical results and simulate match outcomes.

## Project layout
- `app.py` — Streamlit UI (Hot Picks ? Simulator ? League Dashboard).
- `run_matchday_simulation.py` — CLI to run matchday simulations from the merged calendar.
- `Input/` — inputs and intermediate data.
  - `Calendars/` — source calendars per league (CSV).
  - `league_urls.json` — remote result URLs.
  - `simulate_config.json` — leagues/matchdays to simulate via CLI.
  - `results/` — downloaded raw results (ignored by git).
  - `Combined_Country_League_Calendar_25_26.csv`, `combined_results.csv` — merged inputs (ignored by git).
- `Output/` — generated merged outputs (ignored by git).
- Notebooks: `Calendar Updater.ipynb`, `Name Validator.ipynb` for data prep/mapping.

## Setup
```bash
python -m venv .venv
.\.venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

## Usage
- Update inputs (optional):
  - Use `Calendar Updater.ipynb` to merge calendars and download/condense results.
- Run the app:
```bash
streamlit run app.py
```
- Run simulations via CLI:
```bash
python run_matchday_simulation.py --league "Serie A" --matchday 12
```

## Notes
- Input/results and Output CSVs are ignored by git; regenerate locally as needed.
- Large/irrelevant files (e.g., `.cbz`) are ignored by `.gitignore`.
