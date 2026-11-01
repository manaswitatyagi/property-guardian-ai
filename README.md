# Property Guardian AI

Property Guardian AI is an agentic property intelligence system designed for continuous risk monitoring, explainable underwriting, and preventive loss reduction.

> **Data notice:** the bundled dataset (`data/`) is a demo fixture. Only coordinates, year built, square footage and
> historical sale data come from real (2014–2015) King County records; every hazard, neighborhood, cost and risk-score
> field is simulated. Assessments flag this in their confidence reasons. See `data/underwriting_dataset_README.md`.

## How it works

```text
POST /assess {property_id}
        │
  Orchestrator (LangGraph)
        │
  Property Agent ──┬─ Weather Agent ──────┐
                   ├─ Flood Agent ────────┤
                   ├─ Climate Agent ──────┤  (run in parallel)
                   ├─ Neighborhood Agent ─┤
                   └─ Inflation Agent ────┘
                                          │
                     XGBoost risk model + SHAP contributions
                                          │
                     Recommendation Agent  (rule-based, every action cites evidence)
                                          │
                     Memory Agent          (month-over-month change, explained by SHAP deltas)
                                          │
                     Risk Reasoning Agent  → RiskAssessment
                                          │
                     persist snapshot + recommendations to memory
```

Every `RiskAssessment` contains: score and label, scoring method, primary drivers with evidence,
SHAP feature contributions, month-over-month change explanation, confidence (score, level, reasons),
and prioritised recommendations.

### Model governance

`RiskModel.train` runs 5-fold cross-validation **grouped by property** (no property appears in both train and
test) and compares XGBoost against a Random Forest baseline. The resulting `ValidationReport` (MAE, R², ROC-AUC,
precision/recall at the High-risk threshold, caveats) is served at `GET /model/validation`.

## Layout

```text
property_guardian_ai/
  agents/       one module per agent (typed inputs → typed outputs)
  app/          FastAPI application
  memory/       snapshot store protocol + in-memory implementation
  models/       Pydantic schemas
  services/     data repository, risk model, service container, logging
  workflows/    LangGraph orchestrator
tests/
streamlit_app.py  demo dashboard
```

## Quick start

Requires Python 3.12+. This repo uses [uv](https://docs.astral.sh/uv/):

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv -e ".[dev]"
.venv/bin/python -m pytest
.venv/bin/python -m property_guardian_ai.app.main      # API on http://127.0.0.1:8000/docs
.venv/bin/streamlit run streamlit_app.py               # dashboard
```

**macOS:** XGBoost needs the OpenMP runtime (`libomp.dylib`). Either `brew install libomp`, or copy the copy bundled
with scikit-learn next to the Python interpreter:

```bash
cp .venv/lib/python3.12/site-packages/sklearn/.dylibs/libomp.dylib \
   "$(.venv/bin/python -c 'import sys; print(sys.base_prefix)')/lib/"
```

## Demo dashboard

`streamlit_app.py` runs the full agent pipeline in the browser. Search by address, city or ZIP and an
**interactive house** is drawn for that property: click the roof, gutters, foundation, plumbing, walls,
doors & windows or yard to see that component's risk score, evidence, SHAP attribution and the actions that
reduce it (toggle *Risk heatmap* to tint the whole house). Scene details come from the data — rain clouds for
frequent heavy rain, water when the home is within 2 km of it. Below the house: overall score, drivers, SHAP
chart, confidence reasoning, recommendations, 12-month trend with a "why did it change" table, a portfolio
ranking and the model validation report.

```bash
.venv/bin/streamlit run streamlit_app.py
```

### Deploying to Streamlit Community Cloud

1. Sign in at [share.streamlit.io](https://share.streamlit.io) with GitHub.
2. **Create app** → repository `manaswitatyagi/property-guardian-ai`, branch `main`, main file `streamlit_app.py`.
3. Under **Advanced settings** choose **Python 3.12**, then **Deploy**.

Cloud installs use `requirements.txt` (pinned, CPU-only XGBoost) and `packages.txt` (OpenMP runtime).

## API

| Method | Path | Returns |
| --- | --- | --- |
| GET | `/health` | status |
| GET | `/properties` | property ids, addresses, latest score |
| POST | `/assess` | `RiskAssessment` for `{"property_id": "KC-001"}` |
| GET | `/properties/{id}/history` | monthly `RiskSnapshot`s |
| GET | `/properties/{id}/risk-change` | why risk moved last month |
| GET | `/model/validation` | `ValidationReport` |

## Key principles

- strong typing with Pydantic models; agents exchange only typed objects
- independent agent modules
- never a score without drivers, evidence, confidence and recommendations
- evidence-based reasoning without fabricated data: missing inputs stay unknown and lower confidence
