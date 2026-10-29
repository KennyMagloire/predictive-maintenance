# Predictive Maintenance for Industrial Equipment

![Tests](https://github.com/KennyMagloire/predictive-maintenance/actions/workflows/tests.yml/badge.svg)

A cloud-ready prototype that predicts equipment failures and estimates remaining useful life (RUL), presented on a maintenance dashboard.

**Capstone:** *Reducing Industrial Downtime in South Africa Using Machine Learning-Based Failure Prediction*
PGDip ICT (AI and Data Science), Cape Peninsula University of Technology. Supervisor: Dr. Richard.
**Author:** Kenny Magloire Ango

## Built with
![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6.1-F7931E?logo=scikitlearn&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-3.1.0-150458?logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-2.4.6-013243?logo=numpy&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.66.1-FF4B4B?logo=streamlit&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-003B57?logo=sqlite&logoColor=white)
![pytest](https://img.shields.io/badge/pytest-9.2.0-0A9EDC?logo=pytest&logoColor=white)
![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-CI-2088FF?logo=githubactions&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-F37626?logo=jupyter&logoColor=white)
![Google Colab](https://img.shields.io/badge/Google_Colab-Training-F9AB00?logo=googlecolab&logoColor=white)
![PyCharm](https://img.shields.io/badge/PyCharm-IDE-000000?logo=pycharm&logoColor=white)
![Git](https://img.shields.io/badge/Git-F05032?logo=git&logoColor=white)
![GitHub](https://img.shields.io/badge/GitHub-181717?logo=github&logoColor=white)

## What it does
- **Failure prediction:** a Random Forest scores each machine's probability of failing (AI4I 2020 dataset) and turns it into a health score, a status (Critical / Warning / Normal) and a recommended action.
- **Remaining useful life:** a second Random Forest estimates the cycles an engine has left (NASA C-MAPSS FD001).
- **Data quality:** uploaded data is cleaned where safe (spacing, case, duplicates, text-as-number), and rows with missing or physically impossible values are rejected and reported. Missing values are never filled in, because a missing reading may mean a failed sensor.
- **History:** scored batches can be saved to a local SQLite database, so readings accumulate into the historical data a future retraining step would learn from.
- **Dashboard:** Streamlit app with Overview, Machine detail, Alerts, History and Remaining-useful-life tabs, and CSV upload.

## Architecture
| Component | File |
|---|---|
| 1. Data pipeline | `src/data_pipeline.py` |
| 2. ML models | `src/models.py` |
| 3. Prediction engine | `src/prediction_engine.py` |
| 4. Dashboard | `src/dashboard.py` |
| Data layer: history store | `src/storage.py` |

```
data/        datasets (AI4I 2020, C-MAPSS FD001) and small test files
models/      six trained model files (.pkl)
notebooks/   training and evaluation notebooks
figures/     figures produced by the notebooks
src/         application code
tests/       unit, integration, ML-specific and system tests
```

## Run it
Requires Python 3.11.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
python -m streamlit run src/dashboard.py
```

The dashboard opens at `http://localhost:8501`. Without an upload it uses the AI4I 2020 dataset. To try the data-quality checks, upload `data/test_broken.csv` or `data/test_messy.csv`. Use **Save this batch to history** in the sidebar to store a scored batch.

## Test it
```bash
python -m pytest --cov=src --cov-report=term-missing
```
66 automated tests: unit, integration, ML-specific (performance gate, behavioural, robustness, determinism, leakage, version) and system tests that run the dashboard headlessly. Coverage is 95%. The history store was built test-first. The same tests run automatically on every push through GitHub Actions, with a build failure if coverage drops below 90%.

## Results (held-out test set)
| Model | Result |
|---|---|
| Failure classifier (threshold 0.40) | precision 0.97, recall 0.82, F1 0.89 (56 of 68 failures caught, 2 false alarms) |
| RUL regressor | RMSE 18.6 cycles (about 9% of mean engine life), cross-validated R² 0.81 |

## Limitations
- Both datasets are public benchmarks, not real mining telemetry. Failure mechanisms transfer to motor-driven machinery; thresholds do not. Validation on real equipment data is required before any operational use.
- Recall is lower on machine type H (few examples in the data).
- C-MAPSS sensors are anonymised, which limits explanations for the RUL model.
- The system is advisory. It recommends actions and never acts on equipment.

## Not included (future work)
Automatic retraining from the stored history, a REST API, per-machine explanations (SHAP), and per-equipment-class models.

## Data sources
- Matzka, S. (2020) *AI4I 2020 Predictive Maintenance Dataset*. UCI Machine Learning Repository.
- Saxena, A. and Goel, K. (2008) *Turbofan Engine Degradation Simulation Data Set (C-MAPSS)*. NASA Ames Prognostics Center of Excellence.