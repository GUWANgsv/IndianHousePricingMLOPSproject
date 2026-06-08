# Indian House Pricing MLOps Project

Quick notes to run the project locally:

- Install dependencies (recommended in a virtualenv):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

- To run the training pipeline (will create `Artifacts/*`):

```powershell
# from project root
python src/IndianHousePricingMLOPSproject/pipelines/Training_pipeline.py
```

- To run a quick prediction using the saved artifacts:

```powershell
python scripts/test_prediction.py
```

- To run the Flask frontend locally:

```powershell
python app.py
```

Then open http://localhost:5000 in your browser.

- To run the app in Docker on PowerShell, use an absolute host path for the volume mount:

```powershell
docker run --rm -p 5000:5000 -v "$($PWD.Path)\Artifacts:/service/Artifacts" indian-house-price:latest
```

If you are using CMD instead of PowerShell, the equivalent is:

```cmd
docker run --rm -p 5000:5000 -v "%cd%\Artifacts:/service/Artifacts" indian-house-price:latest
```

Notes:
- `xgboost` and `mlflow` are listed in `requirements.txt`. They are optional for running the basic pipeline; the code gracefully skips XGBoost if not installed and avoids failing when MLflow is not configured.
- The training script looks for `Scraped_Data.csv` in a few common locations (e.g. `Artifacts/`, `Notebook/data/`, `data/`). If you place your scraped CSV in one of those locations it will be picked up automatically.
- The container needs the `Artifacts/` folder mounted or copied into it so the backend can load `model.pkl` and `preprocessor.pkl`.