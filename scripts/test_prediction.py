import os
import sys
import pandas as pd

# Ensure project root is on sys.path when running this script
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.IndianHousePricingMLOPSproject.pipelines.Prediction_Pipeline import PredictPipeline

if __name__ == '__main__':
    df = pd.read_csv('Artifacts/test_data.csv').head(2)
    if 'exactPrice' in df.columns:
        X = df.drop(columns=['exactPrice'])
    else:
        X = df

    pipe = PredictPipeline()
    preds = pipe.predict(X)
    print('Predictions:', preds)
