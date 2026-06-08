import os
import sys
import pandas as pd
import numpy as np

# Ensure project root is on sys.path so `from src...` imports work when running
# this script directly.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.IndianHousePricingMLOPSproject.exception import customexception
from src.IndianHousePricingMLOPSproject.logger import logging
from src.IndianHousePricingMLOPSproject.utils.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features: pd.DataFrame):
        """
        features : raw DataFrame with the same columns as the original
                   Scraped_Data.csv (minus exactPrice).
        Returns  : predicted house prices in original rupee scale.
        """
        try:
            preprocessor_path = os.path.join("Artifacts", "preprocessor.pkl")
            model_path = os.path.join("Artifacts", "model.pkl")

            bundle = load_object(preprocessor_path)
            model = load_object(model_path)

            house_preprocessor = bundle["house_preprocessor"]
            scaler = bundle["scaler"]
            feature_columns = bundle["feature_columns"]

            # Domain-specific cleaning & encoding (no target column present at inference)
            cleaned = house_preprocessor.transform(features)

            # Align columns to training order (fill missing with 0)
            cleaned = cleaned.reindex(columns=feature_columns, fill_value=0)

            # Scale
            scaled = scaler.transform(cleaned)

            # Predict (model was trained on log-price, so exp to recover rupees)
            log_pred = model.predict(scaled)
            pred = np.exp(log_pred)

            return pred

        except Exception as e:
            logging.info("Exception occurred in PredictPipeline.predict")
            raise customexception(e, sys)


# ------------------------------------------------------------------
# CustomData — maps form inputs to a DataFrame for prediction
# ------------------------------------------------------------------

class CustomData:
    """
    Holds one inference sample. Call .get_data_as_dataframe() to get a
    single-row DataFrame ready for PredictPipeline.predict().

    Adjust the constructor arguments to match whichever raw features your
    trained model was built on (i.e. the columns present in Scraped_Data.csv
    before transformation).
    """

    def __init__(
        self,
        carpetArea: float,
        carpetAreaUnit: str,
        flrNum,
        postedOn: str,
        # Add / remove fields below to match your actual dataset columns
        **extra_features,
    ):
        self.carpetArea = carpetArea
        self.carpetAreaUnit = carpetAreaUnit
        self.flrNum = flrNum
        self.postedOn = postedOn
        self.extra_features = extra_features

    def get_data_as_dataframe(self) -> pd.DataFrame:
        try:
            data = {
                "carpetArea": [self.carpetArea],
                "carpetAreaUnit": [self.carpetAreaUnit],
                "flrNum": [self.flrNum],
                "postedOn": [self.postedOn],
            }
            data.update({k: [v] for k, v in self.extra_features.items()})
            df = pd.DataFrame(data)
            logging.info("Inference DataFrame assembled")
            return df
        except Exception as e:
            logging.info("Exception occurred in CustomData.get_data_as_dataframe")
            raise customexception(e, sys)