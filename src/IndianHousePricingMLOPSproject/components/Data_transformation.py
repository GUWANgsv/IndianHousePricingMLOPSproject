import os
import sys
import datetime
import pandas as pd
import numpy as np
from dataclasses import dataclass

from src.IndianHousePricingMLOPSproject.exception import customexception
from src.IndianHousePricingMLOPSproject.logger import logging
from src.IndianHousePricingMLOPSproject.utils.utils import save_object

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin


# ------------------------------------------------------------------
# Carpet-area unit conversion map (all units → Sq-ft)
# ------------------------------------------------------------------
CARPET_CONVERSION = {
    'Sq-ft': 1,
    'Sq-yrd': 9,
    'Sq-m': 10.7639,
    'Kanal': 5445,
    'Marla': 272.25,
    'Biswa1': 1350,
    'Biswa2': 900,
    'Rood': 10890,
    'Acre': 43560,
}

# Floor string → numeric map
FLOOR_MAP = {"Ground": 0, "Upper Basement": -1, "Lower Basement": -2}


# ------------------------------------------------------------------
# Custom transformer: all domain-specific cleaning done in one step
# so the sklearn Pipeline stays clean.
# ------------------------------------------------------------------
class HouseDataPreprocessor(BaseEstimator, TransformerMixin):
    """
    Performs:
      1. Replace sentinel 9 / '9' with NaN
      2. CarpetArea unit conversion (carpetArea × unit → Sq-ft)
      3. Drop raw carpetArea, carpetAreaUnit, URLs columns
      4. postedOn  → postedOn_DaysAgo (integer)
      5. flrNum string parsing → float
      6. Median imputation for numerics, 'Missing' for categoricals
      7. Target-mean encoding for categorical columns
      8. Log-transform the target (exactPrice)
      9. Outlier clipping on the target (1st–99th percentile)
    """

    def __init__(self):
        self.cat_means_ = {}          # col → {category: mean(log_price)}
        self.target_q1_ = None
        self.target_q99_ = None

    # ---- helpers ------------------------------------------------

    @staticmethod
    def _days_ago(date_str):
        try:
            d = datetime.datetime.strptime(str(date_str), "%b %d, '%y").date()
            return (datetime.date.today() - d).days
        except Exception:
            return np.nan

    @staticmethod
    def _parse_floor(val):
        if pd.isna(val):
            return np.nan
        val = str(val).strip()
        if val in FLOOR_MAP:
            return float(FLOOR_MAP[val])
        try:
            return float(val)
        except ValueError:
            return np.nan

    def _basic_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # 1. Replace sentinel values
        df = df.replace(9, np.nan).replace("9", np.nan)

        # 2. Carpet-area unit conversion
        if "carpetArea" in df.columns and "carpetAreaUnit" in df.columns:
            df["CarpetArea"] = df["carpetArea"] * df["carpetAreaUnit"].map(CARPET_CONVERSION)
            df = df.drop(columns=["carpetArea", "carpetAreaUnit"])

        # 3. Drop URL column if present
        if "URLs" in df.columns:
            df = df.drop(columns=["URLs"])

        # 4. postedOn → days-ago integer
        if "postedOn" in df.columns:
            df["postedOn_DaysAgo"] = df["postedOn"].apply(self._days_ago)
            df = df.drop(columns=["postedOn"])

        # 5. Floor parsing
        if "flrNum" in df.columns:
            df["flrNum"] = df["flrNum"].apply(self._parse_floor)

        return df

    # ---- fit / transform ----------------------------------------

    def fit(self, df: pd.DataFrame, y=None):
        df = self._basic_clean(df)

        # 6. Impute before encoding
        for col in df.select_dtypes(include=np.number).columns:
            if col != "exactPrice" and df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())
        for col in df.select_dtypes(include="object").columns:
            if df[col].isnull().any():
                df[col] = df[col].fillna("Missing")

        # 8. Log-transform target
        df["exactPrice"] = np.log(df["exactPrice"])

        # 9. Fit outlier bounds on log-price
        self.target_q1_ = df["exactPrice"].quantile(0.01)
        self.target_q99_ = df["exactPrice"].quantile(0.99)

        # Clip outliers for fitting the encoders
        df = df[(df["exactPrice"] >= self.target_q1_) & (df["exactPrice"] <= self.target_q99_)].copy()

        # 7. Fit target-mean encoding on training data
        cat_features = df.select_dtypes(include="object").columns.tolist()
        for col in cat_features:
            self.cat_means_[col] = df.groupby(col)["exactPrice"].mean().to_dict()

        logging.info(f"HouseDataPreprocessor fitted. Categorical columns encoded: {list(self.cat_means_.keys())}")
        return self

    def transform(self, df: pd.DataFrame, y=None):
        df = self._basic_clean(df)

        # 6. Impute
        for col in df.select_dtypes(include=np.number).columns:
            if col != "exactPrice" and df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())
        for col in df.select_dtypes(include="object").columns:
            if df[col].isnull().any():
                df[col] = df[col].fillna("Missing")

        # 8. Log-transform target (only if present — inference may not have it)
        has_target = "exactPrice" in df.columns
        if has_target:
            df["exactPrice"] = np.log(df["exactPrice"])
            # 9. Clip outliers (train split only; test not clipped to avoid data leak)
            df = df[(df["exactPrice"] >= self.target_q1_) & (df["exactPrice"] <= self.target_q99_)].copy()

        # 7. Apply target-mean encoding (unseen categories → global mean of stored means)
        for col, means in self.cat_means_.items():
            if col in df.columns:
                global_mean = np.mean(list(means.values()))
                df[col + "_enc"] = df[col].map(means).fillna(global_mean)
                df = df.drop(columns=[col])

        return df


# ------------------------------------------------------------------
# DataTransformationConfig / DataTransformation
# ------------------------------------------------------------------

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("Artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation(self):
        """
        Returns a fitted-ready HouseDataPreprocessor + StandardScaler pipeline.
        The HouseDataPreprocessor handles all domain logic; the scaler normalises
        the resulting numeric features.
        """
        try:
            logging.info("Data Transformation pipeline initiated")

            # After HouseDataPreprocessor, all columns are numeric.
            # We wrap it in a sklearn Pipeline so save_object / predict pipeline
            # can call .fit_transform() / .transform() transparently.

            pipeline = Pipeline(steps=[
                ("house_preprocessor", HouseDataPreprocessor()),
                # StandardScaler applied inside initialize_data_transformation
                # (after we split X / y) — see below.
            ])

            return pipeline

        except Exception as e:
            logging.info("Exception occurred in get_data_transformation")
            raise customexception(e, sys)

    def initialize_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data complete")
            logging.info(f"Train Dataframe Head:\n{train_df.head().to_string()}")
            logging.info(f"Test Dataframe Head:\n{test_df.head().to_string()}")

            # Step 1: Domain-specific cleaning & encoding
            house_prep = HouseDataPreprocessor()
            train_clean = house_prep.fit(train_df).transform(train_df)
            test_clean = house_prep.transform(test_df)

            logging.info(f"Train shape after preprocessing: {train_clean.shape}")
            logging.info(f"Test shape after preprocessing:  {test_clean.shape}")

            target_column = "exactPrice"

            # Step 2: Split features / target
            X_train = train_clean.drop(columns=[target_column])
            y_train = train_clean[target_column]

            X_test = test_clean.drop(columns=[target_column])
            y_test = test_clean[target_column]

            # Step 3: Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            logging.info("StandardScaler applied to features")

            # Combine into arrays expected by ModelTrainer
            train_arr = np.c_[X_train_scaled, np.array(y_train)]
            test_arr = np.c_[X_test_scaled, np.array(y_test)]

            # Save the full preprocessing object (domain prep + scaler)
            # We save both as a dict so the prediction pipeline can use them.
            preprocessing_bundle = {
                "house_preprocessor": house_prep,
                "scaler": scaler,
                "feature_columns": list(X_train.columns),
            }

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_bundle,
            )

            logging.info("Preprocessing bundle saved to pickle")

            return train_arr, test_arr

        except Exception as e:
            logging.info("Exception occurred in initialize_data_transformation")
            raise customexception(e, sys)