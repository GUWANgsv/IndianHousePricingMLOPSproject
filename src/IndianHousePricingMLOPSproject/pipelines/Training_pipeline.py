import os
import sys

# Ensure project root is on sys.path so `from src...` imports work when running
# this script directly (e.g. `python src/.../Training_pipeline.py`).
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.IndianHousePricingMLOPSproject.components.Data_ingestion import DataIngestion
from src.IndianHousePricingMLOPSproject.components.Data_transformation import DataTransformation
from src.IndianHousePricingMLOPSproject.components.Model_trainer import ModelTrainer
from src.IndianHousePricingMLOPSproject.components.Model_evaluation import ModelEvaluation


# 1. Ingest raw data and split into train / test CSVs
obj = DataIngestion()
train_data_path, test_data_path = obj.initiate_data_ingestion()

# 2. Clean, encode, and scale features
data_transformation = DataTransformation()
train_arr, test_arr = data_transformation.initialize_data_transformation(
    train_data_path, test_data_path
)

# 3. Train and select the best model
model_trainer_obj = ModelTrainer()
model_trainer_obj.initate_model_training(train_arr, test_arr)

# 4. Evaluate with MLflow tracking
model_eval_obj = ModelEvaluation()
model_eval_obj.initiate_model_evaluation(train_arr, test_arr)