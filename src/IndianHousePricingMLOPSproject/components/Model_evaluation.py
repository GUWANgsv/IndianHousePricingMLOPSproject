import os
import sys
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.IndianHousePricingMLOPSproject.utils.utils import load_object


class ModelEvaluation:
    def __init__(self):
        pass

    def eval_metrics(self, actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return rmse, mae, r2

    def initiate_model_evaluation(self, train_array, test_array):
        try:
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            model_path = os.path.join("Artifacts", "model.pkl")
            model = load_object(model_path)

            predicted_prices = model.predict(X_test)

            rmse, mae, r2 = self.eval_metrics(y_test, predicted_prices)

            print(f"RMSE : {rmse:.4f}")
            print(f"MAE  : {mae:.4f}")
            print(f"R2   : {r2:.4f}")

            # MLflow logging is optional and disabled by default so training can complete quickly.
            if os.getenv("ENABLE_MLFLOW", "0") == "1":
                try:
                    import mlflow
                    import mlflow.sklearn
                    from urllib.parse import urlparse

                    mlflow.set_registry_uri("")
                    tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
                    print(f"MLflow tracking store type: {tracking_url_type_store}")

                    with mlflow.start_run():
                        mlflow.log_metric("rmse", rmse)
                        mlflow.log_metric("mae", mae)
                        mlflow.log_metric("r2", r2)
                        if tracking_url_type_store != "file":
                            mlflow.sklearn.log_model(
                                model, "Model", registered_model_name="indian_house_price_model"
                            )
                        else:
                            mlflow.sklearn.log_model(model, "Model")
                except Exception:
                    # Don't fail evaluation just because mlflow logging isn't configured
                    print("MLflow logging skipped due to configuration error")

        except Exception as e:
            raise e