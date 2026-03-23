import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from xgboost import XGBRegressor

from sklearn.model_selection import RandomizedSearchCV

from sklearn.pipeline import Pipeline

from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import (
    DataTransformationArtifacts,
    RegressionMetricArtifact,
    ModelTrainerArtifacts,
)
from src import constants

from src.exception.exception import CustomException
from src.logging.logging import logging

import os
import joblib
import mlflow


class ModelTrainer:
    def __init__(
        self,
        data_transformation_artifacts: DataTransformationArtifacts,
        model_trainer_config: ModelTrainerConfig,
    ):
        try:
            self.data_transformation_artifacts = data_transformation_artifacts
            self.model_trainer_config = model_trainer_config

            self.setup_mlflow()
            
        except Exception as e:
            raise CustomException(e)
    
    def setup_mlflow(self):
        try:
            logging.info("Setting up mlflow tracking.")

            mlflow.set_tracking_uri("sqlite:///mlflow.db")

            experiment_name = "student_performance"

            if mlflow.get_experiment_by_name(experiment_name) is None:
                mlflow.create_experiment(experiment_name)

            mlflow.set_experiment(experiment_name)
            logging.info("MLflow setup completed.")

        except Exception as e:
            logging.error("Error occured while settingup mlflow.")
            raise CustomException(e)

    def track_mlflow(self, model, train_metrics, test_metrics):
        logging.info(f"Logging experiment results to MLflow for model: {type(model).__name__}")
        with mlflow.start_run():
            mlflow.log_param("model_name", type(model).__name__)
            mlflow.log_params(model.get_params())

            mlflow.log_metric("train_rmse", train_metrics.rmse)
            mlflow.log_metric("train_mae", train_metrics.mae)
            mlflow.log_metric("train_r2_score", train_metrics.r2_score)

            mlflow.log_metric("test_rmse", test_metrics.rmse)
            mlflow.log_metric("test_mae", test_metrics.mae)
            mlflow.log_metric("test_r2_score", test_metrics.r2_score)

            mlflow.sklearn.log_model(model, artifact_path="model")
        logging.info(f"Successfully logged metrics and model to MLflow for {type(model).__name__}")

    def get_evaluation_metrics(self, y_true, y_pred):
        try:
            logging.info("Calculating evaluation metrics (RMSE, MAE, R2 Score)")
            rmse = float(np.sqrt(mean_squared_error(y_true=y_true, y_pred=y_pred)))
            mae = mean_absolute_error(y_true=y_true, y_pred=y_pred)
            r2 = r2_score(y_true=y_true, y_pred=y_pred)

            regression_metric = RegressionMetricArtifact(
                r2_score=r2, rmse=rmse, mae=mae
            )

            return regression_metric
        except Exception as e:
            logging.error("Error occurred while calculating evaluation metrics.")
            raise CustomException(e)

    def train_models(self, x_train, y_train, models, params):

        try:
            logging.info(f"Starting hyperparameter tuning for models using RandomizedSearchCV.")
            report = {}

            for model_name, model in models.items():
                random_cv = RandomizedSearchCV(
                    estimator=model,
                    param_distributions=params[model_name],
                    n_iter=5,
                    cv=3,
                    scoring="r2",
                    n_jobs=-1,
                    random_state=42,
                )

                random_cv.fit(x_train, y_train)

                logging.info(f"Training completed for: {model_name}. Best R2 Score: {random_cv.best_score_}")

                report[model_name] = {
                    "score": random_cv.best_score_,
                    "params": random_cv.best_params_,
                    "model": random_cv.best_estimator_,
                }

            return report
        except Exception as e:
            logging.error("Error occurred during the hyperparameter tuning.")
            raise CustomException(e)

    def initiate_model_trainer(self):

        try:
            logging.info("Starting Model Training.")
            logging.info(f"Loading transformed train and test data.")
            train_arr = np.load(
                self.data_transformation_artifacts.transformed_train_file_path
            )
            test_arr = np.load(
                self.data_transformation_artifacts.transformed_test_file_path
            )

            x_train = train_arr[:, :-1]
            y_train = train_arr[:, -1]
            x_test = test_arr[:, :-1]
            y_test = test_arr[:, -1]

            logging.info(f"Data shapes - x_train: {x_train.shape}, y_train: {y_train.shape}, x_test: {x_test.shape}, y_test: {y_test.shape}")

            models = {
                "Random Forest": RandomForestRegressor(random_state=42),
                "Decision Tree": DecisionTreeRegressor(random_state=42),
                "Gradient Boosting": GradientBoostingRegressor(random_state=42),
                "AdaBoost": AdaBoostRegressor(random_state=42),
                "XGBoost": XGBRegressor(random_state=42),
                "Linear Regression": LinearRegression(),
                "Ridge": Ridge(random_state=42),
                "Lasso": Lasso(random_state=42),
                "Elastic Net": ElasticNet(random_state=42),
            }

            params = {
                "Random Forest": {
                    "n_estimators": [200, 300, 400, 500],
                    "max_depth": [None, 10, 20, 30],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                    "max_features": ["sqrt", "log2", 0.3, 0.4, 0.5, 0.7, None],
                    "bootstrap": [True, False],
                },
                "Decision Tree": {
                    "criterion": ["squared_error", "friedman_mse"],
                    "max_depth": [None, 5, 10, 20],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                },
                "Gradient Boosting": {
                    "n_estimators": [100, 200, 300],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "max_depth": [3, 5, 7],
                    "subsample": [0.8, 1.0],
                },
                "AdaBoost": {
                    "n_estimators": [50, 100, 200],
                    "learning_rate": [0.01, 0.1, 0.5],
                },
                "XGBoost": {
                    "n_estimators": [200, 300, 400],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "max_depth": [3, 5, 7],
                    "subsample": [0.8, 1.0],
                    "colsample_bytree": [0.8, 1.0],
                    "gamma": [0, 0.1, 0.2],
                    "reg_alpha": [0, 0.1, 1],
                    "reg_lambda": [1, 1.5],
                },
                "Linear Regression": {"fit_intercept": [True, False]},
                "Ridge": {"alpha": [0.01, 0.1, 1, 10]},
                "Lasso": {"alpha": [0.001, 0.01, 0.1, 1]},
                "Elastic Net": {
                    "alpha": [0.001, 0.01, 0.1, 1],
                    "l1_ratio": [0.2, 0.5, 0.8],
                },
            }

            model_report = self.train_models(
                x_train=x_train, y_train=y_train, models=models, params=params
            )
            logging.info(f"Model Report successfully generated.")

            best_model_name = max(model_report, key=lambda x: model_report[x]["score"])
            model = model_report[best_model_name]["model"]

            logging.info(f"Best Model : {best_model_name} with a score : {model_report[best_model_name]['score']}")

            y_train_pred = model.predict(x_train)
            train_metrics = self.get_evaluation_metrics(
                y_true=y_train, y_pred=y_train_pred
            )

            y_test_pred = model.predict(x_test)
            test_metrics = self.get_evaluation_metrics(
                y_true=y_test, y_pred=y_test_pred
            )

            logging.info(f"Train Score : {train_metrics}")
            logging.info(f"Test Score : {test_metrics}")

            self.track_mlflow(
                model=model, train_metrics=train_metrics, test_metrics=test_metrics
            )

            model_dir_path = os.path.dirname(
                self.model_trainer_config.trained_model_file_path
            )
            os.makedirs(model_dir_path, exist_ok=True)

            joblib.dump(model, self.model_trainer_config.trained_model_file_path)
            logging.info(f"Saved the best model to {self.model_trainer_config.trained_model_file_path}")

            preprocessor = joblib.load(
                self.data_transformation_artifacts.transformed_object_file_path
            )

            final_model = Pipeline(
                steps=[("preprocessor", preprocessor), ("model", model)]
            )

            os.makedirs(constants.FINAL_MODEL_DIR, exist_ok=True)
            joblib.dump(
                final_model,
                os.path.join(constants.FINAL_MODEL_DIR, constants.MODEL_FILE_NAME),
            )
            logging.info(f"Saved the final model (preprocessor + model) to {constants.FINAL_MODEL_DIR}")

            logging.info("Model training pipeline completed successfully.")
            return ModelTrainerArtifacts(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                train_metric_artifact=train_metrics,
                test_metric_artifact=test_metrics,
            )
        
        except Exception as e:
            logging.error("Error occured in initiate_model_trainer method.")
            raise CustomException(e)
