import pandas as pd
import numpy as np
from src.entity.config_entity import DataValidationConfig
from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from src.constants import SCHEMA_FILE_PATH
from scipy.stats import ks_2samp

from src.exception.exception import CustomException
from src.logging.logging import logging
from src.utils.utils import read_yaml, write_yaml

import os


class DataValidation:
    def __init__(self,data_ingestion_artifact: DataIngestionArtifact,data_validation_config: DataValidationConfig):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self.schema_config = read_yaml(file_path=SCHEMA_FILE_PATH)
            logging.info("Schema Loaded successfully.")
        except Exception as e:
            raise CustomException(e)

    def read_df(self, file_path) -> pd.DataFrame:
        try:
            logging.info("Reading dataframe.")
            return pd.read_csv(file_path)
        except Exception as e:
            logging.info("Error occured while reading Dataframe.")
            raise CustomException(e)

    def validate_columns(self, df: pd.DataFrame) -> bool:
        try:
            logging.info("Validating total number of columns received.")
            required_columns = set(self.schema_config["columns"])
            df_columns = set(df.columns)

            missing_columns = required_columns - df_columns
            extra_columns = df_columns - required_columns

            if missing_columns:
                logging.error(f"Missing columns: {missing_columns}.")
                return False

            if extra_columns:
                logging.warning(f"Extra columns: {extra_columns}.")

            return True
        except Exception as e:
            logging.error("Error while validating total number of columns.")
            raise CustomException(e)

    def validate_datatype(self, df: pd.DataFrame) -> bool:
        try:
            logging.info("Validating datatype of each column in the dataframe.")
            required_numeric_cols = set(self.schema_config["numerical_columns"])
            required_categoric_cols = set(self.schema_config["categorical_columns"])

            df_numeric_cols = set(df.select_dtypes(include=["int", "float"]).columns)
            df_categoric_cols = set(df.select_dtypes(include=["object", "category"]))

            if not required_numeric_cols.issubset(df_numeric_cols):
                missing_numeric_cols = required_numeric_cols - df_numeric_cols
                logging.error(f"{missing_numeric_cols} columns are missing in the dataset.")
                return False

            if not required_categoric_cols.issubset(df_categoric_cols):
                missing_categoric_cols = required_categoric_cols - df_categoric_cols
                logging.error(
                    f"{missing_categoric_cols} columns are missing in the dataset."
                )
                return False

            return True
        except Exception as e:
            logging.error("Error while validating datatype of the columns.")
            raise CustomException(e)

    def detect_data_drift(self, base_df: pd.DataFrame, new_df: pd.DataFrame, alpha: float = 0.05, psi_threshold: float = 0.2) -> bool:
        try:
            logging.info("Checking Data Drift.")
            report = {}
            drifted = False

            for col in base_df.columns:
                old = base_df[col].dropna() # removes rows having missing values because, missing values affects Drift detection algos
                new = new_df[col].dropna() 

                if pd.api.types.is_numeric_dtype(old):
                    _, p_value = ks_2samp(old, new)
                    drift = p_value < alpha

                    report[col] = {
                        "type": "numeric",
                        "metric": "ks_test",
                        "value": float(p_value),
                        "drift": bool(drift),
                    }
                else:
                    old_distribution = old.value_counts(normalize=True)
                    new_distribution = new.value_counts(normalize=True)

                    categories = old_distribution.index.union(new_distribution.index)

                    old_ratio = old_distribution.reindex(categories, fill_value=0).values
                    new_ratio = new_distribution.reindex(categories, fill_value=0).values

                    eps = 1e-8
                    psi = np.sum((new_ratio - old_ratio) * np.log((new_ratio + eps) / (old_ratio + eps)))

                    drift = psi > psi_threshold
                    report[col] = {
                        "type": "categorical",
                        "metric": "PSI",
                        "value": float(psi),
                        "drift": bool(drift),
                    }

                if report[col]["drift"]:
                    drifted = True

            result = {
                "summary": {
                    "features_checked": len(report),
                    "drift_status": drifted,
                },
                "features": report,
            }

            drift_report_file_path = (self.data_validation_config.data_validation_data_drift_file_path)
            os.makedirs(os.path.dirname(drift_report_file_path), exist_ok=True)
            write_yaml(file_path=drift_report_file_path, content=result)
            logging.info(f"Drift status: {drifted}")

            return drifted
        except Exception as e:
            logging.error("Error while checking the data drift.")
            raise CustomException(e)

    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            logging.info("Data Validation Initiated.")
            # Read the train - test csv files
            train_df = self.read_df(file_path=self.data_ingestion_artifact.train_file_path)
            test_df = self.read_df(file_path=self.data_ingestion_artifact.test_file_path)
            logging.info("Dataset Loaded.")

            # Validate columns
            train_valid = (self.validate_columns(train_df) and self.validate_datatype(train_df))
            test_valid = (self.validate_columns(test_df) and self.validate_datatype(test_df))


            if not train_valid or not test_valid:
                invalid_dir = self.data_validation_config.data_validation_data_invalid_dir
                os.makedirs(invalid_dir, exist_ok=True)

                train_df.to_csv(self.data_validation_config.data_validation_data_invalid_train_file_path, header=True, index=False)
                test_df.to_csv(self.data_validation_config.data_validation_data_invalid_test_file_path, header=True, index=False)

                return DataValidationArtifact(
                    validation_status = False,
                    valid_train_file_path = None,
                    valid_test_file_path = None,
                    invalid_train_file_path = self.data_validation_config.data_validation_data_invalid_train_file_path,
                    invalid_test_file_path = self.data_validation_config.data_validation_data_invalid_test_file_path,
                    drift_report_file_path = None
                )

            # If valid -> Detect Drift
            drift_status = self.detect_data_drift(base_df=train_df, new_df=test_df)

            valid_dir = self.data_validation_config.data_validation_data_valid_dir
            os.makedirs(valid_dir, exist_ok=True)
            train_df.to_csv(self.data_validation_config.data_validation_data_valid_train_file_path, header=True, index=False)
            test_df.to_csv(self.data_validation_config.data_validation_data_valid_test_file_path, header=True, index=False)
            logging.info("Data Validation Completed.")

            return DataValidationArtifact(
                validation_status = True,
                valid_train_file_path = self.data_validation_config.data_validation_data_valid_train_file_path,
                valid_test_file_path = self.data_validation_config.data_validation_data_valid_test_file_path,
                invalid_train_file_path = None,
                invalid_test_file_path = None,
                drift_report_file_path = self.data_validation_config.data_validation_data_drift_file_path
            )
        except Exception as e:
            logging.error("Error while intializing Data Validation Component.")
            raise CustomException(e)


