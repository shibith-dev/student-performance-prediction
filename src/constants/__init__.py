import os
import numpy as np

# Project Configuration-----------------------------------------------
TARGET_COLUMN: str = "Overall"

ARTIFACT_DIR: str = "artifacts"
FINAL_MODEL_DIR: str = "final_models"


# Raw Data Configuration---------------------------------------------------
DATA_FILE_NAME: str = "ResearchInformation3.csv"
SCHEMA_FILE_PATH: str = os.path.join("data_schema", "schema.yaml")


# Train/Test Split Configuration--------------------------------------------
TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.2



# Saved Objects-------------------------------------------------------------
PREPROCESSING_OBJECT_FILE_NAME: str = "preprocessing.pkl"
MODEL_FILE_NAME: str = "model.pkl"



# Data Ingestion Configuration (Database + Folder Structure)----------------
DATA_INGESTION_DATABASE_NAME: str = "ShibithAI"
DATA_INGESTION_COLLECTION_NAME: str = "StudentData"

DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_FUTURE_STORE_DIR: str = "feature_store"



# Data Validation Configuration (Folder Structure)----------------

DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALIDATION_DRIFT_REPORT_DIR: str = "drift_report"
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME: str = "report.yaml"
DATA_VALIDATION_VALID_DIR: str = "validated"
DATA_VALIDATION_INVALID_DIR: str = "invalid"

# Data Transformation Configuration (Folder Structure)----------------

DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DIR_NAME: str = "transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR_NAME: str = "transformed_object"
DATA_TRANSFORMATION_TRAIN_FILE_PATH: str = "train.npy"
DATA_TRANSFORMATION_TEST_FILE_PATH: str = "test.npy"

# Model Trainer Configuration (Folder Structure)----------------
MODEL_TRAINER_DIR_NAME: str = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR: str = "trained_model"
MODEL_TRAINER_TRAINED_MODEL_NAME: str = "model.pkl"
MODEL_TRAINER_EXPECTED_SCORE: float = 0.6
