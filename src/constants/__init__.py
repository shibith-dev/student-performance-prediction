import os

# Project Configuration-----------------------------------------------
TARGET_COLUMN: str = "Overall"

ARTIFACT_DIR: str = "artifacts"
FINAL_MODEL_DIR: str = "final_models"


# Raw Data Configuration---------------------------------------------------
DATA_FILE_NAME: str = "ResearchInformation3.csv"


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