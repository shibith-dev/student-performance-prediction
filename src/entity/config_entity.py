from datetime import datetime
from src import constants
import os


class TrainingPipelineConfig:
    def __init__(self, timestamp=datetime.now()):
        self.timestamp = timestamp.strftime("%m_%d_%Y_%H_%M_%S") # clean timestamp format
        self.artifact = constants.ARTIFACT_DIR # where all artifacts will be saved
        self.artifact_dir = os.path.join(self.artifact, self.timestamp) # isolated pipeline execution artifacts on each execution
        self.final_model_dir = constants.FINAL_MODEL_DIR

class DataIngestionConfig:
    def __init__(self, training_pipeline_config:TrainingPipelineConfig):
        # Root directory inside artifacts folder to save all artifacts from DataIngestion component:
        self.data_ingestion_dir = os.path.join(training_pipeline_config.artifact_dir, constants.DATA_INGESTION_DIR_NAME)

        # Location where raw data set is stored (feature store)
        self.feature_store_file_path = os.path.join(self.data_ingestion_dir, constants.DATA_INGESTION_FUTURE_STORE_DIR, constants.DATA_FILE_NAME)

        # location where ingested train.csv and test.csv will be stored :
        self.train_file_path = os.path.join(self.data_ingestion_dir, constants.DATA_INGESTION_INGESTED_DIR, constants.TRAIN_FILE_NAME)
        self.test_file_path = os.path.join(self.data_ingestion_dir, constants.DATA_INGESTION_INGESTED_DIR, constants.TEST_FILE_NAME)

        # Train test split ratio:
        self.train_test_split_ratio = constants.DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO

        # database and collection names: (for pulling data)
        self.database_name = constants.DATA_INGESTION_DATABASE_NAME
        self.collection_name = constants.DATA_INGESTION_COLLECTION_NAME





        


    


