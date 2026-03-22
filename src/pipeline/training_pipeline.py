from src.entity.config_entity import (
    TrainingPipelineConfig,
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig
)
from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

from src.exception.exception import CustomException
from src.logging.logging import logging

class TrainingPipeline:
    def __init__(self):
        try:
            self.training_pipeline_config = TrainingPipelineConfig()
        except Exception as e:
            raise CustomException(e)

    def start_data_ingestion(self):
        try:
            data_ingestion_config_obj = DataIngestionConfig(self.training_pipeline_config)
            data_ingestion_obj = DataIngestion(data_ingestion_config=data_ingestion_config_obj)
            data_ingestion_artifacts = data_ingestion_obj.initialize_data_ingestion()
            return data_ingestion_artifacts
        except Exception as e:
            raise CustomException(e)
    
    def start_data_validation(self, data_ingestion_artifacts):
        try:
            data_validation_config_obj = DataValidationConfig(training_pipeline_config=self.training_pipeline_config)
            data_validation_obj = DataValidation(data_validation_config=data_validation_config_obj, data_ingestion_artifact=data_ingestion_artifacts)
            data_validation_artifacts = data_validation_obj.initiate_data_validation()
            return data_validation_artifacts
        except Exception as e:
            raise CustomException(e)
        
    
    def start_data_transformation(self, data_validation_artifacts):
        try:
            data_transformation_config_obj = DataTransformationConfig(training_pipeline_config=self.training_pipeline_config)
            data_transformation_obj = DataTransformation(data_transformation_config=data_transformation_config_obj, data_validation_artifact=data_validation_artifacts)
            data_transformation_artifacts = data_transformation_obj.initiate_data_transformation()
            return data_transformation_artifacts
        except Exception as e:
            raise CustomException(e)
        
    
    def start_model_training(self, data_transformation_artifacts):
        try:
            model_trainer_config_obj = ModelTrainerConfig(training_pipeline_config=self.training_pipeline_config)
            model_trainer_obj = ModelTrainer(model_trainer_config=model_trainer_config_obj, data_transformation_artifacts=data_transformation_artifacts)
            model_trainer_artifacts = model_trainer_obj.initiate_model_trainer()
            return model_trainer_artifacts
        except Exception as e:
            raise CustomException(e)
        
    
    def run_training_pipeline(self):
        try:
            logging.info("Training Pipeline Initiated.")
            data_ingestion_artifacts = self.start_data_ingestion()
            data_validation_artifacts = self.start_data_validation(data_ingestion_artifacts=data_ingestion_artifacts)
            if data_validation_artifacts.validation_status:
                data_transformation_artifacts = self.start_data_transformation(data_validation_artifacts=data_validation_artifacts)
                model_trainer_artifacts = self.start_model_training(data_transformation_artifacts=data_transformation_artifacts)
                logging.info("Training Pipeline Successfully Completed.")
                return model_trainer_artifacts
            else:
                logging.error("Data Validation Failed")
                raise ValueError("Data Validation Failed.")
        except Exception as e:
            logging.error("Error occured in Training Pipline.")
            raise CustomException(e)
        

    

