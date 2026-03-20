from src.entity.config_entity import TrainingPipelineConfig, DataIngestionConfig, DataValidationConfig, DataTransformationConfig, ModelTrainerConfig
from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


pipeline_obj = TrainingPipelineConfig()

data_ingestion_config_obj = DataIngestionConfig(training_pipeline_config=pipeline_obj)
data_ingestion_obj = DataIngestion(data_ingestion_config=data_ingestion_config_obj)
data_ingestion_artifacts = data_ingestion_obj.initialize_data_ingestion()

data_validation_config_obj = DataValidationConfig(training_pipeline_config=pipeline_obj)
data_validation_obj  = DataValidation(data_ingestion_artifact=data_ingestion_artifacts, data_validation_config=data_validation_config_obj)
data_validation_artifacts = data_validation_obj.initiate_data_validation()

print(data_validation_artifacts.validation_status)

if data_validation_artifacts.validation_status:
    data_transformation_config_obj = DataTransformationConfig(training_pipeline_config=pipeline_obj)
    data_transformation_obj = DataTransformation(data_validation_artifact=data_validation_artifacts, data_transformation_config=data_transformation_config_obj)
    data_transformation_artifacts = data_transformation_obj.initiate_data_transformation()

    model_trainer_config_obj = ModelTrainerConfig(training_pipeline_config=pipeline_obj)
    model_trainer_obj = ModelTrainer(data_transformation_artifacts=data_transformation_artifacts, model_trainer_config=model_trainer_config_obj)
    model_trainer_artifacts = model_trainer_obj.initiate_model_trainer()

else:
    print("Validation Failed")