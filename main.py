from src.entity.config_entity import TrainingPipelineConfig, DataIngestionConfig, DataValidationConfig, DataTransformationConfig
from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation


pipeline_obj = TrainingPipelineConfig()
DataIngestion_confg_obj = DataIngestionConfig(training_pipeline_config=pipeline_obj)
DataIngestion_obj = DataIngestion(data_ingestion_config=DataIngestion_confg_obj)
data_ingestion_artifacts = DataIngestion_obj.initialize_data_ingestion()
dataValidation_config_obj = DataValidationConfig(training_pipeline_config=pipeline_obj)
dataValidation_obj  = DataValidation(data_ingestion_artifact=data_ingestion_artifacts, data_validation_config=dataValidation_config_obj)
data_validation_artifacts = dataValidation_obj.initiate_data_validation()
print(data_validation_artifacts.validation_status)
if data_validation_artifacts.validation_status:
    DataTransformation_config_obj = DataTransformationConfig(training_pipeline_config=pipeline_obj)
    DataTransformation_obj = DataTransformation(data_validation_artifact=data_validation_artifacts, data_transformation_config=DataTransformation_config_obj)
    Data_transformation_artifacts = DataTransformation_obj.initiate_data_transformation()
else:
    print("Validation Failed")