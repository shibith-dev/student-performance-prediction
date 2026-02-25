from src.entity.config_entity import TrainingPipelineConfig, DataIngestionConfig
from src.components.data_ingestion import DataIngestion


pipeline_obj = TrainingPipelineConfig()
DataIngestion_confg_obj = DataIngestionConfig(training_pipeline_config=pipeline_obj)
DataIngestion_obj = DataIngestion(data_ingestion_config=DataIngestion_confg_obj)
DataIngestion_obj.initialize_data_ingestion()
