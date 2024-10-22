import os
import sys
from src.exception import CustomException
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from kaggle.api.kaggle_api_extended import KaggleApi
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts',"train.csv")
    test_data_path: str=os.path.join('artifacts',"test.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()
        self.api=KaggleApi()
        self.api.authenticate()
    def download_from_kaggle(self):
        logging.info(f"Downloading the csv files from Kaggle dataset: mercedes-benz-greener-manufacturing")
        try:
            # Downloading the dataset
            self.api.competition_download_file('mercedes-benz-greener-manufacturing','train.csv.zip',path='notebook/data')
            self.api.competition_download_file('mercedes-benz-greener-manufacturing','test.csv.zip',path='notebook/data')

            import zipfile
            with zipfile.ZipFile('notebook/data/train.csv.zip', 'r') as zip_ref:
                zip_ref.extractall('notebook/data')
            with zipfile.ZipFile('notebook/data/test.csv.zip', 'r') as zip_ref:
                zip_ref.extractall('notebook/data')


            logging.info(f"Downloaded the csv files to notebook/data folder")

            df_train=pd.read_csv('notebook/data/train.csv')
            df_test=pd.read_csv('notebook/data/test.csv')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            df_train.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            df_test.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            
            logging.info("Train and Test data ingestion is Done")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)

if __name__=="__main__":
    obj=DataIngestion()
    #downloading from kaggle
    obj.download_from_kaggle()
    train_data,test_data=obj.download_from_kaggle()

    logging.info(" starting transforming data.....") 
    data_transformation=DataTransformation()
    X_train,test_arr,y_train,_=data_transformation.get_data_transformer_object()

    logging.info(" starting training the model.....") 
    modeltrainer=ModelTrainer()
    print(modeltrainer.initiate_model_trainer(X_train,y_train,test_arr))


