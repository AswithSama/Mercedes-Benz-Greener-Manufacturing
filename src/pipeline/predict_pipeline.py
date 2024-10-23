import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
import os


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')

            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)

            data_scaled=preprocessor.transform(features)
            predictions=model.predict(data_scaled)
            return predictions
        
        except Exception as e:
            raise CustomException(e,sys)


