import sys
import os
from src.utils import save_object
from src.exception import CustomException
from src.logger import logging

from dataclasses import dataclass
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder





@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:

    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def create_preprocessor(self, whole_dataset):

        categorical_columns = whole_dataset.select_dtypes(include='object').columns.tolist()

        logging.info("creating a pipeline and column transformer ")

        # Create a pipeline for encoding categorical features
        cat_pipeline = Pipeline(steps=[
            ("one_hot_encoder", OneHotEncoder())
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ("cat_pipeline", cat_pipeline, categorical_columns)
            ],remainder='passthrough'
        )
        logging.info("successfully created preprocessor which can be dumped as a pickle file and can be reused later ")

        return preprocessor



    def get_data_transformer_object(self):
        try:
                
            train_df=pd.read_csv('artifacts/train.csv')
            X_test=pd.read_csv('artifacts/test.csv')

            #ID is not required and X4 is one category dominant (of character type) so we remove both of them
            df2=train_df.drop(['ID','X4'],axis='columns')
            X_test=X_test.drop(['ID','X4'],axis='columns')


            #DECIDED TO YOU INTERQUARTILE RANGE TECHNIQUE TO REMOVE OUTLIERS
            #upper and lower quartiles
            q1=df2['y'].quantile(0.25)
            q3=df2['y'].quantile(0.75)
            #upper and lower limit
            upper_limit=q3+(q3-q1)*1.5
            lower_limit=q1-(q3-q1)*1.5
            #filtering the outliers
            outlier_indices=df2[(df2['y']>upper_limit) | (df2['y'] < lower_limit)].index
            df3=df2.drop(outlier_indices,axis=0).reset_index(drop=True)  



            X_train=df3.drop(['y'],axis=1)
            y_train=df3.y

            #REMOVING THE COLUMNS (binary type) WHICH ARE ONE CATEOGORY DOMINANT AS THEY DONOT HAVE ANY IMPACT ON THE FINAL RESULT AND CERTAINLY AN OVERHEAD
            check_X=X_train.select_dtypes(exclude='object')
                        
            for col in check_X:
                zeroes=(check_X[col]==0).sum()
                ones=(check_X[col]==1).sum()
                zeroes_percentage=(zeroes/(ones+zeroes))*100
                if zeroes_percentage >99 or zeroes_percentage<1:
                    X_train.drop(col,axis=1,inplace=True)
                    X_test.drop(col,axis=1,inplace=True)
  

            #CONCATINATING THE TRAIN AND TEST SETS ARE REQUIRED TO CONSIDER ALL THE CATEGORIES OF ALL FEATURES FROM BOTH TRAIN AND TEST SETS.
            
            # Applying preprocessor pipeline
            whole_dataset=pd.concat([X_train,X_test],axis=0)
            cat_columns = X_train.select_dtypes(include='object').columns.tolist()
            encoded=pd.get_dummies(whole_dataset,columns=cat_columns,dtype=int)
            preprocessing_obj=self.create_preprocessor(whole_dataset)
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            logging.info("successfully created the preprocessing object")

            preprocessing_obj.fit(whole_dataset)
            X_train = preprocessing_obj.transform(X_train)
            X_test = preprocessing_obj.transform(X_test)

            return X_train, X_test, y_train, self.data_transformation_config.preprocessor_obj_file_path
                    
        except Exception as e:
            raise CustomException(e,sys)
    
   