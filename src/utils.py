import os
import sys
from sklearn.model_selection import GridSearchCV
import numpy as np 
import pandas as pd
#import dill
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor

from sklearn.metrics import r2_score

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
def evaluate_models(X_train, y_train,X_test,models):
    try:
        #report = {}

        # Step 1: Define the model
        catboost_model = CatBoostRegressor(verbose=False)

        # Step 2: Define the parameter grid for hyperparameter tuning
        param_grid = {
            'learning_rate': [0.1, 0.05],
            'depth': [4, 6],
            'l2_leaf_reg': [1, 3, 5]
        }

        # Step 3: Initialize GridSearchCV
        grid_search = GridSearchCV(estimator=catboost_model, param_grid=param_grid, cv=3, n_jobs=-1)

        # Step 4: Fit the model to the training data
        grid_search.fit(X_train, y_train)

        # Step 5: Output the best parameters and corresponding score
        print("Optimal hyperparameters:", grid_search.best_params_)
        print("Best cross-validation score:", grid_search.best_score_)
        '''
        for i in range(len(list(models))):
            
            model = list(models.values())[i]
            #para=param[list(models.keys())[i]]

            #gs = GridSearchCV(model,cv=3)
            model.fit(X_train,y_train)

            #model.set_params(**gs.best_params_)
            #model.fit(X_train,y_train)

            #model.fit(X_train, y_train)  # Train model

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)

            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score'''

    except Exception as e:
        raise CustomException(e, sys)
    

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)