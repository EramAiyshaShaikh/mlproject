import os
import sys
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from src.logger import logging
from src.exception import CustomException
from src.utils import save_obj

@dataclass
class DataTransformationConfig():
    preprocessor_obj_file_path=os.path.join("artifacts","preprocessor.pkl")

class DataTransformation():
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformation_obj(self):
        try:
            numerical_cols=["reading_score","writing_score"]
            categorical_cols=['gender','lunch','race_ethnicity','parental_level_of_education','test_preparation_course']

            num_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())
            ])

            cat_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("encoder",OneHotEncoder()),
                    ("scaler",StandardScaler(with_mean=False))
            ])

            preprocessor=ColumnTransformer(
                [
                ("numerical pipeline",num_pipeline,numerical_cols),
                ("categorical pipeline",cat_pipeline,categorical_cols)
                ]
            )

            logging.info("separate numerical and categorical cols and made a pipeline")

            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
            try:
                train_df=pd.read_csv(train_path)
                test_df=pd.read_csv(test_path)

                logging.info("read the train and test datasets")

                preprocessing_obj=self.get_data_transformation_obj()

                target_col="math_score"

                input_cols_train_df=train_df.drop(columns=[target_col],axis=1)
                target_cols_train_df=train_df[target_col]

                input_cols_test_df=test_df.drop(columns=[target_col],axis=1)
                target_cols_test_df=test_df[target_col]

                logging.info("splitting of input features and target feature is done")

                logging.info(
                    f"Applying preprocessing object on training dataframe and testing dataframe."
                )

                input_cols_train_arr=preprocessing_obj.fit_transform(input_cols_train_df)
                input_cols_test_arr=preprocessing_obj.transform(input_cols_test_df)

                train_arr=np.c_[
                    input_cols_train_arr,np.array(target_cols_train_df)
                ]

                test_arr=np.c_[
                    input_cols_test_arr,np.array(target_cols_test_df)
                ]

                logging.info("Saving the preprocessor object")

                save_obj(
                    file_path=self.data_transformation_config.preprocessor_obj_file_path,
                    obj=preprocessing_obj
                )

                return(
                    train_arr,
                    test_arr,
                    self.data_transformation_config.preprocessor_obj_file_path
                )
            except Exception as e:
                raise CustomException(e,sys)