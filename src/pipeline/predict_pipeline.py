import sys
import pandas as pd
import os
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join('artifacts','model.pkl')
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')

            print("Before loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After loading")

            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)

            return preds
        except Exception as e:
            raise CustomException(e, sys)
        
        
class CustomData:
    def __init__(self,
                 gender:str,
                 lunch,
                 race_ethnicity,
                 parental_level_of_education,
                 test_preparation_course,
                 reading_score,
                 writing_score):
        self.gender=gender
        self.lunch=lunch
        self.race_ethnicity=race_ethnicity
        self.parental_level_of_education=parental_level_of_education
        self.test_preparation_course=test_preparation_course
        self.reading_score=reading_score
        self.writing_score=writing_score

    def get_data_as_data_frame(self):
        try:
            custom_data={
                "gender":[self.gender],
                "lunch":[self.lunch],
                "race_ethnicity":[self.race_ethnicity],
                "parental_level_of_education":[self.parental_level_of_education],
                "test_preparation_course":[self.test_preparation_course],
                "reading_score":[self.reading_score],
                "writing_score":[self.writing_score]
            } 
            return pd.DataFrame(custom_data)
        except Exception as e:
            raise CustomException(e, sys)
              

    
        

