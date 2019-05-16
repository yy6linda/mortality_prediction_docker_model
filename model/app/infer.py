from functools import reduce
import datetime
from operator import add
import pandas as pd
import numpy as np
from datetime import datetime
'''for plotting'''
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot
from matplotlib.pyplot import savefig
'''for implementing simple logisticregression'''
import sklearn
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve,roc_auc_score,auc
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
'''for saving models'''
from joblib import load
ROOT="/"


class OmopParser(object):
    '''this structures the omop dataset'''
    def __init__(self):
        self.name = 'omop_assembler'


    def add_demographic_data(self):
        '''add demographic data'''
        person = pd.read_csv('/infer/person.csv')
        cols = ['person_id','gender_concept_id','year_of_birth','race_source_value']
        person = person[cols]
        person['cut_off_date'] = datetime(2017, 7, 5)
        person['year_of_birth'] = pd.to_datetime(person['year_of_birth'], format='%Y')
        person['age'] = person['cut_off_date'] - person['year_of_birth']
        person['age'] = person['age'].apply(lambda x: round(x.days/365.25))
        dummy_columns_gender = pd.get_dummies(person['gender_concept_id'],prefix='gender',drop_first=True)
        ##scaling the age
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        scaled_column = scaler.fit_transform(person[['age']])
        person = pd.concat([person, pd.DataFrame(scaled_column,columns=['scaled_age'])],axis=1)
        dummy_columns_race = pd.get_dummies(person['race_source_value'],prefix='race',drop_first=True)
        person = person[['person_id','scaled_age']]
        person = person.join(dummy_columns_race)
        person = person.join(dummy_columns_gender)
        person.to_csv( '/scratch/demographic_data_infer.csv',index=False)



    def logit_model(self,filename):
        data = pd.read_csv(filename)
        X = data.drop(['person_id'], axis = 1)
        features = X.columns.values
        X = np.array(X)
        clf =  load('/model/baseline.joblib')
        Y_pred = clf.predict_proba(X)[:,1]
        person_id = data.person_id
        output = pd.DataFrame(Y_pred,columns=['confidence'])
        output_prob = pd.concat([person_id,output],axis=1)
        output_prob.to_csv('/output/predictions.csv')





if __name__ == '__main__':
        FOLDER='scratch/'
        FILE_STR = 'infer_cleaned'
        op = OmopParser()
        op.add_demographic_data()
        op.logit_model('/scratch/demographic_data_infer.csv')
