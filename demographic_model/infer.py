from functools import reduce
import datetime
from operator import add
import pandas as pd
import numpy as np
from datetime import datetime
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
        self.name = 'omop_parser'

    def add_demographic_data(self):
        '''find the last visit date for all patients in the inference set'''
        visit=pd.read_csv('/infer/visit_occurrence.csv')
        person_last_visit = visit.sort_values(['person_id','visit_start_date'],ascending=False).groupby('person_id').head(1)
        person_last_visit = person_last_visit[['person_id','visit_start_date']]
        person_last_visit.columns = ["person_id", "last_visit_date"]
        '''add demographic data including age, gender and race'''
        person = pd.read_csv('/infer/person.csv')
        cols = ['person_id','gender_concept_id','year_of_birth','race_concept_id']
        person = person[cols]
        person = pd.merge(person_last_visit, person,on=['person_id'], how='left')
        person['year_of_birth'] = pd.to_datetime(person['year_of_birth'], format='%Y')
        person['last_visit_date'] = pd.to_datetime(person['last_visit_date'], format='%Y-%m-%d')
        person['age'] = person['last_visit_date'] - person['year_of_birth']
        person['age'] = person['age'].apply(lambda x: x.days/365.25)
        person["count"] = 1
        race = person.pivot(index="person_id", columns="race_concept_id", values="count")
        race.reset_index(inplace=True)
        race.fillna(0, inplace = True)
        race = race[['person_id',8516,8515,8527,8557,8657]]
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        scaled_column = scaler.fit_transform(person[['age']])
        person = pd.concat([person, pd.DataFrame(scaled_column,columns=['scaled_age'])],axis=1)
        gender = person.pivot(index="person_id", columns="gender_concept_id", values="count")
        gender.reset_index(inplace=True)
        gender.fillna(0,inplace = True)
        gender = gender[['person_id',8532]]
        person = person[['person_id','scaled_age']]
        person = person.merge(gender, how = "left", on = ['person_id'])
        person = person.merge(race, how = "left", on = ['person_id'])
        person.fillna(0, inplace = True)
        person.to_csv( '/scratch/demographic_data_infer.csv',index=False)


    def logit_model(self,filename):
        '''apply trained logistic regression model on the inferring dataset'''
        data = pd.read_csv(filename,low_memory=False)
        data.fillna(0, inplace = True)
        X = data.drop(['person_id'], axis = 1)
        features = X.columns.values
        X = np.array(X)
        clf =  load('/model/baseline.joblib')
        Y_pred = clf.predict_proba(X)[:,1]
        person_id = data.person_id
        output = pd.DataFrame(Y_pred,columns=['score'])
        output_prob = pd.concat([person_id,output],axis=1)
        output_prob.columns = ["person_id", "score"]
        output_prob.to_csv('/output/predictions.csv', index = False)

if __name__ == '__main__':
    FOLDER='scratch/'
    FILE_STR = 'infer_cleaned'
    op = OmopParser()
    op.add_demographic_data()
    op.logit_model('/scratch/demographic_data_infer.csv')
