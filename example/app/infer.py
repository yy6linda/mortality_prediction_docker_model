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
        '''find the last visit date for all patients in the inference set'''
        visit=pd.read_csv('/infer/visit_occurrence.csv')
        lst =[]
        visit_id = visit[['person_id']].drop_duplicates(keep='first')
        for id in visit_id.person_id:
            individual_visit = visit.loc[visit['person_id']==id]
            individual_visit =individual_visit.sort_values(by ='visit_start_date',ascending=False)
            lst.append([id,individual_visit.iloc[0,1]])
        cols=['person_id','last_visit_date']
        person_last_visit = pd.DataFrame(lst, columns=cols)

        '''add demographic data'''
        person = pd.read_csv('/infer/person.csv')
        cols = ['person_id','gender_concept_id','year_of_birth','race_source_value']
        person = person[cols]
        person = pd.merge(person_last_visit, person,on=['person_id'], how='inner')
        person['year_of_birth'] = pd.to_datetime(person['year_of_birth'], format='%Y')
        person['last_visit_date'] = pd.to_datetime(person['last_visit_date'], format='%Y-%m-%d')
        person['age'] = person['last_visit_date'] - person['year_of_birth']
        person['age'] = person['age'].apply(lambda x: round(x.days/365.25))
        dummy_columns_gender = pd.get_dummies(person['gender_concept_id'],prefix='gender',drop_first=True)
        ##scaling the age
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        scaled_column = scaler.fit_transform(person[['age']])
        person = pd.concat([person, pd.DataFrame(scaled_column,columns=['scaled_age'])],axis=1)
        dummy_columns_gender = pd.get_dummies(person['gender_concept_id']).rename(columns=lambda x: 'gender_' + str(x))
        dummy_columns_gender = dummy_columns_gender[['gender_8532']]
        dummy_columns_race = pd.get_dummies(person['race_source_value']).rename(columns=lambda x: 'race_' + str(x))
        dummy_columns_race = dummy_columns_race[['race_1','race_2','race_3']]
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
        output_prob.to_csv('/output/predictions.csv', index = False)


if __name__ == '__main__':
        FOLDER='scratch/'
        FILE_STR = 'infer_cleaned'
        op = OmopParser()
        op.add_demographic_data()
        op.logit_model('/scratch/demographic_data_infer.csv')
