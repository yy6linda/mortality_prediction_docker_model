
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
from sklearn.preprocessing import MinMaxScaler
'''for saving models'''
from joblib import dump
ROOT="/"


class OmopParser(object):
    '''this structures the omop dataset'''
    def __init__(self):
        self.name = 'omop_assembler'

    def add_prediction_date(self,file_name):
        '''given a patient's visit records, this function returns the prediction_date '''
        '''and whether this patient has a death record (1) or not(0)'''
        '''output is a reduced visit file'''
        visit=pd.read_csv('/train/visit_occurrence.csv')
        cols=['person_id','visit_start_date']
        visit=visit[cols]
        death=pd.read_csv('/train/death.csv')
        cols=['person_id','death_date']
        death=death[cols]
        visit_death=pd.merge(death,visit,on=['person_id'],how='left')
        visit_death['death_date'] = pd.to_datetime(visit_death['death_date'], format='%Y-%m-%d')
        visit_death['visit_start_date'] = pd.to_datetime(visit_death['visit_start_date'], format='%Y-%m-%d')
        visit_death['last_visit_death'] = visit_death['death_date'] - visit_death['visit_start_date']
        visit_death['last_visit_death']= visit_death['last_visit_death'].apply(lambda x: x.days)
        visit_death=visit_death.loc[visit_death['last_visit_death']<=180]
        visit_death=visit_death.drop_duplicates(subset=['person_id'], keep='first')
        visit_death=visit_death[['person_id','visit_start_date']]
        visit_death.columns=['person_id','prediction_date']
        visit_death['death']=np.ones(visit_death.shape[0])
        visit_live=visit[~visit.person_id.isin(visit_death.person_id)]
        visit_live=visit_live[['person_id','visit_start_date']]
        live_id = visit_live[['person_id']].drop_duplicates(keep='first')
        '''
        for patients in the negative case, select patients' latest visit record
        '''
        lst=[]
        for id in live_id.person_id:
            individual_visit = visit_live.loc[visit_live['person_id']==id]
            individual_visit =individual_visit.sort_values(by ='visit_start_date',ascending=False)
            lst.append([id,individual_visit.iloc[0,1]])
        cols=['person_id','prediction_date']
        visit_live = pd.DataFrame(lst, columns=cols)
        visit_live['death']= np.zeros(visit_live.shape[0])
        prediction_date=pd.concat([visit_death,visit_live],axis=0)
        prediction_date.to_csv(file_name[0:-4] + '_prediction_date.csv',index=False)

    def add_demographic_data(self,file_name):
        '''add demographic data'''
        person=pd.read_csv('/train/person.csv')
        prediction_date = pd.read_csv(file_name)
        cols=['person_id','gender_concept_id','year_of_birth','race_source_value']
        person=person[cols]
        person_prediction_date=pd.merge(prediction_date,person,on=['person_id'], how='left')
        person_prediction_date['prediction_date'] = pd.to_datetime(person_prediction_date['prediction_date'], format='%Y-%m-%d')
        person_prediction_date['year_of_birth'] = pd.to_datetime(person_prediction_date['year_of_birth'], format='%Y')
        person_prediction_date['age']=person_prediction_date['prediction_date']-person_prediction_date['year_of_birth']
        person_prediction_date['age']=person_prediction_date['age'].apply(lambda x: round(x.days/365.25))
        dummy_columns_gender = pd.get_dummies(person_prediction_date['gender_concept_id']).rename(columns=lambda x: 'gender_' + str(x))
        dummy_columns_gender = dummy_columns_gender[['gender_8532']]
        dummy_columns_race = pd.get_dummies(person['race_source_value']).rename(columns=lambda x: 'race_' + str(x))
        dummy_columns_race = dummy_columns_race[['race_1','race_2','race_3']]
        ##scaling the age
        scaler = MinMaxScaler(feature_range=(0, 1), copy=True)
        scaled_column = scaler.fit_transform(person_prediction_date[['age']])
        person_prediction_date = pd.concat([person_prediction_date, pd.DataFrame(scaled_column,columns=['scaled_age'])],axis=1)
        dummy_columns_race = pd.get_dummies(person_prediction_date['race_source_value'],prefix='race',drop_first=True)
        person_prediction_date =person_prediction_date[['death','person_id','scaled_age']]
        person_prediction_date = person_prediction_date.join(dummy_columns_race)
        person_prediction_date = person_prediction_date.join(dummy_columns_gender)
        '''
        output demographic_plot.csv is for the function plot_all_demographic_distribution()
        '''
        person_prediction_date.to_csv(file_name[0:-4] + '_plus_demographic_data.csv',index=False)



    def logit_model(self,filename):
        data = pd.read_csv(filename)
        X = data.drop(['death','person_id'], axis = 1)
        features = X.columns.values
        X = np.array(X)
        Y = np.array(data[['death']]).ravel()
        print(X.shape)
        print(Y.shape)
        clf = LogisticRegressionCV(cv = 5, penalty='l2', tol=0.0001, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None,
        max_iter=100, verbose=0, n_jobs=None).fit(X,Y)
        dump(clf, '/model/baseline.joblib')



if __name__ == '__main__':
        FOLDER='scratch/'
        FILE_STR = 'train_cleaned'
        op = OmopParser()
        op.add_prediction_date(ROOT+ FOLDER + FILE_STR + '.csv')
        op.add_demographic_data(ROOT+ FOLDER + FILE_STR + '_prediction_date.csv')
        op.logit_model('/scratch/train_cleaned_prediction_date_plus_demographic_data.csv')
