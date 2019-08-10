import pickle
import math
import re
import csv
import concurrent.futures
import os
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
from sklearn.model_selection import GridSearchCV
import pandas_ml
from pandas_ml import ConfusionMatrix
from joblib import dump

ROOT="/"


class OmopParser(object):
    '''this structures the omop dataset'''
    def __init__(self):
        self.name = 'omop_parser'

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
        print("print visit_death")
        print(visit_death.head(10))
        visit_live=visit[~visit.person_id.isin(visit_death.person_id)]
        visit_live=visit_live[['person_id','visit_start_date']]
        '''
        for patients in the negative case, select patients' latest visit record
        '''
        visit_live = visit_live.sort_values(['person_id','visit_start_date'],ascending=False).groupby('person_id').head(1)
        visit_live = visit_live[['person_id','visit_start_date']]
        visit_live.columns = ["person_id", "prediction_date"]
        visit_live['death']= np.zeros(visit_live.shape[0])
        prediction_date=pd.concat([visit_death,visit_live],axis=0)
        print("print visit_live")
        print(visit_live.head(10))
        print("in training set, #positive #negative " )
        print(prediction_date.groupby('death').count())
        prediction_date.to_csv(file_name[0:-4] + '_prediction_date.csv',index=False)

    def add_demographic_data(self,file_name):
        '''add demographic data'''
        person=pd.read_csv('/train/person.csv')
        prediction_date = pd.read_csv(file_name)
        cols=['person_id','gender_concept_id','year_of_birth','race_concept_id']
        person=person[cols]
        person_prediction_date=pd.merge(prediction_date,person,on=['person_id'], how='left')
        person_prediction_date['prediction_date'] = pd.to_datetime(person_prediction_date['prediction_date'], format='%Y-%m-%d')
        person_prediction_date['year_of_birth'] = pd.to_datetime(person_prediction_date['year_of_birth'], format='%Y')
        person_prediction_date['age']=person_prediction_date['prediction_date']-person_prediction_date['year_of_birth']
        person_prediction_date['age']=person_prediction_date['age'].apply(lambda x: x.days/365.25)
        person["count"] = 1
        gender = person.pivot(index="person_id", columns="gender_concept_id", values="count")
        gender.reset_index(inplace=True)
        gender.fillna(0,inplace = True)
        '''
        check race and gender
        '''
        print("gender component")
        print(person.groupby('gender_concept_id').count())
        print("race component")
        print(person.groupby('race_concept_id').count())
        race = person.pivot(index = "person_id", columns = "race_concept_id", values = "count")
        race.reset_index(inplace = True)
        race.fillna(0,inplace = True)
        race = race[['person_id', 8516, 8515, 8527, 8557, 8657]]
        gender = gender[['person_id', 8532]]
        '''scaling the age'''
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        scaled_column = scaler.fit_transform(person_prediction_date[['age']])
        person_prediction_date = pd.concat([person_prediction_date, pd.DataFrame(scaled_column,columns=['scaled_age'])],axis=1)
        mortality_predictors =person_prediction_date[['death','person_id','scaled_age']]
        mortality_predictors = mortality_predictors.merge(gender, on = ['person_id'], how = 'inner')
        mortality_predictors = mortality_predictors.merge(race, on = ['person_id'], how ='inner')
        mortality_predictors.fillna(0, inplace = True)
        mortality_predictors.to_csv(file_name[0:-4] + '_add_demographic_data.csv',index=False)

    def add_cancer(self,file_name):
        cancer = pd.read_csv('/app/cancer_condition_id.csv')
        cancer['snomed_concept_id'] =cancer['concept_id'].apply(pd.to_numeric,errors='ignore',downcast='signed')
        condition = pd.read_csv('/train/condition_occurrence.csv')
        condition_cancer = pd.merge(condition, cancer, on =['snomed_concept_id'], how = 'inner')
        condition_cancer = condition_cancer.drop_duplicates(subset=['person_id'], keep='first')
        print(condition_cancer.iloc[:,1:5].head(20))
        print("#patients with cancer{} ".format(condition_cancer.shape[0]))
        person_prediction_demographic = pd.read_csv(file_name)
        print("#patients total{}".format(person_prediction_demographic.shape[0]))
        person_prediction_demographic['cancer'] = np.zeros(person_prediction_demographic.shape[0])
        person_prediction_demographic.loc[person_prediction_demographic.person_id.isin(condition_cancer.person_id),'cancer']=1
        person_prediction_demographic.to_csv(file_name[0:-4] + '_cancer.csv',index=False)


    def add_chd(self,file_name):
        chd = pd.read_csv('/app/CHD_condition_id.csv')
        chd['snomed_concept_id'] = chd['concept_id'].apply(pd.to_numeric,errors='ignore',downcast='signed')
        condition = pd.read_csv('/train/condition_occurrence.csv')
        condition_chd = chd.merge(condition, on =['snomed_concept_id'], how = 'inner')
        condition_chd= condition_chd.drop_duplicates(subset=['person_id'], keep='first')
        print(condition_chd.iloc[:,1:5].head(20))
        print("#patients with chd {} ".format(condition_chd.shape[0]))
        person_prediction_demographic = pd.read_csv(file_name)
        person_prediction_demographic['CHD'] = np.zeros(person_prediction_demographic.shape[0])
        person_prediction_demographic.loc[person_prediction_demographic.person_id.isin(condition_chd.person_id),'CHD']=1
        person_prediction_demographic.to_csv(file_name[0:-4] + '_CHD.csv',index=False)

    def add_copd(self,file_name):
        copd = pd.read_csv('/app/COPD_condition_id.csv')
        copd['snomed_concept_id'] = copd['concept_id'].apply(pd.to_numeric,errors='ignore',downcast='signed')
        condition = pd.read_csv('/train/condition_occurrence.csv')
        condition_copd = copd.merge(condition, on =['snomed_concept_id'], how = 'inner')
        condition_copd= condition_copd.drop_duplicates(subset=['person_id'], keep='first')
        print(condition_copd.iloc[:,1:5].head(20))
        print("#patients with copd {} ".format(condition_copd.shape[0]))
        person_prediction_demographic = pd.read_csv(file_name)
        person_prediction_demographic['COPD'] = np.zeros(person_prediction_demographic.shape[0])
        person_prediction_demographic.loc[person_prediction_demographic.person_id.isin(condition_copd.person_id),'COPD']=1
        person_prediction_demographic.to_csv(file_name[0:-4] + '_COPD.csv',index=False)

    def add_t2dm(self,file_name):
        t2dm = pd.read_csv('/app/T2DM_condition_id.csv')
        t2dm['snomed_concept_id'] =t2dm['concept_id'].apply(pd.to_numeric,errors='ignore',downcast='signed')
        condition = pd.read_csv('/train/condition_occurrence.csv')
        condition_t2dm = t2dm.merge(condition, on =['snomed_concept_id'], how = 'inner')
        condition_t2dm = condition_t2dm.drop_duplicates(subset=['person_id'], keep='first')
        print("#patients with t2dm {} ".format(condition_t2dm.shape[0]))
        print(condition_t2dm.iloc[:,1:5].head(20),flush = True)
        person_prediction_demographic = pd.read_csv(file_name)
        person_prediction_demographic['T2DM'] = np.zeros(person_prediction_demographic.shape[0])
        person_prediction_demographic.loc[person_prediction_demographic.person_id.isin(condition_t2dm.person_id),'T2DM']=1
        person_prediction_demographic.to_csv(file_name[0:-4] + '_T2DM.csv',index=False)


    def logit_model(self,file_name):
        data = pd.read_csv(file_name)
        print("logit_model, print data")
        print(list(data.columns.values))
        X = data.drop(['death','person_id'], axis = 1).fillna(0)
        features = X.columns.values
        Y = data[['death']].fillna(0)
        X = np.array(X)
        Y = np.array(data[['death']]).ravel()
        '''gridsearchcv'''
        grid={"max_iter":[100,200,500,1000], "penalty":["l1","l2"]}# l1 lasso l2 ridge
        logreg=LogisticRegression()
        logreg_cv=GridSearchCV(logreg,grid,cv=10)
        logreg_cv.fit(X,Y)
        print("tuned hpyerparameters :(best parameters) ",logreg_cv.best_params_,flush = True)
        print("accuracy :",logreg_cv.best_score_,flush=True)
        clf = LogisticRegressionCV(cv = 5, penalty='l2', tol=0.0001, fit_intercept=True, intercept_scaling=1, class_weight=None, solver ='lbfgs', random_state=None,
        max_iter=100, verbose=0, n_jobs=None).fit(X,Y)
        dump(clf, '/model/baseline.joblib')




if __name__ == '__main__':
    print("start baseline 4 training", flush = True)
    FOLDER='scratch/'
    FILE_STR = 'train_cleaned'
    op = OmopParser()
    print("add prediction date", flush = True)
    op.add_prediction_date(ROOT+ FOLDER + FILE_STR + '.csv')
    op.add_demographic_data(ROOT+ FOLDER + FILE_STR +'_prediction_date.csv')
    op.add_cancer(ROOT +FOLDER+ FILE_STR+'_prediction_date_add_demographic_data.csv')
    op.add_chd(ROOT +FOLDER+ FILE_STR+'_prediction_date_add_demographic_data_cancer.csv')
    op.add_copd(ROOT +FOLDER+ FILE_STR+'_prediction_date_add_demographic_data_cancer_CHD.csv')
    op.add_t2dm(ROOT +FOLDER+ FILE_STR+'_prediction_date_add_demographic_data_cancer_CHD_COPD.csv')
    print("finished adding the 4 types", flush = True)
    op.logit_model(ROOT +FOLDER+ FILE_STR+'_prediction_date_add_demographic_data_cancer_CHD_COPD_T2DM.csv')
    print("finished logit model for training", flush = Ture)
