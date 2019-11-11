import datetime
import pandas as pd
import numpy as np
from datetime import datetime
'''for implementing simple logisticregression'''
import sklearn
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
'''for saving models'''
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
        visit = pd.read_csv('/train/visit_occurrence.csv')
        cols = ['person_id','visit_start_date']
        visit = visit[cols]
        death = pd.read_csv('/train/death.csv')
        cols = ['person_id','death_date']
        death = death[cols]
        visit_death = pd.merge(death,visit,on=['person_id'],how='inner')
        visit_death['death_date'] = pd.to_datetime(visit_death['death_date'], format='%Y-%m-%d')
        visit_death['visit_start_date'] = pd.to_datetime(visit_death['visit_start_date'], format='%Y-%m-%d')
        visit_death['last_visit_death'] = visit_death['death_date'] - visit_death['visit_start_date']
        visit_death['last_visit_death'] = visit_death['last_visit_death'].apply(lambda x: x.days)
        visit_death = visit_death.loc[visit_death['last_visit_death'] <= 180]
        visit_death.drop_duplicates(subset=['person_id'], keep = 'first',inplace = True)
        visit_death = visit_death[['person_id','visit_start_date']]
        visit_death.columns = ['person_id','prediction_date']
        visit_death['death'] = np.ones(visit_death.shape[0])
        visit_live = visit[~visit.person_id.isin(visit_death.person_id)]
        visit_live = visit_live[['person_id','visit_start_date']]
        '''
        for patients in the negative case, select patients' latest visit record
        '''
        visit_live = visit_live.sort_values(['person_id','visit_start_date'],ascending=False).groupby('person_id').head(1)
        visit_live = visit_live[['person_id','visit_start_date']]
        visit_live.columns = ["person_id", "prediction_date"]
        visit_live['death'] = np.zeros(visit_live.shape[0])
        prediction_date = pd.concat([visit_death,visit_live],axis=0)
        prediction_date.to_csv(file_name[0:-4] + '_prediction_date.csv',index=False)

    def add_demographic_data(self,file_name):
        '''add demographic data including age, gender and race'''
        person = pd.read_csv('/train/person.csv')
        prediction_date = pd.read_csv(file_name)
        cols = ['person_id','gender_concept_id','year_of_birth','race_concept_id']
        person = person[cols]
        person_prediction_date = pd.merge(prediction_date,person,on=['person_id'], how='left')
        person_prediction_date['prediction_date'] = pd.to_datetime(person_prediction_date['prediction_date'], format='%Y-%m-%d')
        person_prediction_date['year_of_birth'] = pd.to_datetime(person_prediction_date['year_of_birth'], format='%Y')
        person_prediction_date['age'] = person_prediction_date['prediction_date'] - person_prediction_date['year_of_birth']
        person_prediction_date['age'] = person_prediction_date['age'].apply(lambda x: x.days/365.25)
        person["count"] = 1
        gender = person.pivot(index = "person_id", columns="gender_concept_id", values="count")
        gender.reset_index(inplace = True)
        gender.fillna(0,inplace = True)
        race = person.pivot(index ="person_id", columns="race_concept_id", values="count")
        race.reset_index(inplace = True)
        race.fillna(0,inplace = True)
        race = race[['person_id', 8516, 8515, 8527, 8557, 8657]]
        gender = gender[['person_id',8532]]
        scaler = MinMaxScaler(feature_range = (0, 1), copy = True)
        scaled_column = scaler.fit_transform(person_prediction_date[['age']])
        person_prediction_date = pd.concat([person_prediction_date, pd.DataFrame(scaled_column,columns = ['scaled_age'])],axis=1)
        mortality_predictors = person_prediction_date[['death','person_id','scaled_age']]
        mortality_predictors = mortality_predictors.merge(gender, on = ['person_id'], how = 'left')
        mortality_predictors = mortality_predictors.merge(race, on = ['person_id'], how = 'left')
        mortality_predictors.fillna(0,inplace = True)
        mortality_predictors.to_csv(file_name[0:-4] + '_plus_demographic_data.csv',index = False)

    def logit_model(self,filename):
        '''
        apply logistic regression models for selected demographics features and use GridSearchCV to optimize parameters
        '''
        data = pd.read_csv(filename,low_memory = False).dropna(axis = 0,how = 'any')
        X = data.drop(['death','person_id'], axis = 1)
        features = X.columns.values
        Y = data[['death']]
        X = np.array(X)
        Y = np.array(data[['death']]).ravel()
        clf = LogisticRegressionCV(cv = 20, penalty = 'l2', tol = 0.0001, fit_intercept = True, intercept_scaling = 1, class_weight = None, random_state = None,
        max_iter = 100, verbose = 0, n_jobs = None).fit(X,Y)
        dump(clf, '/model/baseline.joblib')

if __name__ == '__main__':
    FOLDER = 'scratch/'
    FILE_STR = 'train_cleaned'
    op = OmopParser()
    op.add_prediction_date(ROOT + FOLDER + FILE_STR + '.csv')
    op.add_demographic_data(ROOT + FOLDER + FILE_STR + '_prediction_date.csv')
    op.logit_model(ROOT + FOLDER + FILE_STR + '_prediction_date_plus_demographic_data.csv')
