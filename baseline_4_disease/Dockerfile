FROM python:3.5
RUN pip install pandas
RUN pip install numpy
RUN pip install sklearn
RUN pip install matplotlib
RUN pip install datetime
RUN pip install xgboost
RUN pip install keras
RUN pip install tensorflow
RUN pip install eli5
RUN pip install pandas_ml
RUN pip install joblib

RUN mkdir app	

COPY ./baseline_condition/CHD_condition_id.csv /app/
COPY ./baseline_condition/COPD_condition_id.csv /app/
COPY ./baseline_condition/T2DM_condition_id.csv /app/
COPY ./baseline_condition/cancer_condition_id.csv /app/

# Copy our python program for training and inference
COPY ./train.py /app/
COPY ./infer.py /app/

# Copy Bash scripts expected by the IT infrastructure of the EHR DREAM Challenge
COPY ./train.sh /app/
COPY ./infer.sh /app/


RUN chmod +X /app/train.sh
RUN chmod +X /app/infer.sh

