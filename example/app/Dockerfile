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
RUN pip install joblib

RUN mkdir app	


COPY ./train.py /app/
COPY ./infer.py /app/

COPY ./train.sh /app/
COPY ./infer.sh /app/



RUN chmod +X /app/train.sh

