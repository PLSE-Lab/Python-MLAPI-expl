#!/usr/bin/env python
# coding: utf-8

#    # **Deploy our ML model into Docker**
#    
#    for reference check with my git repo
#    https://github.com/satyamuralidhar/ML-Ops_ModelDeployment_k8s
#    
#    
#    **Prerequisites:**
#    ================
# 1. Docker
# 2. Git
# 3. Visual studio code
# 4. DockerHub Account
# 
# ![image.png](attachment:image.png)
# 
# 
# Create a ML model:
# =================
# 
# **For model refer my https://www.kaggle.com/muralidhar123/pima-indians-diabetes-prediction**
# 
# 
# 
# Need to install Docker:
# ======================
# 
# install in windows:
# 
# https://download.docker.com/win/stable/Docker%20Desktop%20Installer.exe
# 
# install in linux:
# 
# #apt-get install docker.io
# 
# 

# In[ ]:



Steps:
=====
1. create a folder named as app  <mkdir app>
    cd /app
2. move These files to app folder like:
    1. app.py
    2.Dockerfile
    3.model.ipynb
    4.model.pkl
    5.requirements.txt
3. Then apply docker Commands
    
    


# In[ ]:



Creating a UI using Flasgger:
============================

# app.py

from flask import Flask, request
import numpy as np
import pickle
import math
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import flasgger
from flasgger import Swagger
app=Flask(__name__)
Swagger(app)
with open('model.pkl', 'rb') as model_pkl:
    rf = pickle.load(model_pkl)
@app.route('/')
def welcome():
    return "Welcome All"
@app.route('/predict',methods=["GET"])
def predict_note_authentication():
    
    """Let's Authenticate the Diabetis
    This is using docstrings for specifications.
    ---
    parameters:
        - name: Pregnancies
          in: query
          type: number
          required: true
        - name: Glucose
          in: query
          type: number
          required: true
        - name: BloodPressure	
          in: query
          type: number
          required: true
        - name: SkinThickness
          in: query
          type: number
          required: true
        - name: Insulin
          in: query
          type: number
          required: true
        - name: BMI
          in: query
          type: number
          required: true
        - name: DiabetesPedigreeFunction	
          in: query
          type: number
          required: true
        - name: Age	
          in: query
          type: number
          required: true
    responses:
        200:
            description: The output values
        
    """
    Pregnancies = request.args.get('Pregnancies')
    Glucose = request.args.get('Glucose')
    BloodPressure = request.args.get('BloodPressure')
    SkinThickness = request.args.get('SkinThickness')
    Insulin = request.args.get('Insulin')
    BMI = request.args.get('BMI')
    DiabetesPedigreeFunction = request.args.get('DiabetesPedigreeFunction')
    Age = request.args.get('Age')
    pred = np.array([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]]).astype(np.float64)
    prediction = rf.predict(pred)
    print(prediction)
    return "result "+str(prediction)
if __name__=='__main__':
    app.run(debug=True,host='0.0.0.0',port=5000)
    


# In[ ]:



Create a requirements.txt file:
===============================

#requirements.txt:
-----------------
flasgger==0.9.3
pandas==0.24.2
jsonschema==2.6.0
certifi==2019.11.28
Click==7.0
Flask==1.1.1
Flask-Cors==3.0.8
gunicorn==20.0.4
itsdangerous==1.1.0
Jinja2==2.10.3
joblib==0.14.0
MarkupSafe==1.1.1
numpy==1.17.4
scikit-learn==0.23.1
scipy==1.3.2
six==1.13.0
Werkzeug==0.16.0
imbalanced-learn==0.6.1


# 
# Create a Dockerfile:
# ====================
# 
# #Dockerfile:
# ============
# 
# FROM python:3.7
# 
# COPY . /app
# 
# WORKDIR /app
# 
# COPY . /model.pkl
# 
# RUN pip3 install -r requirements.txt
# 
# EXPOSE 5000
# 
# CMD python3 app.py
# 
# 

# In[ ]:


NOTE: 
=====
Apply These command where we have Dockerfile

#docker build -t diabetis:v1 . #note v1 '.' is necessary

#docker -d -p 5000:5000 --name diabetis diabetis:v1

first we need to create a account on dockerhub

after building is done .

tag our image to yours dockerhub username:

#docker tag diabetis:v1 <Dockerhub user name>/diabetis:v1

then login

#docker login

give ur username and password

#docker push <Dockerhub user name>/diabetis:v1

