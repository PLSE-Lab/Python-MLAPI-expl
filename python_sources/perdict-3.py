# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
from keras.models import load_model
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.externals import joblib 
from sklearn.metrics import make_scorer, accuracy_score, classification_report
from sklearn.model_selection import train_test_split 
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

        
#KNNC = joblib.load('../input/csvdata/Classifier_KNN.pkl')
#CRFC= joblib.load('../input/csvdata/Classifier_RF.pkl')
#DTC= joblib.load('../input/csvdata/Classifier_DT.pkl')
#NBC= joblib.load('../input/csvdata/Classifier_NB.pkl')
#LinearSVCC= joblib.load('../input/csvdata/Classifier_LinearSVC.pkl')
                       
                       
                       
#split raw data into features and label
data_train = pd.read_csv('../input/csvdata/training.csv')
data_test = pd.read_csv('../input/csvdata/testing.csv')
data_train.sample(1)
y_train = data_train["label"] 
del data_train["label"]
X_train = data_train 
X_test = data_test
X_training, X_valid, y_training, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=0)


'''
#KNN Prediction on Test data
Classifier_KNN = joblib.load('../input/csvdata/Classifier_KNN.pkl')
TEST_Prediction_KNN=Classifier_KNN.predict(X_test)
df=pd.DataFrame(TEST_Prediction_KNN)
df.to_csv('KNN.csv')
'''



##Random Forest Prediction on Test data
Classifier_RF= joblib.load('../input/csvdata/Classifier_RF.pkl')
TEST_Prediction_RF=Classifier_RF.predict(X_test)
df=pd.DataFrame(TEST_Prediction_RF)
df.to_csv('RF.csv')

#Decision Tree Prediction on Test data
Classifier_DT= joblib.load('../input/csvdata/Classifier_DT.pkl')
TEST_Prediction_DT=Classifier_DT.predict(X_test)
df=pd.DataFrame(TEST_Prediction_DT)
df.to_csv('DT.csv')

#Naive Bayes Prediction on Test data

Classifier_NB= joblib.load('../input/csvdata/Classifier_NB.pkl')
TEST_Prediction_NB=Classifier_NB.predict(X_test)
df=pd.DataFrame(TEST_Prediction_NB)
df.to_csv('NB.csv')

#LinearSVC Classifier Prediction on Test data
Classifier_LinearSVC= joblib.load('../input/csvdata/Classifier_LinearSVC.pkl')
TEST_Prediction_LinearSVC=Classifier_LinearSVC.predict(X_test)
df=pd.DataFrame(TEST_Prediction_LinearSVC)
df.to_csv('LinearSVC.csv')

#CNN Prediction on Test data
CNN_model = load_model('../input/csvdata/model.h5')#Load the CNN model 
X_valid_RE=X_valid.values.reshape(-1, 28, 28, 1)#Reshape the input data
X_test=X_test.values.reshape(-1, 28, 28, 1)#Reshape the input data
Prediction_CNN = CNN_model.predict_classes(X_valid_RE)#Evaluate the performance
Accuracy_CNN = accuracy_score(y_valid, Prediction_CNN)
print('Accuracy_CNN:')
print(Accuracy_CNN)
print("Full report is:")
print(classification_report(y_valid, Prediction_CNN))
Prediction_CNN_TEST = CNN_model.predict_classes(X_test)
df=pd.DataFrame(Prediction_CNN_TEST)
df.to_csv('CNN.csv')


