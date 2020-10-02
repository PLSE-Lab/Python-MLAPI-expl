from keras.models import load_model
from sklearn.externals import joblib
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.metrics import classification_report
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import make_scorer, accuracy_score 
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split 

#import data
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        

#split raw data into features and label
data_train = pd.read_csv('../input/csvdata/training.csv')
data_test = pd.read_csv('../input/csvdata/testing.csv')
data_train.sample(1)
y_train = data_train["label"] 
del data_train["label"]
X_train = data_train 
X_test = data_test

#split tranining data into tranining data for tranining model and valid data for evaluating the performance.
X_training, X_valid, y_training, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=0)



#KNN model training
Classifier_KNN = KNeighborsClassifier(n_neighbors=3, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None)
Classifier_KNN.fit(X_training, y_training)#Training Classifier
#Evaluate the performance
Prediction_KNN = Classifier_KNN.predict(X_valid)
Accuracy_KNN = accuracy_score(y_valid, Prediction_KNN)
print("KNeighbors Classifier accuracy is:")
print(Accuracy_KNN)
print("Full report is:")
print(classification_report(y_valid, Prediction_KNN))
joblib.dump(Classifier_KNN, '/kaggle/working/Classifier_KNN.pkl')#Save the model to local file
TEST_Prediction_KNN=Classifier_KNN.predict(X_test)
df=pd.DataFrame(TEST_Prediction_KNN)
df.to_csv('KNN.csv')

#Random Forest model training
Classifier_RF = RandomForestClassifier(n_estimators=100,criterion='gini',max_depth=None,min_samples_split=2,min_samples_leaf=1,min_weight_fraction_leaf=0.0,max_features='sqrt',max_leaf_nodes=None,warm_start=False,class_weight="balanced_subsample")
Classifier_RF.fit(X_training, y_training)#Training Classifier
Prediction_RF = Classifier_RF.predict(X_valid)
Accuracy_RF = accuracy_score(y_valid, Prediction_RF)#Evaluate the performance
print("RandomForest Classifier accuracy is:")
print(Accuracy_RF)
print("Full report is:")
print(classification_report(y_valid, Prediction_RF))
joblib.dump(Classifier_RF, '/kaggle/working/Classifier_RF.pkl')#Save the model to local file
TEST_Prediction_RF=Classifier_RF.predict(X_test)
df=pd.DataFrame(TEST_Prediction_RF)
df.to_csv('RF.csv')


#Naive Bayes Classifier
Classifier_NB =GaussianNB(priors=None, var_smoothing=1e-09)
Classifier_NB.fit(X_training, y_training)#Training Classifier
Prediction_NB = Classifier_NB.predict(X_valid)
Accuracy_NB = accuracy_score(y_valid, Prediction_NB)#Evaluate the performance
print("Naive Bayes Classifier accuracy is:")
print(Accuracy_NB)
print("Full report is:")
print(classification_report(y_valid, Prediction_NB))
joblib.dump(Classifier_NB, '/kaggle/working/Classifier_NB.pkl')#Save the model to local file
TEST_Prediction_NB=Classifier_NB.predict(X_test)
df=pd.DataFrame(TEST_Prediction_NB)
df.to_csv('NB.csv')

#Decision Tree Classifier
Classifier_DT = DecisionTreeClassifier(criterion='entropy',splitter='best',max_depth=None,min_samples_split=2,min_samples_leaf=1,min_weight_fraction_leaf=0.0,max_features=None,max_leaf_nodes=None, class_weight='balanced')
Classifier_DT.fit(X_training, y_training)#Training Classifier
Prediction_DT = Classifier_DT.predict(X_valid)
Accuracy_DT = accuracy_score(y_valid, Prediction_DT)#Evaluate the performance
print("DecisionTree Classifier accuracy is:")
print(Accuracy_DT)
print("Full report is:")
print(classification_report(y_valid, Prediction_DT))
joblib.dump(Classifier_DT, '/kaggle/working/Classifier_DT.pkl')#Save the model to local file
TEST_Prediction_DT=Classifier_DT.predict(X_test)
df=pd.DataFrame(TEST_Prediction_DT)
df.to_csv('DT.csv')

#LinearSVC Classifier
Classifier_LinearSVC = LinearSVC(penalty='l2',loss='hinge', dual=True, tol=0.0001, C=0.1, multi_class='crammer_singer')
Classifier_LinearSVC.fit(X_training, y_training)#Training Classifier
Prediction_LinearSVC = Classifier_LinearSVC.predict(X_valid)
Accuracy_LinearSVC = accuracy_score(y_valid, Prediction_LinearSVC)#Evaluate the performance
print("LinearSVC Classifier accuracy is:")
print(Accuracy_LinearSVC)
print("Full report is:")
print(classification_report(y_valid, Prediction_LinearSVC))
joblib.dump(Classifier_LinearSVC, '/kaggle/working/Classifier_LinearSVC.pkl')#Save the model to local file
TEST_Prediction_LinearSVC=Classifier_LinearSVC.predict(X_test)
df=pd.DataFrame(TEST_Prediction_LinearSVC)
df.to_csv('LinearSVC.csv')
