#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


#To remove depricated warnings
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 


# In[ ]:


#Loading train and test files:
train_path="/kaggle/input/human-activity-recognition-with-smartphones/train.csv"
df_train=pd.read_csv(train_path)
test_path="/kaggle/input/human-activity-recognition-with-smartphones/test.csv"
df_test=pd.read_csv(test_path)
df = pd.concat([df_train,df_test])


#                                                         About Project dataset
# Human activity Recognition using smartphones Data set The experiments have been carried out with a group of 30 volunteers within an age bracket of 19-48 years. Each person performed six activities (WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING) wearing a smartphone (Samsung Galaxy S II) on the waist. Using its embedded accelerometer and gyroscope, we captured 3-axial linear acceleration and 3-axial angular velocity at a constant rate of 50Hz. The experiments have been video-recorded to label the data manually. The obtained dataset has been randomly partitioned into two sets, where 70% of the volunteers was selected for generating the training data and 30% the test data.
# 
# The sensor signals (accelerometer and gyroscope) were pre-processed by applying noise filters and then sampled in fixed-width sliding windows of 2.56 sec and 50% overlap (128 readings/window). The sensor acceleration signal, which has gravitational and body motion components, was separated using a Butterworth low-pass filter into body acceleration and gravity. The gravitational force is assumed to have only low frequency components, therefore a filter with 0.3 Hz cutoff frequency was used. From each window, a vector of features was obtained by calculating variables from the time and frequency domain.
# 
# Attribute Information:
# 
# For each record in the dataset it is provided:
# 
# Triaxial acceleration from the accelerometer (total acceleration) and the estimated body acceleration.
# Triaxial Angular velocity from the gyroscope.
# A 561-feature vector with time and frequency domain variables.
# Its activity label.
# An identifier of the subject who carried out the experiment.
# Data Analysis:
# 
# 1. From the give signals of x,y,z axis of accelrometer and gyroscope, we aim to provide the classification to following groups.
# 
# Laying
# Standing
# Sitting
# Walking
# Walking_Upstairs
# Walking_Downstairs

#                                                       What to expect from this Notebook?
# 1. PreProcessing steps
#     * Reading and understanding the data with basic panda library
#     * Encoding categorical values
#     * Finding outliners using pyod library
#     * Feature selection using PCA
# 2. Model selection and comaprision
#     * Pipeline to create Classifier models
#     * Hyper tuning the models
#     * Display of classification results and comparision of models.
#     

# # Preprocessing

# In[ ]:


df.head(10)


# In[ ]:


df.columns
# Program to remove all whitespaces
import re
# matches all whitespace characters
pattern = r'[()-.,]+'

tempcol = []
col_new = []
for col in df_train.columns:
  new_string = re.sub(pattern, '_', col) 
  col_new.append(new_string)
  tempcol.append(new_string.split('_')[0])
  
df.columns = col_new

#The main features are:
print('The main columns are:')
for temp in list(set(tempcol)):
    print(temp)


# The total 563 columns basically consist of above parameter divided on x, y and z axis.

# In[ ]:


# Returns a Summary dataframe for numeric columns only 
# output will be same as host_df.describe()
df.describe(exclude='O')
# Returns a Summary dataframe 
#  for object type (or categorical) columns only 
df.describe(include='O')
#Cheking data disribtion
#1. Value greater then -1 and 1 , data is higly skewed
#2. Values between -1 to -0.5 and 0.5 to 1 is less skewed
#3. Values between -0.5 to 0.5 is almost symmetrically distributed
df.skew()

# Create correlation matrix
corr_matrix = df.corr().abs()
print(corr_matrix)


# In[ ]:


#Since its a classification problem, its important to know if data is balanced or not?
print(df.Activity.value_counts())
df.Activity.value_counts().plot.bar()


# # Encoding catageorical values

# In[ ]:


from sklearn import preprocessing
le = preprocessing.LabelEncoder()
#le.fit(["WALKING", "LAYING", "STANDING", "SITTING","WALKING_UPSTAIRS","WALKING_DOWNSTAIRS"])
df['Activity'] = le.fit_transform(df['Activity'])
df_en = df.drop(columns=['subject']) #dropping unwaned columns
df_en_data = df_en.drop(columns = ['Activity'])
df_en_target = df_en['Activity']
df_en_target.value_counts()


# In[ ]:


get_ipython().system('pip install pyod')


# # To detect outliners using pyod lib

# In[ ]:



from pyod.models.abod import ABOD
from pyod.models.cblof import CBLOF
#from pyod.models.feature_bagging import FeatureBagging
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lof import LOF

random_state = np.random.RandomState(42)
#Removing 5% outliners
outliers_fraction = 0.05
# Define outlier detection tools to be compared
classifiers = {
        'Angle-based Outlier Detector (ABOD)': ABOD(contamination=outliers_fraction),
        'Cluster-based Local Outlier Factor (CBLOF)':CBLOF(contamination=outliers_fraction,check_estimator=False, random_state=random_state),
        #'Feature Bagging':FeatureBagging(LOF(n_neighbors=35),contamination=outliers_fraction,check_estimator=False,random_state=random_state),
        'Isolation Forest': IForest(contamination=outliers_fraction,random_state=random_state),
        'K Nearest Neighbors (KNN)': KNN(contamination=outliers_fraction),
        'Average KNN': KNN(method='mean',contamination=outliers_fraction)
}

for i, (clf_name, clf) in enumerate(classifiers.items()):
    clf.fit(df_en)
    # predict raw anomaly score
    scores_pred = clf.decision_function(df_en) * -1
        
    # prediction of a datapoint category outlier or inlier
    y_pred = clf.predict(df_en)
    n_inliers = len(y_pred) - np.count_nonzero(y_pred)
    n_outliers = np.count_nonzero(y_pred == 1)
    
    print('OUTLIERS : ',n_outliers,'INLIERS : ',n_inliers, clf_name)
    print('-'*50)


# Scaling the dataset , because many outliners are detected

# In[ ]:


'''
Using Robust based technique to scale data as it has outliners
'''
from sklearn.preprocessing import RobustScaler
robustscaler = RobustScaler()
df_robust = robustscaler.fit_transform(df_en_data)
df_robust


# # Applying PCA

# In[ ]:


#from sklearn.preprocessing import StandardScaler
#df_en_std = StandardScaler().fit_transform(df_en)
print('Covariance matrix \n')
df_en_cov_mat= np.cov(df_robust, rowvar=False)
df_en_cov_mat
df_en_cov_mat = np.cov(df_robust.T)
eig_vals, eig_vecs = np.linalg.eig(df_en_cov_mat)
print('Eigenvectors \n%s' %eig_vecs)
print('\nEigenvalues \n%s' %eig_vals)
tot = sum(eig_vals)
print("\n",tot)
var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
print("\n\n1. Variance Explained\n",var_exp)
cum_var_exp = np.cumsum(var_exp)
print("\n\n2. Cumulative Variance Explained\n",cum_var_exp)
print("\n\n3. Percentage of variance the first 200 principal components each contain\n ",var_exp[0:200])
print("\n\n4. Percentage of variance the first 200 principal components together contain\n",sum(var_exp[0:200]))


# In[ ]:


# Splitting the training and test data


# In[ ]:


from sklearn.decomposition import PCA
pca = PCA(n_components=200)
principalComponents = pca.fit_transform(df_robust)
df_pca = pd.DataFrame(data = principalComponents)

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(df_pca,df_en_target,test_size=0.4, random_state=42)


# In[ ]:


#Apply model
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report


# # Model Selection

# In[ ]:


Classifiers=[LogisticRegression(max_iter=1000, C=0.1,solver= 'newton-cg'),
             DecisionTreeClassifier(class_weight =  'balanced', criterion = 'entropy'),
             RandomForestClassifier(class_weight =  'balanced', criterion = 'entropy'),
             GradientBoostingClassifier(),
             AdaBoostClassifier(),
             ExtraTreesClassifier(),
             KNeighborsClassifier(),
             SVC(kernel="linear",degree=3,C=10,gamma=0.001),
             GaussianNB()]
pipelines = []
for model in Classifiers:
    pipeline = make_pipeline(
              model)
    pipelines.append(pipeline)
for pipeline in pipelines:
    pipeline.fit(xtrain, ytrain)


# In[ ]:


Accuracy_mean = []
Accuracy_std = []
model_names = ['LR','DTC','RFC','GBC','AB','ET','KNN','SVC','GNB']
outcome = []
#models scores
for pipeline in pipelines:
    print(pipeline)
    print('Train Score: ',pipeline.score(xtrain, ytrain))
    print('Test Score: ',pipeline.score(xtest, ytest))
    pred = pipeline.predict(xtest)
    precision_score_temp = precision_score(ytest, pred, average='micro')
    recall_score_temp = recall_score(ytest, pred, average='micro')
    f1_score_temp = f1_score(ytest, pred, average='micro')
    all_accuracies = cross_val_score(estimator=pipeline, X=xtrain, y=ytrain, cv=5)
    print(f'All Accuracies: {all_accuracies}')
    print(f'Mean Accuracies: {all_accuracies.mean()}')
    print(f'Std of Accuracies: {all_accuracies.std()}')
    print(f'Accuracy: {accuracy_score(ytest, pred)}')
    #print(f'Precision: {precision_score_temp}')
    #print(f'Recall: {recall_score_temp}')
    #print(f'f1: {f1_score_temp}')
    print(classification_report(ytest, pred))
    print('Confusion_matrix:')
    print(f'{confusion_matrix(ytest, pred ,labels=[0,1,2,3,4,5])}')
    Accuracy_mean.append(all_accuracies.mean())
    Accuracy_std.append(all_accuracies.std())
    outcome.append(all_accuracies)
    print('*'*50)


# In[ ]:


import matplotlib.pyplot as plt
fig = plt.figure()
fig.suptitle('Machine Learning Model Comparison')
ax = fig.add_subplot(111)
plt.boxplot(outcome)
ax.set_xticklabels(model_names)
plt.show()


# In[ ]:


import matplotlib.pyplot as plt
fig = plt.figure()
fig.suptitle('Machine Learning Model Comparison')
ax = fig.add_subplot(111)
plt.bar(model_names,Accuracy_mean)
ax.set_xticklabels(model_names)
plt.show()


#                             Code For Hyper-tuning the model using grid search and finding the best parameter

# In[ ]:


#Making pipeline for Logestic regression:
#from sklearn.linear_model import LogisticRegression
#steps = [('LR', LogisticRegression())]
#make_pipeline = Pipeline(steps) # define the pipeline object.    
#parameteres = {'LR__max_iter':[1000, 5000],'LR__C':[0.1,10,100,10e5], 'LR__fit_intercept':[True, False], 'LR__class_weight':[None , 'balanced'], 'LR__solver' : ['newton-cg', 'lbfgs', 'liblinear']}
#grid_LR = GridSearchCV(make_pipeline, param_grid=parameteres)
#grid_LR.fit(xtrain, ytrain)

#grid_preds = grid_LR.predict(xtest)
#accuracy = accuracy_score(ytest, grid_preds)
#precision = precision_score(ytest, grid_preds, average='micro')
#recall = recall_score(ytest, grid_preds, average='micro')
#f1 = f1_score(ytest, grid_preds, average='micro')
#['LAYING', 'SITTING', 'STANDING', 'WALKING', 'WALKING_DOWNSTAIRS','WALKING_UPSTAIRS']
#confusion_matrix(ytest, grid_preds ,labels=[0,1,2,3,4,5])
#print('Best params: ', grid_LR.best_params_)
#print('Best score: ', grid_LR.best_score_)
#print('LogisticRegression accuracy: ', accuracy)
#print('LogisticRegression precision: ', precision)
#print('LogisticRegression recall: ', recall)
#print('LogisticRegression f1: ', f1)
#print("score = %3.2f" %(grid_LR.score(xtest,ytest)))
#print(f'Accuracy LogisticRegression classifier on training set {grid_LR.score(xtrain, ytrain)}')
#print(f'Accuracy LogisticRegression classifier on testing set {grid_LR.score(xtest, ytest)}')


# In[ ]:


#Making pipeline for SVC:
#from sklearn.svm import SVC
#steps = [('SVC', SVC())]
#make_pipeline = Pipeline(steps) # define the pipeline object.    
#parameteres = {'SVC__C':[0.001,0.1,10,100,10e5], 'SVC__kernel':['linear', 'poly', 'rbf', 'sigmoid'], 'SVC__degree':[3,4,5], 'SVC__class_weight' : [None,'balanced']}
#grid_SVC = GridSearchCV(make_pipeline, param_grid=parameteres)
#grid_SVC.fit(xtrain, ytrain)

#grid_preds = grid_SVC.predict(xtest)
#accuracy = accuracy_score(ytest, grid_preds)
#precision = precision_score(ytest, grid_preds, average='micro')
#recall = recall_score(ytest, grid_preds, average='micro')
#f1 = f1_score(ytest, grid_preds, average='micro')
#['LAYING', 'SITTING', 'STANDING', 'WALKING', 'WALKING_DOWNSTAIRS','WALKING_UPSTAIRS']
#confusion_matrix(ytest, grid_preds ,labels=[0,1,2,3,4,5])
#print('Best params: ', grid_SVC.best_params_)
#print('Best score: ', grid_SVC.best_score_)
#print('SVC accuracy: ', accuracy)
#print('SVC precision: ', precision)
#print('SVC recall: ', recall)
#print('SVC f1: ', f1)
#print("score = %3.2f" %(grid_SVC.score(xtest,ytest)))
#print(f'Accuracy SVC classifier on training set {grid_SVC.score(xtrain, ytrain)}')
#print(f'Accuracy SVC classifier on testing set {grid_SVC.score(xtest, ytest)}')
#print(classification_report(ytest, grid_preds))
#print(f'{confusion_matrix(ytest, grid_preds ,labels=[0,1,2,3,4,5])}')


# Conclusion: Logestic regression and SVC gave 97% accuracy with hypertuning of models and using PCA
