#!/usr/bin/env python
# coding: utf-8

# # Introduction:
# The challenge/goal of the project is to see how accurately we can predict weather a given patient has diabetes or not. Do we use all the features at our disposal ? or do we choose a subset to corelate with the 'Outcome' field? In this case it will be observed that removing certain features actually increase the accuracy of the prediction. In a previous [runbook](https://www.kaggle.com/arjunshenoymec/diabetes-prediction-using-svm) I attempted an approach of randomly removing variables (mostly based on intuition) to see which combination of features produced the maximum accuracy. The largest I could get was 77.47%.
# 
# In this runbook I am attempting a more organized approach based on statistical techniques to find out two things:
# * A way to methodically select the most appropriate features, such that the particular subset results in maximum accuracy.
# * To see if the accuracy of the models can be pushed beyond 77.4%. 
# 
# 
# ## Loading Libraries:
# All the required import statements were put into this particular block along with the set of statements to initialize the dataset. 

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import os
from matplotlib import pyplot as plt
import math
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

patient_data = pd.read_csv("../input/diabetes.csv")
print(patient_data.dtypes)
print(patient_data.head())
patient_data.describe()    


# ## Removing NaNs:
# The first step is to check if there are any columns with NaN values this can be done with the pandas isnull() function. If there are any depending on they type of data being stores you can use the fillna() function to replace NaN with some appropriate value (Typically median values are used for numeric data). 

# In[ ]:


print(patient_data.isnull().sum())


# ## Univariate Analysis and Binning
# The intention here is to obtain counts and observe distribution of the variables. I used count plots for variables having descrete values (such as number of Pregnancies) and KDE ("kernel density estimation (KDE) is a non-parametric way to estimate the probability density function of a random variable") plots for variables having continuous values (eg: Insulin, BloodPressure). 
# 
# Another Step that I took here is to create 'Bucket' and 'Log' variables for those which I saw skewed distributions. I also created 'AgeGroup' since that would also be more effective in observing trends (if any) rather than looking at raw Age value. 

# In[ ]:


# Note down all Transformations/new column creations here for clarity
patient_data['GlucoseBucket'] = patient_data['Glucose'].map(lambda i: math.ceil(i/10))
patient_data['SkinThicknessBucket'] = patient_data['SkinThickness'].map(lambda i: int(i/10))
patient_data['InsulinBucket'] = patient_data['Insulin'].map(lambda i: int(i/100))
patient_data['InsulinLog'] = patient_data['Insulin'].map(lambda i: np.log(i) if i > 0 else 0)
patient_data['DiabetesPedigreeFunctionLog'] = patient_data['DiabetesPedigreeFunction'].map(lambda i: np.log(i) if i > 0 else 0)
patient_data['DiabetesPedigreeFunctionBucket'] = patient_data['DiabetesPedigreeFunction'].map(lambda i: math.ceil(10*i))
patient_data['AgeGroup'] = patient_data['Age'].map(lambda i: int(i/10))
patient_data.describe()


# In[ ]:


patient_data.loc[:,['Insulin','InsulinLog','InsulinBucket']].describe()


# In[ ]:


patient_data.loc[:,['AgeGroup','GlucoseBucket', 'SkinThicknessBucket', 'DiabetesPedigreeFunctionBucket','InsulinBucket']].describe()


# In[ ]:


patient_data.loc[:,['DiabetesPedigreeFunction','DiabetesPedigreeFunctionLog','DiabetesPedigreeFunctionBucket']].describe()


# In[ ]:


fig, ax = plt.subplots (3,3, figsize=(30,16))
sns.countplot("Pregnancies", data=patient_data, ax=ax[0][0])
ax[0][0].set_title("Pregnancy Distribution")
sns.kdeplot(patient_data['Glucose'], ax=ax[0][1])
ax[0][1].set_title("glucose Concentration")
sns.kdeplot(patient_data['BloodPressure'], ax=ax[0][2])
ax[0][2].set_title("Blood Pressure")
sns.kdeplot(patient_data['SkinThickness'], ax=ax[1][0])
ax[1][0].set_title("Skin Thickness")
sns.kdeplot(patient_data['Insulin'], ax=ax[1][1])
ax[1][1].set_title("Insulin")
sns.countplot(patient_data['Insulin'].map(lambda i: int(i/100)), ax=ax[1][2])
ax[1][2].set_title("Insulin (log10 Transformation)")
sns.kdeplot(patient_data['DiabetesPedigreeFunction'], ax=ax[2][0])
ax[2][0].set_title('DiabetesPedigreeFunction')
sns.kdeplot(patient_data['DiabetesPedigreeFunction'].map(lambda i: np.log(i) if i > 0 else 0), ax=ax[2][1])
ax[2][1].set_title('DiabetesPedigreeFunction Log Transformation')
sns.kdeplot(patient_data['BMI'], ax=ax[2][2])
ax[2][2].set_title('BMI')


# Here it can be seen in the case of insulin that using the log of insulin does not help much as it produces a new challenge of a valley in the middle. 
# 
# ### Correlation between Outcome and Age
# Here we are using the "ageGroup" parameter to see if there is any direct correlation between "Age" and "Outcome".
# [20-30): bucket 2
# [30-40): bucket 3

# In[ ]:


fig, ax = plt.subplots(1,1, figsize=(10, 5))
sns.countplot(patient_data['AgeGroup'], hue=patient_data['Outcome'], ax=ax)


# ## Bi Variate and MultiVariate  Analysis
# 
# My Initial Aim here is to find out if there is any direct correlation between the individual parameters and the "Outcome" variable. 
# 

# In[ ]:


fig, ax = plt.subplots (3,3, figsize=(30,16))
sns.boxplot(x='Outcome', y='Glucose', data=patient_data, ax=ax[0][0])
sns.boxplot(x='Outcome', y='BloodPressure', data=patient_data, ax=ax[0][1])
sns.boxplot(x='Outcome', y='SkinThickness', data=patient_data, ax=ax[0][2])
sns.boxplot(x='Outcome', y='InsulinLog', data=patient_data, ax=ax[2][1]) # insulin Diabetees dependency tough to depict
sns.boxplot(x='Outcome', y='InsulinBucket', data=patient_data, ax=ax[1][0]) 
sns.boxplot(x='Outcome', y='DiabetesPedigreeFunctionLog', data=patient_data, ax=ax[1][1])
sns.boxplot(x='Outcome', y='DiabetesPedigreeFunctionBucket', data=patient_data, ax=ax[1][2])
sns.boxplot(x='Outcome', y='BMI', data=patient_data, ax=ax[2][2])
sns.boxplot(x='Outcome', y='Pregnancies', data=patient_data, ax=ax[2][0])


# The next thing I am trying here is to figure out how the different parameteres are related to each other by using the 'Outcome' variable as the third variable (or hue in terms of plotting terminology). In otherwords how much two variables are related for those who have diabetees vs their relation for those who don't have diabetees. 

# In[ ]:


fig, ax = plt.subplots (3,3, figsize=(40,20))
sns.boxplot(x='Pregnancies', y='BMI', hue='Outcome', data=patient_data, ax=ax[0][0])
ax[0][0].set_title("Pregnancies Vs BMI Vs Outcome")
sns.boxplot(x='GlucoseBucket', y='InsulinBucket', hue='Outcome', data=patient_data, ax=ax[0][1])
sns.boxplot(x='GlucoseBucket', y='DiabetesPedigreeFunctionLog', hue='Outcome', data=patient_data, ax=ax[0][2])
sns.boxplot(x='Pregnancies', y='BMI', data=patient_data, ax=ax[1][0])
sns.boxplot(y='GlucoseBucket', x='SkinThicknessBucket', hue='Outcome', data=patient_data, ax=ax[1][1])
sns.boxplot(y='InsulinBucket', x='SkinThicknessBucket', hue='Outcome', data=patient_data, ax=ax[1][2])
sns.boxplot(x='Outcome', y='GlucoseBucket', data=patient_data, ax=ax[2][0])


# ### Heatmaps
# 
# In a last attempt to see the correlation between different variables in the dataset, I use correlation heatmaps which tell how much the variables effect each other. 

# In[ ]:


fig, ax = plt.subplots (1,1, figsize=(20,10))
f = patient_data.loc[:,['Pregnancies','GlucoseBucket','AgeGroup','DiabetesPedigreeFunctionLog','InsulinBucket','SkinThicknessBucket','BMI','BloodPressure', 'Outcome']].corr()
sns.heatmap(f, annot=True, ax=ax)


# In[ ]:


sns.heatmap(patient_data.loc[:,['Insulin','InsulinLog','InsulinBucket','Outcome']].corr(),annot=True)


# In[ ]:


fig, ax = plt.subplots (1,1, figsize=(10,5))
sns.heatmap(patient_data.loc[:,['DiabetesPedigreeFunction','DiabetesPedigreeFunctionLog','DiabetesPedigreeFunctionBucket','Glucose', 'GlucoseBucket','Outcome']].corr(),annot=True, ax=ax)


# ## Classification
# 
# ### Creating a "Classifier" 
# Instead of creating multiple classifiers and writing similar parts of the code over and over again or passing variables into a series of loosely coupled functions, I am going with the Object Oriented approach of creating a generic "Classifier".
# 
# **constructor:**
# * accepts any basic classifier instance, along with a name (just a string type for later reference) and the set of parameters associated with the classifier (can be ignored if needed). 
# * creates a GridSearchCV instance of the classifier along with the parameter. 
# * sets the accuracy to "None".
# 
# **fit_and_train:**
# * Accepts the subset of the data to be used for training along with the indices of the features to be used for training (As I mentioned in the intro, adding all the variables might not result in the highest accuracy).
# * Performs the fit for the classifier. 
# 
# **get_accuracy:**
# * Accepts the test dataset along with the indices of the features to be used. 
# * Returns the accuracy.
# 
# **get_overall_accuracy:**
# * An extended version of get_accuracy, which performs kfold CV and passes data to **fit_and_train()** and **get_accuracy()** during each iteration. The mean accuracy obtained is set as "accuracy" of the Classifier object.

# In[ ]:


'''
For reference 
0 Pregnancies                         int64
1 Glucose                             int64
2 BloodPressure                       int64
3 SkinThickness                       int64
4 Insulin                             int64
5 BMI                               float64
6 DiabetesPedigreeFunction          float64
7 Age                                 int64
8 Outcome                             int64
9 GlucoseBucket                       int64
10 SkinThicknessBucket                 int64
11 InsulinBucket                       int64
12 InsulinLog                        float64
13 DiabetesPedigreeFunctionLog       float64
14 DiabetesPedigreeFunctionBucket      int64
15 AgeGroup                            int64
'''
print(patient_data.dtypes)
split_num = 8
default_param_set = [1,2,4,5,6,7] # The mysterious feature set which when produced the highest accuracy i've personally seen for the dataset (77.47%)
# Generic class that can be used for any specific classifier
class Classifier:
    
    def __init__(self, clf_type, name, params=None):
        self.name = name
        self.clf = GridSearchCV(clf_type, params)
        self.params = params
        self.accuracy = None #initialized as None 
        print('Initializing done for %s classifier' % (name))
    
    def fit_and_train(self, train, param_set=None):
        print('Fitting and training dataset for %s' % (self.name))
        labels = train.Outcome
        if param_set:
            features = pd.DataFrame(train.iloc[:, param_set])
        else:
            features = pd.DataFrame(train.iloc[:, default_param_set])
        print('Feature initializing complete')
        self.clf.fit(features, labels)
        print('Fit complete')
        
    def get_accuracy(self, test, param_set=None):
        print('Accuracy iteration for %s' % (self.name))
        labels_test = test.Outcome
        if param_set:
            features_test = pd.DataFrame(test.iloc[:, param_set])
        else:
            features_test = pd.DataFrame(test.iloc[:, default_param_set])
        pred = self.clf.predict(features_test)
        print('prediction for test features complete')
        return accuracy_score(pred, labels_test)
    
    def get_overall_accuracy(self, df, param_set=None):
        kf = KFold(n_splits=split_num, random_state=42, shuffle=True)
        scores = []
        for train_index, test_index in kf.split(df):
            train = df.iloc[train_index]
            test = df.iloc[test_index]
            self.fit_and_train(train, param_set)
            scores.append(self.get_accuracy(test, param_set))
        self.accuracy = float(sum(scores)/split_num)


# ### Normalization 
# One final step I am performing here is to scale all the variables that are to be used. So that the influence of each of the variables is equal. I am using a custom function to do so rather than the inbuilt library version. 

# In[ ]:


max_pregn = patient_data.Pregnancies.max()
min_pregn = patient_data.Pregnancies.min()
max_glucose = patient_data.Glucose.max()
min_glucose = patient_data.Glucose.min()
max_bp = patient_data.BloodPressure.max()
min_bp = patient_data.BloodPressure.min()
max_dpf = patient_data.DiabetesPedigreeFunction.max()
min_dpf = patient_data.DiabetesPedigreeFunction.min()
max_ins = patient_data.Insulin.max()
min_ins = patient_data.Insulin.min()
max_bmi = patient_data.BMI.max()
min_bmi = patient_data.BMI.min()
max_age = patient_data.Age.max()
min_age = patient_data.Age.min()
max_st = patient_data.SkinThickness.max()
min_st = patient_data.SkinThickness.min()
max_glucose_bucket = patient_data.GlucoseBucket.max()
min_glucose_bucket = patient_data.GlucoseBucket.min()
max_st_bucket = patient_data.SkinThicknessBucket.max()
min_st_bucket = patient_data.SkinThicknessBucket.min()
max_insulin_bucket = patient_data.InsulinBucket.max()
min_insulin_bucket = patient_data.InsulinBucket.min()
max_dpf_bucket = patient_data.DiabetesPedigreeFunctionBucket.max()
min_dpf_bucket = patient_data.DiabetesPedigreeFunctionBucket.min()
max_age_group = patient_data.AgeGroup.max()
min_age_group = patient_data.AgeGroup.min()

def normalize(srs, max_pregn_val, min_pregn_val, max_glucose_val, min_glucose_val, max_bp_val, min_bp_val, max_dpf_val, min_dpf_val, max_ins_val, min_ins_val, max_bmi_val, min_bmi_val, max_age_val, min_age_val, max_st_val, min_st_val, max_glucose_bucket_val, min_glucose_bucket_val, max_st_bucket_val, min_st_bucket_val, max_insulin_bucket_val, min_insulin_bucket_val, max_dpf_bucket_val, min_dpf_bucket_val, max_age_group_val, min_age_group_val):
    srs.Pregnancies = float((srs.Pregnancies-min_pregn_val)/(max_pregn_val-min_pregn_val))
    srs.Glucose = float((srs.Glucose - min_glucose_val)/(max_glucose_val - min_glucose_val))
    srs.BloodPressure = float((srs.BloodPressure - min_bp_val)/(max_bp_val - min_bp_val))
    srs.DiabetesPedigreeFunction = float((srs.DiabetesPedigreeFunction - min_dpf_val)/(max_dpf_val - min_dpf_val))
    srs.Insulin = float((srs.Insulin - min_ins_val)/(max_ins_val - min_ins_val))
    srs.BMI = float((srs.BMI - min_bmi_val)/(max_bmi_val - min_bmi_val))
    srs.Age = float((srs.Age - min_age_val)/(max_age_val - min_age_val))
    srs.SkinThickness = float((srs.SkinThickness - min_st_val)/(max_st_val - min_st_val))
    srs.GlucoseBucket = float((srs.GlucoseBucket - min_glucose_bucket_val)/(max_glucose_bucket_val - min_glucose_bucket_val))
    srs.SkinThicknessBucket = float((srs.SkinThicknessBucket - min_st_bucket_val)/(max_st_bucket_val - min_st_bucket_val))
    srs.InsulinBucket = float((srs.InsulinBucket - min_insulin_bucket_val)/(max_insulin_bucket_val - min_insulin_bucket_val))
    srs.DiabetesPedigreeFunctionBucket = float((srs.DiabetesPedigreeFunctionBucket - min_dpf_bucket_val)/(max_dpf_bucket_val - min_dpf_bucket_val))
    srs.AgeGroup = float((srs.AgeGroup - min_age_group_val)/(max_age_group_val - min_age_group_val))
    return srs

patient_data = patient_data.apply((lambda x: normalize(x, max_pregn, min_pregn, max_glucose, min_glucose, max_bp, min_bp, max_dpf, min_dpf, max_ins, min_ins, max_bmi, min_bmi, max_age, min_age, max_st, min_st, max_glucose_bucket, min_glucose_bucket, max_st_bucket, min_st_bucket, max_insulin_bucket, min_insulin_bucket, max_dpf_bucket, min_dpf_bucket, max_age_group, min_age_group)), axis='columns')


# In the block below I Instantiate 3 "Classifier" of type  SVM,  DT and XGBoost respectively. 
# 
# In my previous runbook I had only used an SVM to get an accuracy of 77.4%. I pass that same combination of features to get scores of the other classifiers to compare. After which I replace the feature set with the new "engineered" ones to see how the accuracy changes when we use modified features instead of the original ones. 

# In[ ]:


from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

svm_parameters = {'kernel':('linear', 'rbf', 'poly'), 'C':[1, 5, 10, 0.1], 'gamma': ['auto', 'scale']}
dt_parameters = {'criterion': ('entropy', 'gini'), 'max_depth': range(1,25), 'min_samples_split': range(2,10)}
xgb_parameters = {'booster': ['gbtree'], 'gamma': [0.1,0.5,0.2,0.3,0.4], 'n_estimators':[20], 'reg_alpha': range(1,10),'learning_rate': [0.1,0.5,0.2,0.3,0.4,0.6,1.0]}

# Creating 'Classifier' instances
svc_classifier = Classifier(svm.SVC(), 'SVM', svm_parameters)
dt_classifier = Classifier(DecisionTreeClassifier(), 'DecisionTree', dt_parameters)
xgb_classifier = Classifier(XGBClassifier(), 'XGBoost', xgb_parameters)


# In[ ]:


# obtaining scores on raw dataset (without normalization) with the original set of features being considered.
svc_classifier.get_overall_accuracy(patient_data)
svc_original_score = svc_classifier.accuracy

dt_classifier.get_overall_accuracy(patient_data)
dt_original_score = dt_classifier.accuracy

xgb_classifier.get_overall_accuracy(patient_data)
xgb_original_score = xgb_classifier.accuracy
print('Param set 0 complete')

new_param_set = [9,2,10,11,5,14,15] # The new 'engineered' features...
svc_classifier.get_overall_accuracy(patient_data, new_param_set)
svc_score_1 = svc_classifier.accuracy

dt_classifier.get_overall_accuracy(patient_data, new_param_set)
dt_score_1 = dt_classifier.accuracy

xgb_classifier.get_overall_accuracy(patient_data, new_param_set)
xgb_score_1 = xgb_classifier.accuracy
print('Param set 1 complete')

new_param_set = [9,2,11,5,14,15] # Removed SkinThicknessBucket 
svc_classifier.get_overall_accuracy(patient_data, new_param_set)
svc_score_2 = svc_classifier.accuracy

dt_classifier.get_overall_accuracy(patient_data, new_param_set)
dt_score_2 = dt_classifier.accuracy

xgb_classifier.get_overall_accuracy(patient_data, new_param_set)
xgb_score_2 = xgb_classifier.accuracy
print('Param set 2 complete')


# A block to perform a graphical comparison of the accuracies. 

# In[ ]:


accuracy_df = pd.DataFrame({'score_0':[svc_original_score, dt_original_score, xgb_original_score], 'score_1':[svc_score_1, dt_score_1, xgb_score_1], 'score_2':[svc_score_2, dt_score_2, xgb_score_2], 'Classifiers': ['SVM', 'DecisionTree', 'XGBoost']})
fig, ax = plt.subplots (3,1, figsize=(20,8))
sns.barplot('score_0', 'Classifiers', data=accuracy_df, ax=ax[0])
ax[0].set_title('Accuracy comparison with param set 0')
sns.barplot('score_1', 'Classifiers', data=accuracy_df, ax=ax[1])
ax[1].set_title('Accuracy comparison with param set 1')
sns.barplot('score_2', 'Classifiers', data=accuracy_df, ax=ax[2])
ax[2].set_title('Accuracy comparison with param set 2')
accuracy_df.head()

