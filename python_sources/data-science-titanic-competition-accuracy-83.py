#!/usr/bin/env python
# coding: utf-8

# # Define the Problem
# The problem definition is given to us.
# Competition Description
# The sinking of the RMS Titanic is one of the most infamous shipwrecks in history.  On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew. This sensational tragedy shocked the international community and led to better safety regulations for ships.
# 
# One of the reasons that the shipwreck led to such loss of life was that there were not enough lifeboats for the passengers and crew. Although there was some element of luck involved in surviving the sinking, some groups of people were more likely to survive than others, such as women, children, and the upper-class.
# 
# In this challenge, we ask you to complete the analysis of what sorts of people were likely to survive. In particular, we ask you to apply the tools of machine learning to predict which passengers survived the tragedy.
# 
# # Gather the Data
# The dataset is given to us

# # Data Preparation 
# 

# In[2]:


# This Python 3 environment comes with many helpful analytics libraries installed
import sys 
import pandas as pd 
import matplotlib 
import numpy as np 
import scipy as sp
import IPython
from IPython import display 
import sklearn
import random
import time
import warnings
warnings.filterwarnings('ignore')
print('-'*25)
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# In[3]:


from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
from xgboost import XGBClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
from pandas.tools.plotting import scatter_matrix
get_ipython().run_line_magic('matplotlib', 'inline')
mpl.style.use('ggplot')
sns.set_style('white')
pylab.rcParams['figure.figsize'] = 12,8


# In[4]:


data_raw = pd.read_csv('../input/train.csv')
data_val  = pd.read_csv('../input/test.csv')
data1 = data_raw.copy(deep = True)
data_cleaner = [data1, data_val]
print (data_raw.info()) 
data_raw.sample(10)


# In[5]:


print('Train columns with null values:\n', data1.isnull().sum())
print("-"*25)
print('Test/Validation columns with null values:\n', data_val.isnull().sum())
print("-"*25)
data_raw.describe(include = 'all')


# In[6]:


for dataset in data_cleaner:
    dataset['Age'].fillna(dataset['Age'].median(), inplace = True)
    dataset['Embarked'].fillna(dataset['Embarked'].mode()[0], inplace = True)
    dataset['Fare'].fillna(dataset['Fare'].median(), inplace = True)
drop_column = ['PassengerId','Cabin', 'Ticket']
data1.drop(drop_column, axis=1, inplace = True)
print(data1.isnull().sum())
print("-"*25)
print(data_val.isnull().sum())


# In[7]:


# Feature Engineering
for dataset in data_cleaner:
    dataset['FamilySize'] = dataset ['SibSp'] + dataset['Parch'] + 1
    dataset['IsAlone'] = 1 
    dataset['IsAlone'].loc[dataset['FamilySize'] > 1] = 0 
    dataset['Title'] = dataset['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
    dataset['FareBin'] = pd.qcut(dataset['Fare'], 4)
    dataset['AgeBin'] = pd.cut(dataset['Age'].astype(int), 5)

stat_min = 10 
title_names = (data1['Title'].value_counts() < stat_min) 
data1['Title'] = data1['Title'].apply(lambda x: 'Misc' if title_names.loc[x] == True else x)
print(data1['Title'].value_counts())
print("-"*25)
data1.info()
data_val.info()
data1.sample(10)


# In[8]:


label = LabelEncoder()
for dataset in data_cleaner:    
    dataset['Sex_Code'] = label.fit_transform(dataset['Sex'])
    dataset['Embarked_Code'] = label.fit_transform(dataset['Embarked'])
    dataset['Title_Code'] = label.fit_transform(dataset['Title'])
    dataset['AgeBin_Code'] = label.fit_transform(dataset['AgeBin'])
    dataset['FareBin_Code'] = label.fit_transform(dataset['FareBin'])
Target = ['Survived']
data1_x = ['Sex','Pclass', 'Embarked', 'Title','SibSp', 'Parch', 'Age', 'Fare', 'FamilySize', 'IsAlone'] #pretty name/values for charts
data1_x_calc = ['Sex_Code','Pclass', 'Embarked_Code', 'Title_Code','SibSp', 'Parch', 'Age', 'Fare'] #coded for algorithm calculation
data1_xy =  Target + data1_x
print('Original X Y: ', data1_xy, '\n')

data1_x_bin = ['Sex_Code','Pclass', 'Embarked_Code', 'Title_Code', 'FamilySize', 'AgeBin_Code', 'FareBin_Code']
data1_xy_bin = Target + data1_x_bin
print('Bin X Y: ', data1_xy_bin, '\n')

data1_dummy = pd.get_dummies(data1[data1_x])
data1_x_dummy = data1_dummy.columns.tolist()
data1_xy_dummy = Target + data1_x_dummy
print('Dummy X Y: ', data1_xy_dummy, '\n')

data1_dummy.head()


# In[9]:


print('Train columns with null values: \n', data1.isnull().sum())
print("-"*25)
print (data1.info())
print("-"*25)

print('Test/Validation columns with null values: \n', data_val.isnull().sum())
print("-"*25)
print (data_val.info())
print("-"*25)

data_raw.describe(include = 'all')


# In[10]:


train1_x, test1_x, train1_y, test1_y = model_selection.train_test_split(data1[data1_x_calc], data1[Target], random_state = 0)
train1_x_bin, test1_x_bin, train1_y_bin, test1_y_bin = model_selection.train_test_split(data1[data1_x_bin], data1[Target] , random_state = 0)
train1_x_dummy, test1_x_dummy, train1_y_dummy, test1_y_dummy = model_selection.train_test_split(data1_dummy[data1_x_dummy], data1[Target], random_state = 0)

print("Data1 Shape: {}".format(data1.shape))
print("Train1 Shape: {}".format(train1_x.shape))
print("Test1 Shape: {}".format(test1_x.shape))

train1_x_bin.head()


# In[11]:


for x in data1_x:
    if data1[x].dtype != 'float64' :
        print('Survival Correlation by:', x)
        print(data1[[x, Target[0]]].groupby(x, as_index=False).mean())
        print('-'*10, '\n')
print(pd.crosstab(data1['Title'],data1[Target[0]]))


# In[12]:


plt.figure(figsize=[16,12])
plt.subplot(231)
plt.boxplot(x=data1['Fare'], showmeans = True, meanline = True)
plt.title('Fare Boxplot')
plt.ylabel('Fare ($)')

plt.subplot(232)
plt.boxplot(data1['Age'], showmeans = True, meanline = True)
plt.title('Age Boxplot')
plt.ylabel('Age (Years)')

plt.subplot(233)
plt.boxplot(data1['FamilySize'], showmeans = True, meanline = True)
plt.title('Family Size Boxplot')
plt.ylabel('Family Size (#)')

plt.subplot(234)
plt.hist(x = [data1[data1['Survived']==1]['Fare'], data1[data1['Survived']==0]['Fare']], 
         stacked=True, color = ['g','r'],label = ['Survived','Dead'])
plt.title('Fare Histogram by Survival')
plt.xlabel('Fare ($)')
plt.ylabel('# of Passengers')
plt.legend()

plt.subplot(235)
plt.hist(x = [data1[data1['Survived']==1]['Age'], data1[data1['Survived']==0]['Age']], 
         stacked=True, color = ['g','r'],label = ['Survived','Dead'])
plt.title('Age Histogram by Survival')
plt.xlabel('Age (Years)')
plt.ylabel('# of Passengers')
plt.legend()

plt.subplot(236)
plt.hist(x = [data1[data1['Survived']==1]['FamilySize'], data1[data1['Survived']==0]['FamilySize']], 
         stacked=True, color = ['g','r'],label = ['Survived','Dead'])
plt.title('Family Size Histogram by Survival')
plt.xlabel('Family Size (#)')
plt.ylabel('# of Passengers')
plt.legend()


# In[ ]:


fig, saxis = plt.subplots(2, 3,figsize=(16,12))

sns.barplot(x = 'Embarked', y = 'Survived', data=data1, ax = saxis[0,0])
sns.barplot(x = 'Pclass', y = 'Survived', order=[1,2,3], data=data1, ax = saxis[0,1])
sns.barplot(x = 'IsAlone', y = 'Survived', order=[1,0], data=data1, ax = saxis[0,2])

sns.pointplot(x = 'FareBin', y = 'Survived',  data=data1, ax = saxis[1,0])
sns.pointplot(x = 'AgeBin', y = 'Survived',  data=data1, ax = saxis[1,1])
sns.pointplot(x = 'FamilySize', y = 'Survived', data=data1, ax = saxis[1,2])


# In[ ]:


fig, (axis1,axis2,axis3) = plt.subplots(1,3,figsize=(14,12))

sns.boxplot(x = 'Pclass', y = 'Fare', hue = 'Survived', data = data1, ax = axis1)
axis1.set_title('Pclass vs Fare Survival Comparison')

sns.violinplot(x = 'Pclass', y = 'Age', hue = 'Survived', data = data1, split = True, ax = axis2)
axis2.set_title('Pclass vs Age Survival Comparison')

sns.boxplot(x = 'Pclass', y ='FamilySize', hue = 'Survived', data = data1, ax = axis3)
axis3.set_title('Pclass vs Family Size Survival Comparison')


# In[ ]:


fig, qaxis = plt.subplots(1,3,figsize=(14,12))

sns.barplot(x = 'Sex', y = 'Survived', hue = 'Embarked', data=data1, ax = qaxis[0])
axis1.set_title('Sex vs Embarked Survival Comparison')

sns.barplot(x = 'Sex', y = 'Survived', hue = 'Pclass', data=data1, ax  = qaxis[1])
axis1.set_title('Sex vs Pclass Survival Comparison')

sns.barplot(x = 'Sex', y = 'Survived', hue = 'IsAlone', data=data1, ax  = qaxis[2])
axis1.set_title('Sex vs IsAlone Survival Comparison')


# In[ ]:


fig, (maxis1, maxis2) = plt.subplots(1, 2,figsize=(14,12))
sns.pointplot(x="FamilySize", y="Survived", hue="Sex", data=data1,
              palette={"male": "blue", "female": "pink"},
              markers=["*", "o"], linestyles=["-", "--"], ax = maxis1)
sns.pointplot(x="Pclass", y="Survived", hue="Sex", data=data1,
              palette={"male": "blue", "female": "pink"},
              markers=["*", "o"], linestyles=["-", "--"], ax = maxis2)


# In[ ]:


e = sns.FacetGrid(data1, col = 'Embarked')
e.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', ci=95.0, palette = 'deep')
e.add_legend()


# In[ ]:


a = sns.FacetGrid( data1, hue = 'Survived', aspect=4 )
a.map(sns.kdeplot, 'Age', shade= True )
a.set(xlim=(0 , data1['Age'].max()))
a.add_legend()


# In[ ]:


h = sns.FacetGrid(data1, row = 'Sex', col = 'Pclass', hue = 'Survived')
h.map(plt.hist, 'Age', alpha = .75)
h.add_legend()


# In[22]:


data1_x_bin


# In[23]:


cv_split = model_selection.ShuffleSplit(n_splits = 10, test_size = .3, train_size = .6, random_state = 0 ) # run model 10x with 60/30 split intentionally leaving out 10%
#MLA_columns = ['MLA Name', 'MLA Parameters','MLA Train Accuracy Mean', 'MLA Test Accuracy Mean', 'MLA Test Accuracy 3*STD' ,'MLA Time']
#MLA_compare = pd.DataFrame(columns = MLA_columns)
MLA_predict = data1[Target]


##Ron start svm.SVC
cv_results_SVC = model_selection.cross_validate(svm.SVC(probability=True), data1[data1_x_bin], data1[Target], cv  = cv_split)
Time_SVC = cv_results_SVC['fit_time'].mean()
print('Time SVC: \n', Time_SVC)
print("-"*25)
Train_Accuracy_Mean_SVC = cv_results_SVC['train_score'].mean()
print('Train Accuracy Mean SVC: \n', Train_Accuracy_Mean_SVC)
print("-"*25)
Test_Accuracy_Mean_SVC = cv_results_SVC['test_score'].mean()
print('Test Accuracy Mean SVC: \n', Test_Accuracy_Mean_SVC)
print("-"*70)
svm.SVC(probability=True).fit(data1[data1_x_bin], data1[Target])
#MLA_predict[svm.SVC(probability=True).__class__.__name__] = svm.SVC(probability=True).predict(data1_x_bin)
##Ron end svm.SVC

##Ron start XGBClassifier
cv_results_XGBClassifier = model_selection.cross_validate(XGBClassifier(), data1[data1_x_bin], data1[Target], cv  = cv_split)
Time_XGBClassifier = cv_results_XGBClassifier['fit_time'].mean()
print('Time XGBClassifier: \n', Time_XGBClassifier)
print("-"*25)
Train_Accuracy_Mean_XGBClassifier = cv_results_XGBClassifier['train_score'].mean()
print('Train Accuracy Mean XGBClassifier: \n', Train_Accuracy_Mean_XGBClassifier)
print("-"*25)
Test_Accuracy_Mean_XGBClassifier = cv_results_XGBClassifier['test_score'].mean()
print('Test Accuracy Mean XGBClassifier: \n', Test_Accuracy_Mean_XGBClassifier)
print("-"*70)
XGBClassifier().fit(data1[data1_x_bin], data1[Target])
MLA_predict[XGBClassifier().__class__.__name__] = XGBClassifier().predict(data1[data1_x_bin])
##Ron end XGBClassifier




##Ron start neighbors.KNeighborsClassifier
cv_results_KNN = model_selection.cross_validate(neighbors.KNeighborsClassifier(), data1[data1_x_bin], data1[Target], cv  = cv_split)
Time_KNN = cv_results_KNN['fit_time'].mean()
print('Time KNN: \n', Time_KNN)
print("-"*25)
Train_Accuracy_Mean_KNN = cv_results_KNN['train_score'].mean()
print('Train Accuracy Mean KNN: \n', Train_Accuracy_Mean_KNN)
print("-"*25)
Test_Accuracy_Mean_KNN = cv_results_KNN['test_score'].mean()
print('Test Accuracy Mean KNN: \n', Test_Accuracy_Mean_KNN)
print("-"*70)
neighbors.KNeighborsClassifier().fit(data1[data1_x_bin], data1[Target])
#MLA_predict[neighbors.KNeighborsClassifier().__class__.__name__] = neighbors.KNeighborsClassifier().predict(data1[data1_x_bin])
##Ron end neighbors.KNeighborsClassifier


# 
# 
