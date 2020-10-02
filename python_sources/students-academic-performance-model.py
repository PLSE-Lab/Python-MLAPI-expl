#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt  # Matlab-style plotting
import seaborn as sns
color = sns.color_palette()
sns.set_style('darkgrid')

import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)

from scipy import stats
from scipy.stats import norm, skew #for some statistics
from scipy.special import boxcox1p
from sklearn.preprocessing import LabelEncoder

from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score


# In[ ]:


df = pd.read_csv('../input/xAPI-Edu-Data.csv')


# In[ ]:


df.head()


# In[ ]:


df.tail()


# In[ ]:


def DescriptiveStatistics(df):
    print("No of rwos and columns information:",df.shape)
    print("")
    print("---"*20)
    print("")
    print("Columns:")
    print("")
    print(df.columns.values)
    print("---"*20)
    print("")
    print(df.info())
    print("---"*20)
    print("")
    print(df.describe())


# In[ ]:


DescriptiveStatistics(df)


# In[ ]:


def CheckMissingInfo(df):
    print(df.isnull().sum())
    print("---"*20)
    print("")
    df_na = (df.isnull().sum() / len(df)) * 100
    df_na = df_na.drop(df_na[df_na == 0].index).sort_values(ascending=False)[:30]
    missing_data = pd.DataFrame({'Missing Ratio' :df_na})
    print(missing_data)


# In[ ]:


CheckMissingInfo(df)


# In[ ]:


def GetColumnCount(df):
    int_columns = [col for col in df.columns if(df[col].dtype != "object")]
    print("No of integer type columns:",len(int_columns))
    print(int_columns)
    print("")
    obj_columns = [col for col in df.columns if(df[col].dtype == "object")]
    print("No of object type columns:",len(obj_columns))
    print(obj_columns)
    return int_columns,obj_columns


# In[ ]:


int_columns,obj_columns = GetColumnCount(df)


# In[ ]:


def GetCountPlots(df,obj_columns):
    for col in obj_columns:
        if(len(df[col].value_counts()) < 5):
            plt.figure(figsize=(5,5))
        else:
            plt.figure(figsize=(12,6))
        print(sns.countplot(x=col, data=df, palette="muted"))
        plt.show()


# In[ ]:


GetCountPlots(df,obj_columns)


# In[ ]:


def GetCardinality(df,obj_columns):
    for col in obj_columns:
        print("{0} :: {1}".format(col,len(df[col].value_counts())))
        
        print(df[col].value_counts())
        print("")


# In[ ]:


GetCardinality(df,obj_columns)


# In[ ]:


pd.crosstab(df['Class'],df['Topic'])


# In[ ]:


def GetCountPlots_with_hue(df,obj_columns,col_hue):
    for col in obj_columns:
        if(len(df[col].value_counts()) < 5):
            plt.figure(figsize=(5,5))
        else:
            plt.figure(figsize=(12,6))
        #print(sns.countplot(x=col, data=df, palette="muted"))
        sns.countplot(x=col,data = df, hue=col_hue,palette='bright')
        plt.show()


# In[ ]:


GetCountPlots_with_hue(df,obj_columns,'Class')


# In[ ]:


def GetBoxPlots(df,x_col):
    for col in int_columns:
        plt.figure(figsize=(5,5))
        boxplot1 = sns.boxplot(x=x_col, y=col, data=df)
        boxplot1 = sns.swarmplot(x=x_col, y=col, data=df, color=".15")
        plt.show()


# In[ ]:


GetBoxPlots(df,'Class')


# In[ ]:


df['Failed'] = np.where(df['Class']=='L',1,0)


# In[ ]:


df['AbsBoolean'] = df['StudentAbsenceDays']
df['AbsBoolean'] = np.where(df['AbsBoolean'] == 'Under-7',0,1)
df['AbsBoolean'].groupby(df['Topic']).mean()


# In[ ]:


df.head()


# In[ ]:


df.info()


# In[ ]:


def NumaricVariablesDistributions(df):
    int_columns=df.columns[df.dtypes==int]
    plt.figure(figsize=(10,7))
    for i, column in enumerate(int_columns):
        plt.subplot(3,2, i+1)
        sns.distplot(df[column], label=column, bins=10, fit=norm)
        plt.ylabel('Density');


# In[ ]:


NumaricVariablesDistributions(df)


# - Features doesn't have gaussian (normal) distribution.
# - As ML algorithms deal better with values, which are normally distributed, we need to transfrom them closer that view. BoxCox transformation will help us with it

# In[ ]:


def ApplyBoxcoxTransformation(df,columns):
    plt.figure(figsize=(10,7))
    for i, column in enumerate(columns):
        plt.subplot(2,2, i+1)
        df[column]=boxcox1p(df[column], 0.3)
        sns.distplot(df[column], label=column, bins=10, fit=norm)
        plt.ylabel('Density')


# In[ ]:


ApplyBoxcoxTransformation(df,['raisedhands', 'VisITedResources', 'AnnouncementsView', 'Discussion'])


# - However raisedhands and visitedresourses have double gaussian distribution. We can create new binary features for them, where 1 is when values more than its' average, and 0 - less. Then we can look how these features will improve model

# In[ ]:


df['raisedhands_bin']=np.where(df.raisedhands>df.raisedhands.mean(),1,0)
df['VisITedResources_bin']=np.where(df.VisITedResources>df.VisITedResources.mean(),1,0)


# - It appears that no one failed Geology while students in IT, Chemistry, and Math had the highest probability of failing.
# 
# - The boxplot analysis indicates that those who did well were more active in class, and the worst performers were the least active.
# 
# - It is clear that the lowest performers rarely visited the course resources. The swarmplot shapes also indicates that the highest and lowest performers had the most consistent habits with respect to viewing the course resources. 
# 
# - It also appears that less students from all groups viewed course announcements, but there is still a clear pattern with viewing course announcements and how well the student performed.
# 
# - The biggest visual trend can be seen in how frequently the student was absent. Over 90% of the students who did poorly were absent more than seven times, while almost none of the students who did well were absent more than seven times.

# In[ ]:


GetBoxPlots(df,'Class')


# Let's look at correlation between these features:
# - VisitedResources, RaisedHands and AnnouncementViews have medium correlation (0.5-0.7)

# In[ ]:


plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), annot=True, fmt='.1g', cmap='RdBu');


# In[ ]:


sns.pairplot(df);


# In[ ]:


print('Percent of students\' nationality - Kuwait or Jordan: {}'.format(
            round(100*df.NationalITy.isin(['KW','Jordan']).sum()/df.shape[0],2)))


# In[ ]:


target=df['Class']
df=df.drop('Class', axis=1)


# In[ ]:


df.head()


# In[ ]:


#Create new feature - type of topic (technical, language, other)
Topic_types={'Math':'technic', 'IT':'technic','Science':'technic','Biology':'technic',
 'Chemistry':'technic', 'Geology':'technic', 'Arabic':'language', 'English':'language',
 'Spanish':'language','French':'language', 'Quran':'other' ,'History':'other'}
df['Topic_type']=df.Topic.map(Topic_types)


# In[ ]:


df.head()


# In[ ]:


int_columns,obj_columns = GetColumnCount(df)


# In[ ]:


def ApplyScaling(df):
    for column in ['raisedhands', 'VisITedResources', 'AnnouncementsView', 'Discussion']:
        SS=StandardScaler().fit(df[[column]])
        df[[column]]=SS.transform(df[[column]])


# In[ ]:


ApplyScaling(df)


# In[ ]:


df.head()


# In[ ]:


int_columns,obj_columns = GetColumnCount(df)


# In[ ]:


def LabelEncoding(df):
    for column in obj_columns:
        #Binarize and LabelEncode
        if ((df[column].value_counts().shape[0]==2) | (column=='StageID') | (column=='GradeID')):
            le=LabelEncoder().fit(df[column])
            df[column]=le.transform(df[column])
    


# In[ ]:


LabelEncoding(df)


# In[ ]:


df.head()


# In[ ]:


#One-hot encoding
df=pd.get_dummies(df)


# In[ ]:


df.head()


# In[ ]:


df.columns


# In[ ]:


from sklearn.metrics import make_scorer, accuracy_score,roc_auc_score,confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(df, target, test_size=0.20, random_state=42)


# In[ ]:


#using cross_val_score
logis = LogisticRegression()
svm = SVC()
knn = KNeighborsClassifier()
dTmodel = DecisionTreeClassifier()
rForest = RandomForestClassifier()
grBoosting = GradientBoostingClassifier()
    
scores = cross_val_score(logis,x_train,y_train,cv=5)
print("Accuracy for logistic regresion: mean: {0:.2f} 2sd: {1:.2f}".format(scores.mean(),scores.std() * 2))
print("Scores::",scores)
print("\n")

scores2 = cross_val_score(svm,x_train,y_train,cv=5)
print("Accuracy for SVM: mean: {0:.2f} 2sd: {1:.2f}".format(scores2.mean(),scores2.std() * 2))
print("Scores::",scores)
print("\n")

scores3 = cross_val_score(knn,x_train,y_train,cv=5)
print("Accuracy for KNN: mean: {0:.2f} 2sd: {1:.2f}".format(scores3.mean(),scores3.std() * 2))
print("Scores::",scores)
print("\n")

scores4 = cross_val_score(dTmodel,x_train,y_train,cv=5)
print("Accuracy for Decision Tree: mean: {0:.2f} 2sd: {1:.2f}".format(scores4.mean(),scores4.std() * 2))
print("Scores::",scores4)
print("\n")

scores5 = cross_val_score(rForest,x_train,y_train,cv=5)
print("Accuracy for Random Forest: mean: {0:.2f} 2sd: {1:.2f}".format(scores5.mean(),scores5.std() * 2))
print("Scores::",scores5)
print("\n")

scores6 = cross_val_score(grBoosting,x_train,y_train,cv=5)
print("Accuracy for Gradient Boosting: mean: {0:.2f} 2sd: {1:.2f}".format(scores6.mean(),scores6.std() * 2))
print("Scores::",scores6)
print("\n")


# In[ ]:


from sklearn.metrics import roc_auc_score

def roc_auc_score_multiclass(actual_class, pred_class, average = "macro"):

  #creating a set of all the unique classes using the actual class list
  unique_class = set(actual_class)
  roc_auc_dict = {}
  for per_class in unique_class:
    #creating a list of all the classes except the current class 
    other_class = [x for x in unique_class if x != per_class]

    #marking the current class as 1 and all other classes as 0
    new_actual_class = [0 if x in other_class else 1 for x in actual_class]
    new_pred_class = [0 if x in other_class else 1 for x in pred_class]

    #using the sklearn metrics method to calculate the roc_auc_score
    roc_auc = roc_auc_score(new_actual_class, new_pred_class, average = average)
    roc_auc_dict[per_class] = roc_auc

  return roc_auc_dict


# In[ ]:


def modelling(model,model_name):
    print(model)
    print("\n")
    model.fit(x_train, y_train)
    preds=model.predict(x_test)
    preds_proba=model.predict_proba(x_test)
    print('Accuracy = {}'.format(100*round(accuracy_score(y_test,preds),2)))
    print(classification_report(y_test, preds))
    
    print("\n")
    print(model_name)
    lr_roc_auc_multiclass = roc_auc_score_multiclass(y_test, preds)
    print("AUC Score for each lable")
    print(lr_roc_auc_multiclass)
    print("\n")
    plt.figure(figsize=(7,5))
    sns.heatmap(confusion_matrix(y_test,preds), annot=True, vmax=50)
    plt.show()


# In[ ]:


modelling(LogisticRegression(),"Logistic Regression")


# In[ ]:


modelling(GradientBoostingClassifier(),"Gradient Boosting")


# # Hyper parameter tuning

# In[ ]:


# Grid search cross validation

grid={"C":np.logspace(-3,3,7), "penalty":["l1","l2"]}# l1 lasso l2 ridge
logreg=LogisticRegression()
logreg_cv=GridSearchCV(logreg,grid,cv=10)
logreg_cv.fit(x_train,y_train)

print("tuned hpyerparameters :(best parameters) ",logreg_cv.best_params_)
print("accuracy :",logreg_cv.best_score_)


# In[ ]:


modelling(LogisticRegression(C=1.0,penalty='l1'),"Logistic Regression tuned")


# In[ ]:





# 
