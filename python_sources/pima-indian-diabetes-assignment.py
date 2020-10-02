#!/usr/bin/env python
# coding: utf-8

# # Pima Indian Diabetes Dataset 

# # Please Upvote if u like it;)

# In[ ]:


import os
print(os.listdir('../input/'))


# ## CONTENTS::

# #### 1 ) Importing Various Modules and Loading the Dataset

# #### 2 ) Exploratory Data Analysis (EDA)

# #### 3 ) Preprocessing the Data

# #### 4 ) Modelling 

# #### 5 ) Exporting to a CSV file

# In[ ]:





# ## 1 )  Importing Various Modules and Loading the Dataset

# ## 1.1 ) Importing Various Modules

# In[ ]:


# Ignore  the warnings
import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

# data visualisation and manipulation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
import missingno as msno

#configure
# sets matplotlib to inline and displays graphs below the corressponding cell.
get_ipython().run_line_magic('matplotlib', 'inline')
style.use('fivethirtyeight')
sns.set(style='whitegrid',color_codes=True)

#import the necessary modelling algos.
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB

#model selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix,roc_curve,roc_auc_score
from sklearn.model_selection import GridSearchCV

#dim red
from sklearn.decomposition import PCA

#preprocess.
from sklearn.preprocessing import MinMaxScaler,StandardScaler,Imputer,LabelEncoder,OneHotEncoder


# ## 1.2 ) Loading the Dataset

# In[ ]:


train=pd.read_csv(r'../input/pima-indians-diabetes-database/diabetes.csv')


# In[ ]:


train.head(10)


# ## 2 )  Exploratory Data Analysis (EDA)

# ## 2.1 ) The features and the 'Target' variable

# In[ ]:


df=train.copy()


# In[ ]:


df.head(10)


# This shows the first 10 rows or the 'data points' of the dataset.

# In[ ]:


df.shape # this gives the dimensions of the dataset.


# In[ ]:


df.index   


# In[ ]:


df.columns # gives a short description of each feature.


# #### Short description of each feature.

# **Pregnancies:- ** Number of times pregnant
# 
# **Glucose:- ** Plasma glucose concentration a 2 hours in an oral glucose tolerance test
# 
# **BloodPressure:- ** Diastolic blood pressure (mm Hg)
# 
# **SkinThickness:- ** Triceps skin fold thickness (mm)
# 
# **Insulin2:- ** Hour serum insulin (mu U/ml)
# 
# **BMI:- ** Body mass index (weight in kg/(height in m)^2)
# 
# **DiabetesPedigreeFunction:- ** Diabetes pedigree function
# 
# **Age:- ** Age (years)
# 
# **Outcome (The Target):- ** Class variable (0 or 1) 268 of 768 are 1, the others are 0

# #### Note that we have 768 observations each described by 8 features and the target variable is the presence or absence of the diabetes disease.

# ## 2.2 ) Missing Values Treatment

# In[ ]:


# check for null values.
df.isnull().any()   


# Note that we dont have any missing value in any of the column. Hence no imputation is required.

# In[ ]:


msno.matrix(df)  # just to visualize. no missing value.


# ## 2.3 ) Univariate Analysis

# In this section I have performed the univariate analysis. Note that since all of the features are 'numeric' the most reasonable way to plot them would either be a 'histogram' or a 'boxplot'.

# In[ ]:


df.describe()


# In[ ]:





# #### A utility function to plot the distribution of various features.

# In[ ]:


def plot(feature):
    fig,axes=plt.subplots(1,2)
    sns.boxplot(data=df,x=feature,ax=axes[0])
    sns.distplot(a=df[feature],ax=axes[1],color='#ff4125')
    fig.set_size_inches(15,5)


# In[ ]:


plot('Pregnancies')


# #### INFERENCES--
# 
# 1) First of all note that the values are in compliance with that observed from describe method.
# 
# 2) Note that we have a couple of outliers w.r.t. to 1.5 quartile rule (reprsented by a 'dot' in the box plot).
# 
# 3) Also note from the distplot that the distribution seems to be a bit +ve skewed (Right Tailed). 

# #### We can go on for analyzing the other features in the data frame and can draw similar inferences.

# In[ ]:


plot('Glucose')


# In[ ]:


plot('BloodPressure')


# In[ ]:


plot('SkinThickness')


# In[ ]:


plot('Insulin')


# In[ ]:


plot('BMI')


# In[ ]:


plot('DiabetesPedigreeFunction')


# In[ ]:


plot('Age')


# In[ ]:


sns.countplot(data=df,x='Outcome')


# In[ ]:


df['Outcome'].value_counts()


# ####  Note that  we have more number of observations in which Diabetes is not diagnosed which is expected.

# ## 2.4 ) Bivariate Analysis

# ## 2.4.1 ) Plotting the Features against the 'Target' variable

# Here I have just written a small utility function that plots the 'Outcome' column vs the provided feature on a boxplot. In this way I have plotted some of the features against our target variable. This makes it easier to see the effect of the corresponding feature on the 'Outcome'.

# In[ ]:


# drawing features against the target variable.

def plot_against_target(feature):
    sns.factorplot(data=df,y=feature,x='Outcome',kind='box')
    fig=plt.gcf()
    fig.set_size_inches(7,7)


# In[ ]:


plot_against_target('Glucose') # 0 for no diabetes and 1 for presence of it


# #### INFERENCES--
# 
# 1) Firstly note that 0->'absence' and 1->'presence'.
# 
# 2) Note that the boxpot depicts that people having diabetes have higher Glucose levels.

# In[ ]:


plot_against_target('BloodPressure')


# In[ ]:


plot_against_target('SkinThickness')


# In[ ]:


plot_against_target('Age') 


# #### INFERENCES--
# 
# 1) Firstly again note that 0->'absence' and 1->'presence' of diabetes.
# 
# 2) Note that the boxpot depicts that people having diabetes tend to have higher age as expected.

# ####  We can draw similar inferences from the plots plotted ahead for each of the 'feature' aginst the target which is 'Outcome' in our case.

# In[ ]:





# ## 3 ) Preprocessing the Data

# In[ ]:


df.shape


# In[ ]:


df.head(10)


# ## 3.1 ) Normalizing the Features.

# In[ ]:


scaler=MinMaxScaler()
scaled_df=scaler.fit_transform(df.drop('Outcome',axis=1))
X=scaled_df
Y=df['Outcome'].as_matrix()


# ## 3.2) Splitting into Training and Validation sets.

# In[ ]:


x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.20,random_state=42)


# In[ ]:





# ## 4) Modelling

# #### LOGISTIC REGRESSSION

# In[ ]:


clf_lr=LogisticRegression()
clf_lr.fit(x_train,y_train)
pred=clf_lr.predict(x_test)
print(accuracy_score(pred,y_test))


# #### kNN

# In[ ]:


clf_knn=KNeighborsClassifier()
clf_knn.fit(x_train,y_train)
pred=clf_knn.predict(x_test)
print(accuracy_score(pred,y_test))


# #### Support Vector Machine (SVM)

# In[ ]:


clf_svm=SVC()
clf_svm.fit(x_train,y_train)
pred=clf_svm.predict(x_test)
print(accuracy_score(pred,y_test))


# #### DECISION TREE  

# In[ ]:


clf_dt=DecisionTreeClassifier()
clf_dt.fit(x_train,y_train)
pred=clf_dt.predict(x_test)
print(accuracy_score(pred,y_test))


# #### RANDOM FOREST

# In[ ]:


clf_rf=RandomForestClassifier()
clf_rf.fit(x_train,y_train)
pred=clf_rf.predict(x_test)
print(accuracy_score(pred,y_test))


# #### GRADIENT BOOSTING

# In[ ]:


clf_gb=GradientBoostingClassifier()
clf_gb.fit(x_train,y_train)
pred=clf_gb.predict(x_test)
print(accuracy_score(pred,y_test))


# #### We can now move onto comparing the results of various modelling algorithms.

# In[ ]:


models=[LogisticRegression(),LinearSVC(),SVC(kernel='rbf'),KNeighborsClassifier(),RandomForestClassifier(),
        DecisionTreeClassifier(),GradientBoostingClassifier(),GaussianNB()]
model_names=['LogisticRegression','LinearSVM','rbfSVM','KNearestNeighbors','RandomForestClassifier','DecisionTree',
             'GradientBoostingClassifier','GaussianNB']

acc=[]
d={}

for model in range(len(models)):
    clf=models[model]
    clf.fit(x_train,y_train)
    pred=clf.predict(x_test)
    acc.append(accuracy_score(pred,y_test))
     
d={'Modelling Algo':model_names,'Accuracy':acc}


# In[ ]:


acc_frame=pd.DataFrame(d)
acc_frame


# In[ ]:


sns.barplot(y='Modelling Algo',x='Accuracy',data=acc_frame)


# In[ ]:





# ## 5) Exporting to a CSV file

# SVM gives best accuracy on the dataset. So using it to predict on the validation set.

# In[ ]:


clf_svm=SVC()
clf_svm.fit(x_train,y_train)
pred=clf_svm.predict(x_test)
print(accuracy_score(pred,y_test))


# #### Finally saving predictions to a CSV/.XLSX file.

# In[ ]:


ids=[]
for i,obs in enumerate(x_test):
    s='id'+'_'+str(i)
    ids.append(s)


# In[ ]:


# ids
d={'Ids':ids,'Outcome':pred}
final_pred=pd.DataFrame(d)
final_pred.to_csv('predictions.csv',index=False)


# In[ ]:





# ## THE END!!!

# ## Please Upvote if u liked my work ;)

# In[ ]:




