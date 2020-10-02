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

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split,KFold,GridSearchCV
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,precision_recall_curve,auc,roc_auc_score,roc_curve,recall_score,classification_report ,f1_score,precision_score
from sklearn.svm import SVC

plt.style.use('fivethirtyeight')
import itertools
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


df=pd.read_csv("../input/data.csv")


# In[ ]:


df.shape


# In[ ]:


df.head()


# In[ ]:


df.columns


# ## Data Exploration & Prepration
# 

# #### Missing Data

# In[ ]:


missing_cols=(df.isnull().sum()/df.shape[0])*100
missing_cols= missing_cols[missing_cols>0]
missing_cols.plot(kind='bar')
plt.xlabel('Features')
plt.ylabel('Missing Percentage %')
plt.title('Missing Data')
plt.show()


# Some random column " Unamed : 32 " got added , dropping as everything is NaN

# In[ ]:


df.shape


# In[ ]:


df.id.unique().shape[0]


# Observation :
# 
# - Id Column has all unique values , no pattern
# - Unamed : 32 column has all NaN 
# - Dropping these columns

# In[ ]:


df.columns


# In[ ]:


cols_to_drop=['id','Unnamed: 32']
df.drop(cols_to_drop,axis=1,inplace=True)
df.shape


# #### Target Feature
# 

# In[ ]:


f,ax=plt.subplots(1,2,figsize=(12,3))
df['diagnosis'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)
ax[0].set_title('diagnosis')
ax[0].set_ylabel('')
sns.countplot('diagnosis',data=df,ax=ax[1])
ax[1].set_title('classLabel')
plt.show()


# Observation:
#     - balanced dataset
#     - Class distribution: 357 benign, 212 malignant

# #### Variable Identification
# - Variable DataType : Numerical or Categorical

# In[ ]:


df.dtypes


# #### LabelEncoding target variable

# In[ ]:


df['diagnosis'] = df['diagnosis'].map({'M':1,'B':0})
df.head()


# In[ ]:


df.describe()


# - All features are in different scale , we need to make them comparable

# ### Univariate Analysis
# - Histograms
# - Boxplots
# 
# highlight missing and outlier values

# #### Starting with mean  features

# In[ ]:


columns = df.iloc[:,1:11].columns
length  = len(columns)
colors  = ["r","g","b","m","y","c","k","orange",'pink','brown'] 

plt.figure(figsize=(15,20))
for i,j,k in itertools.zip_longest(columns,range(length),colors):
    #plt.subplot(length/3,length/2,j+1)
    plt.subplot(5,3,j+1)
    #print(length/2,length/3,j+1)
    sns.distplot(df.iloc[:,1:11][i],color=k)
    #sns.boxplot(df_num[i+1],color=k)
    plt.title(i)
    plt.subplots_adjust(hspace = .3)
    plt.axvline(df.iloc[:,1:11][i].mean(),color = "k",linestyle="dashed",label="MEAN")
    plt.axvline(df.iloc[:,1:11][i].std(),color = "b",linestyle="dotted",label="STANDARD DEVIATION")
    plt.legend(loc="upper right")
    


# #### Boxplots Mean Features

# In[ ]:


columns = df.iloc[:,1:11].columns
length  = len(columns)
colors  = ["r","g","b","m","y","c","k","orange",'pink','brown'] 

plt.figure(figsize=(15,20))
for i,j,k in itertools.zip_longest(columns,range(length),colors):
    plt.subplot(length/2,length/3,j+1)
    sns.boxplot(df.iloc[:,1:11][i],color=k)
    plt.title(i)
    plt.subplots_adjust(hspace = .4)
    #plt.axvline(df_num[i].mean(),color = "k",linestyle="dashed",label="MEAN")
    #plt.axvline(df_num[i].std(),color = "b",linestyle="dotted",label="STANDARD DEVIATION")
    plt.legend(loc="upper right")
    


# #### Looking at distribution of SE variables

# In[ ]:


columns = df.iloc[:,11:21].columns
length  = len(columns)
colors  = ["r","g","b","m","y","c","k","orange",'pink','brown'] 

plt.figure(figsize=(15,20))
for i,j,k in itertools.zip_longest(columns,range(length),colors):
    #plt.subplot(length/3,length/2,j+1)
    plt.subplot(5,3,j+1)
    #print(length/2,length/3,j+1)
    sns.distplot(df.iloc[:,11:21][i],color=k)
    #sns.boxplot(df_num[i+1],color=k)
    plt.title(i)
    plt.subplots_adjust(hspace = .3)
    plt.axvline(df.iloc[:,11:21][i].mean(),color = "k",linestyle="dashed",label="MEAN")
    plt.axvline(df.iloc[:,11:21][i].std(),color = "b",linestyle="dotted",label="STANDARD DEVIATION")
    plt.legend(loc="upper right")
    


# #### Boxplot SE Features

# In[ ]:


columns = df.iloc[:,11:21].columns
length  = len(columns)
colors  = ["r","g","b","m","y","c","k","orange",'pink','brown'] 

plt.figure(figsize=(15,20))
for i,j,k in itertools.zip_longest(columns,range(length),colors):
    plt.subplot(length/2,length/3,j+1)
    sns.boxplot(df.iloc[:,11:21][i],color=k)
    plt.title(i)
    plt.subplots_adjust(hspace = .4)
    #plt.axvline(df_num[i].mean(),color = "k",linestyle="dashed",label="MEAN")
    #plt.axvline(df_num[i].std(),color = "b",linestyle="dotted",label="STANDARD DEVIATION")
    plt.legend(loc="upper right")
    


# #### Largest Value features

# In[ ]:


columns = df.iloc[:,21:].columns
length  = len(columns)
colors  = ["r","g","b","m","y","c","k","orange",'pink','brown'] 

plt.figure(figsize=(15,20))
for i,j,k in itertools.zip_longest(columns,range(length),colors):
    #plt.subplot(length/3,length/2,j+1)
    plt.subplot(5,3,j+1)
    #print(length/2,length/3,j+1)
    sns.distplot(df.iloc[:,21:][i],color=k)
    #sns.boxplot(df_num[i+1],color=k)
    plt.title(i)
    plt.subplots_adjust(hspace = .3)
    plt.axvline(df.iloc[:,21:][i].mean(),color = "k",linestyle="dashed",label="MEAN")
    plt.axvline(df.iloc[:,21:][i].std(),color = "b",linestyle="dotted",label="STANDARD DEVIATION")
    plt.legend(loc="upper right")
    


# ### Boxplot Wrost Features

# In[ ]:


columns = df.iloc[:,21:].columns
length  = len(columns)
colors  = ["r","g","b","m","y","c","k","orange",'pink','brown'] 

plt.figure(figsize=(15,20))
for i,j,k in itertools.zip_longest(columns,range(length),colors):
    plt.subplot(length/2,length/3,j+1)
    sns.boxplot(df.iloc[:,21:][i],color=k)
    plt.title(i)
    plt.subplots_adjust(hspace = .4)
    #plt.axvline(df_num[i].mean(),color = "k",linestyle="dashed",label="MEAN")
    #plt.axvline(df_num[i].std(),color = "b",linestyle="dotted",label="STANDARD DEVIATION")
    plt.legend(loc="upper right")
    


# Inference :
#     - All features are normally distributed.
#     - No significant outliers.

# #### Mean Values distribution w.r.t Target Varaible

# In[ ]:


columns = df.iloc[:,1:11].columns
length  = len(columns)
#colors  = ["r","g","b","m","y","c","k","orange"] 

plt.figure(figsize=(15,20))
for i,j in itertools.zip_longest(columns,range(length)):
    plt.subplot(length/2,length/3,j+1)
    plt.hist(x = [df[df['diagnosis']==1][i], df[df['diagnosis']==0][i]], 
         stacked=True, color = ['brown','green'],label = ['M','B'])
    plt.title(i)
    plt.subplots_adjust(hspace = .3)
    #plt.axvline(df_num[i].mean(),color = "k",linestyle="dashed",label="MEAN")
    #plt.axvline(df_num[i].std(),color = "b",linestyle="dotted",label="STANDARD DEVIATION")
    plt.legend(loc="upper right")
    


# Observations
# - mean values of radius, perimeter, area, compactness and concave points can be used in classification of the cancer. Larger values of these parameters tends to show a correlation with malignant tumors.
# - Other does not show a particular preference of one diagnosis over the other. Overlapping Distributions.

# #### SE features w.r.t target variable

# In[ ]:


columns = df.iloc[:,11:21].columns
length  = len(columns)
#colors  = ["r","g","b","m","y","c","k","orange"] 

plt.figure(figsize=(15,20))
for i,j in itertools.zip_longest(columns,range(length)):
    plt.subplot(length/2,length/3,j+1)
    plt.hist(x = [df[df['diagnosis']==1][i], df[df['diagnosis']==0][i]], 
         stacked=True, color = ['brown','green'],label = ['M','B'])
    plt.title(i)
    plt.subplots_adjust(hspace = .3)
    #plt.axvline(df_num[i].mean(),color = "k",linestyle="dashed",label="MEAN")
    #plt.axvline(df_num[i].std(),color = "b",linestyle="dotted",label="STANDARD DEVIATION")
    plt.legend(loc="upper right")
    


# Obsertvations :
#     
# - se values of Concave points can also be used in classification of the cancer.Smaller values of these parameters tends to show a correlation with benign tumors.
# - Other does not show a particular preference of one diagnosis over the other. Overlapping Distributions.

# #### Wrost type features w.r.t target variable 
# 

# In[ ]:


columns = df.iloc[:,21:].columns
length  = len(columns)
#colors  = ["r","g","b","m","y","c","k","orange"] 

plt.figure(figsize=(15,20))
for i,j in itertools.zip_longest(columns,range(length)):
    plt.subplot(length/2,length/3,j+1)
    plt.hist(x = [df[df['diagnosis']==1][i], df[df['diagnosis']==0][i]], 
         stacked=True, color = ['brown','green'],label = ['M','B'])
    plt.title(i)
    plt.subplots_adjust(hspace = .3)
    #plt.axvline(df_num[i].mean(),color = "k",linestyle="dashed",label="MEAN")
    #plt.axvline(df_num[i].std(),color = "b",linestyle="dotted",label="STANDARD DEVIATION")
    plt.legend(loc="upper right")
    


# Observations
# - wrost values of radius, perimeter, area, compactness and concave points can be used in classification of the cancer. Larger values of these parameters tends to show a correlation with malignant tumors.
# - Other does not show a particular preference of one diagnosis over the other. Overlapping Distributions.

# **Observation Summary**
# - Mean features to be used radius, perimeter, area, compactness and concave points ( 5 features)
# - SE features to be used concave points ( 1 features)
# - Wrost features to be used radius, perimeter, area, compactness and concave points. (5 features)
# - 11 features 

# In[ ]:


df.columns


# In[ ]:


cols_to_use=['diagnosis', 'radius_mean',  'perimeter_mean',       'area_mean', 'compactness_mean',        'concave points_mean','concave points_se','radius_worst','perimeter_worst', 'area_worst','compactness_worst','concave points_worst']
df1=df[cols_to_use]
df1.shape


# In[ ]:


df1.head()


# In[ ]:


correlation = df1.corr()
plt.figure(figsize=(9,7))
sns.heatmap(correlation,annot=True,edgecolor="k",cmap=sns.color_palette("magma"))
plt.title("CORRELATION BETWEEN VARIABLES")
plt.show()


# Observation 
# - High correlation between mean features & wrost features
# - High correlation within mean features

# In[ ]:


#plt.scatter(x=df1['radius_mean'],y=df['radius_worst'],)
sns.scatterplot(x=df1['radius_mean'], y=df1['radius_worst'], hue=df.diagnosis)
plt.title('Scatter radius_mean vs radius_wrost')
plt.show()


# In[ ]:


#plt.scatter(x=df1['radius_mean'],y=df['radius_worst'],)
sns.scatterplot(x=df1['radius_mean'], y=df1['perimeter_worst'], hue=df.diagnosis)
plt.title('Scatter radius_mean vs perimeter_worst')
plt.show()


# In[ ]:


#plt.scatter(x=df1['radius_mean'],y=df['radius_worst'],)
sns.scatterplot(x=df1['radius_mean'], y=df1['area_worst'], hue=df.diagnosis)
plt.title('Scatter radius_mean vs area_worst')
plt.show()


# In[ ]:


#plt.scatter(x=df1['radius_mean'],y=df['radius_worst'],)
sns.scatterplot(x=df1['concave points_mean'], y=df1['concave points_worst'], hue=df.diagnosis)
plt.title('Scatter concave points_mean vs concave points_worst')
plt.show()


# In[ ]:


#plt.scatter(x=df1['radius_mean'],y=df['radius_worst'],)
sns.scatterplot(x=df1['radius_mean'], y=df1['perimeter_mean'], hue=df.diagnosis)
plt.title('Scatter radius_mean vs perimeter_mean')
plt.show()


# In[ ]:


#plt.scatter(x=df1['radius_mean'],y=df['radius_worst'],)
sns.scatterplot(x=df1['radius_mean'], y=df1['area_mean'], hue=df.diagnosis)
plt.title('Scatter radius_mean vs area_mean')
plt.show()


# **Dropping  correlated features**
# - Dropping radius_worst,perimeter_worst,area_worst,compactness_worst,concave points_worst

# In[ ]:


cols_to_use=['diagnosis', 'radius_mean',         'compactness_mean',        'concave points_se']
df2=df[cols_to_use]
df2.shape


# In[ ]:


correlation = df2.corr()
plt.figure(figsize=(9,7))
sns.heatmap(correlation,annot=True,edgecolor="k",cmap=sns.color_palette("magma"))
plt.title("CORRELATION BETWEEN VARIABLES")
plt.show()


# ## Modelling

# In[ ]:


y=df2.diagnosis
X=df[['radius_mean','compactness_mean','concave points_se']]
X.shape,y.shape


# In[ ]:


# Normalization
scaler = StandardScaler()
X = scaler.fit_transform(X)


# **Splitting Data into training & Testing**

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)


# ## Logistic Regresion

# In[ ]:


model = LogisticRegression()
model.fit(X_train,y_train)
prediction1=model.predict(X_test)
print("__"*50,"\n")
print('The accuracy of the Logistic Regression is',accuracy_score(y_test,prediction1))
print("__"*50,"\n")
print(classification_report(y_test,prediction1))
print("__"*50)
sns.heatmap(confusion_matrix(y_test,prediction1),annot=True,fmt="d")
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# ## Support Vector Machine

# **Linear Kernel**

# In[ ]:


model = SVC(kernel='linear')
model.fit(X_train,y_train)
prediction1=model.predict(X_test)
print("__"*50,"\n")
print('The accuracy of the Linear SVM is',accuracy_score(y_test,prediction1))
print("__"*50,"\n")
print(classification_report(y_test,prediction1))
print("__"*50)
sns.heatmap(confusion_matrix(y_test,prediction1),annot=True,fmt="d")
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# * **Polynomial Kernel of degree 3**

# In[ ]:


model = SVC(kernel='poly',degree=3)
model.fit(X_train,y_train)
prediction1=model.predict(X_test)
print("__"*50,"\n")
print('The accuracy of the Polynomial SVM is',accuracy_score(y_test,prediction1))
print("__"*50,"\n")
print(classification_report(y_test,prediction1))
print("__"*50)
sns.heatmap(confusion_matrix(y_test,prediction1),annot=True,fmt="d")
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# * **Polynomial Kernel of degree 4**

# In[ ]:


model = SVC(kernel='poly',degree=4)
model.fit(X_train,y_train)
prediction1=model.predict(X_test)
print("__"*50,"\n")
print('The accuracy of the Polynomial SVM is',accuracy_score(y_test,prediction1))
print("__"*50,"\n")
print(classification_report(y_test,prediction1))
print("__"*50)
sns.heatmap(confusion_matrix(y_test,prediction1),annot=True,fmt="d")
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# **RBF Kernel**

# In[ ]:


model = SVC(kernel='rbf')
model.fit(X_train,y_train)
prediction1=model.predict(X_test)
print("__"*50,"\n")
print('The accuracy of the SVM is',accuracy_score(y_test,prediction1))
print("__"*50,"\n")
print(classification_report(y_test,prediction1))
print("__"*50)
sns.heatmap(confusion_matrix(y_test,prediction1),annot=True,fmt="d")
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# **Model Accuracy**
# 1. Logistic Regression : 88 %
# 2. Linear SVM : 88 %
# 3. Polynomial Kernel of degree 3 : 87 %
# 4. Polynomial Kernel of degree 4 : 74 %
# 4. RBF Kernel : 88 %

# #### Hyperparameter Tuning

# In[ ]:


get_ipython().run_cell_magic('time', '', "params_dict=\\\n{'C':[0.001,0.01,0.1,1,10,100],\n 'gamma':[0.001,0.01,0.1,1,10,100],\n 'kernel':['linear','rbf']}\nmodel1=GridSearchCV(estimator=SVC(),param_grid=params_dict,scoring='accuracy',cv=10)\nmodel1.fit(X_train,y_train)")


# In[ ]:


#Best parameters for our svc model
model1.best_params_


# In[ ]:


#Let's run our SVC again with the best parameters.
model_final = SVC(C = 0.01, gamma =  0.001, kernel= 'linear')
model_final.fit(X_train,y_train)
prediction1=model_final.predict(X_test)
print("__"*50,"\n")
print('The accuracy of the SVM  is',accuracy_score(y_test,prediction1))
print("__"*50,"\n")
print(classification_report(y_test,prediction1))
print("__"*50)
sns.heatmap(confusion_matrix(y_test,prediction1),annot=True,fmt="d")
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# **Improved accuracy from 88% to 89% by tuning hyperparameters**

# In[ ]:




