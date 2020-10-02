#!/usr/bin/env python
# coding: utf-8

# # Notebook description
#     Role of this notebook would be to see how Exploratory data analysis would affect your model. The first part of the notebook would be model building followed by EDA. We will again fit the model and evaluate both.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # Data visualization
import seaborn as sns #Data visualization

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import warnings 
warnings.filterwarnings('ignore')

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


#importing modules and data
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model.ridge import RidgeClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble.forest import ExtraTreesClassifier
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
# here we have nine models

from sklearn.metrics import accuracy_score,confusion_matrix

train_data=pd.read_csv('../input/learn-together/train.csv',index_col='Id')
test_data=pd.read_csv('../input/learn-together/test.csv',index_col='Id')


# In[ ]:


from sklearn.model_selection import train_test_split
X,y=train_data.iloc[:,:-1],train_data.iloc[:,-1]
X_train,X_test,y_train,y_test=train_test_split(X,y)
mod_val=[]


# # Models without EDA
# These models are built in a hurry without any EDA!

# In[ ]:


#Logistic Regression
model=LogisticRegression()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
accuracy=accuracy_score(y_pred,y_test)
mod_val.append(round(accuracy,4))
print('Accuracy of Logistics Regression:',mod_val[0])


# In[ ]:


#Ridge Classifier
model=RidgeClassifier()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
accuracy=accuracy_score(y_pred,y_test)
mod_val.append(round(accuracy,4))
print('Accuracy of Ridge Classifier:',mod_val[1])


# In[ ]:


#SupportVectorMachine
#It takes a long time and also gives poor results
model=SVC()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
accuracy=accuracy_score(y_pred,y_test)
mod_val.append(round(accuracy,4))
print('Accuracy of Support Vector Classifier:',mod_val[2])


# In[ ]:


#Decision tree Regression
model=DecisionTreeClassifier()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
accuracy=accuracy_score(y_pred,y_test)
mod_val.append(round(accuracy,4))
print('Accuracy of Decision Tree Classifier:',mod_val[3])


# In[ ]:


#GradientBoostingClassifier
model=GradientBoostingClassifier()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
accuracy=accuracy_score(y_pred,y_test)
mod_val.append(round(accuracy,4))
print('Accuracy of Gradient Boosting classifier:',mod_val[4])


# In[ ]:


#Adaboost classifier
model=AdaBoostClassifier()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
accuracy=accuracy_score(y_pred,y_test)
mod_val.append(round(accuracy,4))
print('Accuracy of Ada boost classifier:',mod_val[5])


# In[ ]:


#ExtraTree classifier
model=ExtraTreesClassifier()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
accuracy=accuracy_score(y_pred,y_test)
mod_val.append(round(accuracy,4))
print('Accuracy of Extra Tree classifier',mod_val[6])


# In[ ]:


#RandomForest Classifier
model=RandomForestClassifier()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
accuracy=accuracy_score(y_pred,y_test)
mod_val.append(round(accuracy,4))
print('Accuracy of Random Forest classifier:',mod_val[7])


# In[ ]:


#K nearest Neighbour classifier
model=KNeighborsClassifier()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
accuracy=accuracy_score(y_pred,y_test)
mod_val.append(round(accuracy,4))
print('Accuracy of Nearest neighbour:',mod_val[8])


# Now we constructed the models and trained the data. We find that the ensemble classifiers had higher accuracy. Extra tree classifier had a maximum accuracy among the classifiers. 
# 
# The models built used the data that might include outliers and features that might be irrelevant. We can built an efficient model after EDA.
# 
# There are two things to note here:
# * The dataset did not contain any NaN values else we would have got an error.
# * The metric used here is accuracy. It is valid, if the target variable has equal distribution in all classes

# # Exploratory Data Analysis
# We would run the three traditional EDA commands, describe(), info() and head(). We can then note the observations for each comment and then perform analysis according to the noted observations.

# In[ ]:


train_data.describe().T


# * Here we note that there is no null value in any of the column as expected.
# * We have to see how the data is distributed according to target variable.
# * Two features 'Soil_Type7' and 'Soil_Type15' has all the values as 0, since the min=median=max=0. These features are irrelevant and have to be dropped.
# * Two features 'Soil_type8' and 'Soil_type25' have 0.0066% value as 1.(Since the value is categorical). So we need to further analyze these two columns.

# In[ ]:


print(train_data.info())


# * Wilderness_Area* and Soil_Type* have to be categorical for easy processing.
# * Also the target variable has to be 7 class category.
# * First ten features are continous (Elevation, Aspect, Slope, Horizontal_Distance_To_Hydrology, Vertical_Distance_To_Hydrology, Horizontal_Distance_To_Roadways, Hillshade_9am, Hillshade_Noon, Hillshade_3pm,Horizontal_distance_To_Fire_Points) and we can have scatter plots, correlation matrix with target variable to get necessary informations.

# In[ ]:


print(train_data.head().T)


# Now that we have explored with the three commands we can further process the data with the noted points.

# In[ ]:


# Distribution of Data according to Forest cover type
values=train_data.Cover_Type.value_counts()
plt.bar(values.index,values)
plt.xlabel('Cover types')
plt.ylabel('Counts')
print(values)


# The Data has equal distribution. And so accuracy would also be a better choice for metrics.

# In[ ]:


#Dropping two columns
train_data.drop(['Soil_Type7','Soil_Type15'],axis=1,inplace=True)
train_data.shape


# In[ ]:


# Checking the two features
print(train_data.Soil_Type8.value_counts())
print(train_data.Soil_Type25.value_counts())


# We have only one value of 1 in 'Soil_Type8' and 'Soil_Type25' out of 15119 values. This is too insignificant so we remove this feature also.

# In[ ]:


# Dropping the other two columns as well
train_data.drop(['Soil_Type8','Soil_Type25'],axis=1,inplace=True)
train_data.shape


# In[ ]:


# box plot with the 10 featured column that were noted earlier
Featured=['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology', 
'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm','Horizontal_Distance_To_Fire_Points']
plt.figure(figsize=(15,30))
for i,feature in enumerate(Featured):
    plt.subplot(5,2,i+1)
    sns.boxplot(x='Cover_Type',y=feature,data=train_data).set(title='{} vs Cover_Type'.format(feature))
plt.tight_layout()


# Here we could not find a single feature to clearly distinguish the cover type.
# Lets see some of the correlation matrix of the featured variables.

# In[ ]:


#correlation matrix for featured columns vs others
Feature_Target=pd.concat([train_data[Featured],train_data[['Cover_Type']]],axis=1)
Feature_Target.columns


# In[ ]:


corr=Feature_Target.corr()
plt.figure(figsize=(15,15))
sns.heatmap(corr,square=True,center=0,linewidths=0.5,annot=True)
plt.title('Correlation Matrix')


# Also there are no significant correlation of these columns with the Cover_Type

# Now its time to see how the data are distributed in the categorical variables wilderness and soil type.
# 

# In[ ]:


#Wilderness
train_data.groupby('Cover_Type')['Wilderness_Area1','Wilderness_Area2','Wilderness_Area3','Wilderness_Area4'].sum()


# Here we note that given the forest cover type of 4, the wilderness area will be wilderness area 4 only. 
# Similarly for cover type of 3, the wilderness area would be either 3 or 4.

# In[ ]:


#Soil type. We have 36 soil types now and one cover_type. 
#so in total the last 37 soil types can be used for analyzing soil type
temp=train_data.iloc[:,-37:]
temp.groupby('Cover_Type').sum().T


# These observations implies that bayesian classifier may be of some help.
# We can build a model of categorical variable with naive-bayes classifier.

# In[ ]:


# changing dtype of coulmn
for col in train_data:
    if col[:9]=='Soil_Type' or col[:10]=='Wilderness':
        train_data[col]=train_data[col].astype('category')
train_data['Cover_Type']=train_data['Cover_Type'].astype('category')
train_data.info()


# Note here that we have reduced the memory space from 6.5 MB to 2.5 MB

# # Models After EDA

# We will build the models as before. But there are two difference between the old models and the models to be built now
# 1. We have used similar procedure in building the previous models, so we would write a function for model building and evaluation
# 
# 2. (Very important) We have performed EDA on the data. We have dropped 4 columns!

# In[ ]:


X,y=train_data.iloc[:,:-1],train_data.iloc[:,-1]
X_train,X_test,y_train,y_test=train_test_split(X,y)


# In[ ]:


def model_build_evaluate(classifier,r_list):
    model=classifier()
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    accuracy=accuracy_score(y_pred,y_test)
    r_list.append(round(accuracy,4))
    print('Accuracy of',classifier, ':',round(accuracy,4))
    


# In[ ]:


#list of classifiers
Classifiers=[LogisticRegression,RidgeClassifier,SVC,DecisionTreeClassifier,
             GradientBoostingClassifier,AdaBoostClassifier,ExtraTreesClassifier,
             RandomForestClassifier,KNeighborsClassifier]
eda_val=[]
for classifier in Classifiers:
    model_build_evaluate(classifier,eda_val)


# In[ ]:


ax=sns.scatterplot(x=list(range(9)),y=mod_val)
ax=sns.scatterplot(x=list(range(9)),y=eda_val)
plt.xticks(list(range(9)),Classifiers,rotation=90)
plt.legend(['With out EDA','After EDA'])


# There seems to be a little improvement in the models. We must also include preprocessing here. Except for Ada
# 
# And before that we can construct the bayesian classifier from the catrgorical variable.

# In[ ]:


#Multinomial Naive Bayes is imported. It is more suited for discrete variables
from sklearn.naive_bayes import MultinomialNB


# In[ ]:


# Naive bayes classifier on categorical data
cat_data=train_data.iloc[:,-41:]
print(cat_data.columns)
X,y=cat_data.iloc[:,:-1],cat_data.iloc[:,-1]
X_train,X_test,y_train,y_test=train_test_split(X,y)
model=MultinomialNB()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
accuracy=accuracy_score(y_pred,y_test)
print('Accuracy of Multinomial Naive Bayes classifier',accuracy)


# The multinomial Naive Bayes got third place from last! But wait this classification was made just from wilderness area and soil type. 
# 
# Its time to preprocess and get more closer to the solution.
# 
# In the models we have seen the range of features seems to be different. For elevation the range (diffrence between min and max) is around 2000. 'Horizontal_Distance_To_Roadways' and 'Horizontal_Distance_To_Fire_points' have a range of about 6000 while the range for soil_types are only 1. So there must be a better scaling in the features for better results.

# In[ ]:


#Preprocessing
from sklearn.preprocessing import StandardScaler
#Now we have to make use of the whole model
X,y=train_data.iloc[:,:-1],train_data.iloc[:,-1]
X_train,X_test,y_train,y_test=train_test_split(X,y)
#scaling the first 10 features alone
X_train[Featured]= StandardScaler().fit_transform(X_train[Featured])
X_test[Featured] = StandardScaler().fit_transform(X_test[Featured])


# In[ ]:


# classifying again
pp_val=[]
for classifier in Classifiers:
    model_build_evaluate(classifier,pp_val)


# Here we see there is a significant improvement in SVM classifier from the previous, we can draw scatter plot to further analyze the results.

# In[ ]:


ax=sns.scatterplot(x=list(range(9)),y=mod_val)
ax=sns.scatterplot(x=list(range(9)),y=eda_val)
ax=sns.scatterplot(x=list(range(9)),y=pp_val)
plt.xticks(list(range(9)),Classifiers,rotation=90)
plt.legend(['With out EDA','After EDA','After Preprocessing'])


# Except for ensembled models other models gave a significant improvement on EDA and preprocessing.
# 
# Not to forget it significantly reduced the training time and memory!

# Any comments, feedbacks are welcomed!
