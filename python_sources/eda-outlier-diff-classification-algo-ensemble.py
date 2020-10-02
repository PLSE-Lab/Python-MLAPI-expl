#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # data visualization library  
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import time
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# Exploratory Data Analysis

# In[ ]:


data = pd.read_csv('../input/breast-cancer-wisconsin-data/data.csv')


# In[ ]:


data.shape


# In[ ]:


data.describe()


# Handling the Missing Data

# In[ ]:


pd.DataFrame(data.isna().sum())


# The last column Unnamed: 32 has all the missing value it have to be removed.
# Also the id column need to be remove since its just an ID which wont contribute anything for our prediction

# In[ ]:


data=data.drop("Unnamed: 32", axis =1)
data=data.drop("id", axis =1)


# frequency of cancer stages

# 63% Benign cases compared to 37% Malignant cases, potentially indicating higher number of Benign

# In[ ]:


sns.countplot(data['diagnosis'],label="Count")


# In[ ]:


data["diagnosis"].replace("M",0,inplace = True)
data["diagnosis"].replace("B",1,inplace = True)


# change value of diognose M = 0 and B = 1

# data["diagnosis"].replace("M",0,inplace = True)
# data["diagnosis"].replace("B",1,inplace = True)

# **Discover outliers with Box plot**
# 
# Outlier it will plotted as point in boxplot but other population will be grouped together and display as boxes.

# In[ ]:


data.boxplot()


# As identified there outliers present in the data, now let us check the data for each column for outliers

# In[ ]:


data.boxplot(column=(['radius_mean','texture_mean','perimeter_mean']),vert=False)


# in the box plot we see the outliers

# In[ ]:


data.boxplot(column=(['compactness_mean','concavity_mean','concave points_mean']),vert=False)


# In[ ]:


data.boxplot(column=(['area_mean']),vert=False)


# **Working with Outliers: Correcting, Removing**
# 
# Outliers and should be dropped or correct , as they cause issues when you model your data.
# 
# We are going to use Interquartile Range Method for removing the outliers in the data.
# The IQR is calculated as the difference between the 75th and the 25th percentiles of the data . We can then calculate the cutoff for outliers as 1.5 times the IQR and subtract this cut-off from the 25th percentile and add it to the 75th percentile to give the actual limits on the data.
# 

# In[ ]:


q25, q75 = np.percentile(data['radius_mean'], 25), np.percentile(data['radius_mean'], 75)
iqr = q75 - q25
cut_off = iqr * 1.5
lower, upper = q25 - cut_off, q75 + cut_off

outliers = [x for x in data['radius_mean'] if x < lower or x > upper]
len(outliers)


# For the radius_mean column we have 14 outliers identified which need to be removed.

# In[ ]:


data['radius_mean'] = np.where(data['radius_mean']< lower, np.NaN, data['radius_mean'])
data['radius_mean'] = np.where(data['radius_mean']> upper, np.NaN, data['radius_mean'])
data['radius_mean'].isna().sum()
data['radius_mean'].replace(to_replace =np.NaN, 
                 value =data['radius_mean'].median(),inplace=True)


# All the identified outliers are initially replaced with NaN which is null.
# once the outliers are replaced with NaN , calculatd the Median of the column and replaced with outliers.

# In[ ]:


# Nomally above statement is coded as below, but if we do so it will identify the medean of data which 
# includes the outliers, which will be wrong hence we need to replace the outliers with 0 or NaN then idenfy the median

# outliers = [x for x in data['radius_mean'] if x < lower or x > upper]
# data['radius_mean'].replace(to_replace =[x for x in data['radius_mean'] if x > lower and x < upper], 
#                 value =data['radius_mean'].median(),inplace=True)

#Same need to be performed for all the columns


# In[ ]:


for column in ['texture_mean','perimeter_mean','area_mean','compactness_mean','concavity_mean','concave points_mean','symmetry_mean','fractal_dimension_mean','radius_se','texture_se','perimeter_se','area_se','smoothness_se','compactness_se','concavity_se','concave points_se','symmetry_se','fractal_dimension_se','radius_worst','texture_worst','perimeter_worst','area_worst','smoothness_worst','compactness_worst','concavity_worst','concave points_worst','symmetry_worst','fractal_dimension_worst']:
    q25, q75 = np.percentile(data[column], 25), np.percentile(data[column], 75)
    iqr = q75 - q25
    cut_off = iqr * 1.5
    lower, upper = q25 - cut_off, q75 + cut_off
    data[column].replace(to_replace =[x for x in data[column] if x > lower and x < upper], 
                 value =data[column].median(),inplace=True)
print( 'Outliers are replaced with Median')


# Now all the outliers are removed with median

# In[ ]:


data.head()


# Once the outliers are removed we can look at the correlation between the columns

# In[ ]:


corr = data.corr()
plt.figure(figsize=(18,18))
sns.heatmap(corr, cbar = True,  square = True, annot=True, fmt= '.2f',annot_kws={'size': 10},
           cmap= 'coolwarm')


# AS per the above heatmap we see highly correlated values to be removed, hence either of the one should be removed. Now below are the variables which will use for prediction

# In[ ]:


data_backup=data
data=data.drop(["radius_se"], axis =1)
data.head()
prediction_var = list(['texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean', 'radius_se', 'texture_se',  'area_se', 'smoothness_se', 'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se', 'fractal_dimension_se' ,'radius_worst', 'texture_worst', 'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst'])
#prediction_var 


# Let us see the correlation metrics once again after removing the columns

# In[ ]:


corr = data.corr()
plt.figure(figsize=(18,18))
sns.heatmap(corr, cbar = True,  square = True, annot=True, fmt= '.2f',annot_kws={'size': 10},
           cmap= 'coolwarm')


# Now the data looks good !

# Split data into training and test sets

# In[ ]:


from sklearn.model_selection import train_test_split

#convert the datset into Test and Train data
X_train, X_test, Y_train, Y_test = train_test_split(data.drop('diagnosis', axis=1), data['diagnosis'],                                                    test_size=0.2, random_state=156)


# Now let us execute different models and obtain the results

# **Logistic Regression**

# In[ ]:


from sklearn.linear_model import LogisticRegression

lgr = LogisticRegression(max_iter = 200)
lgr.fit(X_train,Y_train)
ypred=lgr.predict(X_test)
ypred_Log=ypred
print('Accuracy score - ' ,lgr.score(X_test,Y_test))


# **Random Forest**

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
RFmodel=RandomForestClassifier()
result=RFmodel.fit(X_train, Y_train)
ypred=result.predict(X_test)
ypred_RandomForest=ypred
print('Accuracy score - ' ,accuracy_score(Y_test,ypred))


# **Gradient Boosting**

# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier
gb = GradientBoostingClassifier() 
result=gb.fit(X_train, Y_train)
ypred=result.predict(X_test)

print('Accuracy score - ' ,accuracy_score(Y_test,ypred))


# **Gradient Boosting XGBoost**

# In[ ]:


from xgboost import XGBClassifier
XGB = XGBClassifier() 
result=XGB.fit(X_train, Y_train)
ypred=result.predict(X_test)

print('Accuracy score - ' ,accuracy_score(Y_test,ypred))


# **Naive Bayes**

# In[ ]:


from sklearn.naive_bayes import GaussianNB
NB = GaussianNB() 
result=NB.fit(X_train, Y_train)
ypred=result.predict(X_test)

print('Accuracy score - ' ,accuracy_score(Y_test,ypred))


# **Ensemble**
# 
# Now let us do an Ensemble to get the bet model
# Adding back all the predictions got from the diffrent models executed above.

# In[ ]:


lr_diagnosis=lgr.predict(data.drop('diagnosis',axis=1))
rf_diagnosis=RFmodel.predict(data.drop('diagnosis',axis=1))
gb_diagnosis=gb.predict(data.drop('diagnosis',axis=1))
xgb_diagnosis=XGB.predict(data.drop('diagnosis',axis=1))
nb_diagnosis=NB.predict(data.drop('diagnosis',axis=1))


# In[ ]:


data['lr_diagnosis']=lr_diagnosis
data['rf_diagnosis']=rf_diagnosis
data['gb_diagnosis']=gb_diagnosis
data['xgb_diagnosis']=xgb_diagnosis
data['nb_diagnosis']=nb_diagnosis


# In[ ]:


data.head()


# In[ ]:


X_Train2, X_Test2, Y_Train2, Y_Test2 = train_test_split(data.drop("diagnosis",axis=1),data["diagnosis"],test_size=0.25,random_state=123)


# Now let us run  Gradient Boosting with all updated features

# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier
gb = GradientBoostingClassifier() 
result=gb.fit(X_Train2, Y_Train2)
ypred2=result.predict(X_Test2)
accuracy_score(Y_Test2,ypred2)


# In[ ]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix( Y_Test2 ,gb.predict(X_Test2))

f,ax = plt.subplots(figsize = (5,5))
sns.heatmap(cm,annot = True,linewidth = 1,fmt =".0f",ax = ax)


# **Conclusion**
# 
# We have executed 5 models on the data and none of the models were giving above 90% accuracy. Later I have added all 5 predictions back to actual data to improve the final predictions as features. After adding new features again(which is called as Ensamble) I have executed the Gradient boosting method on the updated data and which gave the 97% of accuracy.
