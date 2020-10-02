#!/usr/bin/env python
# coding: utf-8

# In[147]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))



from sklearn.model_selection import train_test_split,GridSearchCV,KFold,LeaveOneOut
from sklearn.metrics import precision_score,recall_score,f1_score,classification_report,accuracy_score,confusion_matrix
from sklearn.tree import DecisionTreeClassifier





# Any results you write to the current directory are saved as output.


# In[110]:


dat=pd.read_csv("../input/WA_Fn-UseC_-HR-Employee-Attrition.csv")


# In[111]:


dat.head(2)


# In[112]:


dat.info()


# In[113]:


dat.nunique()


# In[114]:


dat['Department'].unique()


# In[115]:


dat.groupby(by=["Department",'Attrition']).size().plot(kind='bar')
plt.show()


# In[116]:


dat.groupby(by=['BusinessTravel','Attrition']).size().plot(kind='bar')
plt.show()


# In[117]:


dat.groupby(by=['EducationField','Attrition']).size().plot(kind='bar')
plt.show()


# In[118]:



dat.groupby(by=['EnvironmentSatisfaction','Attrition']).size().plot(kind='bar')
plt.show()


# In[119]:


dat.groupby(by=['Gender','Attrition']).size().plot(kind='bar')
plt.show()


# In[120]:


dat.groupby(by=['JobInvolvement','Attrition']).size().plot(kind='bar')
plt.show()





# In[121]:


dat.groupby(by=['JobLevel','Attrition']).size().plot(kind='bar')
plt.show()


# In[122]:


dat.groupby(by=['JobRole','Attrition']).size().plot(kind='bar')
plt.show()
                        


# In[123]:


dat.groupby(by=['JobSatisfaction','Attrition']).size().plot(kind='bar')
plt.show()



# In[124]:


dat.groupby(by=['MaritalStatus','Attrition']).size().plot(kind='bar')
plt.show()


# In[125]:


print("Average monthly income for males is {}".format(dat[dat['Gender']=='Male']['MonthlyIncome'].mean()))
print("Average monthly income for males is {}".format(dat[dat['Gender']=='Female']['MonthlyIncome'].mean()))

sns.violinplot(x = 'Gender',y = 'MonthlyIncome',data=dat, hue='Attrition',split=True,palette='Set2')
plt.show()


# In[126]:


sns.distplot(dat.Age,kde=False)
plt.show()


# In[127]:


dat.dtypes


# In[128]:


dat['BusinessTravel'] = dat['BusinessTravel'].astype('category')
dat['Department'] = dat['Department'].astype('category')
dat['EducationField'] = dat['EducationField'].astype('category')
dat['EnvironmentSatisfaction'] = dat['EnvironmentSatisfaction'].astype('category')
dat['Gender'] = dat['Gender'].astype('category')
dat['JobInvolvement'] = dat['JobInvolvement'].astype('category')
dat['JobLevel'] = dat['JobLevel'].astype('category')
dat['JobRole'] = dat['JobRole'].astype('category')
dat['JobSatisfaction'] = dat['JobSatisfaction'].astype('category')
dat['MaritalStatus'] = dat['MaritalStatus'].astype('category')
dat['NumCompaniesWorked'] = dat['NumCompaniesWorked'].astype('category')
dat['OverTime'] = dat['OverTime'].astype('category')
dat['RelationshipSatisfaction'] = dat['RelationshipSatisfaction'].astype('category')
dat['StockOptionLevel'] = dat['StockOptionLevel'].astype('category')
dat['WorkLifeBalance'] = dat['WorkLifeBalance'].astype('category')


# In[129]:


#This will return the percentage of Attrition datasets
dat.Attrition.value_counts(normalize=True)*100
#this shows it is an imbalanced dataset


# Spliiting the dataset
# 

# In[130]:


x=dat.drop(columns=['Attrition'])


# In[131]:


y=dat['Attrition']


# In[132]:


x=pd.get_dummies(x)


# In[133]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.33, random_state=42)


# In[134]:


print(X_train.shape)
print(y_train.shape)


# Gini********

# In[135]:


modelgini=DecisionTreeClassifier(criterion='gini')


# In[137]:


modelgini.fit(X_train,y_train)


# In[138]:


predictors_gini=modelgini.predict(X_test)


# Entropy

# In[139]:


modelentropy=DecisionTreeClassifier(criterion='entropy')


# In[140]:


modelentropy.fit(X_train,y_train)


# In[141]:


predictors_entropy=modelentropy.predict(X_test)


# Confusion_Matrix

# In[142]:


Matrix_Gini=confusion_matrix(y_test,predictors_gini)
Matrix_Entropy=confusion_matrix(y_test,predictors_entropy)


print("confusion matrix for gini = \n",Matrix_Gini)

print("confusion matrix for Entropy = \n",Matrix_Entropy)


# In[144]:


#Accuracy Score
print("Accuracy Score for Gini :",accuracy_score(y_test,predictors_gini))
print("Accuracy Score for Entropy :",accuracy_score(y_test,predictors_entropy))


# In[148]:


#Classification Report
print(classification_report(y_test,predictors_gini))
print(classification_report(y_test,predictors_entropy))


# Pruning the Tree

# In[151]:


clf_pruned=DecisionTreeClassifier(criterion='gini',max_depth=3,max_leaf_nodes=5)
clf_pruned.fit(X_train,y_train)


# In[153]:


predictors_pruned=clf_pruned.predict(X_test)


# In[154]:


#Confusion matrix

mat_pruned = confusion_matrix(y_test,predictors_pruned)

print("confusion matrix = \n",mat_pruned)


# In[156]:


#Accuracy Score
print("Accuracy Score is : {}".format(accuracy_score(y_test,predictors_pruned)))


# In[159]:


print(classification_report(y_test,predictors_pruned))


# In[ ]:




