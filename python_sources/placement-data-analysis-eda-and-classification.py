#!/usr/bin/env python
# coding: utf-8

# #                 **Campus Recruitment Data Analysis**
# In this Note we use the data provided by Ben Rosan D on campus recruitment process and try to solve the question arise from the dataset.
# These Questions are
# 1. Which factor influenced a candidate in getting placed?
# 1. Does percentage matters for one to get placed?
# 1. Which degree specialization is much demanded by corporate?
# 
# In addition the these given questions there are some questions we have to solve these are
# 1. Classification of Test data into Placed or Not Placed Category.
# 1. Is there any correlation between the different exams percentage with each other (one to one) and with salary

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# Importing the Libraries

# In[ ]:


from pandas import plotting
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.figure_factory as ff
import plotly.offline as py
from plotly.offline import iplot
import plotly.graph_objs as go
from plotly.offline import plot


# Reading the Dataset

# In[ ]:


df = pd.read_csv("/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv")
df.head()


# In[ ]:


placed=df.dropna()
N_placed=df[df["status"]=="Not Placed"]


# In[ ]:


placed.head()


# In[ ]:


N_placed.head()


# **Visualisation**

# In[ ]:


fig=plt.figure(figsize=(18,10))
plt.subplot(1,3,1)
plt.pie(df["gender"].value_counts(),labels={"Male"," Female"},colors={"cyan", "yellow"},
        shadow=True,autopct = '%.2f%%')
plt.title("Total Student")
plt.subplot(1,3,2)
plt.pie(placed["gender"].value_counts(),labels={"Male"," Female"},colors={"blue", "pink"},
        shadow=True,autopct = '%.2f%%')
plt.title("Placed Student")
plt.subplot(1,3,3)
plt.pie(N_placed["gender"].value_counts(),labels={"Male"," Female"},colors={"green", "orange"},
        shadow=True,autopct = '%.2f%%')
plt.title("Not Placed Student")


# From the above pie charts we can conclude the following points-
# * Male Students have a higher number of placed students in the aspect of their representation in the  total population.
# * Female Students have a lower number of placed students in the aspects of their representation in the total population.

# In[ ]:


sns.pairplot(data=df,kind="scatter",hue="gender")


# In the above pairplot we can concludes the following points-
# * Mean of ssc percentage is around 60% for Male students and around 80% for Female students.
# * Mean of hsc percentage is around 60% for Male students and around 60% for Female students. 
# * Mean of degree percentage is around 60% for Male students and around 65% for Female students.
# * Mean of mba percentage is around 55% for Male students and around 65% for Female students.
# * Mean of Entrance Test  percentage is around 55% for Male students and around 55% for Female students.
# There is positive correlation between the different exam percentage.
# There is not any correlation between salary and exam percentage.
# We can also see that the mean average salary of male students are higher than mean average salary of female students.

# In[ ]:


sns.pairplot(data=df,kind="scatter",hue="status")


# From the above pair plot we can conclude that those students who consistently score lower percentages in their different examinations are not placed. It is strongly evident in the SSC, HSC  and Degree percentage but in Etest and Mba there is not good evidence to support the claim.

# In[ ]:


gen=px.scatter_3d(df,x="ssc_p",y="hsc_p",z="degree_p",color="status")
iplot(gen)


# In the above 3d plot the same information can be extracted that in the Not placed students there are more numbers of the students who scored less than 60 percentage in their SSC,HSC and Degree exam.

# In[ ]:


gen=px.scatter_3d(df,x="mba_p",y="etest_p",z="degree_p",color="status")
iplot(gen)


# In[ ]:


fig=plt.figure(figsize=(12,6))
sns.countplot("specialisation", hue="status", data=df)
plt.show()


# Marketing and Finance are the most demanded specialisation among the two specialisation according to the given data.

# In[ ]:


plt.figure(figsize =(18,6))
sns.boxplot("salary", "gender", data=df)
plt.show()


# From the above box plot we can say that there is more no of outlier in Male students than Female students in terms of Salary.

# 

# In[ ]:


plt.figure(figsize =(18,6))
sns.boxplot("salary", "workex", data=df)
plt.show()


# # **Classification**

# In[ ]:


df.drop("hsc_b",inplace=True,axis=1)
df.drop("ssc_b",inplace=True,axis=1)
df.drop("sl_no",inplace=True,axis=1)
X=df.iloc[:,:-2].values
Y=df.iloc[:,-2].values


# In[ ]:


X


# In[ ]:


Y


# **Encoding the data **

# In[ ]:


from sklearn.preprocessing import LabelEncoder
labelencoder_X=LabelEncoder()
X[:,3]=labelencoder_X.fit_transform(X[:,3])
X[:,0]=labelencoder_X.fit_transform(X[:,0])
X[:,5]=labelencoder_X.fit_transform(X[:,5])
X[:,6]=labelencoder_X.fit_transform(X[:,6])
X[:,8]=labelencoder_X.fit_transform(X[:,8])
Y=labelencoder_X.fit_transform(Y)


# Train and Test Data split
# Train = 60% of given data.
# Test= 40% of the given data.

# In[ ]:


X


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.4,random_state=0)


# Standard Scaling of the data

# In[ ]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# # *Decison Tree Classifier*

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy',)
classifier.fit(X_train, Y_train)
Y_pred = classifier.predict(X_test)
from sklearn.metrics import roc_auc_score
auc = roc_auc_score(Y_test, Y_pred)
print(auc)


# In[ ]:


from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
print(classification_report(Y_test,Y_pred))


# **Confusion Matrix**

# In[ ]:


print(confusion_matrix(Y_test,Y_pred))


# We can say that our model predict
# * TP(Actual Yes and Predicted Yes)=51
# * FP(Actual No and Predicted Yes)=7
# * FN(Actual Yes and Predicted No) =8
# * TN (Actual No and Predicted No) =20

# In[ ]:


accuracy_score(Y_test, Y_pred)


# **We have a pretty decent accuracy of 82 percentage.**

# **Feature Importance**

# In[ ]:


imp=classifier.feature_importances_*100
Fec=pd.DataFrame(imp,columns=["Importance"])

Nam=["Gender","SSC %","HSC %","HSC Stream","Degree % ","Degree Stream",
              "Work Ex","Entrance %"," Specialisation","Mba %"]
Fec["Features"]=Nam
Fec.head(10)


# In[ ]:


fig=plt.figure(figsize=(12,6))
sns.barplot(Fec.Features,Fec.Importance)


#  From the above bar chart we know that If we have a 5 percentage minimum confidence bound then, 
#      The important features which decide one probability of getting placed are- SSC%, HSC%, Degree%, MBA% .
#  

# # Random Forest Classification 

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
#Using Random Forest Algorithm
random_forest = RandomForestClassifier(n_estimators=30,random_state=0)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
from sklearn.metrics import roc_auc_score
auc = roc_auc_score(Y_test, Y_pred)
print(auc)


# In[ ]:


from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
print(classification_report(Y_test,Y_pred))


# In[ ]:


print(confusion_matrix(Y_test,Y_pred))


# We can say that our model predict
# * TP(Actual Yes and Predicted Yes)=52
# * FP(Actual No and Predicted Yes)=7
# * FN(Actual Yes and Predicted No) =8
# * TN (Actual No and Predicted No) =19

# In[ ]:


accuracy_score(Y_test, Y_pred)


# In[ ]:


imp=random_forest.feature_importances_*100
Fec=pd.DataFrame(imp,columns=["Importance"])

Nam=["Gender","SSC %","HSC %","HSC Stream","Degree % ","Degree Stream",
              "Work Ex","Entrance %"," Specialisation","Mba %"]
Fec["Features"]=Nam
Fec.head(10)


# In[ ]:


fig=plt.figure(figsize=(12,6))
sns.barplot(Fec.Features,Fec.Importance)


# From the above bar chart we know that If we have a 5 percentage minimum confidence bound then, The important features which decide one probability of getting placed are- SSC%, HSC%, Degree%, MBA%, Work EX and Entrance % .
