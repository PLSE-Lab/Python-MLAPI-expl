#!/usr/bin/env python
# coding: utf-8

# Business Objective: Use machine learning to create a model that predicts which passengers survived the Titanic shipwreck.
# 
# 

# In[ ]:


#Data Analysis Liabrarise
import numpy as np 
import pandas as pd

#Data visualization liabraries
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import preprocessing 



# **Loading Data**
# 
# 

# In[ ]:


import pandas as pd
test = pd.read_csv("../input/titanic-machine-learning-from-disaster/test.csv")
train = pd.read_csv("../input/titanic-machine-learning-from-disaster/train.csv")


# In[ ]:


#Take backup of data set for some other use if needed.
train_data = train.copy()
test_data  = test.copy()


# In[ ]:


train.head(5)


# **Pre Processing**

# In[ ]:


train.dtypes


# **Null value treatment**

# In[ ]:


train.isna().sum()


# In[ ]:


train = train.fillna(method='ffill')


# In[ ]:


train.isna().sum()


# In[ ]:


#Dropping columns not required in data set
train = train.drop(['Cabin','Ticket','Parch'], axis = 1)


# In[ ]:


train.isna().sum()


# **Outlier treatment**
# 

# In[ ]:


train.describe().T


# In[ ]:


sns.boxplot(x = train['Fare'])


# In[ ]:


def outliers_transform(base_dataset):
    for i in base_dataset.var().sort_values(ascending=False).index[0:10]:
        x=np.array(base_dataset[i])
        qr1=np.quantile(x,0.25)
        qr3=np.quantile(x,0.75)
        iqr=qr3-qr1
        utv=qr3+(1.5*(iqr))
        ltv=qr1-(1.5*(iqr))
        y=[]
        for p in x:
            if p <ltv or p>utv:
                y.append(np.median(x))
            else:
                y.append(p)
        base_dataset[i]=y


# In[ ]:


outliers_transform(train)


# In[ ]:


sns.boxplot(x = train['Fare'])


# In[ ]:


train.describe().T


# ** Label encoders**

# In[ ]:


train.drop('PassengerId',axis=1,inplace=True)
train.drop('Name',axis=1,inplace=True)


# In[ ]:


label_encoder = preprocessing.LabelEncoder() 
train['Embarked']= label_encoder.fit_transform(train['Embarked'])
train['Sex']= label_encoder.fit_transform(train['Sex'])


# In[ ]:


train


# ** Univariate analysis (EDA)** 

# In[ ]:


for i in train.var().index:
    sns.distplot(train[i],kde=False)
    plt.show()


# **Bivariate analysis (EDA)**

# In[ ]:


plt.figure(figsize=(20,10))
sns.heatmap(train.corr())


# **Model Building**

# Spliting the train data

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score

target = train['Survived']
predictors = train.drop(['Survived'], axis = 1) 

x_train,x_test,y_train,y_test=train_test_split(predictors, target, test_size=0.20, random_state=43)


# In[ ]:


x_train.shape


# In[ ]:


x_test.shape


# In[ ]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier



# In[ ]:


classifier = DecisionTreeClassifier()
# Train Decision Tree Classifer
classifier = classifier.fit(x_train,y_train)
#Predict the response for test dataset
y_pred = classifier.predict(x_test)
print("Accuracy:",accuracy_score(y_test, y_pred))

