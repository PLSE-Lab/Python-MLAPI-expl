#!/usr/bin/env python
# coding: utf-8

# In[2]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np  # linear algebra
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score 


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[3]:


titanic_train = pd.read_csv('../input/train.csv') 
titanic_test = pd.read_csv('../input/test.csv')


# In[4]:


titanic_data= pd.concat([titanic_train,titanic_test])


# In[5]:


titanic_data.head()


# In[6]:


titanic_data.tail()


# In[7]:


titanic_train.isnull().sum()


# In[8]:


titanic_test.isnull().sum()


# In[9]:


titanic_data.isnull().sum()


# In[10]:


titanic_data.dtypes


# Data cleaning:

# It can be seen that the data has nulls in columns (Age, Cabin, Ebarked)

# In[18]:


import seaborn as sns
sns.heatmap(titanic_data.corr(),annot=True,cmap='RdYlGn',linewidths=0.3) #data.corr()-->correlation matrix
fig=plt.titanic_data()
fig.set_size_inches(12,10)
fig.show()


# In[ ]:


#snsn.heatmap.(titanic_train.isnull(),yticklabels***False)
#import seaborn as sns 
#sns.heatmap(titanic_data.isnull(), xticklabels=True, yticklabels=True)
#titanic_data()


# In[19]:


titanic_data["Survived"].fillna('0', inplace=True)
titanic_data["Fare"].fillna('0', inplace=True)
titanic_data["Cabin"].fillna('U', inplace=True)


# In[20]:


mean = titanic_data['Age'].mean()


# In[21]:


titanic_data["Age"].fillna(mean, inplace=True)


# In[22]:


titanic_data.head()


# In[23]:


titanic_data.tail()


# In[24]:


titanic_data.Survived = titanic_data.Survived.apply(int)
titanic_data.Age = titanic_data.Age.apply(int)
titanic_data.Fare = titanic_data.Fare.apply(int)


# In[25]:


titanic_data.dtypes


# In[ ]:


#titanic_data.dropna(inplace=True)


# In[ ]:


#sns.heatmap(titanic_data.isnull(), xticklabels=True, yticklabels=True)


# In[ ]:


Data visulization: 


# In[29]:


sns.boxplot(x ='Pclass', y= 'Age', data= titanic_data)


# It can be seen that  passengers who are traveling in class 1 and class 2 are tend to be older that thoes who travel in class 3
# belong to the older age range 

# In[27]:


#number of passengers: 
print(" number of passengers in original data:" +str(len(titanic_data)))


# In[28]:


import seaborn as sns
sns.countplot(x= "Survived", data= titanic_data)


# It can be seen that the majority of passengers did not survived compared to those who survived.

# In[30]:


sns.countplot(x= "Survived", hue="Sex", data= titanic_data)


#  The plot shows that on average females are more likely to survive than males 

# In[31]:


# the number of survived passengers among different Pclasses:  
import seaborn as sns
sns.countplot(x='Survived', hue='Pclass', data=titanic_data)


# The graph shows that the majority of survived passengers do belong to the first class, so  first  class passengers 
# have a better chance to survive more that second and third class passengers.

# In[ ]:


#histogram for age : 


# In[32]:


import seaborn as sns
sns.countplot(x='Ticket', hue='Pclass', data=titanic_data)


# In[33]:


#Defing the age ditribution : 
titanic_data["Fare"].plot.hist(bins=20 , figsize=(10,5)) 


# It can be seen that the fair size is between 0 -100

# In[34]:


sns.countplot(x= "SibSp", data= titanic_data)


# The majority of passengers have 0 value of siblings and spouses.

# In[ ]:


#sns.countplot(x = "Cabin", data = titanic_data)


# In[35]:


sns.countplot(x = "Parch", data = titanic_data)


# In[36]:


titanic_data.head()


# In[37]:


#dropping unneccary columns :
titanic_data.drop(['Ticket','Name','Cabin'], axis = 1, inplace = True) 


# In[38]:


titanic_data.head()


# In[39]:


titanic_data = pd.get_dummies(titanic_data , drop_first=True)
titanic_data.head()

Train Data:
#1:Defining the predicted variable and the independed variables.
#2:splitting data into  training and testing .
#3:Creating a model.
#4:Predictions.
#5:Evaluating the performane of the model(classification report). 
#6:Testing the score accuracy.
# In[52]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score 


# In[41]:


#step1:
X = titanic_data.drop('Survived', axis=1)
y = titanic_data['Survived']


# In[42]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[43]:


#step2: feature scaling :
Sc_X = StandardScaler()
X_train = Sc_X.fit_transform(X_train) 
X_test = Sc_X.transform(X_test)

# X_train_mm.shape, X_test_mm.shape
#model = KNeighborsClassifier(n_neighbors=426, )
#model.fit(X_train_mm, y_train)
#model.score(X_test_mm, y_test)


# In[45]:


len(y)


# In[46]:


import math 
math.sqrt(len(y_test))


# In[49]:


#step3: define the model using KNeigbors classifier: init K_NN:
model = KNeighborsClassifier(n_neighbors=19,p=2,)
model.fit(X_train, y_train)


# In[50]:


y_pred=model.predict(X_test)
y_pred


# In[56]:


model.score(X_test, y_test)


# In[53]:


#step 4: Evaluate model:
cm = confusion_matrix(y_test,y_pred)
print (cm)


# In[54]:


#step5: f-score :
print(f1_score(y_test, y_pred))


# In[55]:


print(accuracy_score(y_test,y_pred))

