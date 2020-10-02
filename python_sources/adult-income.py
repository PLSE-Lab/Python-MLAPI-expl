#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#load dataset
data = pd.read_csv('../input/adult.csv')


# In[3]:


#view some data
data.head()


# In[4]:


data.shape


# In[5]:


#information about data type
data.info()


# In[6]:


#Count the values of target variables
print(data.income.value_counts())
sb.countplot(x='income', data=data)
plt.show()


# ### Analysis on missing variable

# In[7]:


#replace missing variable('?') into null variable using numpy
data = data.replace('?', np.NaN)


# In[8]:


#let's count the how many variable missing
data.isnull().sum()


# In[9]:


#plotting of Null variable
plt.figure(figsize=(10,6))
sb.heatmap(data.isnull())
plt.show()


# In[10]:


#let's fill null variable 
var = data['native-country'].mode()
data['native-country'] = data['native-country'].replace(np.NaN,var[0])

var1 = data.workclass.mode()[0]
data.workclass = data.workclass.replace(np.NaN, var1)

var2 = data.occupation.mode()[0]
data.occupation = data.occupation.replace(np.NaN,var2)


# In[11]:


#again check there is null value or not
print(list(data.isnull().sum()))
plt.figure(figsize=(10,6))
sb.heatmap(data.isnull())
plt.show()


# In[12]:


#convert string into integer
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
cols = ['workclass', 'education','marital-status', 'occupation',
       'relationship', 'race', 'gender','native-country','income']
for col in cols:
    data[col] = le.fit_transform(data[col])


# In[13]:


data.head()


# In[14]:


data.describe()


# In[15]:


#Correlation between attributes
corr = data.corr()
plt.figure(figsize=(20, 10))
sb.heatmap(corr, annot=True)
plt.show()


# In[16]:


#Violin plot
plt.style.use('ggplot')
cols = ['age', 'fnlwgt', 'education', 'marital-status', 'occupation', 'relationship', 'gender']
result = {0:'<=50k', 1:'>50'}
print(result)
for col in cols:
    plt.figure(figsize=(12, 5))
    plt.title(str(col) +' with' + ' income')
    sb.violinplot(x=data.income, y=data[col], data=data)
    plt.show()


# In[26]:


X = data.iloc[:,:14]
X = X.values
y = data['income'].values


# In[32]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)


# ## Let's apply the algorithms

# In[34]:


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


# In[42]:


algo = {'LR': LogisticRegression(), 
        'DT':DecisionTreeClassifier(), 
        'RFC':RandomForestClassifier(n_estimators=100), 
        'SVM':SVC(gamma=0.01),
        'KNN':KNeighborsClassifier(n_neighbors=10)
       }

for k, v in algo.items():
    model = v
    model.fit(X_train, y_train)
    print('Acurracy of ' + k + ' is {0:.2f}'.format(model.score(X_test, y_test)*100))

