#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import all need libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#load dataset
data = pd.read_csv('../input/horse.csv')


# In[3]:


#view some data
data.head()


# In[4]:


#shape
data.shape


# In[5]:


#information about the datatype of datesets
data.info()


# In[6]:


data.describe()


# # Analysis on datasets

# ## Lets's analysis on missing data

# In[7]:


data.isnull().any()


# In[8]:


#let's count the mising values from each features
null = data.isnull().sum() #Count the missing value
#let's see the missing values in percetange format
null = null/len(data)*100
null = null[null>0]
null.sort_values(inplace=True, ascending=False)
null


# In[9]:


null = null.to_frame() #convert into dataframe
null.columns = ['count'] #add count column
#add new feature
null.index.name = ['Feature'] 
null['Feature'] = null.index


# In[10]:


#ploting missing value of each attributes by percentage
plt.figure(figsize=(20, 10))
sb.set(font_scale=1.2)
plt.style.use('ggplot')
sb.barplot(x='Feature', y='count', data=null)
plt.show()


# In[11]:


## replacing surgery and age
data = data.replace({'no':0, 'yes': 1, 'adult':1, 'young':2})


# ### Fill the missing value

# In[12]:


# rectal_temp
data.rectal_temp = data.rectal_temp.fillna(value=data.rectal_temp.mode()[0])


# In[13]:


# pulse
data.pulse = data.pulse.fillna(value=data.pulse.mean())


# In[14]:


# respiratory_rate
data.respiratory_rate = data.respiratory_rate.fillna(value=data.respiratory_rate.mean())


# In[15]:


# abdomo_protein
data.abdomo_protein = data.abdomo_protein.fillna(value=data.abdomo_protein.mode()[0])


# In[16]:


# total_protein
data.total_protein = data.total_protein.fillna(value=data.total_protein.mean())


# In[17]:


# packed_cell_volume
data.packed_cell_volume = data.packed_cell_volume.fillna(value=data.packed_cell_volume.mean())


# In[18]:


# nasogastric_reflux_ph
data.nasogastric_reflux_ph = data.nasogastric_reflux_ph.fillna(value=data.nasogastric_reflux_ph.mean())


# In[19]:


#filling all object data type with their mode values
col = null.index
for i in col:
    data[i] = data[i].fillna(data[i].mode()[0])


# In[20]:


#let's check there is missing value or not throught plot
plt.figure(figsize=(12, 8))
sb.heatmap(data.isnull())
plt.show()


# #### This means there is no any missing value in this datasets

# ### Converting String to interger

# In[21]:


# Label Encoding
from sklearn.preprocessing import LabelEncoder
col = data.columns
for i in col:
    lb = LabelEncoder()
    lb.fit(data[i].values)
    data[i] = lb.transform(data[i].values)


# ### Analysis on Target variable.
# 
# In this dataset, target varible is outcome

# In[23]:


#count the Classifing values
data.outcome.value_counts()


# In[24]:


#plotint of above count
plt.figure(figsize=(8, 5))
sb.countplot(x='outcome', data=data)
plt.show()


# In[25]:


#Separate data into feature and target
X = data.drop('outcome', axis=1).values
y = data['outcome'].values


# In[26]:


#split the data into train and test perpose
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=10)


# ## Applying alogrithm

# In[27]:


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


# In[28]:


algo = {'LR':LogisticRegression(),
        'DT':DecisionTreeClassifier(),
        'RFC':RandomForestClassifier(n_estimators=100),
        'SVM':SVC(gamma=0.001),
        'KNN':KNeighborsClassifier(n_neighbors=10)}


# In[29]:


for k, v in algo.items():
    model = v
    model.fit(X_train, y_train)
    print('Acurracy of ' + k + ' is {0:.2f}'.format(model.score(X_test, y_test)*100))


# #### Here, Maximum accuracy is Random Frorest . So, i make here confusion matric of RFC

# In[30]:


model = RandomForestClassifier(n_estimators=100)
model.fit(X_test, y_test)
y_pred = model.predict(X_test)


# In[31]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_pred, y_test)
sb.set(font_scale=1.3)
sb.heatmap(cm, annot=True)
plt.show()


# In[34]:


print("Prediction value " + str(model.predict([X_test[3]])))
print("Real value " + str(y_test[3]))


# ###'died':0, 'euthanized':1, 'lived':2
