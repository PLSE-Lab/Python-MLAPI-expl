#!/usr/bin/env python
# coding: utf-8

# **Importing python libraries**

# In[1]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# **Importing machine learning libraries**

# In[2]:


from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, cross_val_score

import keras 
import tensorflow 
from keras import Sequential
from keras.layers import Dense,Activation


# Let's read the dataset using the pandas

# In[3]:


breastData=pd.read_csv("../input/data.csv")


# In[4]:


breastData.info()


# In[5]:


breastData.head()


# In[6]:


breastData.describe()


# In[7]:


breastData.shape


# There's a column named as 'Unnamed: 32' which doesn't seem to be of much importance. so lets remove it

# In[8]:


breastData.drop('Unnamed: 32',axis=1,inplace=True)


# In[9]:


breastData.shape


# Also, Lets check if there is any null entries in the dataframe

# In[10]:


breastData.isnull().sum()


# As the 'id' column is unique, lets drop that column from our primary dataframe

# In[11]:


breastDataid=breastData['id']
breastData.drop('id',axis=1,inplace=True)


# In[12]:


breastData.shape


# Let's find the frequency of the cacncer stages

# In[13]:


sns.countplot(x=breastData['diagnosis'])


# As we can see from the above plot that the count of benign is higher

# Let's plot a correlation graph between all the features 

# In[14]:


plt.figure(figsize=(14,14))
sns.heatmap(breastData.corr(), cbar = True,  square = True, annot=False,cmap= 'coolwarm')


# Following observations are made from the above correlation graph 
# 
# 1. Radius, parameter and area are highly correlated
# 2. compactness_mean, concavity_mean and concavepoint_mean are highly correlated

# Also Lets plot the pair plot between all the features 

# In[15]:


# sns.pairplot(breastData)


# In[16]:


plt.rcParams['figure.figsize'] = (18, 16)

plt.subplot(2, 2, 1)
sns.boxenplot(x = 'diagnosis', y = 'radius_mean', data = breastData, palette = 'rainbow')
plt.title('Diagnosis vs radius_mean', fontsize = 20)

plt.subplot(2, 2, 2)
sns.boxenplot(x = 'diagnosis', y = 'texture_mean', data = breastData, palette = 'summer')
plt.title('Diagnosis vs texture_mean', fontsize = 20)

plt.subplot(2, 2, 3)
sns.boxenplot(x = 'diagnosis', y = 'perimeter_mean', data = breastData, palette = 'spring')
plt.title('Diagnosis vs perimeter_mean', fontsize = 20)

plt.subplot(2, 2, 4)
sns.boxenplot(x = 'diagnosis', y = 'area_mean', data = breastData, palette = 'deep')
plt.title('Diagnosis vs area_mean', fontsize = 20)

plt.show()


# In[17]:


plt.rcParams['figure.figsize'] = (18, 16)

plt.subplot(2, 2, 1)
sns.violinplot(x = 'diagnosis', y = 'smoothness_mean', data = breastData, palette = 'Reds')
plt.title('Diagnosis vs smoothness_mean', fontsize = 20)

plt.subplot(2, 2, 2)
sns.violinplot(x = 'diagnosis', y = 'compactness_mean', data = breastData, palette = 'Oranges')
plt.title('Diagnosis vs compactness_mean', fontsize = 20)

plt.subplot(2, 2, 3)
sns.violinplot(x = 'diagnosis', y = 'concavity_mean', data = breastData, palette = 'Purples')
plt.title('Diagnosis vs concavity_mean', fontsize = 20)

plt.subplot(2, 2, 4)
sns.violinplot(x = 'diagnosis', y = 'concave points_mean', data = breastData, palette = 'Greens')
plt.title('Diagnosis vs concave points_mean', fontsize = 20)

plt.show()


# In[18]:


plt.rcParams['figure.figsize'] = (18, 16)

plt.subplot(2, 2, 1)
sns.swarmplot(x = 'diagnosis', y = 'symmetry_mean', data = breastData, palette = 'magma')
plt.title('Diagnosis vs symmetry_mean', fontsize = 20)

plt.subplot(2, 2, 2)
sns.swarmplot(x = 'diagnosis', y = 'fractal_dimension_mean', data = breastData, palette = 'Blues')
plt.title('Diagnosis vs fractal_dimension_mean', fontsize = 20)

plt.subplot(2, 2, 3)
sns.swarmplot(x = 'diagnosis', y = 'radius_se', data = breastData, palette = 'viridis')
plt.title('Diagnosis vs radius_se', fontsize = 20)

plt.subplot(2, 2, 4)
sns.swarmplot(x = 'diagnosis', y = 'texture_se', data = breastData, palette = 'plasma')
plt.title('Diagnosis vs texture_se', fontsize = 20)


# In[19]:


plt.rcParams['figure.figsize'] = (18, 16)

plt.subplot(2, 2, 1)
sns.boxplot(x = 'diagnosis', y = 'perimeter_se', data = breastData, palette = 'Greys')
plt.title('Diagnosis vs perimeter_se', fontsize = 20)

plt.subplot(2, 2, 2)
sns.boxplot(x = 'diagnosis', y = 'smoothness_se', data = breastData, palette = 'PuRd')
plt.title('Diagnosis vs smoothness_se', fontsize = 20)

plt.subplot(2, 2, 3)
sns.boxplot(x = 'diagnosis', y = 'area_se', data = breastData, palette = 'RdPu')
plt.title('Diagnosis vs area_se', fontsize = 20)

plt.subplot(2, 2, 4)
sns.boxplot(x = 'diagnosis', y = 'compactness_se', data = breastData, palette = 'BuPu')
plt.title('Diagnosis vs compactness_se', fontsize = 20)

plt.show()


# In[20]:


encoder=LabelEncoder()

breastData['diagnosis']=encoder.fit_transform(breastData['diagnosis'])


# In[21]:


sns.countplot(breastData['diagnosis'])


# Let's separate the features accordingly

# In[22]:


X=breastData.drop('diagnosis',axis=1)
y=breastData['diagnosis']


# Now, Let's Separate the data into a train test split

# In[23]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[24]:


print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# In[25]:


sc=StandardScaler()

X_train=sc.fit_transform(X_train)
X_test=sc.fit_transform(X_test)


# **Logistic Regression**

# In[26]:


model=DecisionTreeClassifier()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)


# In[27]:


print("Training accuracy :", model.score(X_train, y_train))
print("Testing accuarcy :", model.score(X_test, y_test))


# In[28]:


cr = classification_report(y_test, y_pred)
print(cr)

# confusion matrix 
plt.rcParams['figure.figsize'] = (5, 5)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot = True, cmap = 'coolwarm')


# **Random forest classifier**

# In[29]:


model=RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


# In[30]:


print("Training accuracy :", model.score(X_train, y_train))
print("Testing accuarcy :", model.score(X_test, y_test))


# In[31]:


cr = classification_report(y_test, y_pred)
print(cr)

# confusion matrix 
plt.rcParams['figure.figsize'] = (5, 5)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot = True, cmap = 'coolwarm')


# **Support vector machine**

# In[32]:


model=SVC()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)


# In[33]:


print("Training accuracy :", model.score(X_train, y_train))
print("Testing accuarcy :", model.score(X_test, y_test))


# In[34]:


cr = classification_report(y_test, y_pred)
print(cr)

# confusion matrix 
plt.rcParams['figure.figsize'] = (5, 5)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot = True, cmap = 'coolwarm')


# **Hyperparameter tuning for SVM**

# Let's use grid search cv for SVM

# In[35]:


param = {
    'C': [0.8,0.9,1,1.1,1.2,1.3,1.4],
    'kernel':['linear', 'rbf'],
    'gamma' :[0.1,0.8,0.9,1,1.1,1.2,1.3,1.4]
}


# In[36]:


grid_svc = GridSearchCV(model, param_grid = param, scoring = 'accuracy', cv = 10,verbose=2)


# In[37]:


grid_svc.fit(X_train, y_train)
print("Best Parameters: ", grid_svc.best_params_)
print("Best Accuarcy: ", grid_svc.best_score_)


# In[38]:


grid_svc.best_params_


# Now, the above params are the best for the given model, so lets use these params

# In[39]:


gridModelSVC=SVC(C=1.3,gamma=0.1,kernel='linear')
gridModelSVC.fit(X_train,y_train)
y_pred=gridModelSVC.predict(X_test)


# In[40]:


print(classification_report(y_test, y_pred))
print("Testing accuarcy :", model.score(X_test, y_test))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot = True, cmap = 'coolwarm')


# **Nearest Neighbours Classifier**

# In[41]:


model=KNeighborsClassifier(n_neighbors=5)
model.fit(X_train,y_train)
y_pred=model.predict(X_test)


# In[42]:


print(classification_report(y_test, y_pred))
print("Testing accuarcy :", model.score(X_test, y_test))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot = True, cmap = 'coolwarm')


# Neural Network Classifier

# In[43]:


neuralModel = Sequential()
neuralModel.add(Dense(input_dim=30, output_dim=30))
neuralModel.add(Dense(input_dim=30, output_dim=30))
neuralModel.add(Dense(input_dim=30, output_dim=2))
neuralModel.add(Activation("softmax"))

neuralModel.compile(loss='sparse_categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])


# In[44]:


neuralModel.fit(X_train,y_train,epochs=30,batch_size=16,validation_data=[X_test,y_test])


# So the neural network predicted with an accuracy of 98.3%

# Algos test

# In[46]:


from sklearn.ensemble import RandomForestClassifier


# In[52]:


randomforestModel=RandomForestClassifier(n_jobs=-1,criterion='gini')
randomforestModel.fit(X_test,y_test)
y_pred=randomforestModel.predict(X_test)


# In[61]:


print(classification_report(y_test,y_pred))

