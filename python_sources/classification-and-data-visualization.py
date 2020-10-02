#!/usr/bin/env python
# coding: utf-8

# # Classification and Data Visualization
# 
# The notebook break down a problem of classification based on a weather in australia's data set. The idea of this work is to show different aproaches in how to visualize a data set, besides the idea is to develop different kind of Machine Learning Algorithms as Random Forest, SVM and Neural Networks.

# ## 1. Getting Started

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sb
import category_encoders as ce
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score,confusion_matrix
from sklearn.metrics import accuracy_score

get_ipython().run_line_magic('matplotlib', 'inline')


# ### 1.1 Loading file

# In[ ]:


australia = pd.read_csv('../input/weatherauscsv/weatherAUS.csv')
australia.head()


# ### 1.2 Dropping some columns

# In[ ]:


australia  = australia.drop(['Location','Date','Evaporation','Sunshine', 'Cloud9am','Cloud3pm',
                           'WindGustDir','WindGustSpeed','WindDir9am','WindDir3pm','WindSpeed9am',
                           'WindSpeed3pm'], axis=1)


# ### 1.3 Splitting 'Y' vector and 'X' matrix

# In[ ]:


Y =  australia.RainTomorrow
X = australia.drop(['RainTomorrow'], axis=1)


# ### 1.4 Plotting balance between positive and negative classes

# In[ ]:


plot_sb = sb.countplot(Y, label='Total')
Rain, NotRain = Y.value_counts()
print('Rain: ',Rain)
print('Not Rain : ',NotRain)


# ### 1.5 Changing boolen values and handling NaN values

# In[ ]:


X = X.replace({'No':0, 'Yes':1})
X = X.fillna(0)
Y = Y.replace({'No':0, 'Yes':1})
Y = Y.fillna(0)


# ### 1.6 Scaling Data

# In[ ]:


X_scaled = (X - X.mean()) / (X.std())
X_scaled.head()


# ## 2. Data Visualization

# ### 2.1 Preparing data to plot

# In[ ]:


# Concatenate the target frame with just 20 columns from corpus_scaled
#X_plot = pd.concat([Y, X_scaled], axis=1) 
X_plot = pd.concat([Y, X_scaled.iloc[:,0:20]], axis=1) 

# Reshaping the frame
X_plot = pd.melt(X_plot, id_vars="RainTomorrow", var_name="Features", value_name='Values')
X_plot.head()


# ### 2.2 Violin Plot

# In[ ]:


# Setting the plt object
plt.figure(figsize=(10,10))
# Setting the violinplot objetc with respecitve atributes
sb.violinplot(x="Features", y="Values", hue="RainTomorrow", data=X_plot, split=True, inner="quart")
# Rotation of x ticks
plt.xticks(rotation=90)


# ### 2.3 Joint plot

# In[ ]:


# Correlation is taken from Pearsonr value, 1 is totally correlated.
sb.jointplot(X_scaled.loc[:,'MinTemp'], 
              X_scaled.loc[:,'MaxTemp'], kind="regg", color="#ce1414")


# > ### 2.4 Correlation Matrix

# In[ ]:


f, ax = plt.subplots(figsize=(18, 18))
sb.heatmap(X_scaled.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.xticks(rotation=90)


# ## 3. Classification Algorithms

# ### 3.1 Splitting train and test set

# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(X_scaled, Y, test_size=0.25, random_state=28)


# ### 3.2 Random Forest

# In[ ]:


clf_rf = RandomForestClassifier(random_state=23)      
clr_rf = clf_rf.fit(x_train,y_train)


# In[ ]:


y_predict = clf_rf.predict(x_test)
accuracy = accuracy_score(y_test, y_predict )
print('Accuracy: ', accuracy)


# In[ ]:


conf_matrix = confusion_matrix(y_test, y_predict)
sb.heatmap(conf_matrix, annot=True, fmt="d")


# ### 3.3 Suppor Vector Machine (SVM)

# In[ ]:


clf_svm = SVC(kernel='linear', random_state=12)
clf_svm = clf_svm.fit(x_train, y_train)


# In[ ]:


y_predict = clf_svm.predict(x_test)
accuracy = accuracy_score(y_test, y_predict)
print('Accuracy: ', accuracy)


# In[ ]:


conf_matrix = confusion_matrix(y_test, y_predict)
sb.heatmap(conf_matrix, annot=True, fmt="d")


# In[ ]:




