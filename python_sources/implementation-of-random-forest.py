#!/usr/bin/env python
# coding: utf-8

# <h1><center><b><font color="Purple">RANDOM FOREST</font></b></center></h1>

# - <u><b>`Random Forest`</b></u> is a learning method that operates by constructing multiple decision trees. The final decision is made based on the majority of the trees and is chosen by the random forest.<br><br>
# - Random Forest runs efficiently in large databases and produces highly accurate predictions by estimating missing data.<br><br>
# - Random forests or random decision forests are an ensemble learning method for classification, regression and other tasks that operates by constructing a multitude of decision trees at training time and outputting the class that is the mode of the classes (classification) or mean prediction (regression) of the individual trees.<br><br>
# - Random decision forests correct for decision trees' habit of overfitting to their training set.<br><br>
# - In general, the more trees in the forest the more robust the forest looks like. In the same way in the random forest classifier, the higher the number of trees in the forest gives the high accuracy results.<br><br>

# The dataset consist of various columns showing the symptoms for a patient to be suffering from brest cancer.

# ### Importing Libraries

# In[ ]:


import pandas as pd   
# for reading the dataset

import matplotlib.pyplot as plt 
# for drawing the graph
get_ipython().run_line_magic('matplotlib', 'inline')
# used so that the graph is drawn in the confined boundaries

from sklearn.ensemble import RandomForestClassifier
# used to enable the use of random forest algorithm on the dataset


# ### Loading the dataset

# In[ ]:


ds = pd.read_csv('../input/breastcancer/breastcancer_test.csv')
ds


# In[ ]:


ds.info()

# This function is used to get a concise summary of the dataframe.


# In[ ]:


ds.describe()

# The .describe() method is use to give a descriptive exploration on the dataset


# In[ ]:


ds.shape

# Returns the number of rows and columns


# In[ ]:


ds.dtypes

# Returns the datatype of the values in each column in the dataset


# In[ ]:


ds.isnull().sum()

# .isnull() is used to check for null values in the dataset. It returns result in true/false manner.

# .sum() used with isnull() gives the combined number of null values in the dataset if any.


# In[ ]:


ds.hist(grid=True, figsize=(20,10), color='c')


# In[ ]:


ds1 = ds.drop(['Class'], axis='columns')    # Independent Variable
ds2 = ds.Class        # Dependent Variable


# In[ ]:


import seaborn as sns

# importing seaborn library for interpretation of visual data


# In[ ]:


x, y = ds1['Cl.thickness'], ds1['Cell.size']
with sns.axes_style("white"):
    sns.jointplot(x=x, y=y, kind="hex", color="k")


# In[ ]:


sns.jointplot(x=ds1['Cl.thickness'], y=ds1['Cell.shape'], data=ds1, kind="kde")


# In[ ]:


sns.pairplot(ds1)


# In[ ]:


g = sns.PairGrid(ds1)
g.map_diag(sns.kdeplot)
g.map_offdiag(sns.kdeplot, n_levels=6)


# #### Dividing the dataset into Train and Test Partition 

# In[ ]:


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(ds1,ds2, test_size=0.3, random_state=0) 


# ##### Initializing the model with the Random Forest Classifier to be used for testing and training

# In[ ]:


model = RandomForestClassifier(n_estimators=100)
model.fit(x_train, y_train)


# ##### Checking for the score or accuracy of the model implemented

# In[ ]:


model.score(x_test, y_test)


# In[ ]:


y_predicted = model.predict(x_test)


# ##### Implementing Confusion Matrix

# In[ ]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_predicted)
cm


# In[ ]:


import seaborn as sn
#importing seaborn for graphical representation of the data
f,ax = plt.subplots(figsize=(9,6))
#plt.figure(figsize=(10,7))
sn.heatmap(cm, annot=True,fmt="d",linewidths=.5,ax=ax)
plt.xlabel('Predicted')
plt.ylabel('Actual')


# In[ ]:





# In[ ]:




