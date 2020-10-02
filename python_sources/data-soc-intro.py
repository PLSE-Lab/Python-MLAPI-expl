#!/usr/bin/env python
# coding: utf-8

# **Welcome datasoc second python(pandas) workshop**

# <h1>Author: Richard<h1>
# ***Master of computer science***

# <p>Gao jian support some script<p>

# In[1]:


from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'


# In[2]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# **Read dataset**

# <p>Pandas can read different kinds of file, such as csv, json, sql.<p><br>
#     More info can be found in https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html

# In[4]:


df = pd.read_csv('../input/iris.csv')
df.head()


# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.describe.html

# In[5]:


df.describe()


# In[13]:


df.columns
df.species.describe()
df.species.unique()


# **plot data**

# In[6]:


import matplotlib.pyplot as plt
# The following line just for the convience of seeing the plot in the notebook sheet
get_ipython().run_line_magic('matplotlib', 'inline')


# In[7]:


from random import randint, seed
seed(20)
x = [i for i in range(1, 21)]
y = [randint(20, 60) for _ in range(20)]
y_x = list(zip(x, y))
print(y_x)
plt.xticks(np.arange(0, 21, 1));
plt.plot(x,y);


# https://matplotlib.org/gallery/lines_bars_and_markers/simple_plot.html#sphx-glr-gallery-lines-bars-and-markers-simple-plot-py

# **bar chart**

# In[8]:


plt.bar(x, y);


# https://matplotlib.org/gallery/lines_bars_and_markers/barchart.html#sphx-glr-gallery-lines-bars-and-markers-barchart-py

# In[9]:


plt.ylabel('people')
plt.xlabel('language')
plt.bar(['Java', 'c++', 'c', 'python', 'R'], [10, 80, 70, 80, 50], color = 'orange');


# **plot on the data**

# In[ ]:


column_name = df.columns
print(column_name)
df[column_name[0]].plot(legend = True, xticks = np.arange(0, 151, 10));
df[column_name[1]].plot(legend = True, xticks = np.arange(0, 151, 10));
df.plot(xticks = np.arange(0, 151, 10));


# **separate by label**

# In[14]:


df.species.unique()


# Query the dataframe:
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.query.html

# In[16]:


setosa_df = df.query('species == "setosa"')
versicolor_df = df.query('species == "versicolor"')
virginica_df = df.query('species == "virginica"')
setosa_df.describe()
versicolor_df.describe()
virginica_df.describe()


# In[15]:


setosa_df = df[df.species == 'setosa']
versicolor_df = df[df.species == 'versicolor']
virginica_df = df[df.species == 'virginica']
setosa_df.describe()
versicolor_df.describe()
virginica_df.describe()


# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.plot.html

# In[17]:


setosa_df.plot(yticks=np.arange(0, 9, 1))
versicolor_df.plot(use_index = False, yticks=np.arange(0, 9, 1))
virginica_df.plot(use_index = False, yticks=np.arange(0, 9, 1))


# **sepal_length and sepal_width**

# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.plot.scatter.html

# In[22]:


# Plot a scatter chart using x='sepal_length', y='sepal_width', and separate colors for each of the three dataframes
ax = setosa_df.plot.scatter(x='sepal_length', y='sepal_width', label='setosa')
ax = versicolor_df.plot.scatter(x='sepal_length', y='sepal_width', label='versicolor', color='green', ax=ax)
ax = virginica_df.plot.scatter(x='sepal_length', y='sepal_width', label='virginica', color='red', ax=ax)


# In[23]:


# Plot a scatter chart using x='sepal_length', y='sepal_width', and separate colors for each of the three dataframes
ax = setosa_df.plot.scatter(x='petal_length', y='petal_width', label='setosa')
ax = versicolor_df.plot.scatter(x='petal_length', y='petal_width', label='versicolor', color='green', ax=ax)
ax = virginica_df.plot.scatter(x='petal_length', y='petal_width', label='virginica', color='red', ax=ax)


# **Get the column we want**

# In[18]:


df2 = df.copy()
df2.describe()
df.describe()


# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.drop.html

# In[19]:


df2.drop(['sepal_length', 'sepal_width'], inplace = True, axis = 1)
# axis = 1 is column, axis = 0 is row
df2.describe()


# **Data preparation**

# In[25]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html<br>
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html<br>
# More info can be found in:https://scikit-learn.org/stable/index.html

# **Seperate feature and label**

# x as feature, y as the label<br> Our final goal is using x to predict the label.

# In[20]:


y = df2.pop('species')
X = df2


# In[21]:


X.head()
y.head()


# **    split data in training set and test set**

# **test_size** will be how many percentage of data you want to set as test data, the rest will be training data.<br>
# Normally, we do want to have **at least half-half**, and the training data will usually more than the test data.<br>
# Random_state is how to randomly select the training data and test data.

# In[27]:



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=42)
X_train.head()
X_train.describe()
y_train.head()


# In[28]:


# define a model
clf = LogisticRegression(random_state=0, solver='lbfgs',
                         multi_class='multinomial')
# fit the data, training
clf.fit(X_train, y_train)


# In[29]:


# predict the data base on the 
y_predict = clf.predict(X_test)


# **How well is the prediction**

# In[30]:


from sklearn.metrics import accuracy_score


# In[31]:


accuracy_score([1, 1, 0], [1, 1, 1])


# In[32]:


accuracy_score(y_test, y_predict)


# In[34]:


# can also use the model score function
clf.score(X_test, y_test)


# **use all feature to fit**

# In[35]:


y = df.pop('species')


# In[37]:


X = df
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=42)

clf2 = LogisticRegression(random_state=0, solver='lbfgs',
                         multi_class='multinomial')
# fit the data, training
clf2.fit(X_train, y_train)
clf2.score(X_test, y_test)


# In[39]:


X = df
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

clf2 = LogisticRegression(random_state=0, solver='lbfgs',
                         multi_class='multinomial')
# fit the data, training
clf2.fit(X_train, y_train)
clf2.score(X_test, y_test)


# **store model**

# https://docs.python.org/3/library/pickle.html#module-pickle

# In[40]:


import pickle


# In[41]:


pickle.dump(clf2, open('logs_model.pkl', 'wb'))


# pickle can also store the csv, sql etc data file.

# **load the model**

# In[42]:


loaded_model = pickle.load(open('logs_model.pkl', 'rb'))
result = loaded_model.score(X_test, y_test)
result

