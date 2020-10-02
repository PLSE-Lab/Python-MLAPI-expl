#!/usr/bin/env python
# coding: utf-8

# # TPOT Automated ML Exploration with Mushroom Classification 
# ## By Jeff Hale
# 
# This is my experimentation with the TPOT automated machine learning algorithm with the Mushroom classification task. For more information see [this Medium article](https://medium.com/p/4c063b3e5de9/) I wrote discussing TPOT. 

# In[ ]:


# import the usual packages
import time
import gc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import timeit
import category_encoders

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from tpot import TPOTClassifier

pd.options.display.max_columns = 200
pd.options.display.width = 200

get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(font_scale=1.5, palette="colorblind")

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import os
print(os.listdir("../input"))


# Read in the data, encode it, and split it into training and test sets.

# In[ ]:


df = pd.read_csv('../input/agaricus-lepiota.csv')

X = df.reindex(columns=[x for x in df.columns.values if x != 'class'])        # separate out X
X = X.apply(LabelEncoder().fit_transform)  # encode the x columns string values as integers

y = df.reindex(columns=['class'])   # separate out y
print(y['class'].value_counts())
y = np.ravel(y)                     # flatten the y array
y = LabelEncoder().fit_transform(y) # encode y column strings as integer

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, test_size=0.25, random_state=10) 


# Check out the data 

# In[ ]:


print(X_train.describe())
print(X_train.info())


# Instantiate TPOTclassifier object - the generation and population_size determine how many populations are made.

# In[ ]:


tpot = TPOTClassifier(verbosity=3, 
                      scoring="accuracy", 
                      random_state=10, 
                      periodic_checkpoint_folder="tpot_mushroom_results", 
                      n_jobs=-1, 
                      generations=2, 
                      population_size=10)
times = []
scores = []
winning_pipes = []

# run several fits 
for x in range(10):
    start_time = timeit.default_timer()
    tpot.fit(X_train, y_train)
    elapsed = timeit.default_timer() - start_time
    times.append(elapsed)
    winning_pipes.append(tpot.fitted_pipeline_)
    scores.append(tpot.score(X_test, y_test))
    tpot.export('tpot_mushroom.py')

# output results
times = [time/60 for time in times]
print('Times:', times)
print('Scores:', scores)   
print('Winning pipelines:', winning_pipes)


# Make a data frame of the time to fit thirty pipelines ten times  are from a previous uncommitted run. All scores on the test set were 1.0.

# In[ ]:


# timeo = [1.6234928817333032, 1.162914126116084, 0.6119730584498029, 0.9018127734161681, 
#          2.0324099983001362, 0.45596561313335165, 0.4123572280164808, 1.9914514322998003, 
#          0.31134609155027043, 2.268216603050435]  # previous times
timeo = np.array(times)
df = pd.DataFrame(np.reshape(timeo, (len(timeo))))
df= df.rename(columns={0: "Times"})
df = df.reset_index()
df = df.rename(columns = {"index": "Runs"})
print(df)


# Make a seaborn barplot of the TPOT fit times for 10 pipelines.

# In[ ]:


ax = sns.barplot(x= np.arange(1, 11), y = "Times", data = df)
ax.set(xlabel='Run # for Set of 30 Pipelines', ylabel='Time in Minutes')
plt.title("TPOT Run Times for Mushroom Dataset")
plt.show()

