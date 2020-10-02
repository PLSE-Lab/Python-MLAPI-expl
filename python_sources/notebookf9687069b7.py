#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


data = pd.read_csv('../input/train.csv')
print(data.head())


# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
survived_fare = data.loc[data.Survived == 1,['Fare']]   .apply(np.log)   .replace([np.inf, -np.inf], np.nan)   .dropna(how="all")
dead_fare = data.loc[data.Survived == 0,['Fare']]   .apply(np.log)   .replace([np.inf, -np.inf], np.nan)   .dropna(how="all")

freq = np.linspace(0, 7, 56)
#survived_fare.head()
print(survived_fare.count())
plt.hist(survived_fare['Fare'], freq, alpha=0.5, label='Survived', cumulative=True);
plt.hist(dead_fare['Fare'], freq, alpha=0.5, label='Dead', cumulative=True);
plt.legend(loc='upper right')
plt.xlabel('log of fare')
plt.ylabel('ticket count')
plt.show()


# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
survived_men = data.loc[(data.Survived == 1) & (data.Sex == 'male'),['Age','Gender']]   .apply(np.log)   .replace([np.inf, -np.inf], np.nan)   .dropna(how="all")
dead_men = data.loc[(data.Survived == 0) & (data.Sex == 'male'),['Age','Gender']]   .apply(np.log)   .replace([np.inf, -np.inf], np.nan)   .dropna(how="all")
    
survived_women = data.loc[(data.Survived == 1) & (data.Sex == 'female'),['Age','Gender']]   .apply(np.log)   .replace([np.inf, -np.inf], np.nan)   .dropna(how="all")
dead_women = data.loc[(data.Survived == 0) & (data.Sex == 'female'),['Age','Gender']]   .apply(np.log)   .replace([np.inf, -np.inf], np.nan)   .dropna(how="all")

freq = np.linspace(0, 5, 25)
#survived_fare.head()
print(survived_men.count())
plt.hist(survived_men['Age'], freq, alpha=0.5, label='Survived Men');
plt.hist(dead_men['Age'], freq, alpha=0.5, label='Dead Men');
plt.hist(survived_women['Age'], freq, alpha=0.5, label='Survived Women');
plt.hist(dead_women['Age'], freq, alpha=0.5, label='Dead Women');
plt.legend(loc='upper right')
plt.xlabel('log of age')
plt.ylabel('age count')
plt.show()


# In[ ]:


def convert_for_clf(data):
    survival_log = data.loc[:,["Survived","Fare","Sex","Age"]]
    survival_log["Fare"] = survival_log["Fare"]     .apply(np.log)     .replace([np.inf, -np.inf], np.nan)
    survival_log["Sex"] = survival_log["Sex"]     .replace("male",1)     .replace("female",0)
    survival_log["Age"] = survival_log["Age"]     .apply(np.log)     .replace([np.inf, -np.inf], np.nan)

    survival_log = survival_log.dropna(how="any")

    X = survival_log.loc[:,["Fare","Sex","Age"]].values
    y = survival_log.loc[:,["Survived"]].values
    
    return (X,y)


# In[ ]:


X,y = convert_for_clf(data)
print(X[0:10])


# In[ ]:


from sklearn import svm
linearSvc_clf = svm.LinearSVC()
linearSvc_clf.fit(X,y)


# In[ ]:


test_data = pd.read_csv('../input/test.csv')
print(test_data.head())


# In[ ]:


test_X, test_y = convert_for_clf(test_data)
print(test_X[0:10])

