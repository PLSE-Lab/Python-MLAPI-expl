#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import statsmodels.api as sm
from sklearn.feature_selection import RFE
from sklearn.svm import SVC

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv("../input/Speed Dating Data.csv", encoding="ISO-8859-1")
input_vars = ['attr', 'sinc', 'intel', 'fun', 'amb', 'shar', 'like', 'prob']


# In[ ]:



# female model
f = df.loc[df.gender == 0, :]
f_data = f.copy()
f_data = f.dropna(subset=input_vars)
f_model = sm.OLS(f_data.dec, sm.add_constant(f_data.loc[:, input_vars]))
f_results = f_model.fit()
f_results.params
f_results.pvalues  
# all p-values are less than 0.05 which means all of features are significant in the model.


# In[ ]:


# Now let's rank the features based on their importance in the model using RFE method, for women.
estimator = SVC(kernel= "linear", C=0.1)
selector = RFE(estimator,1)
cfl=selector.fit(f_data.loc[:, input_vars], f_data.dec)
ranking_f=cfl.ranking_
feature_ranking_f=zip(ranking_f, input_vars)
list(feature_ranking_f)


# In[ ]:


# male model
m = df.loc[df.gender == 1, :]
m_data = m.copy()
m_data = m_data.dropna(subset=input_vars)
m_model = sm.OLS(m_data.dec, sm.add_constant(m_data.loc[:, input_vars]))
m_results = m_model.fit()
m_results.params
m_results.pvalues


# In[ ]:


# feature ranking for male participants.
estimator = SVC(kernel= "linear", C=0.1)
selector = RFE(estimator,1)
cfl=selector.fit(m_data.loc[:, input_vars], m_data.dec)
ranking_m=cfl.ranking_
feature_ranking_m=zip(ranking_m, input_vars)
list(feature_ranking_m)
# you can see that being funny is more important for women than man in selecting a partner!  
# Being sincere on the other hand is more important to men than women. 

