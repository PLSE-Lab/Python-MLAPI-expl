#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import statsmodels.api as sm

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# This is a simple test-run of creating a model to determine what factors men and women in the experiment can be used to predict if a "Yes" decision was made.  In this first pass, we will just use an OLS model.

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
f_results.summary()


# In[ ]:


# male model
m = df.loc[df.gender == 1, :]
m_data = m.copy()
m_data = m_data.dropna(subset=input_vars)
m_model = sm.OLS(m_data.dec, sm.add_constant(m_data.loc[:, input_vars]))
m_results = m_model.fit()
m_results.summary()

