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


# read in our data
fiveDaySurveyData = pd.read_csv("../input/5day-data-challenge-signup-survey-responses/anonymous-survey-responses.csv")
kaggleSurveyData = pd.read_csv('../input/kaggle-survey-2017/multipleChoiceResponses.csv', encoding="ISO-8859-1")


# ## What is the Chi-Squared Test?
# The chi-squared goodness-of-fit test is used to test whether the distribution of sample categorical data matches an expected distribution. For example you could use it to check if the race demographics in your community matches the entire US population or whether the political affiliation of your Facebook friends matches those of the US as a whole. 
# 

# In[ ]:


import scipy.stats as sta # statistics
cs_data = kaggleSurveyData['CodeWriter'].value_counts()
print(sta.chisquare(cs_data))
prog_data = fiveDaySurveyData["Do you have any previous experience with programming?"].value_counts()
print(sta.chisquare(prog_data))

contingencyTable = pd.crosstab(fiveDaySurveyData["Do you have any previous experience with programming?"],
                               kaggleSurveyData['CodeWriter'])
           
sta.chi2_contingency(contingencyTable)


# In[ ]:




