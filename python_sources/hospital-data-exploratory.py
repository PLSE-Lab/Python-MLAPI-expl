#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as scipy
from scipy import stats

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


hosp = pd.read_csv('../input/mimic3d.csv')
# getting general information
hosp.info()


# In[ ]:


hosp.describe()


# In[ ]:


hosp.dtypes


# In[ ]:


hosp.shape


# In[ ]:


hosp.head(20)


# In[ ]:


hosp.tail(20)


# In[ ]:


hosp['AdmitDiagnosis'].unique().shape


# In[ ]:


hosp['age'].unique().shape


# In[ ]:


scipy.stats.describe(hosp.age)


# In[ ]:


scipy.stats.kurtosis(hosp.age)


# Checking for missing values

# In[ ]:


hosp.isnull().sum()


# In[ ]:


plt.figure(figsize=(20,10))
sns.countplot(x="age", data=hosp, palette="bwr")
plt.title('Distibution of Age')
plt.xticks(rotation=90)
plt.show()


# In[ ]:


#corelation matrix
plt.figure(figsize=(20,20))
sns.heatmap(cbar=False,annot=True,data=hosp.corr()*100,cmap='coolwarm')
plt.title('% Corelation Matrix')
plt.show()


# The chi square test tests the null hypothesis that the categorical data has the given frequencies.

# In[ ]:


scipy.stats.chisquare(hosp.age)


# Pearson correlation coefficient and p-value for testing non-correlation.
# 
# The Pearson correlation coefficient [1] measures the linear relationship between two datasets. The calculation of the p-value relies on the assumption that each dataset is normally distributed. (See Kowalski [3] for a discussion of the effects of non-normality of the input on the distribution of the correlation coefficient.) Like other correlation coefficients, this one varies between -1 and +1 with 0 implying no correlation. Correlations of -1 or +1 imply an exact linear relationship. Positive correlations imply that as x increases, so does y. Negative correlations imply that as x increases, y decreases.
# 
# The p-value roughly indicates the probability of an uncorrelated system producing datasets that have a Pearson correlation at least as extreme as the one computed from these datasets.

# In[ ]:


scipy.stats.pearsonr(hosp.age,hosp.NumRx)


# Compute the sample skewness of a data set.
# 
# For normally distributed data, the skewness should be about 0. For unimodal continuous distributions, a skewness value > 0 means that there is more weight in the right tail of the distribution. The function skewtest can be used to determine if the skewness value is close enough to 0, statistically speaking.

# In[ ]:


scipy.stats.skew(hosp.age, axis=0, bias=True, nan_policy='propagate')


# In[ ]:


from scipy import stats
stats.describe(hosp.age)


# In[ ]:


np.histogram(hosp.age, bins=40)
plt.show()


# In[ ]:


import matplotlib.pyplot as plt
plt.figure(figsize=(20,20))

n, bins, patches = plt.hist(x=hosp.age, bins='auto', color='#0504aa',
                            alpha=0.7, rwidth=0.85)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Age Distribution')
plt.text(23, 45, r'$\mu=15, b=3$')
maxfreq = n.max()

#plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)

