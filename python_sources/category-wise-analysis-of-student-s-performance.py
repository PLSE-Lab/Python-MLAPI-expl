#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


data = pd.read_csv('../input/StudentsPerformance.csv')
data.head()


# In[ ]:


data.info()


# # Visualization

# ### Multicollinearity among scores
# In statistics, multicollinearity (also collinearity) is a phenomenon in which one predictor variable in a multiple regression model can be linearly predicted from the others with a substantial degree of accuracy. In this situation the coefficient estimates of the multiple regression may change erratically in response to small changes in the model or the data. Multicollinearity does not reduce the predictive power or reliability of the model as a whole, at least within the sample data set; it only affects calculations regarding individual predictors. That is, a multivariate regression model with collinear predictors can indicate how well the entire bundle of predictors predicts the outcome variable, but it may not give valid results about any individual predictor, or about which predictors are redundant with respect to others.
# 
# In the case of perfect multicollinearity (in which one independent variable is an exact linear combination of the others) the design matrix **X** has less than full rank, and therefore the moment matrix cannot be inverted. Under these circumstances, for a general linear model ![](https://wikimedia.org/api/rest_v1/media/math/render/svg/b4b37cd053fae656404e567945563b5f80630073)
# ![](https://wikimedia.org/api/rest_v1/media/math/render/svg/e894f0fc3ac07d9cb43194002d48d893b92f04a7) doesn't exist.
# 
# Though we don't have to apply OLS here so a better idea would be to add all the scores and check its variation with other features.

# In[ ]:


import warnings
warnings.filterwarnings('ignore')
sns.pairplot(data,hue = 'gender');


# In[ ]:





# #### Count plots for all scores

# In[ ]:


plt.figure(figsize = (15,30))
plt.subplot(311)
data['math score'].value_counts().sort_index().plot(kind = 'bar');
plt.xlabel('Math Score',fontsize = 15);
plt.ylabel('Number of Students',fontsize = 15);
plt.title('Distribution of Math Score',fontsize = 20);
plt.subplot(312)
data['reading score'].value_counts().sort_index().plot(kind = 'bar');
plt.xlabel('Reading Score',fontsize = 15);
plt.ylabel('Number of Students',fontsize = 15);
plt.title('Distribution of Reading Score',fontsize = 20);
plt.subplot(313)
data['writing score'].value_counts().sort_index().plot(kind = 'bar');
plt.xlabel('Writing Score',fontsize = 15);
plt.ylabel('Number of Students',fontsize = 15);
plt.title('Distribution of Writing Score',fontsize = 20);


# #### All these graphs are slightly skewed towards left so let's put some passing marks in each subject for further analysis, Here on, we'll consider 40 to be the passing marks out of 100.
# #### So, let's see how many of them passed in Math.

# In[ ]:


data['math_result'] = np.where(data['math score']>40, 'Passed', 'Failed')
print(data['math_result'].value_counts())
plt.figure(figsize = (10,10))
data['math_result'].value_counts().plot(kind = 'pie');
plt.xlabel('Math Score',fontsize = 15);
plt.ylabel('');
plt.legend();


# #### Reading result.

# In[ ]:


data['reading_result'] = np.where(data['reading score']>40, 'Passed', 'Failed')
print(data['reading_result'].value_counts())
plt.figure(figsize = (10,10))
data['reading_result'].value_counts().plot(kind = 'pie');
plt.xlabel('Reading Score',fontsize = 15);
plt.ylabel('');
plt.legend();


# #### Writing Result

# In[ ]:


data['writing_result'] = np.where(data['writing score']>40, 'Passed', 'Failed')
print(data['writing_result'].value_counts())
plt.figure(figsize = (10,10))
data['writing_result'].value_counts().plot(kind = 'pie');
plt.xlabel('Writing Score',fontsize = 15);
plt.ylabel('');
plt.legend();


# # Now let's see how students perform based on their gender.
# #### It is clear from the graphs below that females have more passing count in all three subjects while males have more failing count in all subjects  but math.

# In[ ]:


plt.figure(figsize = (12,7))
sns.countplot(x = 'math_result',hue = 'gender',data = data);
plt.xlabel('Math Result',fontsize = 15);


# In[ ]:


plt.figure(figsize = (12,7))
sns.countplot(x = 'reading_result',hue = 'gender',data = data);
plt.xlabel('Reading Result',fontsize = 15);


# In[ ]:


plt.figure(figsize = (12,7))
sns.countplot(x = 'writing_result',hue = 'gender',data = data);
plt.xlabel('Writing Result',fontsize = 15);


# ### Performance of Students Based on Race/Ethnicity

# In[ ]:


plt.figure(figsize = (12,7))
sns.countplot(x = 'math_result',hue = 'race/ethnicity',data = data);
plt.xlabel('Math Result',fontsize = 15);


# In[ ]:


plt.figure(figsize = (12,7))
sns.countplot(x = 'reading_result',hue = 'race/ethnicity',data = data);
plt.xlabel('Reading Result',fontsize = 15);


# In[ ]:


plt.figure(figsize = (12,7))
sns.countplot(x = 'writing_result',hue = 'race/ethnicity',data = data);
plt.xlabel('Writing Result',fontsize = 15);


# ## Performance of Students based on Parental level of education

# In[ ]:


plt.figure(figsize = (12,7))
sns.countplot(x = 'math_result',hue = 'parental level of education',data = data);
plt.xlabel('Math Result',fontsize = 15);


# In[ ]:


plt.figure(figsize = (12,7))
sns.countplot(x = 'reading_result',hue = 'parental level of education',data = data);
plt.xlabel('Reading Result',fontsize = 15);


# In[ ]:


plt.figure(figsize = (12,7))
sns.countplot(x = 'writing_result',hue = 'parental level of education',data = data);
plt.xlabel('Writing Result',fontsize = 15);


# ## Performance of student depending on whether they took a test preparation course or not.

# In[ ]:


plt.figure(figsize = (12,7))
sns.countplot(x = 'math_result',hue = 'test preparation course',data = data);
plt.xlabel('Math Result',fontsize = 15);


# In[ ]:


plt.figure(figsize = (12,7))
sns.countplot(x = 'reading_result',hue = 'test preparation course',data = data);
plt.xlabel('Reading Result',fontsize = 15);


# In[ ]:


plt.figure(figsize = (12,7))
sns.countplot(x = 'writing_result',hue = 'test preparation course',data = data);
plt.xlabel('Writing Result',fontsize = 15);


# In[ ]:




