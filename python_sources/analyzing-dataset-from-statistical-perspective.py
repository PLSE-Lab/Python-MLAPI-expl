#!/usr/bin/env python
# coding: utf-8

# # Statistical analysis
# Advertising dataset
# Code author: Chinmay Upadhye

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import seaborn as sns


from sklearn.preprocessing import scale
import sklearn.linear_model as skl_lm
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
import statsmodels.formula.api as smf

get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('seaborn-white')


# In[ ]:


adv = pd.read_csv('../input/Advertising.csv', usecols=[1,2,3,4])
adv.info()


# In[ ]:


adv.describe()


# Statistical overview of advertisement dataset tell us about how each attribute behaves. This is essentially a **snapshot of the behaviour of data**. We have keep an eye out for outliers and random points (noise) which don't belong in **Inter Quartile Range** of variables.

# In[ ]:


adv.head()


# ## Finding values of fields
# The describe function gives a good overview of all the given attributes. Now I'll show all the mentioned fields below. Generally I look for anamolies and try to normalize them before I process any further but that depends on requirements of the model.
# First I'll look at the data directly first and then check how much element each relevant class represents.

# ### Linear regression fit on Sales vs TV data

# In[ ]:


sns.regplot(adv.TV, adv.Sales, order=1, ci=None, scatter_kws={'color':'r', 's':9})
plt.xlim(0,310)
plt.ylim(ymin=0);


# In[ ]:


estTV = smf.ols('Sales ~ TV', adv).fit()
estTV.summary().tables[1]


# Additional statistical test. We can also see the results of other tests with negative skewed and sharpness of the curve by kurtosis. For Data scientists, Gaussian distribution **(Normal Distribution) is bastion** of unknown phenonmenons. It is often used in natural sciences to represent random variables whose distibution isn't known.

# In[ ]:


estTV = smf.ols('Sales ~ TV', adv).fit()
estTV.summary().tables[2]


# For Radio & Newspaper

# In[ ]:


estRD = smf.ols('Sales ~ Radio', adv).fit()
estRD.summary().tables[1]


# In[ ]:


estRD = smf.ols('Sales ~ Radio', adv).fit()
estRD.summary().tables[2]


# In[ ]:


estNP = smf.ols('Sales ~ Newspaper', adv).fit()
estNP.summary().tables[1]


# In[ ]:


estNP = smf.ols('Sales ~ Newspaper', adv).fit()
estNP.summary().tables[2]


# In[ ]:


estAll = smf.ols('Sales ~ TV + Radio + Newspaper', adv).fit()
estAll.summary()


# In[ ]:


adv.corr()


# **Even though correlation provides us with how independent attributes are and how to scale the model as a whole, causality is one thing that we as a designer should consciously take into account. The data has insights about localized patterns but we are the link who can understand those localized patterns for taking decision in real world.** --- Chinmay Upadhye

# 

# 

# In[ ]:





# In[ ]:





# # Analysing Auto dataset

# In[ ]:


auto = pd.read_csv('../input/Auto.csv', na_values='?').dropna();
auto.head()


# In[ ]:


auto.info()


# In[ ]:


auto.describe()


# In[ ]:


sns.regplot(auto.acceleration, auto.mpg, order=1, ci=None, scatter_kws={'color':'r', 's':9})
plt.show()


# In[ ]:


sns.regplot(auto.displacement, auto.mpg, order=1, ci=None, scatter_kws={'color':'r', 's':9})
plt.show()


# In[ ]:


sns.regplot(auto.horsepower, auto.mpg, order=1, ci=None, scatter_kws={'color':'r', 's':9})
plt.show()


# In[ ]:


s = sns.PairGrid(auto)
s.map(plt.scatter)


# In[ ]:


sns.regplot(auto.horsepower, auto.mpg, order=1, ci=None)
plt.xlim(xmin=0)
plt.ylim(ymin=0)


# 1) We can see here that for there is relationship between predictor and response. It is fairly linear.
# 2) The relationship is strong as change in horsepower proportionally impacts miles per gallon
# 3) The mpg decreases as we increase horsepower so we can say that their relationship is negative. There is one important phenomenon here that for horsepower below 75 mpg consumption is high similarly after 200 mpg starts to increase. In between these rangess they have almost linear changes. It is almost parabolic response. We can derive that from 75 to 175 the fuel consumption is moderate so it is most recommended range of operation.
# 4) For 98 horsepower, the miles per gallon predicted is approximately 26-28, but the actual values are between 25-27.
# 5) The plot already contains linear regression line.

# 

# In[ ]:


auto = pd.read_csv('../input/Auto.csv', na_values='?').dropna();
auto.head()


# In[ ]:


auto.info()


# In[ ]:


auto.describe()


# In[ ]:


auto.corr()


# In[ ]:


f, ax = plt.subplots(figsize=(10, 8))
corr = auto.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax)


# Q4 D 1) Yes there is relationship between predictors and response but some of the variable might not be causal though they are correlated. Acceleration, year and origin are positively correlated while all others are negatively correlated.
#      2) We can see from heatmap variables like cylinders, displacement, horsepower & weight have greater impact. Hence they have more statistically significant relationship. Also we can see there's fair impact of acceleration, year and origin but not as above variables.
#      3) Even though year suggests one of highly correlated coefficient 0.580541, the causality is less

# In[ ]:





# In[ ]:





# In[ ]:




