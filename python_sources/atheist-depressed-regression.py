#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt


# In[ ]:


import seaborn as sns
sns.set(color_codes=True)


# In[ ]:


population_df = pd.read_csv('../input/API_SP.POP.TOTL_DS2_en_csv_v2.csv', skiprows = range(0,4))
population_df.head()
USA_df = population_df.iloc[249]
print(USA_df)


# Looking at Data from PewResearch.org, we see that those not affiliated to religion in the time bracket 2007-2014 we take data from those who are religiously unaffiliated to be a proxy for those who do not believe in a higher deity, however this is a little crude. We hope it gives us a good rough understanding. 
# 
# Reference :-
# (1) http://www.pewresearch.org/fact-tank/2016/06/01/10-facts-about-atheists/ [Data for years 2014]
# (2) http://www.pewforum.org/2012/10/09/nones-on-the-rise/ [Data for years 2007-2012]
# 
# 2007 - 15.3% of the population of the country.
# 2008 - 16.0% of the population of the country.
# 2009 - 16.8% of the population of the country.
# 2010 - 17.4% of the population of the country.
# 2011 - 18.6% of the population of the country.
# 2012 - 19.6% of the population of the country. 
# 2013 - extrapolated to be the average of 2012 and 2014 - 21.2% of the population of the country. 
# 2014 - 22.8% of the population of the country.
# 
# 

# In[ ]:


Atheist_US_= [0.153,0.160,0.168,0.174,0.186,0.196,0.212,0.228];


# In[ ]:


USA2007_2014_df = USA_df.iloc[51:59]
USA2007_2014_df


# In[ ]:


USA2007_2014_Population_Atheist = USA2007_2014_df[0:]*Atheist_US_
USA2007_2014_Population_Atheist


# In[ ]:


Atheist = pd.DataFrame()
Atheist_US_= [4.60884e+07,4.8655e+07,5.15376e+07,5.38266e+07,5.79694e+07,6.15437e+07,6.70354e+07,7.26325e+07];
Atheist = Atheist.append(Atheist_US_)
Atheist

#Data :-
#2007    4.60884e+07
#2008     4.8655e+07
#2009    5.15376e+07
#2010    5.38266e+07
#2011    5.79694e+07
#2012    6.15437e+07
#2013    6.70354e+07
#2014    7.26325e+07


# Data from Centers for Disease Control and Prevention in the United States :- 
# Data from 2001-2002 :- 16.1% of non-institutionalized adults had a major depressive disorder at some point in their lifetime. https://www.cdc.gov/nchs/data/hus/hus07.pdf
# 
# Data from 2004 :- 25% of adults report having mental illness in the previous year. 
# https://www.cdc.gov/mmwr/preview/mmwrhtml/su6003a1.htm
# 
# Pew Research Center Data :- 
# 2002-2006 :- online searches for issues on mental health, 22%.
# 2008 :- rose to 28%, a statistically significant increase according to Pew research.
# http://www.pewinternet.org/2009/06/11/depression-anxiety-stress-or-mental-health-issues/
# 
# NIMH Statistic :- 
# Looking at the National Institute of Mental Health Statistic page in the US we see :- https://www.nimh.nih.gov/health/statistics/index.shtml
# 
# Broken into three sections :- 
# (1) Mental Illness - includes depression
# (2) Any anxiety disorder - includes Obsessive-Compulsive Disorder
# (3) Attention Deficit/Hyperactivity Disorder - includes suicide and eating disorders.
# 
# Looking at the data we see :-
# Mental Illness - (18.3% of All US Adults) in 2016.
# Major Depression - (6.7% of All US Adults) in 2016. 
# Post Traumatic Stress Disorder - (5% of All US Adults) study conducted in the time bracket [2001- 2004].
# Anxiety Disorder - (19.1% of All US Adults) study conducted in the time bracket [2001-2004].
# Bipolar Disorder - 2.8% of US Adults in the period [2001-2003]
# Obsessive Compulsive Disorder - 1.2% of US Adults had OCD in the past year [2001-2003].
# ADHD (Attention Deficit Hyperactivity Disorder) - 7.8% of Children in 2003 and 11.0% of Children in 2011.
# Eating Disorders - 2.8% of US Adults between 2001- 2003.
# Personality Disorders - 9.1% of US Adults had any personality disorder between 2001-2003.
# Suicide Rates per 100,000
# 2007 - 11.3 per 100,000
# 2008 - 11.6 per 100,000
# 2009 - 11.8 per 100,000
# 2010 - 12.1 per 100,000
# 2011 - 12.3 per 100,000
# 2012 - 12.6 per 100,000
# 2013 - 12.6 per 100,000
# 2014 - 13.0 per 100,000
# 
# https://www.nimh.nih.gov/health/statistics/suicide.shtml
# 
# Although it is clear there is a increasing trend towards increasing suicides as well as mental illnesses using the multitude of studies available, we will take the suicide rates as a proxy for the increasing levels of mental illness as this is the most clear data available.

# In[ ]:


Suicide = pd.DataFrame()
Suicide_US_= [11.3,11.6,11.8,12.1,12.3,12.6,12.6,13.0];
Suicide = Suicide.append(Suicide_US_)
Suicide


# In[ ]:


import seaborn as sns  
grid = sns.JointGrid(Atheist,Suicide, space=0, size=6, ratio=50)
grid.plot_joint(plt.scatter,color="g")


# In[ ]:


from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from pandas import Series, DataFrame
from sklearn.neighbors import KNeighborsRegressor

X = Atheist
y = Suicide

X_train_val, X_test, Y_train_val, Y_test = train_test_split(X, y, test_size=0.15, random_state=0)
X_train, X_val, Y_train, Y_val = train_test_split(X_train_val, Y_train_val, test_size=0.15/0.85, random_state=0)

model = KNeighborsRegressor(n_neighbors=3)
model.fit(X_train, Y_train)
print(model.score(X_train, Y_train))
plt.scatter(X_train, Y_train, marker='.')
plt.xlabel('Atheism')
plt.ylabel('SuicideRates')


# In[ ]:


y_pred = model.predict(X_val)
y_actual = Y_val
mean_squared_error(y_actual, y_pred)


# In[ ]:


y_pred_test = model.predict(X_test)
y_actual_test = Y_test
mean_squared_error(y_actual_test, y_pred_test)


# In[ ]:


# Test R^2
print(model.score(X_test, y_actual_test))
plt.scatter(y_pred_test, y_actual_test, marker='.')
plt.xlabel('Predicted y')
plt.ylabel('Actual y')
plt.show()

