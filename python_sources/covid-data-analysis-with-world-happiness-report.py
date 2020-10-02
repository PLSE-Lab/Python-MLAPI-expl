#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt 

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # IMPORTING COVID DATASET 

# In[ ]:


covid=pd.read_csv('/kaggle/input/covid-data100-days/covid19_Confirmed_dataset.csv')
covid


# ## Deleting the useless columns

# In[ ]:


covid_1=covid.drop(['Lat','Long'],axis=1)


# In[ ]:


covid_1


# ## AGGREGATING DATA BY COUNTRIES

# In[ ]:


covid_2=covid_1.groupby("Country/Region").sum()


# In[ ]:


covid_2


# In[ ]:


countries=covid_2.index
countries


# ## Visualizing data related to a country

# In[ ]:


covid_2.loc['US'].plot()
covid_2.loc['Spain'].plot()
plt.legend(loc=2)


# ## Calculating a good measure:
# After a lot of research, I came to a conclusion that plotting a curve for derivative of the first curve with time and determining the maximum point will give me the highest rate of increase in covid cases for a particular country

# ### caculating the first derivative of the curve

# In[ ]:


covid_2.loc['China'].diff().plot()
covid_2.loc['US'].diff().plot()
plt.legend()


# ### find maxmimum infection rate for a Country

# In[ ]:


covid_2.loc['China'].diff().max()


# In[ ]:


covid_2.loc['US'].diff().max()


# ### find maximum infection rate for all of the countries

# In[ ]:


max_infection_rate=[]
for country in countries:
    max_infection_rate.append(covid_2.loc[country].diff().max())


# In[ ]:


covid_2['Max Infection Rate']=max_infection_rate
covid_2.head()


# ### create a new dataframe with only needed column

# In[ ]:


Covid=pd.DataFrame(covid_2['Max Infection Rate'])


# In[ ]:


Covid.head(20)


# ### Importing world happiness report dataset

# In[ ]:


happiness_report=pd.read_csv('/kaggle/input/world-happiness/worldwide_happiness_report.csv')


# In[ ]:


happiness_report.head()


# ###  Dropping the useless columns

# In[ ]:


useless=['Regional indicator','Standard error of ladder score','upperwhisker','lowerwhisker','Generosity','Perceptions of corruption']


# In[ ]:


happiness_report=happiness_report.drop(useless,axis=1)


# In[ ]:


happiness_report.head()


# ### Changing the indices of the dataframe

# In[ ]:


happiness_report=happiness_report.set_index('Country name')
happiness_report


# ### Joining Covid and happiness_report

# In[ ]:


data= Covid.join(happiness_report,how='inner')
data.head()


# ## Visualization of the results

# ### Plotting GDP vs Maximum Infection Rate

# In[ ]:


y1=data['Max Infection Rate']
x1=data['Logged GDP per capita']


# In[ ]:


sns.regplot(x1,np.log(y1))
plt.title("Plot of Max Infection Rate Vs. GDP")


# NOTE : A POSITIVE CORRELATION HAS BEEN ESTABLISHED BETWEEN GDP per Capita AND MAX INFECTION RATE. 
# CONCLUSION: COVID-19 SPREAD IS ADVERSE IN MORE DEVELOPED COUNTRIES

# ### Plotting Healthy life expectancy vs maximum Infection rate

# In[ ]:


x2=data['Healthy life expectancy']
y2=data['Max Infection Rate']


# In[ ]:


sns.regplot(np.log(y2),x2)


# NOTE : A POSITIVE CORRELATION IS ESTABLISHED BETWEEN HEALTHY LIFE EXPECTANCY AND MAX INFECTION RATE.
# CONCLUSION: COVID-19 SPREAD IS MORE IN COUNTRIES WITH HIGHER HEALTHY LIFE EXPECTANCY.

# ### Plotting happiness index vs max infection rate

# In[ ]:


x5=data['Ladder score']
y5=data['Max Infection Rate']


# In[ ]:


sns.regplot(np.log(y5),x5)


# NOTE: A POSITIVE CORRELATION IS ESTABLISHED BETWEEN HAPPINESS INDEX AND MAX INFECTION RATE.
# CONCLUSION: THE COVID VIRUS SPREAD IS HIGHER IN COUNTRIES THAT ARE MORE ECONOMICALLY AND PSYCOLOGICALLY WELL PLACED THAN THE OTHERS.

# In[ ]:




