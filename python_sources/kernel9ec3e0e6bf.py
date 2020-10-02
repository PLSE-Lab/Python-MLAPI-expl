#!/usr/bin/env python
# coding: utf-8

# <h1> Analyzing Effect of various pollution indicators on Mortality Rate </h1>
# <p> In this module we analyse and visualise the effect of various pollution indicators on Mortality rate (number of deaths per hundred thousand (or <i> lakh </i>) people). <br>
# Data is obtained from <a href = "https://www.kaggle.com/c/predict-impact-of-air-quality-on-death-rates/data"> Kaggle </a>. Data contains the following fields:
# <ol>
# <li>Ozone concentration</li>
# <li>PM10 level</li>
# <li>PM2.5 level</li>
# <li>NO<sub>2</sub> level</li>
# <li>Temperature Mean</li>
# <li>Mortality Rate</li>
# </ol>
# 
# </p>

# <h2>Importing Dependencies</h2>
# <br>
# We import the following python libraries:
#     <ol>
#     <li> Pandas: For reading and manipulating dataframes</li>
#     <li> Numpy: For numeric computation</li>
#     <li> OS: Changing, getting path</li>
#     <li> Matplotlib: Plotting and visualising</li>
#     <li> Itertool: Creating combinations</li>
#     <li> Seaborn: Plotting and visualising</li>
#     <li> scipy: Python library for statistics</li>
#     </ol>

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import os
from matplotlib import pyplot as plt
import itertools
from scipy import stats
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


## change directory to the directory in which data is present
os.chdir(os.getcwd() + '\\Data')


# <h2> Reading Data </h2>
# <p>We read data present in comma separated value (CSV) format using Pandas library. Dataframe consists of 18,403 data points and 9 variables (or <i>features</i>). Data is stored in the pandas dataframe format
# </p>
# 

# In[ ]:


## reading file
train = pd.read_csv('train.csv')
train.head()
print("Number of data points =", train.shape[0], "\nNumber of Variables =", train.shape[1])


# <h2> Analysing Distribution of Variables </h2>
# <p> We analyze the distribution of the variables present in the dataset by studying their distibutions, the describe method in pandas library goves the description (count, mean, standard deviation, mean, minimum, maximum, quantiles, and median of the dataset). <br> <br>
# After obtaining the paratmeters, in order to visualise the distribution, we plot the histogram for different variables in the dataset
# </p>

# In[ ]:


## drop id as it is not a feature
train.drop(['Id'], axis = 1, inplace = True)

## describe method gives the description of each of the variables in the dataframe
train.describe()


# <h2> Visualisation of Distribution </h2>
# <p> We use histogram to visualise the distribution of various variables in the dataset.</p>

# In[ ]:


## variables to be analyzed
variables = ['O3', 'PM10', 'PM25', 'NO2', 'T2M','mortality_rate']

## colors for different histograms
colors = itertools.cycle(["r", "b", "g", "y", "m", "k"])

## creating plot
f = plt.figure()

f, axes = plt.subplots(nrows = 6, figsize = (30, 60))

## creating subplots
for i in range(len(variables)):
    dist = train[variables[i]].fillna(train[variables[i]].mean())
    sc = axes[i].hist(dist, bins = 20, color = next(colors))
    axes[i].set_xlabel(variables[i])


# <h2> Correlation among features </h2>
# <p>
# To analyze correlation among the features we find the Pearson's correlation coeffecient for all pairs of variables.
# </p>
# <img src = "data:image/png;base64,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", alt = formula>
# <p>
# Pearson's coeffecient lies between -1 and 1, indicating positive and negative correlation, while 0 indicates no correlation.
# </p>

# In[ ]:


train.corr()


# In[ ]:


## heatmap of correlation
sns.heatmap(train.corr())


# <h2> Visualisation of correlation among features </h2>
# <p> We analyze the correlation among different features of the dataset using scatter plots. The mortality rate is plotted on Y-axis while other features are plotted on X-axis.
# </p>

# In[ ]:


## features in the dataset
features = ['O3', 'PM10', 'PM25', 'NO2', 'T2M']

## colors for plotting
colors = itertools.cycle(["r", "b", "g", "y", "m"])

## creating plot
f = plt.figure()    

## creating subplots
f, axes = plt.subplots(nrows = 5, figsize = (30, 60))

## Mortality Rate versus O3
sc = axes[0].scatter(train[features[0]], train.mortality_rate, marker = '.', color = next(colors))
axes[0].set_xlabel(features[0], labelpad = 5)

## Mortality Rate versus PM10
sc = axes[1].scatter(train[features[1]], train.mortality_rate, marker = '.', color = next(colors))
axes[1].set_xlabel(features[1], labelpad = 5)

## Mortality Rate versus PM2.5
sc = axes[2].scatter(train[features[2]], train.mortality_rate, marker = '.', color = next(colors))
axes[2].set_xlabel(features[2], labelpad = 5)

## Mortality Rate versus NO2
sc = axes[3].scatter(train[features[3]], train.mortality_rate, marker = '.', color = next(colors))
axes[3].set_xlabel(features[3], labelpad = 5)

## Mortality Rate versus mean temperature
sc = axes[4].scatter(train[features[4]], train.mortality_rate, marker = '.', color = next(colors))
axes[4].set_xlabel(features[4], labelpad = 5)


# <h2> Pollution level in Mumbai </h2>
# <p>We canalyze the pollution levels in Maharashtra. <a href = "https://timesofindia.indiatimes.com/city/mumbai/mumbai-4th-most-polluted-megacity-in-world-9-in-10-people-breathe-bad-air/articleshow/63993044.cms">Mumbai is known to be one of the most polluted cities in the world. </a>. While London has low to moderate pollution levels, Mumbai suffers from very severe pollution</p>

# In[ ]:


maharashtra = pd.read_csv('maharashtra.csv')
maharashtra.head()


# In[ ]:


so2 = (maharashtra.groupby('City/Town/Village/Area')['SO2'].mean())
no2 = (maharashtra.groupby('City/Town/Village/Area')['NO2'].mean())
pm10 = (maharashtra.groupby('City/Town/Village/Area')['RSPM/PM10'].mean())
pm25 = (maharashtra.groupby('City/Town/Village/Area')['PM 2.5'].mean())


# <h2> Comparing the distribution of pollution for Maharashtra and England 
# </h2>
# <p> We use ZTest to find if the values belong to the same distribution given by</p>
# 
# <img src = "http://homework.uoregon.edu/pub/class/es202/zz.jpg">
# 

# In[ ]:


'''
Input : 2 distributions (2 pandas series to be compared)
Output : Z-Value (single float)
'''
def ZTest(distribution1 ,distribution2):
    return (np.mean(distribution1) - np.mean(distribution2)) / np.sqrt(np.std(distribution1) ** 2  /  len(distribution1) + np.std(distribution2) ** 2 / len(distribution2))


# In[ ]:


print(ZTest(maharashtra['NO2'], train['NO2']))


# The difference is <b> highly significant </b>

# In[ ]:


plt.hist(maharashtra['NO2'].fillna(maharashtra['NO2'].mean()), bins = 20, label = 'Maharashtra in 2015')
plt.hist(train['NO2'].fillna(train['NO2'].mean()), bins = 20, label = 'England in 2007-2010')
plt.title('NO2 levels in Maharashtra and England')
plt.legend()


# Also we can see from the plot that difference is highly significant, between the 2 distributions.

# In[ ]:


print(ZTest(maharashtra['RSPM/PM10'], train['PM10']))


# Again, the difference is highly significant

# In[ ]:


plt.hist(maharashtra['RSPM/PM10'].fillna(maharashtra['RSPM/PM10'].mean()), bins = 20, label = 'Maharashtra in 2015')
plt.hist(train['PM10'].fillna(train['PM10'].mean()), bins = 20, label = 'England in 2007-2010')
plt.title('PM10 levels in Maharashtra and England')
plt.legend()


# The plot shows high difference

# <h2> Most polluted cities of Maharashtra at higher risk </h2>
# <p>We rank the cities according to different pollution parameters (averaged over the entire year) </p>

# In[ ]:


## SO2 level
print('List of cities by SO2 level:\n', so2.sort_values()[::-1])


# In[ ]:


print('List of cities by NO2 level:\n', no2.sort_values()[::-1])


# In[ ]:


print('List of cities by PM10 level:\n', pm10.sort_values()[::-1])


# <h2> Conclusion </h2>
# <p> Maharashtra has significantly high pollution levels as compared to England. In case of England somewhat less correlation is found between mortality rate and pollution parameters. However, same cannot be concluded for Maharashtra as pollution rate is significantly higher in Maharashtra. </p>
# <p> Similar data can be collected in the state of Maharashtra to analyse the effect of high pollution on mortality rate and take steps to decrease pollution. This will lead to lower mortality rate and well being of the people.
# </p>
