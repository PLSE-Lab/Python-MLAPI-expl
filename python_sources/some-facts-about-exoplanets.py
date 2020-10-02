#!/usr/bin/env python
# coding: utf-8

# ## Learning some facts about exo planets
# 
# In this notebook I tried to learn some useful information using the raw data provided. I explored a few things such as
# 
# * Planets similar to earth
# * Planets that are too far
# * Correlation of the planetary features and 
# 
# Finally I tried to compare the various Discovery methods.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


inp = pd.read_csv('../input/oec.csv')
inp['extra'] = 1


# In[ ]:


# Defining some constants
AvgHumanLifeExpectancy = 71 # https://en.wikipedia.org/wiki/List_of_countries_by_life_expectancy
ParsecInLightYears = 3.26 # Source : Google
EarthAgeGyr = 4.54 # https://en.wikipedia.org/wiki/Age_of_the_Earth
EarthSurfaceTemp = 287


# In[ ]:


# Number of discoveries per year
plt.xticks(rotation=90)
sns.countplot(inp.DiscoveryYear)


# * We can see that the number has gone up rapidly in the last couple of years, although it dropped in 2015. 
# Let's see the data corresponding to 2015

# In[ ]:


d2015 = inp[inp.DiscoveryYear == 2015]
d2015.describe()


# ### Planets younger than Earth
# 
# The age of earth is 4.54 Gyr (Source: Wikipedia)

# In[ ]:


inp[inp.AgeGyr < 4.54]
inp[inp.AgeGyr < 4.54][['AgeGyr', 'HostStarAgeGyr']]


# ### Planets like earth
# 
# For this section I will try to find planets whose surface temp is similar to earth

# In[ ]:


f = ['PlanetIdentifier', 'DistFromSunParsec', 'PlanetaryMassJpt',    'RadiusJpt', 'PeriodDays', 'SemiMajorAxisAU', 'HostStarTempK', 'SurfaceTempK']
d = inp[pd.DataFrame.abs(inp.SurfaceTempK - EarthSurfaceTemp) < 15]
d[f].describe()


# These planets have a similar surface temperature to earth and looks like their mean HostStarTempK, Period Days are closer to ours. Now all we need is an atmosphere and a way to reach them (since they are at a mean distance of nearly 350 light years)

# ### Planets too far
# 
# This is a list of planets that are so far that any image of the planet that we see during our lifetime would be the state of the planet prior to the earth. For example, a person born in the year 2000 would never be able to see the state of the planet after the year 2000, in his lifetime. 
# 
# The reason for this is, the fact that it takes light, the fastest moving entity in the universe, more than 71 years to reach us from that planet. So, the latest information we obtain of the planet would be 71 years old or more by the time it reaches us.
# 
# Of course, this is assuming the person is on earth throughout his lifetime and that there are no sufficient advancements in Science.

# In[ ]:


inp[inp.DistFromSunParsec * ParsecInLightYears > AvgHumanLifeExpectancy]


# ### Analyzing features
# 
# Let's see the correlation between some numeric features of exo planets

# In[ ]:


sns.heatmap(inp.select_dtypes([np.float64]).corr())


# ### Which discovery method is better ?
# 
# For this, I went through the Wikipedia article describing the discovery methods. I found that some methods we better at determining some properties (like mass, radius, eccentricity etc). For this section, I will assume the following:
# 
# "In the data, if the fraction of non-null entries for a property is more for one method than the others, then this method is better at determining this property" 
# 
# Let's go ahead and test this hypothesis on eccentricity. Wikipedia says that RV (Radial Velocity) method is better at identifying eccentricity.

# In[ ]:


sns.countplot(inp[inp.Eccentricity.notnull()].DiscoveryMethod)


# Looks like the hypothesis worked out well for eccentricity

# In[ ]:


inp.columns


# In[ ]:


# All features were are interested in for comparing Discovery methods
features = ['PlanetaryMassJpt', 'RadiusJpt', 'PeriodDays', 'SemiMajorAxisAU',           'Eccentricity','LongitudeDeg','AscendingNodeDeg', 'InclinationDeg',           'RightAscension', 'Declination', 'DistFromSunParsec', 'AgeGyr']

# Create a colname for check if the feature is null or nor
featuresbool = ['is'+feature+'NotNull' for feature in features]
for featurebool, feature in zip(featuresbool, features):
    inp[featurebool] = inp[feature].notnull()


# In[ ]:


d = inp.groupby('DiscoveryMethod').agg(np.mean)[featuresbool]


# In[ ]:


d.T.plot(kind='bar', stacked=True)


# ### Observations
# 
# * Transit is bad at identifying planetary mass but good at identifying radius.
# * RV is good at identifying SemiMajorAxis, Planetary mass, period days, eccentricity but bad at identifying radius.
# * All of them seem to be good at determining RightAscension, Declination and DistanceFromSun. Either that or these attributes are computed using the previously discussed properties.
# * Timing method seems to be closer to RV in terms of identifying properties. In fact it is sometimes better than RV.
# * Despite other methods being better, transit is used the most according to the below chart. I wonder why ?

# In[ ]:


a = inp.groupby(['DiscoveryYear', 'DiscoveryMethod'])[['DiscoveryMethod']].agg(['count'])


# In[ ]:


a.unstack().plot(kind='bar', stacked='True')


# Now lets try to see which methods are good at what distances from Sun

# In[ ]:


d = inp.groupby('DiscoveryMethod')['DistFromSunParsec'].agg(['mean','min','max'])


# In[ ]:


d.plot(kind='bar')


# Microlensing and transit can detect plants upto very far distances, but transit can detect for a wider range of distances. Maybe this explains why transit is used more often.
