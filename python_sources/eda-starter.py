#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
from scipy.stats import pearsonr
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pandas.api.types import union_categoricals
from matplotlib import pyplot as plt 
import seaborn as sns

from os import listdir
from os import path
from sklearn.decomposition import KernelPCA


# In[ ]:


train_data = pd.read_csv(path.join("..", "input", "learn-together", "train.csv"))


# In[ ]:


soil_cols = []
other_cols = []
for x in train_data.columns:
    if x.startswith('Soil'):
        soil_cols.append(x)
    else:
        other_cols.append(x)


# In[ ]:


train_data.head()


# In[ ]:


train_descriptives = train_data.describe()
print(train_descriptives.filter(items = other_cols))


# In[ ]:


#Missingness check 
print('It is {:s} that there are missing data.'.format(
    str(any(train_descriptives.loc['count'] != train_descriptives.loc['count'].Id))))


# In[ ]:


sns.distplot(train_data.Cover_Type, kde = False, rug = False)


# The distribution of cover types is relatively uniform, so we won't have to worry about bias due to class imbalance.

# In[ ]:


#Are there any patches where there are multiple soil types?
soil_frame_train = train_data.filter(items = soil_cols)
print('It is {:s} that each patch has one and only one soil type.'.format(str(all(soil_frame_train.agg('sum', axis = 'columns') == 1))))


# In[ ]:


(soil_frame_train.agg('mean')).sort_values()


# Each patch has one and only one soil type, so the multinomial distribution is appropriate here; this means that there's an intrinsic covariance between soil types. For example, soil type 32 has mean 0.045635 and soil type 33 has mean 0.040741, so they both have a variance of about $0.96 \times 0.04 = 0.038$ and a covariance of approximately $-1 \times\left( 0.04\right)^2 = -0.0016$ so they'll be correlated at roughly $-\frac{0.04}{0.96}$, or about $-0.04$, which is highly statistically significant in this dataset, despite not telling us anything other than that a patch has only one soil type. 

# In[ ]:


pearsonr(train_data.Soil_Type32, train_data.Soil_Type33)


# In[ ]:


#Elevation histogram
sns.distplot(train_data.Elevation, kde = False, rug = True)


# The elevation histogram _looks_ trimodal; maybe the modes reflect the different wilderness areas.

# In[ ]:


wilderness_category = (train_data.filter(like = "Wildern").apply(axis = 1, func = np.flatnonzero) )
wilderness_category = (wilderness_category.astype('int32')).astype('category')
train_data['wilderness_cat'] = wilderness_category


# In[ ]:


sns.boxplot(data = train_data, x = "wilderness_cat", y = "Elevation", notch = True)


# In[ ]:


sns.swarmplot(data = train_data, x = "wilderness_cat", y = "Elevation")


# Mapping this to the three modes of elevation is actually relatively straightforward, although wilderness areas 0 and 2 (here, which are Rawah and Comanche Peak, respectively) show substantial overlap in elevations.

# In[ ]:


#Aspect histogram
sns.distplot(train_data.Aspect, kde = False, rug = False)


# The distribution of aspects is bimodal with a trough at around 270 degrees.

# In[ ]:


#Slope histogram
sns.distplot(train_data.Slope  , kde = True, rug = False)


# The distribution of slopes has an odd peak at around 30 degrees.

# In[ ]:


sns.pairplot(train_data.filter(items = ['Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways','Horizontal_Distance_To_Fire_Points', 'Elevation']))


# In[ ]:


(train_data.filter(items = ['Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways','Horizontal_Distance_To_Fire_Points','Elevation'])).corr()


# Because of the physical layout of each patch, there's correlation between their horizontal & vertical distances to hydrology, as well as their elevation. Distances to fire points seems independent of those things.

# In[ ]:


sns.pairplot(train_data.filter(items = ['Aspect', 'Slope', 'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm']))


# In[ ]:


(train_data.filter(items = ['Aspect', 'Slope', 'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm'])).corr()


# In[ ]:


(train_data.filter(items = ['Aspect', 'Slope', 'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm'])).cov()


# The relationships among aspect, slope, and different measures of hillshade have some kind of nonlinear constraints that cause them to be related. Correlation coefficients are able to account for some of these relationships. Kernel PCA might achieve some dimension reduction here.

# In[ ]:


pca = KernelPCA(
                n_components = 3)
solar_pca = pca.fit_transform(train_data.filter(items = ['Aspect', 'Slope', 'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm']))


# In[ ]:


pca.lambdas_


# In[ ]:


sns.pairplot(pd.DataFrame(solar_pca))


# In[ ]:


def crosstabber(soil_column):#
    ## add try catch for 
    try:
        return(np.ravel(pd.crosstab(train_data[soil_column], train_data.Cover_Type).iloc[1].astype('int')))
    except: 
        return(np.zeros(7).astype('int'))


# In[ ]:


print(crosstabber("Soil_Type3"))


# In[ ]:


soil_dictionary = {}
for stype in soil_cols:
    soil_dictionary[stype] = crosstabber(stype)

soil_tabulation  = (pd.DataFrame(soil_dictionary)).T


# In[ ]:


soil_tabulation


# # New Features
