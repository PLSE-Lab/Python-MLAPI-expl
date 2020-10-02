#!/usr/bin/env python
# coding: utf-8

# # HERE IS MY WORK ON BOSTON DATA FOR MY REFERENCE ,AS A BEGINNER I WORKED ON IT THROUGH THE DATASET.

# # IMPORTING LIBRARIES.

# In[ ]:


import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split


# # IMPORTING DATA FROM MY WORKSHOP.

# In[ ]:


rawBostonData = pd.read_csv('https://raw.githubusercontent.com/PacktWorkshops/The-Data-Science-Workshop/master/Chapter02/Dataset/Boston.csv')


# # DATA PREPROCESSING

# In[ ]:


rawBostonData = rawBostonData.dropna()


# In[ ]:


rawBostonData = rawBostonData.drop_duplicates()


# In[ ]:


renamedBostonData = rawBostonData.rename(columns = {'CRIM':'crimeRatePerCapita',
 ' ZN ':'landOver25K_sqft',
 'INDUS ':'non-retailLandProptn',
 'CHAS':'riverDummy',
 'NOX':'nitrixOxide_pp10m',
 'RM':'AvgNo.RoomsPerDwelling',
 'AGE':'ProptnOwnerOccupied',
 'DIS':'weightedDist',
 'RAD':'radialHighwaysAccess',
 'TAX':'propTaxRate_per10K',
 'PTRATIO':'pupilTeacherRatio',
 'LSTAT':'pctLowerStatus',
 'MEDV':'medianValue_Ks'})


# # TRAIN AND TEST DATA.

# In[ ]:


X = renamedBostonData.drop('crimeRatePerCapita', axis = 1)
y = renamedBostonData[['crimeRatePerCapita']]
seed = 10 
test_data_size = 0.3 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_data_size, random_state = seed)
train_data = pd.concat([X_train, y_train], axis = 1)
test_data = pd.concat([X_test, y_test], axis = 1)


# # LOGLINEARMODEL.

# In[ ]:


logLinearModel = smf.ols(formula='np.log(crimeRatePerCapita) ~ medianValue_Ks', data=train_data)


# In[ ]:


logLinearModResult = logLinearModel.fit()


# In[ ]:


print(logLinearModResult.summary())


# # MULTILOGLINEARMODEL

# In[ ]:


multiLogLinMod = smf.ols(formula='np.log(crimeRatePerCapita) ~ (pctLowerStatus + radialHighwaysAccess + medianValue_Ks + nitrixOxide_pp10m)**2',data=train_data)


# In[ ]:


multiLogLinModResult = multiLogLinMod.fit()


# In[ ]:


print(multiLogLinModResult.summary())


# In[ ]:




