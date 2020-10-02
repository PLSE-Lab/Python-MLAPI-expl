#!/usr/bin/env python
# coding: utf-8

# Sometimes with problems like this it is difficult to see the wood for the trees So let us concentrate on the data for id = 2047

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
get_ipython().run_line_magic('matplotlib', 'inline')


# Lets us load the data specific to id = 2047

# In[ ]:


low_y_cut = -0.086093
high_y_cut = 0.093497


# In[ ]:


directory = '../input/'
with pd.HDFStore(directory+'train.h5') as train:
    fullset = train.get('train')
fullset = fullset[['id', 'technical_20', 'technical_30', 'y']]
print(fullset.shape)
fullset.fillna(fullset.median(), inplace=True)
fullset = fullset[fullset['id'] == 2047]
fullset = fullset[fullset.y < high_y_cut]
fullset = fullset[fullset.y > low_y_cut]
print(fullset.shape)


# As you can see let us concentrate on data for 1790 rows just using technical_20 and technical_30

# I looked at the data and found it fitted poorly with my genetic program using just one point.  However when using a window of three points I got much better accuracy. So let us window the data.

# In[ ]:


y = fullset.iloc[1:-1]['y'].values
fullset.drop('y', inplace=True, axis=1)
wdw = pd.DataFrame()
for i in range(3):
    if(i == 0):
        wdw['technical_20_Row_Offset_0'] =             fullset['technical_20'].iloc[0:-2].values
        wdw['technical_30_Row_Offset_0'] =             fullset['technical_30'].iloc[0:-2].values
    elif(i == 1):
        wdw['technical_20_Row_Offset_1'] =             fullset['technical_20'].iloc[1:-1].values
        wdw['technical_30_Row_Offset_1'] =             fullset['technical_30'].iloc[1:-1].values
    else:
        wdw['technical_20_Row_Offset_2'] =             fullset['technical_20'].iloc[2:].values
        wdw['technical_30_Row_Offset_2'] =             fullset['technical_30'].iloc[2:].values
wdw['y'] = y


# In[ ]:


print(fullset.head())


# In[ ]:


print(wdw.head())


# All I have done is added the data from the previous row and the next row

# In[ ]:


def r_score(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    r = np.sign(r2) * np.sqrt(np.abs(r2))
    return max(-1, r)


# Now define a simple model (I did this using GP but you can use whatever you wish!)

# In[ ]:


def GPTechnicalPrediction(data):
    p = (((((8.0) * (data["technical_20_Row_Offset_2"] + (data["technical_30_Row_Offset_1"] - (data["technical_20_Row_Offset_1"] + data["technical_30_Row_Offset_2"])))) - (data["technical_30_Row_Offset_1"] - ((((data["technical_30_Row_Offset_1"] + ((((data["technical_30_Row_Offset_1"] + data["technical_20_Row_Offset_1"]) * data["technical_30_Row_Offset_1"]) + data["technical_20_Row_Offset_1"])/2.0))/2.0) + data["technical_20_Row_Offset_2"])/2.0)))) )
    return p.values.clip(low_y_cut,high_y_cut)


# Now let us get the predictions

# In[ ]:


yhat = GPTechnicalPrediction(wdw)


# What is the score?

# In[ ]:


print('R Score: ',r_score(wdw.y.values, yhat))


# The score isn't bad but obviously we are only using one id and we know the next values for technical_20 and technical_30 which we don't know for the test set.  However you may want to investigate either a couple of models to predict the values for the next technical_20;technical_30 tuple or use a simple look up table based on id and timestamp.  From some initial investigation is does seem to me that is easier to predict future technical_20 and technical_30 values rather than just a straight prediction on y.

# In[ ]:


plt.figure(figsize=(8,8))
plt.plot(wdw.y.values)
plt.plot(yhat)

