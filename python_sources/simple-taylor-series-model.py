#!/usr/bin/env python
# coding: utf-8

# # Load the training data form the csv

# In[ ]:


import numpy as np
import pandas as pd
train_data = pd.read_csv('/kaggle/input/covid19-local-us-ca-forecasting-week-1/ca_train.csv')
# we don't need the test data
# test_data = pd.read_csv('/kaggle/input/covid19-local-us-ca-forecasting-week-1/ca_test.csv')


# # Check the training data

# In[ ]:


print(train_data)


# # Throw away useless columns and cases
# First of all lets check how many entries each column has

# In[ ]:


useless_cols = []
for col in train_data.columns:
    entries_num = len(set(train_data[col]))
    if entries_num < 2:
        useless_cols.append(col)
    print(f'{col} -- has {entries_num} unic entries')


# The geographical data -- like Latitude, Longitude, etc -- has only one entry 'cause all data is from California. We can freely drop this data.
# 
# We also don't need Id and Date for prediction.

# In[ ]:


train_data.drop(useless_cols, axis=1, inplace=True)
train_data.drop(['Id', 'Date'], axis=1, inplace=True)


# # Drop useless rows
# The first 48 days there were no cases, we should drop this rows too

# In[ ]:


X_train = train_data[48:]
print(X_train)


# # Creating features
# Let's create some features which we will use for our models

# In[ ]:


# Approximation of the confirmed cases first and second derivatives
diff_conf, conf_old = [], 0 
dd_conf, dc_old = [], 0

# Approximation of the fatalities first and second derivatives
diff_fat, fat_old = [], 0
dd_fat, df_old = [], 0

# Approximation of exponential grow coefficient
exp_conf, exp_fat = [], []

# Approximation of the ration between fatalities and confirmed cases
ratio = []

# Calc the features' arrays
for row in X_train.values:
    diff_conf.append(row[0]-conf_old)
    conf_old=row[0]
    
    diff_fat.append(row[1]-fat_old)
    fat_old=row[1]
    
    exp_conf.append(diff_conf[-1]/row[0])
    exp_fat.append(diff_fat[-1]/row[1])
    
    ratio.append(row[1]/row[0])
    
    dd_conf.append(diff_conf[-1]-dc_old)
    dc_old=diff_conf[-1]
    
    dd_fat.append(diff_fat[-1]-df_old)
    df_old=diff_fat[-1]

# Insert features into training set
X_train['diff_confirmed'] = diff_conf
X_train['diff_fatalities'] = diff_fat
X_train['exp_confirmed'] = exp_conf
X_train['exp_fatalities'] = exp_fat
X_train['fatalities_to_confirmed'] = ratio
X_train['dd_confirmed'] = dd_conf
X_train['dd_fatalities'] = dd_conf
print(X_train)


# # Calculate means
# We are throwing away some values of the derivatives lists -- one value for the first derivative and two for the second.

# In[ ]:


exp_c = X_train.exp_confirmed.drop(48).mean()
print(f'exp_c: {exp_c}')
exp_f = X_train.exp_fatalities.drop(48).mean()
print(f'exp_f: {exp_f}')
ratio = X_train.fatalities_to_confirmed.drop(48).mean()
print(f'ratio: {ratio}')
d_c = X_train.diff_confirmed.drop(48).mean()
print(f'd_c: {d_c}')
dd_c = X_train.dd_confirmed.drop(48).drop(49).mean()
print(f'dd_c: {dd_c}')
d_f = X_train.diff_fatalities.drop(48).mean()
print(f'd_f: {d_f}')
dd_f = X_train.dd_fatalities.drop(48).drop(49).mean()
print(f'dd_f: {dd_f}')


# # The first week of the prediction overlap with the training data
# Copying the first week from the training set

# In[ ]:


pred_c, pred_f = list(X_train.ConfirmedCases.loc[50:56]), list(X_train.Fatalities.loc[50:56])


# # Model #1
# We know that almost any signal can be represented as an series of it's derivatives: taylor series.
# 
# f(x) = df(a) * (x - a) + ddf(a) * (x - a)^2 / 2 + dddf(a) * (x - a)^3 / 6 + ...
# 
# 
# We will use the first two members of the taylor series to predict the future. 
# 
# In theory the more members we choose the more accurate our model will be, unfortunatly there is not enough data to calculate the third derivative accuratly.

# In[ ]:


for i in range(1, 44 - 7):
    # use taylor series to predict confirmed cases
    pred_c.append((X_train.ConfirmedCases[56] + d_c*i + 0.5*dd_c*(i**2)))
    # use taylor series to predict fatalities
    # pred_f.append((X_train.Fatalities[56] + d_f*i + 0.5*dd_f*(i**2)))
    
    # We can also try to use ratio of fatalities to cases instead
    pred_f.append(pred_c[-1] * ratio)


# # Model #2
# Instead of using taylor series we can use simple exponential model (see https://www.youtube.com/watch?v=Kas0tIxDvrg)

# In[ ]:


# for i in range(1, 44 - 7):
#     pred_c.append(X_train.ConfirmedCases[56] * ((1 + exp_c) ** i ))
#     pred_f.append(pred_c[-1] * ratio)


# # Submit results

# In[ ]:


my_submission = pd.DataFrame({'ForecastId': list(range(1,44)), 'ConfirmedCases': pred_c, 'Fatalities': pred_f})
my_submission.to_csv('submission.csv', index=False)
print(my_submission)


# In[ ]:




