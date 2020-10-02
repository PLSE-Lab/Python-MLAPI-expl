#!/usr/bin/env python
# coding: utf-8

# This tries to estimate hard drive failure probability, from the data on last day. Set up one of many prediction problems, clean the data a bit, notice some areas for dataset improvement and likely *introduce a leak* into my prediction task(some eda to find it would be a good Kernel for someone to write). There are a few data processing improvements for this Dataset, to make it more usable.
# 

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import ensemble, metrics 
import gc


# In[ ]:


hdd = pd.read_csv('../input/harddrive.csv')


# In[ ]:


hdd.head()
hdd.serial_number.unique().shape


# In[ ]:


np.int64(hdd['capacity_bytes'].at[5])


# In[ ]:


print(hdd['capacity_bytes'].at[5])
print(hdd['capacity_bytes'].at[5] * 10**10)


# **note** capacity is in bytes has been loaded as a very small number(very close to zero)

# In[ ]:


hdd.shape


# In[ ]:


# number of hdd
hdd['serial_number'].value_counts().shape


# In[ ]:


# drop constant columns
hdd = hdd.loc[:, ~hdd.isnull().all()]


# In[ ]:


# number of different types of harddrives
hdd['model'].value_counts().shape


# In[ ]:


print(hdd.groupby('model')['failure'].sum().sort_values(ascending=False).iloc[:30])


# In[ ]:


# ST4000DM000 refers to Seagate SATA 6Gb/s 3.5-Inch 4TB Desktop HDD 
# Since it makes up a majority of the data, and the features should be the same for 
# drives of the same model, we will start by predicting failure in these drives. 
# combining them with other similar seagate drives might be an easy way to use more of the data

hdd_st4 = hdd.query('model == "ST4000DM000"')


# In[ ]:


del hdd
gc.collect()


# In[ ]:


# number of drives in the reduced data
hdd_st4['serial_number'].value_counts().shape


# In[ ]:


# out of the 35k drives there are 131 failures, so this is definitly an imbalanced dataset. 
# note the output says 139 1 labeled but this is incorrect as 8 are duplicates. I drop them later
# because dropping at the begginning crashed the Kernel
hdd_st4['failure'].value_counts()


# In[ ]:


# more constant columns
hdd_st4['capacity_bytes'].value_counts()


# In[ ]:


# drop them 
hdd_st4 = hdd_st4.loc[:, ~hdd_st4.isnull().all()]


# In[ ]:


hdd_st4.shape


# In[ ]:


# these have similar exponents as the size of harddrive. 
# I am also pretty sure these variables are something like total read or total write. 
# The scientific format is interprating the exponents as negative when they should likely be positive
# A fractional byte does not make sense. 
hdd_st4.iloc[:5,13:15] 


# The normalized values are not correcting for the format issue. That is 1.2*10^317, should be larger then 1.01*10^315
# And as can be seen above it is not. **NOTE** I only process four columns that have very small values where there should be large values. The others should be found and corrected(though reversing order is probably not horrible and training on these would probably work but probably better to have them corrected) 

# In[ ]:


# removed normalized values, and model, and capacity, since they are constants
hdd_st4 = hdd_st4.select(lambda x: x[-10:] != 'normalized', axis=1)
hdd_st4 = hdd_st4.drop(['model', 'capacity_bytes'], axis=1)
gc.collect()


# In[ ]:


# no null values left. 
# thanks to the scripts by Alan Pryor(https://www.kaggle.com/apryor6) foor this and other nice ways of doing things 
hdd_st4.isnull().any()


# In[ ]:


# there are more constant columms but skipping this for run time reasons
# hdd_st4['smart_3_raw'].value_counts()


# In[ ]:


# remove more constant columns(anyone have a fast one liner for this?)
# for i in hdd_st4.columns:
#    if len(hdd_st4.loc[:,i].unique()) == 1:
#        hdd_st4.drop(i, axis=1, inplace=True)


# In[ ]:


# hdd_st4.head()


# In[ ]:


# now to deal with the issue of really small byte numbers. If there is an easy way to do this please
# feel free to point it out, after a quick search turned up nothing this was the first solution that 
# I came up with. 


# In[ ]:


# turns number into a string, then extracts, base and exponent
def convert_large_number(large_num, min_exponent):
    str_num = str(large_num)
    
    base_end = str_num.find('e')
    base  = np.float64((str_num[:base_end]))
    
    # if i remember correctly this is equivelent to dividing by a constant
    exponent = np.int64(str_num[base_end+2:]) - (min_exponent-1)
    return base*10**exponent

# just fetches the exponent
def get_exp(large_num):
    str_num = str(large_num)
    base_end = str_num.find('e')
    
    exponent = np.int64(str_num[base_end+2:])
        
    return exponent

# finds the minimum exponenet for a series
def min_exp(series_of_large_num):
    exps = series_of_large_num.apply(get_exp)
    return exps.min()

# scales a series down but subtracting the min observed exponent from exponent.  
def scale_large_num_col(series, min_exponent):
    return series.apply(convert_large_number, min_exponent=min_exponent)
    


# In[ ]:


# smart_241_raw contains a single 0 which messes up my method of conversion
s241_mean = hdd_st4['smart_241_raw'].mean()
hdd_st4['smart_241_raw'].replace(0.0, s241_mean, inplace=True)


# In[ ]:


# just find the min exponent of the whole of all the columns so I can adjust all the data together
# or you can just look at the columns
# mins = []
# for i in range(3, len(hdd_st4.columns)):
#     if hdd_st4.iloc[0,i] < 10**-10 and hdd_st4.iloc[0,i] > 0:
#             mins.append(min_exp(hdd_st4.iloc[:,i]))

# this takes a little to long to run

# minimum exponent for these is 309
        


# In[ ]:


# transform data so it is a more managable size
# alternativly they could be stored as full length integers
for i in range(3, len(hdd_st4.columns)):
    if hdd_st4.iloc[0,i] < 10**-10 and hdd_st4.iloc[0,i] > 0:
       hdd_st4.iloc[:,i] =  scale_large_num_col(hdd_st4.iloc[:,i], 308)


# In[ ]:


gc.collect()
hdd_st4.head()


# In[ ]:


# Since we are trying to predict drive failure, we randomly select a set of drives. 
# note that if there is some relationship between the drives, say a large group are in the same building. Then failure 
# between drives won't be indepentent

hdd_st4.loc[:, 'date'] = pd.to_datetime(hdd_st4.loc[:,'date'])
hdd_st4['day_of_year'] = hdd_st4['date'].dt.dayofyear

hdd_st4.plot(kind='scatter', x='day_of_year', y='failure', title='Hard drive failures over time')
plt.show()


# looks like there is a gap in the data, and the failure distrubtion looks roughly uniform
# but this should be more carefully checked.

# In[ ]:


# note this could be done earlier but it doesn't work on Kernels because of memory limitations
hdd_st4 = hdd_st4.drop_duplicates()


# In[ ]:


# lets try to predict the probability of failure from data only on the day of failure
# it would be good to see how this probability relates to the probability of failure using previous days data only
hdd_group = hdd_st4.groupby('serial_number')
hdd_last_day = hdd_group.nth(-1) # take the last row from each group
del hdd_st4
gc.collect()


# In[ ]:


# the number of drives in the dataset
uniq_serial_num = pd.Series(hdd_last_day.index.unique())
uniq_serial_num.shape


# In[ ]:


# hold out 25% of data for testing
test_ids = uniq_serial_num.sample(frac=0.25)


# In[ ]:


train = hdd_last_day.query('index not in @test_ids')
test = hdd_last_day.query('index in @test_ids')


# In[ ]:


test['failure'].value_counts()


# In[ ]:


train['failure'].value_counts()


# In[ ]:


# close enough to stratified sampling for me. 
131/4


# In[ ]:


train_labels = train['failure']
test_labels = test['failure']
train = train.drop('failure', axis=1)
test = test.drop('failure', axis=1)


# In[ ]:


train['day_of_year'].value_counts()


# In[ ]:


# the last day has most of the data without failures. This makes sense because I chose
# to use the last day as a feature and most drives are still working on last day.
# looks like I wasn't careful enough about possible leaks. 
# in this dataset date, and by extension number of samples will be a leak, as those 
# harddrives which failed in the dataset will likely have less days available. 
print(train_labels.reindex(train.query('day_of_year == 120').index).shape[0],
      train_labels.reindex(train.query('day_of_year == 120').index).sum())


# In[ ]:


# 
print(train_labels.reindex(train.query('day_of_year != 120').index).shape[0],
      train_labels.reindex(train.query('day_of_year != 120').index).sum())


# In[ ]:


#drop date related features maybe this will prevent leakage;)
train = train.drop(['day_of_year', 'date'], axis=1)
test = test.drop(['day_of_year', 'date'] , axis=1)


# In[ ]:


# remove more constant columns(anyone have a fast one liner for this?)
# could have done this earlier
for i in train.columns:
    if len(train.loc[:,i].unique()) == 1:
        train.drop(i, axis=1, inplace=True)
        test.drop(i, axis=1, inplace=True)


# In[ ]:


train.head().columns


# In[ ]:


rf = ensemble.RandomForestClassifier()
rf.fit(train, train_labels)
preds = rf.predict_proba(test)

print('logloss', metrics.log_loss(y_true=test_labels, y_pred=preds[:,1]))
print('roc_auc', metrics.roc_auc_score(y_true=test_labels, y_score=preds[:,1]))


# This looks good but I am immediately skeptical about this result given that the last day has almost all the negative examples. There may be a another variable which is a proxy for time. It would be good to **prove there is a leak** or exhaust as many possibilities as possible. This is where eda is very useful. I have gone over the amount of time I wanted to spend on this Kernel so some one else will need to follow up on data quality stuff.  
# 
# A better next step might be to use the first day instead of the last day(if i have time I will try this otherwise someone else give it a try) 
