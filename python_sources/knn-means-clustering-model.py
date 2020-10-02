#!/usr/bin/env python
# coding: utf-8

# I was curious to see if we can directly apply unsupervised learning to a two classes competition. 
# In this Kernel, the unsupervised learning KNN algorithm was applied to test data from TalkingDataAdTracking Fraud.
# The KNN was applied directly to test data and it was tried to find two distinct classes.
# A submit file with all equal class 1 was used to find out the ration of classes in the test data. As it was expected the test data contained half class 1 and the other half class 0.
# Although the KNN divided the test data into two classes with almost the same numbers, the results were not promising (52%).
# I thought to share this Kernel with the fellow here at Kaggle (Kaggeleres). 
# Any feedback is appreciated.

# # libraries

# In[ ]:


# import statements
from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt
# import KMeans
from sklearn.cluster import KMeans
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import pyplot


# # Functions

# In[ ]:


def Parse_time(df):
    df['day'] = df['click_time'].dt.day.astype('uint8')
    df['hour'] = df['click_time'].dt.hour.astype('uint8')
    df['minute'] = df['click_time'].dt.minute.astype('uint8')
    df['second'] = df['click_time'].dt.second.astype('uint8')
    
def Drop_cols(df, x):
    Num_of_line = 100
    print(Num_of_line*'=')
    print('Before drop =\n', df.head(3))
    print(Num_of_line*'=')
    df.drop(labels = x, axis = 1, inplace = True)
    print('After drop =\n', df.head(3))
    return df


# # Read data

# In[ ]:


address_test = '../input/talkingdata-adtracking-fraud-detection/test.csv'
df_test = pd.read_csv(address_test, parse_dates=['click_time'])


# # Parse data and drop columns

# In[ ]:


Parse_time(df_test)
colmn_names = ["click_time", "click_id", "ip"]
df_test = Drop_cols(df_test, colmn_names); df_test.head()


# # Fit KNN clustering model

# In[ ]:


# create kmeans object
kmeans = KMeans(n_clusters=2)
# fit kmeans object to data
kmeans.fit(df_test)
# print location of clusters learned by kmeans object
print(kmeans.cluster_centers_)
# save new clusters for chart
y_km = kmeans.fit_predict(df_test)


predict = pd.DataFrame(y_km)
data_to_submit = pd.DataFrame()
data_to_submit['click_id'] = range(0, len(df_test))
data_to_submit['is_attributed'] = predict
print('data_to_submit = \n', data_to_submit.head(5))
pyplot.hist(data_to_submit['is_attributed'], log = True)


# # Save submit data

# In[ ]:


data_to_submit.to_csv('Unsuper_csv_to_submit.csv', index = False)


# See the submit data

# In[ ]:


data_to_submit

