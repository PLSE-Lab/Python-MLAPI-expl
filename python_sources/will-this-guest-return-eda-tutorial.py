#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# essentials for getting the input files
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


# importing the essentials libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


# looks like we have only one dataset. 
# importing the dataset into the code
df = pd.read_csv("/kaggle/input/hotel-booking-demand/hotel_bookings.csv")


# In[ ]:


# checking the dimensions of dataset
df.shape


# In[ ]:


df.dtypes


# Seems like we have a dataset with numerical, categorical as well as date datatypes

# In[ ]:


df.describe()


# We can generate a lot of hypothesis. 
# 
# Right now, let us try to find whether a guest is repeated guest or not. In other words, **let us find whether a guest will come again to the particular hotel or not**.

# We need to pick only the essential features. Not everything ! 
# 
# Let us get the features that might be related to the feature "is_repeated_guest".

# In[ ]:


df.corr(method = 'pearson')


# In[ ]:


sns.heatmap(df.corr())


# We are interested about this particular feature "is_repeated_guest"

# In[ ]:


# removing missing values
df.isnull().sum()


# We can remove the field "company" and "agent" for now. We shall add it and play with it later on. 

# In[ ]:


df = df.drop(["company", "agent"], axis = 1)


# In[ ]:


df = df.dropna()


# In[ ]:


df.isnull().sum()

Perfect !!! Now we can play !
# Let us get rid of categorical data for now

# In[ ]:


df.head()


# In[ ]:


df.shape


# In[ ]:


# importing the label encoder from sklearn
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
def encoding(df_column):
    if (df_column.dtype == 'O'):
        return encoder.fit_transform(df_column)
    return df_column
df_encoded = df.apply(encoding)


# In[ ]:


df_encoded.head()


# We can remove
# 1. lead_time
# 2. arrival_date_year
# 3. reservation_status_date
# 4. adr

# In[ ]:


df_encoded = df_encoded.drop(['lead_time', 'arrival_date_year', 'reservation_status_date', 'adr'], axis = 1)


# In[ ]:


df_encoded.corr(method = 'pearson')['is_repeated_guest']


# Let us use statistical tests for selecting features that contribute the most for the target feature.

# In[ ]:


input_features = df_encoded.drop(["is_repeated_guest"], axis = 1)
target_feature = df_encoded[["is_repeated_guest"]]


# In[ ]:


input_features.info(), target_feature.info()


# In[ ]:


input_features.describe()


# In[ ]:


# importing the SelectKBest feature selection
from sklearn.feature_selection import SelectKBest, chi2
best_features = SelectKBest(score_func = chi2, k = 'all')
new_best = best_features.fit(input_features, target_feature)


# In[ ]:


scores_df = pd.DataFrame(new_best.scores_)
columns_df = pd.DataFrame(input_features.columns)
final_dataframe = pd.concat([columns_df, scores_df], axis = 1)
final_dataframe.columns = ['features', 'scores']
final_dataframe


# In[ ]:


# sorting them in descending order
final_dataframe.sort_values(by = 'scores', ascending = False)


# Let us pick the features that return th highest scores for our final dataset to model

# In[ ]:


input_features = df_encoded.drop(['arrival_date_month', 
                                  'customer_type', 
                                  'babies', 
                                  'total_of_special_requests', 
                                  'arrival_date_day_of_month', 
                                  'booking_changes',
                                  'is_repeated_guest'], axis = 1)
target_feature = df_encoded[["is_repeated_guest"]]


# In[ ]:


input_features.info()


# In[ ]:


# converting the children feature from float to int
print(input_features['children'].dtype)
input_features['children'] = pd.to_numeric(input_features['children'].values, downcast = 'integer')
print(input_features['children'].dtype)


# In[ ]:


input_features.info(), target_feature.info()


# Splitting the training and testing dataset

# In[ ]:


from sklearn.model_selection import train_test_split
train_X, test_X, train_y, test_y = train_test_split(input_features, target_feature, test_size = 0.2, random_state = 101)


# In[ ]:


print(train_X.shape)
print(train_y.shape)
print(test_X.shape)
print(test_y.shape)


# In[ ]:


# let us use KNN classification
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = int(np.sqrt(input_features.shape[0]).round()))


# In[ ]:


classifier.fit(train_X, train_y.values)


# In[ ]:


predicted_classes = classifier.predict(test_X)


# In[ ]:


# let us check the accuracy of the model
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
print(confusion_matrix(test_y, predicted_classes))


# In[ ]:


(22987 + 131) / (22987 + 21 + 641 + 131)


# In[ ]:


print(classification_report(test_y, predicted_classes))


# f1 scores are good !!!

# In[ ]:


print(accuracy_score(test_y, predicted_classes))


# We can find whether a particular guest return or not to a particular hotel with an accuracy of 97%. :) 

# Creative ideas and Comments are welcome !!! 
