#!/usr/bin/env python
# coding: utf-8

# Table of contents
# * [Data Cleaning](#section-three)
# * [Data Modelling](#section-ten)
# * [Data Validation](#section-fifteen)
# * [Probability of on-time arrivals for an evening flight from JFK to ATL over a range of days](#section-eighteen)
# * [Summary](#section-twenty)

# In[ ]:


get_ipython().system('curl https://topcs.blob.core.windows.net/public/FlightData.csv -o flightdata.csv')


# In[ ]:


import pandas as pd

df = pd.read_csv('flightdata.csv')
df.head()


# # Cleaning and preparing data

# In[ ]:


df.shape


# # Column Details

# ![image.png](attachment:image.png)

# ![image.png](attachment:image.png)

# ![image.png](attachment:image.png)
# 

# In[ ]:


df.isnull().sum()


# The 26th column ("Unnamed: 25") contains 11,231 missing values, which equals the number of rows in the dataset.So it can be removed.

# In[ ]:


df = df.drop('Unnamed: 25', axis=1)
df.isnull().sum()


# # Taking only the features that are necessary

# In[ ]:


df = df[["MONTH", "DAY_OF_MONTH", "DAY_OF_WEEK", "ORIGIN", "DEST", "CRS_DEP_TIME", "ARR_DEL15"]]
df.isnull().sum()


# In[ ]:


df = df.fillna({'ARR_DEL15': 1})


# In[ ]:


df


# In[ ]:


df.isnull().sum()


# # Binning the departure times

# In[ ]:


import math

for index, row in df.iterrows():
    df.loc[index, 'CRS_DEP_TIME'] = math.floor(row['CRS_DEP_TIME'] / 100)
df.head()


# In[ ]:


df = pd.get_dummies(df, columns=['ORIGIN', 'DEST'])
df.head()


# # Building Model

# In[ ]:


from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(df.drop('ARR_DEL15', axis=1), df['ARR_DEL15'], test_size=0.2, random_state=42)


# In[ ]:


train_x.shape


# In[ ]:


test_x.shape


# In[ ]:


train_y.shape


# In[ ]:


test_y.shape


# In[ ]:


from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(random_state=30)
model.fit(train_x, train_y)


# In[ ]:


predicted = model.predict(test_x)
model.score(test_x, test_y)


# In[ ]:


from sklearn.metrics import roc_auc_score
probabilities = model.predict_proba(test_x)
probabilities


# In[ ]:


roc_auc_score(test_y, probabilities[:, 1])


# **Roc < Accuracy because data is skewed**

# In[ ]:


from sklearn.metrics import confusion_matrix
confusion_matrix(test_y, predicted)


# In[ ]:


from sklearn.metrics import precision_score

train_predictions = model.predict(train_x)
precision_score(train_y, train_predictions)


# In[ ]:


from sklearn.metrics import recall_score

recall_score(train_y, train_predictions)


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()


# In[ ]:


from sklearn.metrics import roc_curve

fpr, tpr, _ = roc_curve(test_y, probabilities[:, 1])
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], color='grey', lw=1, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')


# **The below function takes as input a date and time, an origin airport code, and a destination airport code, and returns a value between 0.0 and 1.0 indicating the probability that the flight will arrive at its destination on time.**

# In[ ]:


def predict_delay(departure_date_time, origin, destination):
    from datetime import datetime

    try:
        departure_date_time_parsed = datetime.strptime(departure_date_time, '%d/%m/%Y %H:%M:%S')
    except ValueError as e:
        return 'Error parsing date/time - {}'.format(e)

    month = departure_date_time_parsed.month
    day = departure_date_time_parsed.day
    day_of_week = departure_date_time_parsed.isoweekday()
    hour = departure_date_time_parsed.hour

    origin = origin.upper()
    destination = destination.upper()

    input = [{'MONTH': month,
              'DAY': day,
              'DAY_OF_WEEK': day_of_week,
              'CRS_DEP_TIME': hour,
              'ORIGIN_ATL': 1 if origin == 'ATL' else 0,
              'ORIGIN_DTW': 1 if origin == 'DTW' else 0,
              'ORIGIN_JFK': 1 if origin == 'JFK' else 0,
              'ORIGIN_MSP': 1 if origin == 'MSP' else 0,
              'ORIGIN_SEA': 1 if origin == 'SEA' else 0,
              'DEST_ATL': 1 if destination == 'ATL' else 0,
              'DEST_DTW': 1 if destination == 'DTW' else 0,
              'DEST_JFK': 1 if destination == 'JFK' else 0,
              'DEST_MSP': 1 if destination == 'MSP' else 0,
              'DEST_SEA': 1 if destination == 'SEA' else 0 }]

    return model.predict_proba(pd.DataFrame(input))[0][0]


# **Lets check for 1 value**

# In[ ]:


predict_delay('1/10/2018 21:45:00', 'JFK', 'ATL')


# The probability is 0.82

# # Probability of on-time arrivals for an evening flight from JFK to ATL over a range of days

# In[ ]:


import numpy as np

labels = ('Oct 1', 'Oct 2', 'Oct 3', 'Oct 4', 'Oct 5', 'Oct 6', 'Oct 7')
values = (predict_delay('1/10/2018 21:45:00', 'JFK', 'ATL'),
          predict_delay('2/10/2018 21:45:00', 'JFK', 'ATL'),
          predict_delay('3/10/2018 21:45:00', 'JFK', 'ATL'),
          predict_delay('4/10/2018 21:45:00', 'JFK', 'ATL'),
          predict_delay('5/10/2018 21:45:00', 'JFK', 'ATL'),
          predict_delay('6/10/2018 21:45:00', 'JFK', 'ATL'),
          predict_delay('7/10/2018 21:45:00', 'JFK', 'ATL'))
alabels = np.arange(len(labels))

plt.bar(alabels, values, align='center', alpha=0.5)
plt.xticks(alabels, labels)
plt.ylabel('Probability of On-Time Arrival')
plt.ylim((0.0, 1.0))


# # Summary

# Things done in this notebook :
# 
# Import data using curl, 
# Use Pandas to clean and prepare data, 
# Use scikit-learn to build a machine-learning model, 
# Use Matplotlib to visualize the results

# In[ ]:




