#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, cross_val_score

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# In[ ]:


import warnings
warnings.filterwarnings('ignore')


# After taking a look at the data in my exploratory data analysis notebook: https://www.kaggle.com/kypygy/eda-w-visualizations, I decided to see if I could predict whether a stop resulted in an arrest or not.

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='darkgrid')
sns.set(palette='Set1')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# Read in the data
df = pd.read_csv('/kaggle/input/police-pedestrian-stops-and-vehicle-stops/police_pedestrian_stops_and_vehicle_stops.csv')


# In[ ]:


# Take another look
df.head()


# In[ ]:


# Check for NaN values
df.isnull().sum()


# Personally, I don't like having to capitalize the whole word every time I reference a column, so I'm going to lowercase every column with a list comprehension just to make my life a bit easier.

# In[ ]:


df.columns = [col.lower() for col in df.columns]


# In[ ]:


# That's much better!
df.columns


# As I did in my EDA notebook, I'm going to make a new column that populates with a 1 if the stop resulted in an arrest, and a 0 if it did not. This will be used as our label. To do this, I'll need to parse call_disposition to return anything with the word 'arrest' in it, as the wording is not entirely standardized. If we wanted to be really thorough, we could check each of the 207 values for call_disposition to make sure we're not missing anything, but for now I think having the word 'arrest' will suffice.

# In[ ]:


# Take a quick glance at call_disposition as a value count
df.call_disposition.value_counts().head(20)


# In[ ]:


# Create the arrest column based on call_disposition to establish our label in 1s and 0s
# Checking a lowercase version ensures capitalization won't throw us off
df['arrest_made'] = df.call_disposition.apply(lambda x: 1 if 'arrest' in x.lower() else 0)


# In[ ]:


# Let's see what percentage of stops resulted in an arrest
df.arrest_made.mean()*100


# From this we can tell that our majority class (non-arrest) makes up 87.4% of all cases. This is useful to know, as if our model does not predict with at least that accuracy, it is no better than choosing the majority class every time, making it not very useful.
# 
# Now we should start our feature selection process. I'll approach it like this:
# 
# 1) Determine which pieces of information might be good features
# 
# 2) Remove the columns we don't think we'll need
# 
# 3) Establish dummy data in numerical form for our categorical features
# 
# To start, I think whether a stop is a vehicle stop or a subject stop, what time (day, month, day of week, year) the stop occurred, which neighborhood it occurred in, and which police precinct was responsible could all be relevant.

# In[ ]:


# Let's break down the time_phonepickup column into more specific time chunks

# First we should convert the date as a string into a datetime object
df.time_phonepickup = pd.to_datetime(df.time_phonepickup)

# Then we can use datetime attributes to find hour (0-23), day of the week (0 is Monday, Sunday is 6), month (1-12), and year
df['hour'] = df.time_phonepickup.apply(lambda x: x.hour)
df['day_of_week'] = df.time_phonepickup.apply(lambda x: x.weekday()) # isoweekday() returns Monday starting as 1 and Sunday as 6
df['month'] = df.time_phonepickup.apply(lambda x: x.month)
df['year'] = df.time_phonepickup.apply(lambda x: x.year)


# ![](http://)I have visualizations of the breakdowns for those at: https://www.kaggle.com/kypygy/eda-w-visualizations

# In[ ]:


# Let's take a look at the precinct
df.precinct_id.value_counts()


# Looks like 18,037 records are 'None' for precinct_id. We checked for NaN values early on, but 'None' here is a string, so it wasn't caught. I'll drop the records with 'None' as a precinct_id value.

# In[ ]:


df = df[df['precinct_id'] != 'None']


# In[ ]:


# That got me thinking, what if there are 'None' values in other columns. To find out, let's loop through each column and check.
for column in df.columns:
    if any(df[column] == 'None'):
        print(column)
        print(len(df[df[column]=='None']))


# In[ ]:


# It comes out to less than .2% of our dataframe records, so I think it's safe to drop them. We can do this in one line by returning the df only if all values
# Within each row are not 'None'
df = df[(df[df.columns] != 'None').all(axis=1)]


# For categorical data, we will use what's known as one hot encoding. Essentially, we don't want our model thinking that there is a meaningful numerical condition associated with neighborhood names, for instance. So we will binarize that data instead. But first, let's drop all the columns we don't plan on using.

# In[ ]:


# Here's what we want to keep

to_keep = ['problem', 'arrest_made', 'hour', 'day_of_week', 'month', 'year', 'neighborhood_name', 'precinct_id']
new_df = df[to_keep]
"""
Here I tried label encoding as opposed to one hot, to no avail
new_df['neighborhood_name'] = df.neighborhood_name.astype('category').cat.codes
new_df['precinct_id'] = df.precinct_id.astype('category').cat.codes
"""
to_keep.pop(1)


# In[ ]:


# Now let's use pandas get_dummies function to one hot encode the categorical columns
encoded_df = pd.get_dummies(new_df, columns=to_keep, drop_first=True)


# It's time for us to try out a few models. I'm using K Nearest Neighbors, Logistic Regession, and Random Forest. 

# In[ ]:


#test_df = encoded_df.sample(100000)

# We establish our features variable (X) and our label variable (y). Notice X is everything but 'arrest_made'.

kf = KFold(n_splits=3, random_state = 1)

X = encoded_df.drop('arrest_made', axis=1)
y = encoded_df.arrest_made
print(X.shape)
print(y.shape)


# In[ ]:


# # Now we instantiate our class with the number of neighbors we want to try. 
# knn = KNeighborsClassifier(n_neighbors=20)
# scores = cross_val_score(knn, X, y, cv=kf, scoring='accuracy').mean()
# print(scores)


# In[ ]:


### If we had time, we could loop over a range for n_neighbors to find the best fit

# k_range = (1,30)
# k_scores = []
# for k in k_range:
#     knn = KNeighborsClassifier(n_neighbors=k)
#     scoring = cross_val_score(knn, X, y, cv=kf, scoring='accuracy').mean()
#     k_scores.append(scoring)
# print(k_scores)


# In[ ]:


def get_accuracy(model):
    if model == KNeighborsClassifier:
        return cross_val_score(model(n_neighbors=20), X, y, cv=kf, scoring='accuracy').mean()
    else:
        return cross_val_score(model, X, y, cv=kf, scoring='accuracy').mean()


# In[ ]:


for model in [LogisticRegression, RandomForestClassifier]:#, KNeighborsClassifier]:
    mean = get_accuracy(model())
    print("Model: {} - mean: {}".format(model.__name__, mean))


# In the end, I wasn't able to beat the majority class. I tried tweaking my features, using label encoding instead of one hot encoding, but that didn't help. I probably did not choose features very well, and this problem is likely not easily solved. Tuning didn't seem like it was going to help much, so I didn't go crazy. But, I learned a lot doing it! Please feel free to let me know how you'd improve this, or any clear mistakes I made along the way!
