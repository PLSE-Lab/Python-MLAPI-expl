#!/usr/bin/env python
# coding: utf-8

# # Predicting No-Shows in Hospitals
# In this notebook I look at the ["Medical Appointment No Shows" dataset](https://www.kaggle.com/joniarroba/noshowappointments) on www.kaggle.com
# After cleaning the data, I will use Machine Learning models to predict future no shows.

# # Step 1: Data Wrangling
# After reading in my data, I'm going to look for missing values and invalid entries.

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pprint import pprint

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df_raw = pd.read_csv('../input/noshowappointments/KaggleV2-May-2016.csv')


# In[ ]:


df_raw.info()


# Evidently there are no missing values.

# ### Clean up datatypes
# Now I check for mismatched datatypes and convert each category appropriately.

# In[ ]:


# PatientId should be int64, not float64
df_raw['PatientId'] = df_raw['PatientId'].astype('int64')

# Convert ScheduledDay and AppointmentDay to datetime64[ns]
df_raw['ScheduledDay'] = pd.to_datetime(df_raw['ScheduledDay']).dt.date.astype('datetime64[ns]')
df_raw['AppointmentDay'] = pd.to_datetime(df_raw['AppointmentDay']).dt.date.astype('datetime64[ns]')


# In[ ]:


# check the head of dataset
df_raw.head()


# ### Check for typos

# In[ ]:


# rename typo columns
df_raw.rename(columns={"Hipertension": "Hypertension","Handcap":"Handicap",
                      "SMS_received": "SMSReceived", "No-show": "NoShow"},inplace=True)


# In[ ]:


# check for typos
print(sorted(df_raw['Neighbourhood'].unique()))


# Looks clean, so I check other categories

# In[ ]:


# Check Age
print(sorted(df_raw['Age'].unique()))


# In[ ]:


df_raw[df_raw['Age'] == -1]


# In[ ]:


df_raw[df_raw['Age'] == 115]


# * "Age" column has negative value and anomalous entries of 115. These entries are too far-removed from the rest of dataset. A Google search has confirmed that there were no 115-year-olds alive in Brazil at the timestamps stated. See reference [1] for confirmation.
# * I will remove these entries.

# In[ ]:


# Remove erroneous entries
df_raw = df_raw[(df_raw['Age'] < 115) & (df_raw['Age'] > 0)]


# * "PatientId" and "AppointmentID" columns are random system generated numbers. I will delete these.

# In[ ]:


df_raw = df_raw.drop(['PatientId','AppointmentID'],axis=1)


# * Now I clean up "ScheduledDay" and "AppointmentDay" columns

# In[ ]:


df_raw['ScheduledMonth'] = df_raw['ScheduledDay'].dt.month
df_raw['ScheduledDayofWeek'] = df_raw['ScheduledDay'].dt.day_name()
df_raw['ScheduledHour'] = df_raw['ScheduledDay'].dt.hour


# In[ ]:


df_raw['AppointmentMonth'] = df_raw['AppointmentDay'].dt.month
df_raw['AppointmentDayofWeek'] = df_raw['AppointmentDay'].dt.day_name()
df_raw['AppointmentHour'] = df_raw['AppointmentDay'].dt.hour


# # Step 2: EDA
# * Time for some Exploratory Data Analysis.

# In[ ]:


sns.countplot(x='Gender', hue='NoShow', data=df_raw)


# In[ ]:


plt.figure(figsize=(30,12))
fig = sns.countplot(x='Neighbourhood',hue='NoShow',data=df_raw)
fig.set_xticklabels(fig.get_xticklabels(), rotation=90);


# In[ ]:


sns.heatmap(df_raw.corr(), vmin=-0.9, vmax=0.9,cmap='coolwarm')


# # Step 3: Preprocessing
# * Next I prepare data for modelling, creating logical variables and dropping redundant features

# In[ ]:


df_raw['AppointmentDayofWeek'] = df_raw['AppointmentDay'].dt.weekday
df_raw['ScheduledDayofWeek'] = df_raw['ScheduledDay'].dt.weekday


# ### Get dummy variables
# Now I convert non-numeric variables to logical variables

# In[ ]:


df_raw['NoShow'] = pd.get_dummies(df_raw['NoShow'])['Yes']


# In[ ]:


no_show = len(df_raw[df_raw['NoShow'] == 1])
print(f'No-shows: {no_show}')

total = len(df_raw)
print(f'Percentage no-show: {(no_show/total) * 100}')


# * To help with following plots, I now create a logical variable for gender

# In[ ]:


# skewed towards female entries
print(f"Gender entries: {df_raw['Gender'].unique()}")
print(df_raw['Gender'].describe())
df_raw['Male'] = pd.get_dummies(df_raw['Gender'])['M']
      
df_raw = df_raw.drop('Gender',axis=1)


# In[ ]:


# get dummy variables for neighbourhood
neighbourhoods = pd.get_dummies(df_raw['Neighbourhood'])

# join dummy neighbourhood columns and drop string neighbourhood column
df_raw = df_raw.join(neighbourhoods).drop('Neighbourhood',axis=1)


# ### Drop redundant variables

# * Now I drop the "AppointmentDay" "ScheduledDay" columns, as we have no more use for these

# In[ ]:


df = df_raw.drop(['AppointmentDay','ScheduledDay'],axis=1)


# ## Standardise Variables
# Lastly, I'm going to standardise the variables to prepare data for modelling.

# In[ ]:


# import StandardScaler from Scikit learn
from sklearn.preprocessing import StandardScaler

# create StandardScaler object
scaler = StandardScaler()

# fit scaler to features
scaler.fit(df.drop(['NoShow'],axis=1))


# In[ ]:


# use .transform() to transform features to scaled version
scaled_features = scaler.transform(df.drop('NoShow',axis=1))


# In[ ]:


df_feat = pd.DataFrame(scaled_features)
df_feat.head()


# # Step 4: Modelling
# * Now I'll make some predictions using Machine Learning models

# In[ ]:


# Import train_test_split function
from sklearn.model_selection import train_test_split

X = df_feat  # Features
y = df['NoShow']  # Labels

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3) # 70% training and 30% test


# ## Decision Tree Model
# I'll start by training a single decision tree first.

# In[ ]:


from sklearn.tree import DecisionTreeClassifier

dtree = DecisionTreeClassifier()

# fit to data
dtree.fit(X_train,y_train)

# get predictions
dtree_pred = dtree.predict(X_test)


# In[ ]:


#Import scikit-learn metrics module for accuracy calculation
from sklearn.metrics import classification_report,confusion_matrix


# In[ ]:


print(classification_report(y_test,dtree_pred))


# In[ ]:


print("Confusion matrix:\n",confusion_matrix(y_test, dtree_pred))


# ## Random Forest
# We got 73% accuracy with a single decision tree, let's see if we can improve that with a random forest classifier

# In[ ]:


#Import Random Forest Model
from sklearn.ensemble import RandomForestClassifier

#Create a Gaussian Classifier
rfc = RandomForestClassifier(n_estimators=100,verbose=5)

#Train the model using the training sets y_pred=clf.predict(X_test)
rfc.fit(X_train,y_train)

rfc_pred = rfc.predict(X_test)


# In[ ]:


print(classification_report(y_test,rfc_pred))


# In[ ]:


# Model Accuracy, how often is the classifier correct?
print("Confusion matrix:\n",confusion_matrix(y_test, rfc_pred))


# ## Logistic Regression
# Lastly I'll try a logistic regression model.

# In[ ]:


from sklearn.linear_model import LogisticRegression

# Instantiate model
logmodel = LogisticRegression(max_iter=1000)

# Train model
logmodel.fit(X_train,y_train)

# Get predictions
log_pred = logmodel.predict(X_test)


# In[ ]:


print(classification_report(y_test,log_pred))


# In[ ]:


print("Confusion matrix:\n",confusion_matrix(y_test, log_pred))


# # Conclusion
# * Logistic regression produced the predictions with the highest accuracy, at 79%, though only slightly higher than the random forest, which scored 78%. The single decision tree performed worst at 73%.
# * Further investigation is needed to determine why it outperformed the decision tree-based models.

# # References
# [1] Wikipedia List of verified oldest people: https://en.wikipedia.org/wiki/List_of_the_verified_oldest_people
