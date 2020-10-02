#!/usr/bin/env python
# coding: utf-8

# 
# Data Dictionary
# PatientId - Identification of a patient
# AppointmentID - Identification of each appointment 
# Gender = Male or Female .
# AppointmentDay = The day of the actual appointment when they have to visit the doctor. 
# ScheduledDay = The day someone called to register for an appointment, this is before appointment date.
# Age = How old is the patient. 
# Neighbourhood = Where the appointment takes place. 
# Scholarship = (True/False) A benefit of a particular amount is given when the income for a household is below the poverty line.   
# Hipertension = True or False 
# Diabetes = True or False
# Alcoholism = True or False 
# Handicap = True or False 
# SMS_received = 1 or more messages sent to the patient.
# No-show = True or False.
# 
# Courtesy : https://www.kaggle.com/joniarroba/noshowappointments
# 
# This dataset consist of appointment records of 62,299 patients with 14 features among which few are potential factors for patients not showing up for their appointment. 
# 

# In[13]:


"""Importing required libraries"""

import pandas as pd
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


data = pd.read_csv('../input/KaggleV2-May-2016.csv')
data.head()


# **DATA PROCESSING**

# In[4]:


#Renaming Columns
data = data.rename(columns={'PatientId':'patient_id','AppointmentID':'appointment_id','Gender': 'sex', 'ScheduledDay': 'scheduled_day', 'AppointmentDay': 'appointment_day', 
                                    'Age': 'age', 'Neighbourhood': 'neighbourhood', 'Scholarship': 'scholarship', 
                                    'Hipertension': 'hypertension', 'Diabetes': 'diabetic', 'Alcoholism': 'alcoholic', 
                                    'Handcap': 'handicap', 'No-show': 'no_show'})


# In[5]:


#binarizing columns
data['no_show'] = data['no_show'].map({'No': 1, 'Yes': 0})
data['sex'] = data['sex'].map({'F': 0, 'M': 1})
data['handicap'] = data['handicap'].apply(lambda x: 2 if x > 2 else x)


# In[6]:


#Converting the AppointmentDay and ScheduledDay into a date and time format 
data['scheduled_day'] = pd.to_datetime(data['scheduled_day'], infer_datetime_format=True)
data['appointment_day'] = pd.to_datetime(data['appointment_day'], infer_datetime_format=True)


# In[7]:


data.describe()


# Upon executing data.describe(), we noticed that the minimum age is -1 and maximum age is 115. Though there are people who are living/lived past the age of 100, it is usually rare and not common. Thus, we decided to exclude these values. 
# 

# In[8]:


data.drop(data[data['age'] <= 0].index, inplace=True)
data.drop(data[data['age'] >100].index, inplace=True)


# The next column we added was neighbourhood_enc. From the data, we observed that there are around 80 neighbourhoods and it could potentially contribute to the patients not showing up at the doctor(for example: because of distance they had to travel to visit the doctor). We used LabelEncoder to encode neighbourhoods to numbers.
# 

# In[9]:


encoder_neighbourhood = LabelEncoder()
data['neighbourhood_enc'] = encoder_neighbourhood.fit_transform(data['neighbourhood'])


# Upon analyzing the features, we came to the conclusion that adding new features will help us in obtaining better results for predicting a patient not showing up.

# Thus, we added a new column waiting_time. For this, we calculated the number of days the patient had to wait to visit the doctor. After plotting a graph, we noticed there was a negative waiting time, which upon looking in the dataset we realized that there were incorrect entries. That is, we observed that ScheduledDay(the day when the appointment was booked) was after AppointmentDay(the day of the appointment), which we excluded.  
# 

# In[11]:


data['waiting_time'] = list(map(lambda x: x.days+1 , data['appointment_day'] - data['scheduled_day']))
data.drop(data[data['waiting_time'] <= -1].index, inplace=True)

"""We are adding the days of week column in order to find out the days when the patient is not likely to show up for the appointment. 
For example: A patient might not show up because it is a weekend."""
data['appointment_dayofWeek'] = data['appointment_day'].map(lambda x: x.dayofweek)


# We are finding out the probability of a person not likely to show up for an appointment. For this we calculated: (no.of times a patient didn't show up/total no. of appointments he/she booked) and added it as a risk_score.

# In[12]:


data['no_of_noshows'] = data.groupby('patient_id')[['no_show']].transform('sum')
data['total_appointment'] = data.groupby('patient_id')[['no_show']].transform('count')

data['risk_score'] = data.no_of_noshows / data.total_appointment


# **Plotting some features with no_show**

# In[14]:


sns.countplot(x='sex', hue='no_show', data=data, palette='RdBu')
plt.show();
sns.countplot(x='handicap', hue='no_show', data=data, palette='RdBu')
plt.show();
sns.countplot(x='alcoholic', hue='no_show', data=data, palette='RdBu')
plt.show();
sns.countplot(x='hypertension', hue='no_show', data=data, palette='RdBu')
plt.show();
sns.countplot(x='diabetic', hue='no_show', data=data, palette='RdBu')
plt.show();
sns.countplot(x='scholarship', hue='no_show', data=data, palette='RdBu')
plt.show();


# In[15]:


#Removing columns that are not necessary for prediction
data.drop(['scheduled_day','appointment_day','neighbourhood','patient_id','appointment_id'], axis=1, inplace=True)


# In[16]:


#Splitting the data into training and testing sets.
X = data.drop(['no_show'], axis=1)
y = data['no_show']
X.head()
y.head()
X_train, X_test, y_train, y_test = train_test_split(X, y)


# In[23]:


#Modeling : Random Forest
clf = RandomForestClassifier(n_estimators=300)
clf.fit(X_train, y_train)

#Performance Check
print("Mean Accuracy:")
print(clf.score(X_test, y_test))
print("Confusion Matrix:")
print(confusion_matrix(y_test, clf.predict(X_test)))
print("Classification Report:")
print(classification_report(y_test, clf.predict(X_test)))

#Plotting feature importance
feat_importances = pd.Series(clf.feature_importances_, index=X.columns)
feat_importances = feat_importances.nlargest(20)
feat_importances.plot(kind='barh')
plt.show()


# 

# 
