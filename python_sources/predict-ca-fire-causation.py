#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import datetime as dt
import sqlite3
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

conn = sqlite3.connect('../input/FPA_FOD_20170508.sqlite')
c = conn.cursor()

c.execute("""SELECT FIRE_Size, FIRE_SIZE_CLASS, DISCOVERY_DATE, 
DISCOVERY_TIME, CONT_DATE, CONT_TIME, STAT_CAUSE_DESCR, FIPS_NAME,
LATITUDE, LONGITUDE
from Fires
where STATE = 'CA'""")

#return one row of data
#for row in c.fetchone():
    #print row
    
column_names = ['fire_size', 'fire_size_class', 
                'discovery_date', 'discovery_time', 
                'cont_date', 'cont_time', 
                'stat_cause_descr', 'county_name', 
                'latitude', 'longitude']

raw_data = c.fetchall()
#type(raw_data) #returns a list
data_ar = np.array(raw_data) #turns list into array
df = pd.DataFrame(data_ar, columns = [column_names]) # turns array into dataframe






#DATAFRAME PREP WORKING WITH DATES
#Convert discovery date from julian to standard date
df['disc_clean_date'] = pd.to_datetime(df['discovery_date'] - pd.Timestamp(0).to_julian_date(), unit='D')
#Convert containment date from julian to standard date
df['cont_clean_date'] = pd.to_datetime(df['cont_date'] - pd.Timestamp(0).to_julian_date(), unit='D')
#Day of month string
df['discovery_month'] = df['disc_clean_date'].dt.strftime('%b')
#Returns the weekday string
df['discovery_weekday'] = df['disc_clean_date'].dt.strftime('%a')
#Merge discovery date and time 
df['disc_date_final'] = pd.to_datetime(df.disc_clean_date.astype('str') + ' ' + df.discovery_time, errors='coerce')
#Merge containment date and time 
#"Unknown string Error". Need to find a way locate bad strings instead of simply deleting them
df['cont_date_final'] = pd.to_datetime(df.cont_clean_date.astype('str') + ' ' + df.cont_time, errors='coerce')

#Only keep rows that have valid containment times
#df = df[df['cont_time'].between('0000', '2400', inclusive=True)]






#FEATURE CREATION
#Create Boolean column for Arson Yes/No
df['is_arson'] =  df['stat_cause_descr'].map(lambda x: True if x =='Arson' else False)

#Create column for stat_cause_descr_grp categories
stat_cause_descr_grp = []
for row in df.stat_cause_descr:
    if row in ['Lightning', 'Debris Burning', 'Campfire','Equipment Use',
              'Children', 'Smoking', 'Railroad', 'Fireworks', 'Structure',
              'Powerline']:
        stat_cause_descr_grp.append('accidental')
    elif row in ['Arson']: 
        stat_cause_descr_grp.append('crime')
    elif row in ['Miscellaneous', 'Missing/Undefined']:
        stat_cause_descr_grp.append('other')
    else:
        stat_cause_descr_grp.append('n_a')
df['stat_cause_descr_grp'] = stat_cause_descr_grp

#Create column for time_of_day_grp categories
time_of_day_grp = []
df['discovery_time'] = pd.to_numeric(df.discovery_time, errors='coerce')
#df['discovery_time'] = df.discovery_time.astype(dtype='int32', errors='ignore')
for row in df.discovery_time:
    if row >= 0 and row < 400:
        time_of_day_grp.append('early_morning')
    elif row >= 400 and  row < 800:
        time_of_day_grp.append('mid_morning')
    elif row >= 800 and  row < 1200:
        time_of_day_grp.append('late_morning')
    elif row >=1200 and  row <1600:
        time_of_day_grp.append('afternoon')
    elif row >=1600 and  row <2000:
        time_of_day_grp.append('evening')
    elif row >=2000 and  row <2400:
        time_of_day_grp.append('night')
    else:
        time_of_day_grp.append('n_a')
df['time_of_day_grp'] = time_of_day_grp

#Number of hours to containment
#Not sure how to return hour and min as float
df['time_to_cont'] = (df.cont_date_final - df.disc_date_final).astype('timedelta64[m]')
#Fill NaN with mean values instead of deleting rows
df['time_to_cont'] = df.time_to_cont.fillna(df.time_to_cont.mean())









#DROP UNEEDED COLUMNS
df.drop(['discovery_date', 'discovery_time', 'disc_clean_date', 'disc_date_final',
        'cont_date', 'cont_time', 'cont_clean_date', 'cont_date_final'], axis=1, inplace=True)
#df = df[df['cont_time'].between('0000', '2400', inclusive=True)]

#TYPE CONVERSIONS
df['fire_size'] = df.fire_size.astype(float)



#DATA EXPLORATION
sns.set(style="darkgrid", palette="Set3")

#Count distribution by Object variable
dist_fig1 = sns.countplot(y='stat_cause_descr', data=df,
                     order=df['stat_cause_descr'].value_counts(normalize=True).index)

#Distribution by Cause Grp
dist_fig2 = sns.countplot(x='stat_cause_descr_grp', data=df,
                     order=df['stat_cause_descr_grp'].value_counts(normalize=True).index)
#Distribution by Weekday
dist_fig3 = sns.countplot(x='discovery_weekday', data=df,
                     order=['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'])

dist_fig3 = sns.countplot(x='discovery_weekday', data=df, 
                     order=['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'])

sns,



sns.boxplot(x=df['fire_size'])
#Distribution by Month
dist_fig4 = sns.countplot(x='discovery_month', data=df, 
                          order=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul',
                           'Aug', 'Sep','Oct', 'Nov', 'Dec'])
#Distribution by State
#dist_fig5 = sns.countplot(x='state', data=df, order=df['state'].value_counts().index)


#Scattter Plot
#The smaller the fire the least amount of time it takes to contain
sns.regplot(x='fire_size', y='time_to_cont', data=df, scatter=True)

                         

#MACHINE LEANRING
#first we must take our potential categorical variables and turn them into numbers

df = pd.get_dummies(df, prefix='dm', prefix_sep='_', columns=['time_of_day_grp', 'discovery_month', 
                                                       'discovery_weekday'], drop_first=True)



features = ['fire_size', 'time_to_cont', 
            'dm_Mon', 'dm_Tue', 'dm_Wed',
           'dm_Thu', 'dm_Sat', 'dm_Sun']
target = 'is_arson'

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=0)

cls = RandomForestClassifier(n_estimators=50)
cls.fit(X_train, y_train)

print(cls.score(X_train,y_train))
print(cls.score(X_test,y_test))


#print ('Random forest Train: ', cross_val_score(cls, X_train, y_train, scoring='roc_auc', cv=5).mean())
#print 'Random forest Test: ', cross_val_score(cls, X_test, y_test, scoring='roc_auc', cv=5).mean()


logr = LogisticRegression()
logr.fit(X, y)
print 'Logistic regression: ', cross_val_score(logr, X_train, y_train, scoring='roc_auc', cv=5).mean()







#DATAFRAME CHECKS
df.head(5)
#df.shape
#df.isnull().sum()
#df.isnull().any()
#df.dtypes
df.columns
#df.index
#df.info()
#df.describe()

type(df.fire_size)

#type(df) #Ensure DF was created
#type(df.fire_name[9]) 
#type(df.fire_size[9]) #float
#type(df.time_to_cont[9]) #float
#type(df.fire_size_class[9])
#type(df.disc_clean_date[9])
#type(df.cont_clean_date[9])
#type(df.discovery_month[9])
#type(df.discovery_weekday[9])
#type(df.discovery_time[9]) #string
#type(df.cont_time[9]) #string
#type(df.stat_cause_descr[9])
#type(df.county_name[9])
#type(df.latitude[9]) #float
#type(df.longitude[9]) #float






# In[ ]:




