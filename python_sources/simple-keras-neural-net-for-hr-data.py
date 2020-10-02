#!/usr/bin/env python
# coding: utf-8

# In my first notebook: https://www.kaggle.com/dsloet/first-data-exploration-hr-data/editnb I did a data exploration of the HR core data set. This notebook however, is going to try and predict the chance someone in the organisation gets a higher performance score than 'Fully Meets'. So therefore it becomes a classification issue because the classification will be between fully meets or below (needs improvement/ PIP) and exceeds or higher.
# 
# I use a very basic Keras neural net than proves to be very effective (90%) on the full dataset. 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Dense
import numpy
numpy.random.seed(7)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv('../input/core_dataset.csv')


# Based on my previous notebook I already know a lot about this data and I will clean up the data in the following lines.

# In[ ]:


df['Date of Termination'] = df['Date of Termination'].fillna("None")


# In[ ]:


df = df[df.Position.notnull()]


# In[ ]:


HispLat_map = {'Yes': 1, 'yes': 1, 'No': 0, 'no': 0}
df['Hispanic/Latino'] = df['Hispanic/Latino'].replace(HispLat_map)


# In[ ]:


MaritalDesc_map = {'Divorced': 0, 'Married': 1, 'Separated': 2,'Single': 3, 'widowed': 4}
df['MaritalDesc'] = df['MaritalDesc'].replace(MaritalDesc_map)


# In[ ]:


PerformanceScore_map = {'N/A- too early to review': 2,
                        'Needs Improvement': 1,
                        'Fully Meets': 2,
                        '90-day meets': 2,
                        'Exceeds': 3,
                        'Exceptional': 4,
                        'PIP': 1}
df['Performance Score'] = df['Performance Score'].replace(PerformanceScore_map)


# In[ ]:


Sex_map = {'Male': 0,
           'male': 0,
           'Female': 1,
           'female': 1}
df['Sex'] = df['Sex'].replace(Sex_map)


# In[ ]:


RaceDesc_map = {'American Indian or Alaska Native': 0,
                'Asian': 1,
                'Black or African American': 2,
                'Hispanic': 3,
                'Two or more races': 4,
                'White': 5}
df['RaceDesc'] = df['RaceDesc'].replace(RaceDesc_map)


# In[ ]:


CitizenDesc_map = {'Eligible NonCitizen': 0,
                   'Non-Citizen': 0,
                   'US Citizen': 1}
df['CitizenDesc'] = df['CitizenDesc'].replace(CitizenDesc_map)


# In[ ]:


EmploymentStatus_map = {'Active': 0,
                        'Future Start': 1,
                        'Leave of Absence': 2,
                        'Terminated for Cause': 3,
                        'Voluntarily Terminated': 4}
df['Employment Status'] = df['Employment Status'].replace(EmploymentStatus_map)


# In[ ]:


Department_map = {'Admin Offices': 0,
                  'Executive Office': 1,
                  'IT/IS': 2,
                  'Production       ': 3,
                  'Sales': 4,
                  'Software Engineering': 5,
                  'Software Engineering     ': 5}
df['Department'] = df['Department'].replace(Department_map)


# In[ ]:


del df['DOB']
del df['Date of Hire']


# In[ ]:


le = preprocessing.LabelEncoder()


# In[ ]:


le.fit(df['State'])
State1 = pd.DataFrame({'State1': le.transform(df['State'])})
df = pd.concat([df, State1], axis=1)
del df['State']


# In[ ]:


le.fit(df['Position'])
Position1 = pd.DataFrame({'Position1': le.transform(df['Position'])})
df = pd.concat([df, Position1], axis=1)
del df['Position']


# In[ ]:


le.fit(df['Manager Name'])
ManagerName1 = pd.DataFrame({'ManagerName1': le.transform(df['Manager Name'])})
df = pd.concat([df, ManagerName1], axis=1)
del df['Manager Name']


# In[ ]:


le.fit(df['Employee Source'])
EmployeeSource1 = pd.DataFrame({'EmployeeSource1': le.transform(df['Employee Source'])})
df = pd.concat([df, EmployeeSource1], axis=1)
del df['Employee Source']


# In[ ]:


le.fit(df['Reason For Term'])
ReasonForTerm1 = pd.DataFrame({'ReasonForTerm1': le.transform(df['Reason For Term'])})
df = pd.concat([df, ReasonForTerm1], axis=1)
del df['Reason For Term']


# In[ ]:


del df['Employee Name']


# In[ ]:


del df['Date of Termination']
del df['Zip']


# In[ ]:


Performance1 = pd.DataFrame({'Performance1': df['Performance Score']})
df = pd.concat([df, Performance1], axis=1)
del df['Performance Score']


# Now the Performance scores will become a binary classification feature where the performance scores 1 and 2 become a zero and performance scores 3 and 4 become ones.

# In[ ]:


PerfHigh = pd.DataFrame({'PerfHigh': df['Performance1']})
df = pd.concat([df, PerfHigh], axis=1)
PerfHigh_map = {2: 0,
                  1: 0,
                  3: 1,
                  4:1}
df['PerfHigh'] = df['PerfHigh'].replace(PerfHigh_map)


# In[ ]:


df.head()


# Lastly deleting some features we won't use.

# In[ ]:


del df['Performance1']


# In[ ]:


del df['Employee Number']


# In[ ]:


del df['Hispanic/Latino']


# In[ ]:


df.head()


# Checking the shape

# In[ ]:


df.shape


# In[ ]:


df.info()


# In[ ]:


df.to_csv('dataset.csv', header=0)


# This can be done better I suppose, but I don't know how...

# In[ ]:


dataset = numpy.genfromtxt("dataset.csv", delimiter=',', skip_header=1)


# In[ ]:


dataset.shape


# In[ ]:


X = dataset[:,0:14]
Y = dataset[:,14]


# In[ ]:


X.shape


# In[ ]:


model = Sequential()
model.add(Dense(12, input_dim=14, activation='relu'))
model.add(Dense(14, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


# In[ ]:



model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[ ]:


model.fit(X, Y, epochs=450, batch_size=15, verbose=0)


# In[ ]:


scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


# In[ ]:


model.summary()


# In[ ]:


model.get_config()


# In[ ]:


from pydot import graphviz
from keras.utils import plot_model
plot_model(model, to_file='model.png')


# And there you have it. With new data we should now be able to predict ones Performance Score.
