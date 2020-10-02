#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
#Linear Algebra
import numpy as np
#Data preprocessing
import pandas as pd

#setting display options
pd.set_option('display.max_rows', 5000)
pd.set_option('display.max_columns', 500)
pd.set_option('max_colwidth', 500)
np.set_printoptions(linewidth =400)

from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
#Advance-style plotting
import seaborn as sns
color =sns.color_palette()
sns.set_style('darkgrid')

#Ignore annoying warning from sklearn and seaborn
import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn

#other libraiaries
import os
import copy
from collections import defaultdict
from collections import Counter
from sklearn import metrics
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
import re
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)


# In[ ]:


description = pd.read_csv('/kaggle/input/widsdatathon2020/WiDS Datathon 2020 Dictionary.csv'); description


# In[ ]:


#read the data
train = pd.read_csv('/kaggle/input/widsdatathon2020/training_v2.csv')
test  = pd.read_csv('/kaggle/input/widsdatathon2020/unlabeled.csv')


# UNIVARIATE ANALYSIS - it takes the data (both train and test), summarizes the data (both train and test) and finds patterns in the data
# for further analysis

# In[ ]:


train.columns


# In[ ]:


print(train.shape , test.shape)


# In[ ]:


#column 1 Unique identifier associated with a patient unit stay
print (train['encounter_id'].nunique() , test['encounter_id'].nunique())


# Comment/Hint : Since there the id's are unique across train and test data; they have no or little significant importance to our model; drop them to avoid noisy parameters when training your model.. but had it been the patient unit stay was diagnose for another ailment this might be a good feature 
# 
# summary: drop encounter id

# In[ ]:


#column 2 Unique identifier associated with a hospital
print (train['hospital_id'].nunique() , test['hospital_id'].nunique())


# #Comment: Since an Hospital id appear several times, definitely several patient has visited an hospital and might have been diagnosed of the disease that leads to hospital death. 
# 
# #Hint Generate new features fro these
# Question 1? how many hospital death was recorded when using an hospital with respect to some diseases like cirosis, aids
# Question 2? was hospital death common to an hospital id or not
# 
# #create a new feature (frequency of hospital_id)
# 
# test['hospital_idcount']=test['hospital_id'].map(test['hospital_id'].value_counts().to_dict())

# In[ ]:


#column 3 Unique identifier associated with a patient
print (train['patient_id'].nunique() , test['patient_id'].nunique())


# Comment/Hint: Same as above (encounter id); every patient appear once in the data
# 
# summary: drop patient id

# In[ ]:


#column 4
Yes = len(train[train.hospital_death ==1])
No = len(train[train.hospital_death ==0])
Total = len(train)
print ('There are imbalanace datset with a %i/%i ratio'%((No/Total*100), (Yes/Total*100)+1))


# In[ ]:


sns.catplot(x ='hospital_death', kind ='count',palette='pastel', data = train);


# #Comment: its an imbalance data set but its still normal to work with; because there should be a low probaility of hospital death in the real sense. Except in some certains season/country whereby high numbers of hospital death was recorded for a period of time of which such data point doesnt exist here...
# 
# #Hint1: oversampling/undersampling is not a good techniques for this task; kindly use a robust algorithm like xgboost, lightgbm etc
# #Hint2: also since there is a lot of missing data point... oversampling/undersampling is not a good techniques for this task

# In[ ]:


#columnn 5
train['age'].describe()


# #Comment/Hints: there is a maximum age of 89 and a minimum of 16 age; classifying age into categories will enhance our model peformance
# #a patient diagnose of a chronic disease around old age will have a high probability of dying irrespectively of the hospital death record
# #define a function to classifiy age: it will be a good feature
# 
# 
# #code
#    if x >= 15 and x <= 24:
#         return 'igen'
#     elif x >= 25 and x <= 54:
#       return 'Prime_working_Age'
#     elif x >= 55 and x <= 64:
#         return 'Mature_working_Age'
#     else:
#         return 'Elderly_working_Age'
# train['Age_category'] = train['age'].apply(age_category)
# x =train[['age','Age_category']]
# 
# 

# In[ ]:


#Hint: ensure all units of each columns are having relationship with respect to each other..

