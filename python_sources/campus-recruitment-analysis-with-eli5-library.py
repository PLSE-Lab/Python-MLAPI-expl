#!/usr/bin/env python
# coding: utf-8

# **Welcome to my Analysis**
# 
# Thank you for visiting my Kernel. I am a newbie to this field and very much interested to learn.
# Do analyze my work and feedback is very much appreciated.
# 
# Do suggest different ways to improve so that I can learn the most.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


data = pd.read_csv("/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv")


# In[ ]:


data.head()


# In[ ]:


data.isnull().sum()


# In[ ]:


data1 = data.fillna(0)
data1


# In[ ]:


from sklearn.preprocessing import LabelEncoder

my_label_Encoder = LabelEncoder()

data1['status'] = my_label_Encoder.fit_transform(data1['status'])

data1.dtypes


# In[ ]:


# "1" represent Male
# "0" represent Female
data1['status'].value_counts()


# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

values = sns.countplot(data['status'],hue=data['gender'])



for total_count in values.patches:
    height = total_count.get_height()
    width=total_count.get_x()+total_count.get_width()/2.
    values.text(total_count.get_x()+total_count.get_width()/2.,height + 3,'{:1.2f}'.format(height),ha="center") 


    
total_students = data1['gender'].value_counts()
print("Number of male and female\n",total_students)


# In[ ]:


values = sns.countplot(data['status'],hue=data['degree_t'])

# The below code is to see the total values of each bar on the top
for total_count in values.patches:
    height = total_count.get_height()
    width=total_count.get_x()+total_count.get_width()/2.
    values.text(total_count.get_x()+total_count.get_width()/2.,height + 3,'{:1.2f}'.format(height),ha="center")
    
degrees = data1['degree_t'].value_counts()
print("Students with different degrees: \n",degrees)


# In[ ]:


values = sns.countplot(data['status'],hue=data['specialisation'])

# The below code is to see the total values of each bar on the top
for total_count in values.patches:
    height = total_count.get_height()
    width=total_count.get_x()+total_count.get_width()/2.
    values.text(total_count.get_x()+total_count.get_width()/2.,height + 3,'{:1.2f}'.format(height),ha="center") 
    
total_students_specialisation = data1['specialisation'].value_counts()
print("Number of students in each specialisation\n",total_students_specialisation)


# In[ ]:


values = sns.countplot(data['status'],hue=data['hsc_b'])

# The below code is to see the total values of each bar on the top
for total_count in values.patches:
    height = total_count.get_height()
    width=total_count.get_x()+total_count.get_width()/2.
    values.text(total_count.get_x()+total_count.get_width()/2.,height + 3,'{:1.2f}'.format(height),ha="center") 

hsc_board = data1['hsc_b'].value_counts()
print("Number of students in different hsc boards\n",hsc_board)


# In[ ]:


sns.regplot(x='ssc_p',y='hsc_p',data=data1)

print("We can see the students perfomed well in ssc also performed well in hsc as well and students who did not perform well in ssc did not perform well in hsc but some performed well in hsc")


# In[ ]:


sns.regplot(x='hsc_p',y='degree_p',data=data1)


# In[ ]:


sns.regplot(x='degree_p',y='mba_p',data=data1)


# In[ ]:


sns.catplot(x="status", y="ssc_p", data=data,kind="swarm",hue='gender')
sns.catplot(x="status", y="hsc_p", data=data,kind="swarm",hue='gender')
sns.catplot(x="status", y="degree_p", data=data,kind="swarm",hue='gender')
sns.catplot(x="status", y="mba_p", data=data,kind="swarm",hue='gender')


# We can see that the people below 60% are mostly not placed.

# In[ ]:


sns.catplot(x="status", y="etest_p", data=data,kind="swarm",hue='gender')


# In[ ]:


data1.head()


# In[ ]:


y = data1['status']
df = data1.drop(['salary','gender','degree_t','hsc_s','hsc_b','ssc_b','specialisation','workex','status'],axis=1)
x = df


# In[ ]:


from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import accuracy_score as acc_score
from sklearn.ensemble import RandomForestClassifier as rfc

X_train,X_test,y_train,y_test = tts(x,y,test_size=0.33)
model = rfc()
model.fit(X_train,y_train)
Z = model.predict(X_test)
print (acc_score(y_test,Z))


# So, This is the final accuracy score I got.
# 
# From this data analysis,
# 
# We see students who got less percentage in ssc, continued the same in hsc and some in degree too.
# But very little students who got less percentage in ssc did best in hsc and degree.
# 
#  1. The "MKT&Fin" students with degree background "Comm & Mgmt" students got more placed. The students with degree backgound "Sci & tech " also perfomed well in placements with better placed percentage than Comm & Mgmt.
# 
#  2. Although, The analysis also told us that ssc% and hsc% are also important for strong foundation. There are more students less than 60% in hsc_p and ssc_p who did not get placed. As we see there are only little students who performed well in etest_p and mba_p did not get placed.
#  
#  3. The corporate intrested to hire students who preferred MKT&Fin specialisation though this specialisation has more students.
# 
# 

# In[ ]:


import eli5
from eli5.sklearn import PermutationImportance
feature_names = [i for i in data.columns if data[i].dtype in [np.int64]]
perm = PermutationImportance(model, random_state=1).fit(X_test, y_test)
eli5.show_weights(perm, feature_names = X_test.columns.tolist())


# [edit]
# This concept is permutation importance
# The values towards the top are the most important features, and those towards the bottom matter least.

# 
