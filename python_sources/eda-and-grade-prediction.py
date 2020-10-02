#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


fpmath = '/kaggle/input/student-alcohol-consumption/student-mat.csv'
fportugese = '/kaggle/input/student-alcohol-consumption/student-por.csv'


# * *school - Student's school (binary: 'GP' - Gabriel Pereira or 'MS' - Mousinho da Silveira)
# * *sex - Student's sex (binary: 'F' - female or 'M' - male)
# * #age - Student's age (numeric: from 15 to 22)
# * *address - Student's home address type (binary: 'U' - urban or 'R' - rural)
# * *famsize - Family size (binary: 'LE3' - less or equal to 3 or 'GT3' - greater than 3)
# * *Pstatus - Parent's cohabitation status (binary: 'T' - living together or 'A' - living apart)
# * #Medu - Mother's education (numeric: 0 - none, 1 - primary education (4th grade), 2 - 5th to 9th grade, 3 - secondary education, or 4 - higher education)
# * #Fedu - Father's education (numeric: 0 - none, 1 - primary education (4th grade), 2 - 5th to 9th grade, 3 - secondary education, or 4 - higher education)
# * @Mjob - Mother's job (nominal: 'teacher', 'health' care related, civil 'services' (e.g. administrative or police), 'at_home' or 'other')
# * @Fjob - Father's job (nominal: 'teacher', 'health' care related, civil 'services' (e.g. administrative or police), 'at_home' or 'other')
# * @reason - Reason to choose this school (nominal: close to 'home', school 'reputation', 'course' preference or 'other')
# * @guardian - Student's guardian (nominal: 'mother', 'father' or 'other')
# * #traveltime - Home to school travel time (numeric: 1 - <15 min., 2 - 15 to 30 min., 3 - 30 min. to 1 hour, or 4 - >1 hour)
# * #studytime - Weekly study time (numeric: 1 - <2 hours, 2 - 2 to 5 hours, 3 - 5 to 10 hours, or 4 - >10 hours)
# * #failures - Number of past class failures (numeric: n if 1<=n<3, else 4)
# * *schoolsup - Extra educational support (binary: yes or no)
# * *famsup - Family educational support (binary: yes or no)
# * *paid - Extra paid classes within the course subject (Math or Portuguese) (binary: yes or no)
# * *activities - Extra-curricular activities (binary: yes or no)
# * *nursery - Attended nursery school (binary: yes or no)
# * *higher - Wants to take higher education (binary: yes or no)
# * *internet - Internet access at home (binary: yes or no)
# * *romantic - With a romantic relationship (binary: yes or no)
# * #famrel - Quality of family relationships (numeric: from 1 - very bad to 5 - excellent)
# * #freetime - Free time after school (numeric: from 1 - very low to 5 - very high)
# * #goout - Going out with friends (numeric: from 1 - very low to 5 - very high)
# * #Dalc - Workday alcohol consumption (numeric: from 1 - very low to 5 - very high)
# * #Walc - Weekend alcohol consumption (numeric: from 1 - very low to 5 - very high)
# * #health - Current health status (numeric: from 1 - very bad to 5 - very good)
# * #absences - Number of school absences (numeric: from 0 to 93)
# * G1 - First period grade (numeric: from 0 to 20)
# * G2 - Second period grade (numeric: from 0 to 20)
# * G3 - Final grade (numeric: from 0 to 20, output target)

# In[ ]:


datamath = pd.read_csv(fpmath)
dataport = pd.read_csv(fportugese)


# In[ ]:


datamath.columns


# In[ ]:


binary_features = [
    
        'school', 'sex', 'address', 'famsize', 'Pstatus','schoolsup', 'famsup', 'paid', 
        'activities', 'nursery', 'higher', 'internet', 'romantic'
    
]

numeric_features = [
    
        'age', 'Medu', 'Fedu', 'traveltime', 'studytime','failures', 'famrel', 
        'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences'
    
]

nominal_features = [
    
        'Mjob', 'Fjob', 'reason', 'guardian'
    
]

y = ['G1', 'G2', 'G3']


# In[ ]:


datamath[binary_features]


# In[ ]:


datamath[numeric_features]


# In[ ]:


datamath[nominal_features]


# In[ ]:


datamath[y]


# In[ ]:


datamath.info()


# In[ ]:


#for col in datamath.columns:
 #   print(col+'    \t', len(datamath[col].unique()), ' categories')


# In[ ]:


# creating initial dataframe
#bridge_types = ('Arch','Beam','Truss','Cantilever','Tied Arch','Suspension','Cable')
#bridge_df = pd.DataFrame(bridge_types, columns=['Bridge_Types'])
# converting type of columns to 'category'
#bridge_df['Bridge_Types'] = bridge_df['Bridge_Types'].astype('category')
# Assigning numerical values and storing in another column
#bridge_df['Bridge_Types_Cat'] = bridge_df['Bridge_Types'].cat.codes
#bridge_df


# In[ ]:


adata = pd.read_csv(fpmath)


# In[ ]:


for col in adata[nominal_features]:
    adata[col] = adata[col].astype('category')
    # Assigning numerical values and storing in another column
    #data[col+'_coded'] = data[col].cat.codes


# In[ ]:


for col in adata[numeric_features]:
    adata[col] = adata[col].astype('int')


# In[ ]:


adata['age'] = adata['age'].astype('int')


# In[ ]:


adata.info()


# In[ ]:


datanew = pd.get_dummies(adata)


# In[ ]:


datanew


# In[ ]:


datanew.shape


# In[ ]:


X = [
    [
        'age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 'famrel',
        'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences', 'school_GP', 'school_MS', 'sex_F', 'sex_M', 'address_R',
        'address_U', 'famsize_GT3', 'famsize_LE3', 'Pstatus_A', 'Pstatus_T',
        'Mjob_at_home', 'Mjob_health', 'Mjob_other', 'Mjob_services',
        'Mjob_teacher', 'Fjob_at_home', 'Fjob_health', 'Fjob_other',
        'Fjob_services', 'Fjob_teacher', 'reason_course', 'reason_home',
        'reason_other', 'reason_reputation', 'guardian_father',
        'guardian_mother', 'guardian_other', 'schoolsup_no', 'schoolsup_yes',
        'famsup_no', 'famsup_yes', 'paid_no', 'paid_yes', 'activities_no',
        'activities_yes', 'nursery_no', 'nursery_yes', 'higher_no',
        'higher_yes', 'internet_no', 'internet_yes', 'romantic_no',
        'romantic_yes'
    ]
]
y = [['G1', 'G2', 'G3']]


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=10)
clf = clf.fit(X, y)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
#X = [[0, 0], [1, 1]]
#Y = [0, 1]
clf = RandomForestClassifier(n_estimators=150, max_features=7)
clf = clf.fit(X_train, y_train)


# In[ ]:


pd.DataFrame(clf.predict(X_test), columns=['G1','G2','G3'], index=X_test.index)
#help(pd.DataFrame)


# In[ ]:


corr = datanew.corr()
plt.figure(figsize=(100,100))
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, annot= True)


# In[ ]:


datanew.columns


# In[ ]:




