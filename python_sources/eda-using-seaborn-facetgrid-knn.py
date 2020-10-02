#!/usr/bin/env python
# coding: utf-8

# ![image.png](attachment:image.png)

# **Campus placement or campus recruiting is a program conducted within universities or other educational institutions to provide jobs to students nearing completion of their studies. In this type of program, the educational institutions partner with corporations who wish to recruit from the student population. Today the competition is at another level and now companies uses academic score as the first step to screen the students. Therefore, it becomes important for a job seeking student to maintain his academic in order to clear the screening. In this notebook different analysis techniques have been used to conclusively say which factor is important for the placement.**

# In[ ]:


#important modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sea
import plotly.express as px
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df = pd.read_csv('../input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')


# In[ ]:


df.head()


# In[ ]:


df.info()


# Dropping the serial number column .

# In[ ]:


df = df.drop(['sl_no'], axis = 1)


# There are some missing values in salary column. These null values are for those who did not get placed. Therefore, we can fill the null values in salary with numeric value 0.

# In[ ]:


df['salary'] = df['salary'].fillna(0)


# To make visualization more easier and conclusive let's convert some continuous features into categorical. Using percentage bands instead of percentage itself will be more effective. Below four new features have been created.

# In[ ]:


#let's convert percentage into percentage bands.
# 0 - Less than 60
# 1 - Between 60 and 80
# 2 - Between 80 and 100

df.loc[df['ssc_p'] <= 60, 'ssc_p_band'] = 0
df.loc[(df['ssc_p'] > 60) & (df['ssc_p'] <= 80), 'ssc_p_band'] = 1
df.loc[(df['ssc_p'] > 80) & (df['ssc_p'] <= 100), 'ssc_p_band'] = 2
df = df.drop(['ssc_p'], axis = 1)

df.loc[df['hsc_p'] <= 60, 'hsc_p_band'] = 0
df.loc[(df['hsc_p'] > 60) & (df['hsc_p'] <= 80), 'hsc_p_band'] = 1
df.loc[(df['hsc_p'] > 80) & (df['hsc_p'] <= 100), 'hsc_p_band'] = 2
df = df.drop(['hsc_p'], axis = 1)

df.loc[df['degree_p'] <= 60, 'degree_p_band'] = 0
df.loc[(df['degree_p'] > 60) & (df['degree_p'] <= 80), 'degree_p_band'] = 1
df.loc[(df['degree_p'] > 80) & (df['degree_p'] <= 100), 'degree_p_band'] = 2
df = df.drop(['degree_p'], axis = 1)

df.loc[df['mba_p'] <= 60, 'mba_p_band'] = 0
df.loc[(df['mba_p'] > 60) & (df['mba_p'] <= 80), 'mba_p_band'] = 1
df.loc[(df['mba_p'] > 80) & (df['mba_p'] <= 100), 'mba_p_band'] = 2
df = df.drop(['mba_p'], axis = 1)

df.loc[df['etest_p'] <= 60, 'etest_p_band'] = 0
df.loc[(df['etest_p'] > 60) & (df['etest_p'] <= 80), 'etest_p_band'] = 1
df.loc[(df['etest_p'] > 80) & (df['etest_p'] <= 100), 'etest_p_band'] = 2
df = df.drop(['etest_p'], axis = 1)


# In[ ]:


df.head()


# # Gender

# In[ ]:


x = df[(df['gender'] == 'M') & (df['status'] == 'Placed')]
print('Total number of male who got placed = ',x.groupby(['gender'])['status'].count()[0])
print('Maximum salary offered to  male = ',x['salary'].max())
print('Minimum salary offered to  male = ',x['salary'].min())
print('Mean salary offered to  male = ',x['salary'].mean())
sea.kdeplot(x['salary'])

x = df[(df['gender'] == 'F') & (df['status'] == 'Placed')]
print('Total number of female who got placed = ',x.groupby(['gender'])['status'].count()[0])
print('Maximum salary offered to  female = ',x['salary'].max())
print('Minimum salary offered to  female = ',x['salary'].min())
print('Mean salary offered to  female = ',x['salary'].mean())
sea.kdeplot(x['salary'])
plt.legend(['Male', 'Female'])


# 1. Male got more placement offers than female.
# 2. Average salary offered to male is higher than offered to female.

# # SSC Board

# In[ ]:


print(df['ssc_b'].value_counts())
print(df.groupby(['ssc_p_band'])['salary'].mean())
grid = sea.FacetGrid(df, row = 'ssc_b', col = 'ssc_p_band')
grid.map(sea.countplot, 'status', order = ['Placed', 'Not Placed'], color = 'cyan')


# 1. High ssc percentage increases chances of getting placement.
# 2. Average salary offered to students with high ssc percentage is higher.

# # HSC Board

# In[ ]:


print(df['hsc_b'].value_counts())
print(df['hsc_s'].value_counts())
print(df.groupby(['hsc_p_band'])['salary'].mean())
print(df.groupby(['hsc_s'])['salary'].mean())
grid = sea.FacetGrid(df, row = 'hsc_b', col = 'hsc_p_band', hue = 'hsc_s')
grid.map(sea.countplot, 'status', order = ['Placed', 'Not Placed']).add_legend()


# 1. Science stream got more placement offers.
# 2. Average salary offered to students with high hsc percentage is higher.

# # Degree

# In[ ]:


print(df['degree_t'].value_counts())
print(df.groupby(['degree_p_band'])['salary'].mean())
print(df.groupby(['degree_p_band'])['salary'].mean())
fig, ax = plt.subplots(1,2, figsize = (10,4))
sea.countplot(df['degree_t'], hue = df['status'], ax = ax[0])
sea.countplot(df['degree_p_band'], hue = df['status'], ax = ax[1])
grid = sea.FacetGrid(df, col = 'degree_p_band', height = 4)
grid.map(plt.hist, 'salary', bins = 20, color = 'green')


# 1. Commerce and management field got more placement offers.
# 2. Science and technology field has high average salary.
# 3. Average salary offered to students with high degree percentage is higher.

# # Work Experience

# In[ ]:


print(df['workex'].value_counts())
print(df.groupby(['workex'])['salary'].mean())
sea.countplot('workex', hue = 'status', data = df)
grid = sea.FacetGrid(df, col = 'workex', height = 4)
grid.map(plt.hist, 'salary', bins = 20, color = 'red')


# 1. Having work experience is important as it incerase chances of getting placed and high salary

# # Specialisation
# 

# In[ ]:


print(df['specialisation'].value_counts())
print(df.groupby(['specialisation'])['salary'].mean())
sea.countplot('specialisation', hue = 'status', data = df)
grid = sea.FacetGrid(df, col = 'specialisation', height = 4)
grid.map(plt.hist, 'salary', bins = 20, color = 'orange')


# 1. Student with specialisation in marketing and finance is more likely to get placed and get high salary

# In[ ]:


print(df.groupby(['mba_p_band'])['salary'].mean())
print(df.groupby(['etest_p_band'])['salary'].mean())
fig, ax = plt.subplots(1,2, figsize = (10,4))
sea.countplot(df['mba_p_band'], hue = df['status'], ax = ax[0], palette = 'Set2')
sea.countplot(df['etest_p_band'], hue = df['status'], ax = ax[1], palette = 'Set3')


# 1. Students who scored good percentage in MBA and ETEST are more likely to get placed.

# # Final Conclusion :-
# 
# 1. Scoring high percentage in ssc board examination is important.
# 2. Students of science field in HSC board got more offers.
# 3. HSC percentage is not that important.
# 4. Scoring high degree percentage is important.
# 5. Having work experience increases chance of getting placed with high salary.
# 6. Marketing and Finance specialisation students got more placement offers.
# 7. Scoring high MBA percentage and ETEST percentage is important.
# 8. Board of SSC and HSC is not important.

# # Prediction

# In[ ]:


df.drop(['ssc_b','hsc_b', 'salary'], axis=1, inplace=True)
df["gender"] = df.gender.map({"M":0,"F":1})
df["workex"] = df.workex.map({"No":0, "Yes":1})
df["specialisation"] = df.specialisation.map({"Mkt&HR":0, "Mkt&Fin":1})
df["status"] = df.status.map({"Not Placed":0, "Placed":1})
df["hsc_s"] = df.hsc_s.map({"Commerce":0, "Science":1, "Arts" : 2})
df["degree_t"] = df.degree_t.map({"Sci&Tech":0, "Comm&Mgmt":1, "Others" : 2})


# In[ ]:


from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y = train_test_split(df.drop(['status'], axis = 1), df['status'],test_size = 0.2)
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics


# In[ ]:


error_rate = []
for i in range(1,40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(train_x,train_y)
    pred_i = knn.predict(test_x)
    error_rate.append(np.mean(pred_i != test_y))

plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', 
         marker='o',markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
print("Minimum error:-",min(error_rate),"at K =",error_rate.index(min(error_rate)))


# In[ ]:


acc = []
# Will take some time
from sklearn import metrics
for i in range(1,40):
    neigh = KNeighborsClassifier(n_neighbors = i).fit(train_x, train_y)
    yhat = neigh.predict(test_x)
    acc.append(metrics.accuracy_score(test_y, yhat))
    
plt.figure(figsize=(10,6))
plt.plot(range(1,40),acc,color = 'blue',linestyle='dashed', 
         marker='o',markerfacecolor='red', markersize=10)
plt.title('accuracy vs. K Value')
plt.xlabel('K')
plt.ylabel('Accuracy')
print("Maximum accuracy:-",max(acc),"at K =",acc.index(max(acc)))


# ![image.png](attachment:image.png)
