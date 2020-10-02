#!/usr/bin/env python
# coding: utf-8

# <h1>Introduction</h1>
# <h3>This data set has information about patients with heart disease. For this analysis, I want to find relationships between variables and visualize these insights if possible. This set contains data like age, sex, cholesterol, etc.
# I will go over the columns below.<h3>

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

#imports
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
import seaborn as sns
sns.set(color_codes=True)
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


#importing heart.csv as a DataFrame called, heart
heart = pd.read_csv('../input/heart.csv')


# <h2>The Table</h2>
# <h3>This table has 303 observations (rows) and 14 columns

# In[ ]:


heart.info()


# In[ ]:


heart.head()


# <font size=6>About the data</font>
# <h3>Columns Explained
# - age age in years
# - sex (1 = male; 0 = female)
# - cp chest pain type (4 values)
# - trestbps resting blood pressure (in mm Hg on admission to the hospital)
# - chol serum cholestoral in mg/dl
# - fbs (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)
# - restecg resting electrocardiographic results
# - thalach maximum heart rate achieved
# - exang exercise induced angina (1 = yes; 0 = no)
# - oldpeak ST depression induced by exercise relative to rest
# - slope the slope of the peak exercise ST segment
# - ca number of major vessels (0-3) colored by flourosopy
# - thal 3 = normal; 6 = fixed defect; 7 = reversable defect
# - target 1 or 0

# In[ ]:


#function to make labeling visualizations easier
def set_title_x_y(axis,title,x,y,size=10):
    """Function that sets title, x, and, y"""
    axis.set_title(title,size=size)
    axis.set_xlabel(x,size=size)
    axis.set_ylabel(y,size=size)


# <h1>Descriptive</h1>
# <h2>How many females and males are in this data set?</h2>

# In[ ]:


#Creating male and female dfs
female = heart[heart.sex==0]
male = heart[heart.sex==1]
len(male), len(female)


# <h2>**There are 207 Males and 96 Females**

# <h1>Age
# <h2>What is the age range among Male and Females?</h2>
# <h3>**For females, the youngest is 34, the oldest is 76, the average is 55.67, and the standard deviation is 9.4 years**
# <h3>**For males, the youngest is 29, the oldest is 77, the average is 53.75, and the standard deviation is 8.88 years **

# In[ ]:


fig,ax=plt.subplots(1,1,figsize=(12,8),constrained_layout=False)
#ax[0].hist(female.age,stacked=True)
ax.hist([female.age,male.age],stacked=False,color=['green','blue'],edgecolor='black', linewidth=1.5,bins=8)
ax.grid(axis='y')
ax.legend(['Female','Male'])
set_title_x_y(ax,'Men and Woman Heart Disease by Age','Age','Frequency',15)
print('Female'),print(female.age.describe(),'\n\nMale'), print(male.age.describe())


# <h1>Cholesterol</h1>
# <h2>What is condsiered high cholesterol?</h2>
# Desirable: Less than 200 mg/dL
# Borderline high: 200-240 mg/dL
# <br><b>High: Over 240 mg/dL</b>
# <br>I will look at how many males and females have over 240 mg/dL
# <br>With the following information I will conduct a Chi Square test to see if there is a relationship between <b>high serum cholesterol and sex.</b>

# In[ ]:


fig,ax = plt.subplots(1,2,figsize=(15,8))
ax[0].hist(male.chol);
ax[0].set_title('Male');
ax[1].hist(female.chol, color='green');
ax[1].set_title('Female');
for a in ax:
    a.set_xlabel('Cholesterol Level',size=14)


# <h2>Cholesterol and Sex(M/F)</h2>
# <h3>The graph above shows that females have higher cholesterol
# <h3>The  cell below prints how many Males and Females have a cholesterol over 240 mg/dL

# In[ ]:


heart.groupby('sex').chol.mean()
len(heart[heart.chol>240])
print('Cholesterol over 240 mg/dL\n')
print('Male')
print(((male.chol>240).value_counts()).sort_index(ascending=False))
print('\nFemale')
print(((female.chol>240).value_counts()))


# <h2>Question: Do Males and Females have have a statistically different Cholesterol level?

# In[ ]:


high_cholesterol = [57,94]
low_cholesterol = [39,113]
observed = np.array([high_cholesterol,low_cholesterol])
chi2, p, dof, expected = stats.chi2_contingency(observed, correction=False)
print('(Chi2 statistic: {}, probability: {}, degrees of freedom: {})'.format(chi2,p,dof))


# <h1>Analysis For Cholesterol and Sex</h1>
# <br>The number of participants with high cholesterol statistically differed by sex 
# <h3>X2 (1, N = 303)=5.12, p < .05</h3>
# <br>The sample included 303 people who suffered from heat disease, 207 who were male, 96 who were female.
# <br>The number of participants with **high cholesterol statistically differed by sex at significance level of a=.05**
#  <h2>**X2 (1, N = 303) = 5.12, p = .024**

# <h1>Maximum Heart Rate</h1>
# 
# <h2>The column 'thalach' records the maximum heart rate achieved by patients

# In[ ]:


plt.figure(figsize=(10,10));
heart.thalach.hist(),male.thalach.hist(),female.thalach.hist();
plt.legend(['All','Male','Female']);
plt.xlabel('Maximum Heart Rate', size=15);
plt.ylabel('Frequency',size=15);


# <h1>Taking a look at age</h1>
# <h3>I will be categorizing age into four groups to see what group contributes to most of the data and to see if there are any dependent variables that differ from one another significantly 

# In[ ]:


all_ages = np.array(((heart.age)).values)


# In[ ]:


heart['age_group'] = pd.cut(all_ages,4, labels=['28 - 40','41 - 52', '53 - 64', '65 - 76']);


# In[ ]:


plt.figure(figsize=(6,7));
((heart.age_group.value_counts()).sort_index()).plot(kind='bar');
plt.xticks(rotation=0);


# Categorizing ages into 4 groups, although these groups are not 

# In[ ]:


fig,ax = plt.subplots(1,3,figsize=(20,9))
heart_rate_age_group = ((heart.groupby('age_group').thalach.value_counts()))
blood_pressure_age_group = ((heart.groupby('age_group').trestbps.value_counts()))
cholesterol_age_group = ((heart.groupby('age_group').chol.value_counts()))
groups = [heart_rate_age_group,blood_pressure_age_group,cholesterol_age_group]
x_labels = ['Heart Rate', 'Blood Pressure', 'Cholesterol ']
for group in (heart.age_group.unique()).sort_values():
    ax[0].scatter(x=heart_rate_age_group.loc[group].index, y=heart_rate_age_group.loc[group].values,s=80)
    ax[1].scatter(x=blood_pressure_age_group.loc[group].index, y=blood_pressure_age_group.loc[group].values,s=80)
    ax[2].scatter(x=cholesterol_age_group.loc[group].index, y=cholesterol_age_group.loc[group].values,s=80,)
for axis, x_label in zip(ax, x_labels):
    axis.set_xlabel(x_label, size=14)
ax[0].set_ylabel('Frequency',size=14)
plt.legend(heart.age_group.unique().sort_values());
heart.age_group.value_counts()


# <h1>Cholesterol and Resting Blood Pressure</h1>
# <h3>Is there are relationship between the two?

# In[ ]:


gs = gridspec.GridSpec(5,5)
plt.figure(figsize=(10,10));

x = heart.chol.values
y = heart.trestbps.values

ax = plt.subplot(gs[1:7,:4]);
ax_2 = plt.subplot(gs[1:7,4:7]);
ax_3 = plt.subplot(gs[:1,:4]);

#plots
ax.scatter(x,y);
ax_2.hist(y, orientation='horizontal',bins=10);
ax_3.hist(x,bins=10);

ax_2.tick_params(axis='y',which='both',left=False,labelleft=False,labelright=True);
ax_3.tick_params(axis='x',bottom=False,which='both',labelbottom=False, labeltop=True);

ax.set_xlabel('Serum Cholesterol Level', size=14);
ax.set_ylabel('Resting Blood Pressure', size=14);

ax.grid(True)
ax_2.grid(axis='y')
ax_3.grid(axis='x')
x.mean(), y.mean()

slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
plot = [((i*slope)+intercept)for i in range(600)]
ax.plot(plot,color='green');
print('The slope of the regression is {}, The line intercepts the y-axis at {}'.format(slope,intercept))


# <h2>Creating a visualization with Matplotlib and Gridspec (above) vs Seaborn (below)

# In[ ]:


sns.jointplot(x=x, y=y, kind="reg", height=10,);
plt.xlabel('Serum Cholesterol Level',size=15);
plt.ylabel('Resting Blood Pressure', size=15);
plt.axhline(y.mean());
plt.axvline(x.mean());


# In[ ]:


# A graph like above, but in its own way
# fig,ax = plt.subplots(1,1,figsize=(10,10))
# #plt.figure(figsize=(10,10))
# plt.axhline(y.mean())
# plt.axvline(x.mean())
# sns.regplot(x,y);
# plt.grid(True)
# set_title_x_y(ax,'','Serum Cholesterol Level','Resting Blood Pressure',15)


# In[ ]:


heart[heart.fbs==1].head()


# <h1>Blood Sugar
# <h2>45 people in this data set have a blood sugar over 120 mg/dl, this accounts for about 14.85% of people.

# In[ ]:


#how many people in this data set have a fasting blood sugar over 120
len(heart[heart.fbs==1])/len(heart)


# <h1>Chest Pain
# <h2>Is there a relationship with having chest pain and '*variable*'?

# In[ ]:


fig,ax = plt.subplots(1,1,figsize=(8,8));
ax.hist(heart.cp,);
ax.set_xlabel('Chest Pain', size=14);


# In[ ]:


print('Age')
print(heart.groupby('cp').age.mean())
print('\nMaximum Heart Rate')
print(heart.groupby('cp').thalach.mean())
print('\nResting Blood Pressre')
print(heart.groupby('cp').trestbps.mean())
print('\nCholesterol')
print(heart.groupby('cp').chol.mean())


# <h3>From looking at the averages above, there is nothing that stands out.

# <h1>Conclusion</h1>
# <br><font size=3>From this data set I was able to pull descriptive insights and some inferential insights as well. Graphs are very useful tool when it comes to analyzing data as it can simplifiy insights and make digestable for a given audience. The two most notable findings in this analysis is the relationship between high cholesterol and sex(M/F) and the positive correlation between cholesterol and resting blood pressure.
#     

# In[ ]:




