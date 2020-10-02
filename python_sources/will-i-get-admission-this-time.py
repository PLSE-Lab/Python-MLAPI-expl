#!/usr/bin/env python
# coding: utf-8

# This dataset is created for prediction of Graduate Admissions from an Indian perspective.
# 
# The dataset contains several parameters which are considered important during the application for Masters Programs.
# The parameters included are :
# 
# - GRE Scores ( out of 340 )
# - TOEFL Scores ( out of 120 )
# - University Rating ( out of 5 )
# - Statement of Purpose and Letter of Recommendation Strength ( out of 5 )
# - Undergraduate GPA ( out of 10 )
# - Research Experience ( either 0 or 1 )
# - Chance of Admit ( ranging from 0 to 1 )

# <H1><font color=red>If you like my work please consider upvoting.</H1>

# ## Importing libraries

# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import time
import pandas as pd
import warnings
import seaborn as sns
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.linear_model import Lasso,Ridge,BayesianRidge,ElasticNet,HuberRegressor,LinearRegression,LogisticRegression,SGDRegressor
from sklearn.metrics import mean_squared_error


# ## Importing data

# In[ ]:


admission = pd.read_csv("../input/graduate-admissions/Admission_Predict.csv")
admission.head()


# In[ ]:


#Utility function to make change of admission as Admitted if the probablity was greater than 0.5 and Rejected otherwise
def admitted_or_not(chance):
    if chance > 0.5:
        return "Admitted"
    else :
        return "Rejected"


# In[ ]:


admission.columns


# We can change the output based on the probablity. If the probablity is greater than 0.5 we change the value to admitted and if not we change the value to rejected.

# In[ ]:



admission['Chance of Admit Catagory'] = admission['Chance of Admit '].map(admitted_or_not)


# In[ ]:


admission.head()


# In[ ]:


admission.describe()


# We dont need serial number as it is simialar to roll number. Lets drop it.

# In[ ]:


admission.drop(['Serial No.'], axis=1, inplace=True)


# ## Exploratory data analysis
# ### Contribution of Gre score in selection

# In[ ]:


from matplotlib.pyplot import figure
figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
sns.distplot(admission['GRE Score'], bins=10)
plt.grid()


# - We can see that GRE score is distributed between 290 and 340

# In[ ]:


f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,5))
sns.distplot(admission[(admission['Chance of Admit Catagory'] == 'Admitted')]['GRE Score'], color='green', bins=5, ax=ax1)
sns.distplot(admission[(admission['Chance of Admit Catagory'] == 'Rejected')]['GRE Score'], color='red', bins=5, ax=ax2)
ax1.title.set_text("Selected")
ax2.title.set_text("Rejected")
ax1.grid()
ax2.grid()


# - Most of the selected students have an average score of 320 and rejected students have an average of 300

# In[ ]:


figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
sns.violinplot(y="GRE Score", x=admission['Chance of Admit Catagory'], data=admission)
plt.grid()


# - The 50th persentile value of GRE Score of selected candidates is more than 100th persentile of the rejected students

# In[ ]:


print('  0th persentile value of seleced candidates {}'.format(np.percentile(admission[(admission['Chance of Admit Catagory'] == 'Admitted')]['GRE Score'].values, 0)))
print(' 25th persentile value of seleced candidates {}'.format(np.percentile(admission[(admission['Chance of Admit Catagory'] == 'Admitted')]['GRE Score'].values, 25)))
print(' 50th persentile value of seleced candidates {}'.format(np.percentile(admission[(admission['Chance of Admit Catagory'] == 'Admitted')]['GRE Score'].values, 50)))
print(' 75th persentile value of seleced candidates {}'.format(np.percentile(admission[(admission['Chance of Admit Catagory'] == 'Admitted')]['GRE Score'].values, 75)))
print('100th persentile value of seleced candidates {}'.format(np.percentile(admission[(admission['Chance of Admit Catagory'] == 'Admitted')]['GRE Score'].values, 100)))


# In[ ]:


print('  0th persentile value of rejected candidates {}'.format(np.percentile(admission[(admission['Chance of Admit Catagory'] == 'Rejected')]['GRE Score'].values, 0)))
print(' 25th persentile value of rejected candidates {}'.format(np.percentile(admission[(admission['Chance of Admit Catagory'] == 'Rejected')]['GRE Score'].values, 25)))
print(' 50th persentile value of rejected candidates {}'.format(np.percentile(admission[(admission['Chance of Admit Catagory'] == 'Rejected')]['GRE Score'].values, 50)))
print(' 75th persentile value of rejected candidates {}'.format(np.percentile(admission[(admission['Chance of Admit Catagory'] == 'Rejected')]['GRE Score'].values, 75)))
print('100th persentile value of rejected candidates {}'.format(np.percentile(admission[(admission['Chance of Admit Catagory'] == 'Rejected')]['GRE Score'].values, 100)))


# In[ ]:


figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
sns.kdeplot(label='Admitted', data=admission[(admission['Chance of Admit Catagory'] == 'Admitted')]['GRE Score'], color='green', shade=True, legend=True)
sns.kdeplot(label='Rejected', data=admission[(admission['Chance of Admit Catagory'] == 'Rejected')]['GRE Score'], color='red', shade=True, legend=True)
plt.grid()


# - We cant tell the acceptence or rejection based on just the GRE score as the distributions are overlapping

# In[ ]:


sns.lmplot(x='GRE Score', y='Chance of Admit ', data=admission, hue='Chance of Admit Catagory', height=8, aspect=1)


# - There is a positive correlation between the GRE score and the chance of admission
# - Students with high GRE score have higher chance of admission

# ### Distribution of TOEFL Score

# In[ ]:


figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
sns.distplot(admission['TOEFL Score'], bins=10)
plt.grid()


# - Score ranges between 92 and 120 and most people have scored 110

# In[ ]:


f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,5))
sns.distplot(admission[(admission['Chance of Admit Catagory'] == 'Admitted')]['TOEFL Score'], color='green', bins=5, ax=ax1)
sns.distplot(admission[(admission['Chance of Admit Catagory'] == 'Rejected')]['TOEFL Score'], color='red', bins=5, ax=ax2)
ax1.title.set_text("Selected")
ax2.title.set_text("Rejected")
ax1.grid()
ax2.grid()


# - Most of the selected people have TOEFL score around 110 and most of the rejected students have a score of 98

# In[ ]:


figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
sns.violinplot(y="TOEFL Score", x=admission['Chance of Admit Catagory'], data=admission)
plt.grid()


# The 75th persentile of rejected students TOEFL score is less than the 25th persentile of accepted students persentile score

# In[ ]:


print('  0th persentile value of seleced candidates {}'.format(np.percentile(admission[(admission['Chance of Admit Catagory'] == 'Admitted')]['TOEFL Score'].values, 0)))
print(' 25th persentile value of seleced candidates {}'.format(np.percentile(admission[(admission['Chance of Admit Catagory'] == 'Admitted')]['TOEFL Score'].values, 25)))
print(' 50th persentile value of seleced candidates {}'.format(np.percentile(admission[(admission['Chance of Admit Catagory'] == 'Admitted')]['TOEFL Score'].values, 50)))
print(' 75th persentile value of seleced candidates {}'.format(np.percentile(admission[(admission['Chance of Admit Catagory'] == 'Admitted')]['TOEFL Score'].values, 75)))
print('100th persentile value of seleced candidates {}'.format(np.percentile(admission[(admission['Chance of Admit Catagory'] == 'Admitted')]['TOEFL Score'].values, 100)))


# In[ ]:


print('  0th persentile value of rejected candidates {}'.format(np.percentile(admission[(admission['Chance of Admit Catagory'] == 'Rejected')]['TOEFL Score'].values, 0)))
print(' 25th persentile value of rejected candidates {}'.format(np.percentile(admission[(admission['Chance of Admit Catagory'] == 'Rejected')]['TOEFL Score'].values, 25)))
print(' 50th persentile value of rejected candidates {}'.format(np.percentile(admission[(admission['Chance of Admit Catagory'] == 'Rejected')]['TOEFL Score'].values, 50)))
print(' 75th persentile value of rejected candidates {}'.format(np.percentile(admission[(admission['Chance of Admit Catagory'] == 'Rejected')]['TOEFL Score'].values, 75)))
print('100th persentile value of rejected candidates {}'.format(np.percentile(admission[(admission['Chance of Admit Catagory'] == 'Rejected')]['TOEFL Score'].values, 100)))


# In[ ]:


figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
sns.kdeplot(label='Admitted', data=admission[(admission['Chance of Admit Catagory'] == 'Admitted')]['TOEFL Score'], color='green', shade=True, legend=True)
sns.kdeplot(label='Rejected', data=admission[(admission['Chance of Admit Catagory'] == 'Rejected')]['TOEFL Score'], color='red', shade=True, legend=True)
plt.grid()


# - We cant tell the acceptence or rejection based on just the TOEFL score as the distributions are overlapping

# In[ ]:


sns.lmplot(x='TOEFL Score', y='Chance of Admit ', data=admission, hue='Chance of Admit Catagory', height=8, aspect=1)


# - On case of TOEFL scores also these is good correlation between the score and probablity of admission

# ### Distribution of University Rating

# In[ ]:


figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
sns.distplot(admission['University Rating'], bins=10)
plt.grid()


# - University rating is between 1 and 5

# In[ ]:


f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,5))
sns.distplot(admission[(admission['Chance of Admit Catagory'] == 'Admitted')]['University Rating'], color='green', ax=ax1, kde_kws={'bw': 1})
sns.distplot(admission[(admission['Chance of Admit Catagory'] == 'Rejected')]['University Rating'], color='red',  ax=ax2, kde_kws={'bw': 1})
ax1.title.set_text("Selected")
ax2.title.set_text("Rejected")
ax1.grid()
ax2.grid()


# - None of the rejected students have full score

# In[ ]:


f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,5))
sns.countplot(admission[(admission['Chance of Admit Catagory'] == 'Admitted')]['University Rating'], color='green', ax=ax1)
sns.countplot(admission[(admission['Chance of Admit Catagory'] == 'Rejected')]['University Rating'], color='red', ax=ax2)
ax1.title.set_text("Selected")
ax2.title.set_text("Rejected")
ax1.grid()
ax2.grid()


# - Both the selected and rejected students have scored the lowest mark.
# - Only the selected students have scored the maximum mark

# In[ ]:


figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
sns.violinplot(y="University Rating", x=admission['Chance of Admit Catagory'], data=admission)
plt.grid()


# - Most of the accepted students have a average score of 3 and rejected students have an avegare of 2.

# In[ ]:


print('  0th persentile value of seleced candidates {}'.format(np.percentile(admission[(admission['Chance of Admit Catagory'] == 'Admitted')]['University Rating'].values, 0)))
print(' 25th persentile value of seleced candidates {}'.format(np.percentile(admission[(admission['Chance of Admit Catagory'] == 'Admitted')]['University Rating'].values, 25)))
print(' 50th persentile value of seleced candidates {}'.format(np.percentile(admission[(admission['Chance of Admit Catagory'] == 'Admitted')]['University Rating'].values, 50)))
print(' 75th persentile value of seleced candidates {}'.format(np.percentile(admission[(admission['Chance of Admit Catagory'] == 'Admitted')]['University Rating'].values, 75)))
print('100th persentile value of seleced candidates {}'.format(np.percentile(admission[(admission['Chance of Admit Catagory'] == 'Admitted')]['University Rating'].values, 100)))


# In[ ]:


print('  0th persentile value of seleced candidates {}'.format(np.percentile(admission[(admission['Chance of Admit Catagory'] == 'Rejected')]['University Rating'].values, 0)))
print(' 25th persentile value of seleced candidates {}'.format(np.percentile(admission[(admission['Chance of Admit Catagory'] == 'Rejected')]['University Rating'].values, 25)))
print(' 50th persentile value of seleced candidates {}'.format(np.percentile(admission[(admission['Chance of Admit Catagory'] == 'Rejected')]['University Rating'].values, 50)))
print(' 75th persentile value of seleced candidates {}'.format(np.percentile(admission[(admission['Chance of Admit Catagory'] == 'Rejected')]['University Rating'].values, 75)))
print('100th persentile value of seleced candidates {}'.format(np.percentile(admission[(admission['Chance of Admit Catagory'] == 'Rejected')]['University Rating'].values, 100)))


# Most of the selected students have an average rating of 3 and rejected students have an average rating of 2

# In[ ]:


sns.lmplot(x='University Rating', y='Chance of Admit ', data=admission, hue='Chance of Admit Catagory', height=8, aspect=1)


# - There is a positive correlation for selected students and a slight negative correlation for rejected students university rating for selection

# ## Distribution of SOP

# In[ ]:


figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
sns.distplot(admission['SOP'], bins=10)
plt.grid()


# - SOP score varies between 1 and 5
# - The maximum score is around 4

# In[ ]:


f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,5))
sns.distplot(admission[(admission['Chance of Admit Catagory'] == 'Admitted')]['SOP'], color='green', bins=10, ax=ax1)
sns.distplot(admission[(admission['Chance of Admit Catagory'] == 'Rejected')]['SOP'], color='red', bins=10, ax=ax2)
ax1.title.set_text("Selected")
ax2.title.set_text("Rejected")
ax1.grid()
ax2.grid()


# - Most of the selected students scored between 3 and 4
# - Most of the rejected students scored between 1 and 2

# In[ ]:


f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,5))
sns.countplot(admission[(admission['Chance of Admit Catagory'] == 'Admitted')]['SOP'], color='green', ax=ax1)
sns.countplot(admission[(admission['Chance of Admit Catagory'] == 'Rejected')]['SOP'], color='red', ax=ax2)
ax1.title.set_text("Selected")
ax2.title.set_text("Rejected")
ax1.grid()
ax2.grid()


# - Some of the rejected students have full score
# - Some of the selected students have lowest score

# In[ ]:


figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
sns.violinplot(y="SOP", x=admission['Chance of Admit Catagory'], data=admission)
plt.grid()


# - The 75th persentile of rejected students SOP is less than  25th persentile of students accepted SOP. Looks like we have a clear distinction here.

# In[ ]:


figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
sns.kdeplot(label='Admitted', data=admission[(admission['Chance of Admit Catagory'] == 'Admitted')]['SOP'], color='green', shade=True, legend=True)
sns.kdeplot(label='Rejected', data=admission[(admission['Chance of Admit Catagory'] == 'Rejected')]['SOP'], color='red', shade=True, legend=True)
plt.grid()


# - We cant classify with the use of just SOP as the distribution plots are overlapping.

# In[ ]:


sns.lmplot(x='SOP', y='Chance of Admit ', data=admission, hue='Chance of Admit Catagory', height=8, aspect=1)


# - The observation is similar to University rating

# ### Distribution of LOR

# In[ ]:


figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
sns.distplot(admission['LOR '], bins=10)
plt.grid()


# - The value of LOR varies between 1 and 5
# - Most students scored near 3

# In[ ]:


f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,5))
sns.distplot(admission[(admission['Chance of Admit Catagory'] == 'Admitted')]['LOR '], color='green', bins=5, ax=ax1)
sns.distplot(admission[(admission['Chance of Admit Catagory'] == 'Rejected')]['LOR '], color='red', bins=5, ax=ax2)
ax1.title.set_text("Selected")
ax2.title.set_text("Rejected")
ax1.grid()
ax2.grid()


# In[ ]:


f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,5))
sns.countplot(admission[(admission['Chance of Admit Catagory'] == 'Admitted')]['LOR '], color='green', ax=ax1)
sns.countplot(admission[(admission['Chance of Admit Catagory'] == 'Rejected')]['LOR '], color='red', ax=ax2)
ax1.title.set_text("Selected")
ax2.title.set_text("Rejected")
ax1.grid()
ax2.grid()


# - The maximim LOR of the rejected student is 3.5 and most student have scored an average score of 2.
# - Some of the admitted students have scored full marks.

# In[ ]:


figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
sns.violinplot(y="LOR ", x=admission['Chance of Admit Catagory'], data=admission)
plt.grid()


# - The 75th persentile value of the rejected students is near thr 25th persentile of rejected students.

# In[ ]:


figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
sns.kdeplot(label='Admitted', data=admission[(admission['Chance of Admit Catagory'] == 'Admitted')]['LOR '], color='green', shade=True, legend=True)
sns.kdeplot(label='Rejected', data=admission[(admission['Chance of Admit Catagory'] == 'Rejected')]['LOR '], color='red', shade=True, legend=True)
plt.grid()


# - The distribution of LOR score is also overlapped.

# In[ ]:


sns.lmplot(x='LOR ', y='Chance of Admit ', data=admission, hue='Chance of Admit Catagory', height=8, aspect=1)


# ### Distribution of CGPA

# In[ ]:


figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
sns.distplot(admission['CGPA'], bins=10)
plt.grid()


# - The CGPA score is distributed between 7 and 10.
# - Most of the students have a score between 8 and 9

# In[ ]:


f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,5))
sns.distplot(admission[(admission['Chance of Admit Catagory'] == 'Admitted')]['CGPA'], color='green', bins=10, ax=ax1)
sns.distplot(admission[(admission['Chance of Admit Catagory'] == 'Rejected')]['CGPA'], color='red', bins=10, ax=ax2)
ax1.title.set_text("Selected")
ax2.title.set_text("Rejected")
ax1.grid()
ax2.grid()


# - The rejected students have a maximim CGPA of 8.5 and some of the admitted student have score near 10.
# - Only the rejected students have scored less than 7

# In[ ]:


figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
sns.violinplot(y="CGPA", x=admission['Chance of Admit Catagory'], data=admission)
plt.grid()


# - Based on the persentile values, we can see some good gap.

# In[ ]:


print('  0th persentile value of seleced candidates {}'.format(np.percentile(admission[(admission['Chance of Admit Catagory'] == 'Admitted')]['CGPA'].values, 0)))
print(' 25th persentile value of seleced candidates {}'.format(np.percentile(admission[(admission['Chance of Admit Catagory'] == 'Admitted')]['CGPA'].values, 25)))
print(' 50th persentile value of seleced candidates {}'.format(np.percentile(admission[(admission['Chance of Admit Catagory'] == 'Admitted')]['CGPA'].values, 50)))
print(' 75th persentile value of seleced candidates {}'.format(np.percentile(admission[(admission['Chance of Admit Catagory'] == 'Admitted')]['CGPA'].values, 75)))
print('100th persentile value of seleced candidates {}'.format(np.percentile(admission[(admission['Chance of Admit Catagory'] == 'Admitted')]['CGPA'].values, 100)))


# In[ ]:


print('  0th persentile value of rejected candidates {}'.format(np.percentile(admission[(admission['Chance of Admit Catagory'] == 'Rejected')]['CGPA'].values, 0)))
print(' 25th persentile value of rejected candidates {}'.format(np.percentile(admission[(admission['Chance of Admit Catagory'] == 'Rejected')]['CGPA'].values, 25)))
print(' 50th persentile value of rejected candidates {}'.format(np.percentile(admission[(admission['Chance of Admit Catagory'] == 'Rejected')]['CGPA'].values, 50)))
print(' 75th persentile value of rejected candidates {}'.format(np.percentile(admission[(admission['Chance of Admit Catagory'] == 'Rejected')]['CGPA'].values, 75)))
print('100th persentile value of rejected candidates {}'.format(np.percentile(admission[(admission['Chance of Admit Catagory'] == 'Rejected')]['CGPA'].values, 100)))


# In[ ]:


figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
sns.kdeplot(label='Admitted', data=admission[(admission['Chance of Admit Catagory'] == 'Admitted')]['CGPA'], color='green', shade=True, legend=True)
sns.kdeplot(label='Rejected', data=admission[(admission['Chance of Admit Catagory'] == 'Rejected')]['CGPA'], color='red', shade=True, legend=True)
plt.grid()


# - The graphs are some what overlapped

# In[ ]:


sns.lmplot(x='CGPA', y='Chance of Admit ', data=admission, hue='Chance of Admit Catagory', height=8, aspect=1)


# - CGPA seems to have a very good correlation with the chance of admission

# ### Distribution of Research score

# In[ ]:


figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
sns.distplot(admission['Research'], bins=10)
plt.grid()


# - There are only two possible values for the research score. 

# In[ ]:


figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
sns.countplot(x="Research", hue="Chance of Admit Catagory", data=admission)


# - Most of the students have the Research score as 1.
# - Students having 0 research score have got admission.

# In[ ]:


print('  0th persentile value of seleced candidates {}'.format(np.percentile(admission[(admission['Chance of Admit Catagory'] == 'Admitted')]['Research'].values, 0)))
print(' 25th persentile value of seleced candidates {}'.format(np.percentile(admission[(admission['Chance of Admit Catagory'] == 'Admitted')]['Research'].values, 25)))
print(' 50th persentile value of seleced candidates {}'.format(np.percentile(admission[(admission['Chance of Admit Catagory'] == 'Admitted')]['Research'].values, 50)))
print(' 75th persentile value of seleced candidates {}'.format(np.percentile(admission[(admission['Chance of Admit Catagory'] == 'Admitted')]['Research'].values, 75)))
print('100th persentile value of seleced candidates {}'.format(np.percentile(admission[(admission['Chance of Admit Catagory'] == 'Admitted')]['Research'].values, 100)))


# In[ ]:


print('  0th persentile value of rejected candidates {}'.format(np.percentile(admission[(admission['Chance of Admit Catagory'] == 'Rejected')]['Research'].values, 0)))
print(' 25th persentile value of rejected candidates {}'.format(np.percentile(admission[(admission['Chance of Admit Catagory'] == 'Rejected')]['Research'].values, 25)))
print(' 50th persentile value of rejected candidates {}'.format(np.percentile(admission[(admission['Chance of Admit Catagory'] == 'Rejected')]['Research'].values, 50)))
print(' 75th persentile value of rejected candidates {}'.format(np.percentile(admission[(admission['Chance of Admit Catagory'] == 'Rejected')]['Research'].values, 75)))
print('100th persentile value of rejected candidates {}'.format(np.percentile(admission[(admission['Chance of Admit Catagory'] == 'Rejected')]['Research'].values, 100)))


# ### Is there any relationship between GRE and TOEFL Score ?

# In[ ]:


sns.lmplot(x='GRE Score', y='TOEFL Score', data=admission, hue='Chance of Admit Catagory', height=8, aspect=1)


# - As expected there is a good correlation between the GRE and TOFEL score. This is related as both checks the language fluency.

# ### How is GRE Score related to CGPA ?

# In[ ]:


sns.lmplot(x='GRE Score', y='CGPA', data=admission, hue='Chance of Admit Catagory', height=8, aspect=1)


# - People having high CGPA have high GRE score. 

# ### How is TOFEL Score related to CGPA ?

# In[ ]:


sns.lmplot(x='TOEFL Score', y='CGPA', data=admission, hue='Chance of Admit Catagory', height=8, aspect=1)


# - Similar obseravation as seen with GRE Score. Bright students have good TOFEL Score

# ### Does University rating  has influence on CGPA ?

# In[ ]:


sns.lmplot(x='University Rating', y='CGPA', data=admission, hue='Chance of Admit Catagory', height=8, aspect=1)


# - People with high CGPA tend to have high University ratings.

# ### Does Letter of recommendation has influence on CGPA ?

# In[ ]:


sns.lmplot(x='LOR ', y='CGPA', data=admission, hue='Chance of Admit Catagory', height=8, aspect=1)


# - This is also as expected. Letter of recommendation strength is more for students with good CGPA.

# ### Which are the most important factors affecting the admission ?

# In[ ]:


correlation_matrix = admission.corr()
f, x = plt.subplots(figsize=(10,10))
dropSelf = np.zeros_like(correlation_matrix)
dropSelf[np.triu_indices_from(dropSelf)] = True
sns.heatmap(correlation_matrix, vmax=1, annot=True, mask=dropSelf)


# - The CGPA seems to have a very good correlation with the chance of admission
# - After CGPA, the GRE and TOFEL score seems to have importance in the chance of admission
# - Research have least significance

# ### Distribution of Chance of Admit

# In[ ]:


figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
sns.countplot(x='Chance of Admit Catagory', data=admission)


# In[ ]:


admission['Chance of Admit Catagory'].value_counts()


# Here the dataset is heavly imbalanced. There are 365 admitted case and 35 rejected cases. 

# ### How models perform with imbalanced data ?

# In[ ]:


X = admission.drop(['Chance of Admit ', 'Chance of Admit Catagory'], axis=1)
y = admission['Chance of Admit ']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.20, shuffle=False)


# In[ ]:


models = [['DecisionTree :',DecisionTreeRegressor()],
           ['Linear Regression :', LinearRegression()],
           ['RandomForest :',RandomForestRegressor()],
           ['KNeighbours :', KNeighborsRegressor(n_neighbors = 2)],
           ['SVM :', SVR()],
           ['AdaBoostClassifier :', AdaBoostRegressor()],
           ['GradientBoostingClassifier: ', GradientBoostingRegressor()],
           ['Xgboost: ', XGBRegressor()],
           ['CatBoost: ', CatBoostRegressor(logging_level='Silent')],
           ['Lasso: ', Lasso()],
           ['Ridge: ', Ridge()],
           ['BayesianRidge: ', BayesianRidge()],
           ['ElasticNet: ', ElasticNet()],
           ['HuberRegressor: ', HuberRegressor()]]

print("Results...")


for name,model in models:
    model = model
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print(name, (np.sqrt(mean_squared_error(y_test, predictions))))


# ### Which feature is most important ?

# In[ ]:


classifier = RandomForestRegressor()
classifier.fit(X,y)
feature_names = X.columns
importance_frame = pd.DataFrame()
importance_frame['Features'] = X.columns
importance_frame['Importance'] = classifier.feature_importances_
importance_frame = importance_frame.sort_values(by=['Importance'], ascending=True)


# In[ ]:


plt.barh([1,2,3,4,5,6,7], importance_frame['Importance'], align='center', alpha=0.5)
plt.yticks([1,2,3,4,5,6,7], importance_frame['Features'])
plt.xlabel('Importance')
plt.title('Feature Importances')
plt.show()


# ## Summary
# - The dataset is heavly unbalanced
# - All the features or columns have overlapped kde plots, which means its difficut to distingush the selection or rejection just with one of these values.
# - The CGPA seems to have a very good correlation with the chance of admission
# - After CGPA, the GRE and TOFEL score seems to have importance in the chance of admission
# - Research has least significance
# - The GRE, TOFEL, University Rating and LOR has positive correlation with CGPA. 
# - Students who have scored good in GRE have scored good marks in TOFEL
# 
