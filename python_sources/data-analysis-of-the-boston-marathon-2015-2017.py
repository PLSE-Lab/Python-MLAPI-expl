#!/usr/bin/env python
# coding: utf-8

# # Data Analysis of the Boston Marathon 2015-2017

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import colorsys
import seaborn as sns
import warnings
from datetime import datetime
get_ipython().run_line_magic('matplotlib', 'inline')
warnings.simplefilter("ignore")


# # Dataset of the 2015-2017 Boston Marathon

# In[ ]:


Boston_15 = pd.read_csv('../input/marathon_results_2015.csv', sep=',')
Boston_16 = pd.read_csv('../input/marathon_results_2016.csv', sep=',')
Boston_17 = pd.read_csv('../input/marathon_results_2017.csv', sep=',')
# The total dataset:
Boston_15_to_17 = pd.concat([Boston_15, Boston_16, Boston_17], ignore_index=True, sort=False).set_index('Name')
Boston_15_to_17.head()


# In[ ]:


#Checking the existence of the null values in the dataset
Boston_15_to_17.isnull().sum(axis=0)


# In[ ]:


Boston_15_to_17.columns


# In[ ]:


Boston = Boston_15_to_17.drop(['Pace','Unnamed: 0','Bib', 'Gender','Unnamed: 9', 'Division', 'State', 'Citizen','Proj Time','City', 'Unnamed: 8','5K','15K', '10K', '25K', '20K', 'Half', '30K', '35K', '40K', 'Overall'], axis='columns')
Boston.head()


# In[ ]:


#Checking the existence of the null values in the dataset
Boston.isnull().sum(axis=0)


# In[ ]:


# Changing the str columns to time form 
Boston['Official Time'] = pd.to_timedelta(Boston['Official Time'])
# Transforming the time in minutes:
Boston['Official Time'] = Boston['Official Time'].astype('m8[m]').astype(np.int32)


# In[ ]:


Boston.info()


# In[ ]:


Boston.describe()


# In[ ]:


print('The oldest person finishing the Boston Marathon 2015-2017 was {} years old.\nThe youngest person was {} years old.'.format(Boston['Age'].max(), Boston['Age'].min()))


# # Distribution of finishers based on Ages

# In[ ]:


plt.figure(figsize=(8,6))
hage = sns.distplot(Boston.Age, color='g')
hage.set_xlabel('Ages',fontdict= {'size':14})
hage.set_ylabel(u'Distribution',fontdict= {'size':14})
hage.set_title(u'Distribution of finishers for Ages',fontsize=18)
plt.show()
warnings.simplefilter("ignore")


# The Boston Marathon finishers are mostly in the 35-50 age range.

# # Number of Finishers for Age

# In[ ]:


plt.figure(figsize=(20,10))
agecont = sns.countplot('Age',data=Boston, palette=sns.color_palette("RdPu", n_colors=len(Boston['Age'].value_counts())))
agecont.set_title('Ages Counting', fontsize=18)
agecont.set_xlabel('Ages', fontdict= {'size':16})
agecont.set_ylabel('Total of People', fontdict= {'size':16})
plt.show()


# # Gender Distribuition

# In[ ]:


plt.figure(figsize=(25,25))
d = sns.countplot(x='Age', hue='M/F', data=Boston, palette={'F':'r','M':'b'}, saturation=0.6)
d.set_title('Number of Finishers for Age and Gender', fontsize=25)
d.set_xlabel('Ages',fontdict={'size':20})
d.set_ylabel('Number of Finishers',fontdict={'size':20})
d.legend(fontsize=16)
plt.show()


# In[ ]:


plt.figure(figsize=(6,6))
l = Boston['M/F'].value_counts().index
plt.pie(Boston['M/F'].value_counts(), colors =['b','r'], startangle = 90, autopct='%.2f', textprops=dict(color="w"))
#plt.axes().set_aspect('equal','datalim')
plt.legend(l, loc='upper right')
plt.title("Gender",fontsize=18)
plt.show()


# In[ ]:


Boston_1 = Boston.copy()
bins = [17, 34, 39, 44, 49, 54, 59, 64, 69, 74, 79, 90]
Boston_1['Ranges'] = pd.cut(Boston_1['Age'],bins,labels=["18-34", "35-39", "40-44", "45-49", "50-54", "55-59", "60-64",
                                          "65-69", "70-74", "75-79", "< 80"]) 

Boston_2 = pd.crosstab(Boston_1.Ranges,Boston_1['M/F']).apply(lambda r: (r/r.sum()) * 100 , axis=1)

ax1 = Boston_2.plot(kind = "bar", stacked = True, color = ['r','b'], figsize=(9,6),
                      fontsize=12, position=0.5)
ax1.get_legend_handles_labels
ax1.legend(bbox_to_anchor = (1.3, 1))
ax1.set_xlabel('Age Ranges', fontdict={'size':14})
ax1.set_ylabel('Percentages (%)', fontdict={'size':14})
ax1.set_title('Gender Finishers x Age Ranges', fontsize=18)
plt.show()


# The plots above show a significant female participation in the Boston Marathon. In particular, we can observe that up to the age of 38 the female participation is greater than the male participation. After this age the scenario reverses and the male participation overcomes the female participation. Around the age of 35 is the age that many women are preparing to get pregnant, this decline in participation can perhaps be explained by this.

# In[ ]:


FM_mean = Boston.groupby('M/F').mean()
FM_mean


# In[ ]:


print('The average age of the female finishers in Boston is {:.2f} years old.'.format(FM_mean['Age'][0]))
print('The average age of the male finishers in Boston is {:.2f} years old.'.format(FM_mean['Age'][1]))
print('The average finishing time of the female finishers in Boston is {:.2f} hours.'.format(FM_mean['Official Time'][0] / 60))
print('The average finishing time of the male finishers in Boston is {:.2f} hours.'.format(FM_mean['Official Time'][1] / 60))


# # Age x Performance

# In[ ]:


plt.figure(figsize=(12,10))
Boston_copy = Boston.copy()
Boston_copy = Boston_copy[Boston_copy['Age'].isin(range(0,85))]

x = Boston_copy.Age
y = Boston_copy['Official Time']


plt.plot(x, y, '.')
plt.xlabel("Age", fontsize=16)
plt.ylabel("Official Time (min)",fontsize=16)
plt.title("Official Time for Age",fontsize=20)
plt.show()


# As we can see from the plot above, using the total datas of the dataset, Boston, it is difficult to find a relation between age and performance.
# 
# Then, we will consider the mean and median of the official time for each age.

# In[ ]:


# The mean of official time for the set of Age 
mean_age_time = Boston.groupby('Age').mean().set_index(np.arange(67))
mean_age_time['Age'] = mean_age_time.index 
mean_age_time.head()


# In[ ]:


# The median of official time for the set of Age 
median_age_time = Boston.groupby('Age').median().set_index(np.arange(67))
median_age_time['Age'] = median_age_time.index 
median_age_time.head()


# In[ ]:


# Plotting the results

plt.figure(figsize=(12,10))

x = mean_age_time['Age']
y = mean_age_time['Official Time']

plt.plot(x, y, '.')

xx = median_age_time['Age']
yy = median_age_time['Official Time']


plt.plot(xx, yy, '.', color = 'r')


plt.xlabel("Age", fontsize=16)
plt.ylabel("Official Time (min)",fontsize=16)
plt.title("Official Time for Age",fontsize=20)
plt.legend(['Mean', 'Median'])
plt.show()


# Therefore, we can observe that a polynomial approximation (such as a quadratic, cubic polinomials) seems to be the better way to relate Age and Official time. 

# # Predicting the Official Time

# In[ ]:


from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
warnings.simplefilter("ignore", category=FutureWarning)


# In[ ]:


# Defining the dependent and independent variables 
X = mean_age_time.drop(['Official Time'], axis=1)
Y = mean_age_time['Official Time']


# In[ ]:


# Separeting the dataset in training and test datasets:
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.33, random_state = 42)
print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)


# In[ ]:


# Constructing the model 
model = make_pipeline(PolynomialFeatures(degree=2), Ridge())
model


# In[ ]:


# Training the model with the training variables:
model.fit(X_train, Y_train)


# In[ ]:


# We show to the model the unknown test variables, in order to predict the results. 
# Then we can compare to the respective Y_test dependent variable, and check the error of the model. 
pred_test = model.predict(X_test)


# In[ ]:


# Plotting the error of the model:
plt.figure(figsize=(8,6))
#plt.scatter(pred_training, pred_training - Y_training, c = 'b', alpha = 0.5)
plt.scatter(pred_test,  pred_test - Y_test, c = 'g', alpha = 0.5)
plt.title(u"The Model Error Function", fontsize=15)
plt.show()


# Since the difference between the test data predicted by the model and the data that we expected to obtain are mostly close to zero, it can be said that the model considered revalues a good prediction of time by age of each athlete.
