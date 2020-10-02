#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import random
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix


# In[ ]:


data = pd.read_csv('/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')


# In[ ]:


data.head()


# In[ ]:


data.shape


# In[ ]:


random.seed(42)


# # Data:
# 
# 215 college graduates with their grades from different stages of education and test, together with job placement outcomes.
# 
# ## Column meaning:
# 
# 	sl_no - Serial Number
#     
#     ssc_p - Secondary Education percentage- 10th Grade
#     ssc_b - Board of Education- Central/ Others
#     hsc_p - Higher Secondary Education percentage- 12th Grade  
#     hsc_b - Board of Education- Central/ Others
#     hsc_s - Specialization in Higher Secondary Education
#     degree_p - Degree Percentage
#     degree_t - Under Graduation(Degree type)- Field of degree education
#     workex - Work Experience
#     etest_p - Employability test percentage ( conducted by college)
#     specialisation - Post Graduation(MBA)- Specialization
#     mba_p - MBA percentage
#     status - Status of placement- Placed/Not placed
#     salary - Salary offered by corporate to candidates

# # GOAL:
#     examine gender gap
#     predict job placement after college
#     predict salary

# ## Explore columns, check missing values.
# We will remove outliers when we will build the models.

# In[ ]:


# Label numerical and categorical columns
num_col = ['ssc_p', 'hsc_p', 'degree_p', 'etest_p', 'mba_p', 'salary']
cat_col = ['gender', 'ssc_b', 'hsc_b', 'hsc_s', 'degree_t', 'workex', 
           'specialisation', 'status']


# In[ ]:


data.groupby('gender').sl_no.count()


# There are twice more males than females in the dataset. 

# In[ ]:


na_values = []
for column in data.columns:
    na_values.append(data[column].isna().sum())
print(na_values)


# In[ ]:


data[data.salary.isna()].status.unique()


# In[ ]:


data[~data.salary.isna()].status.unique()


# We observe that the only missing values in the data are in the salary column. They correspond exactly to the non-placed candidates. We can fill the salary as 0. This will also make it harder to predict salary, since the dataset of people with salary consists of roughly 150 entries, which might not be enough for good model.

# In[ ]:


data.salary.fillna(0, inplace=True)


# In[ ]:


data[cat_col].describe()


# All categorical data seems intact, without strange or unexpected categories.

# In[ ]:


data[num_col].describe(percentiles=[0.1, 0.5, 0.9])


# We see that there are some salary outliers (maximal is 940K, while 90% percentile is at 350K). The rest of the data seems balanced. (we can throw the outlier salaries later, when we train our model)

# ## Exploration of categorical columns

# In[ ]:


# We create a dataframe consisting only of placed workers, for which we will like to analyyze the salary.
placed = data[data.status=='Placed']


# In[ ]:


placed.salary.describe(percentiles=[0.25, 0.5, 0.75 ,0.95])


# In[ ]:


# Introduce a new column indicating that a person is in top quartile of the salary.
placed.eval('top_sal = salary>300000', inplace=True)


# Let us look at pairplot of the numerical values. The colors represent people with salary in top quartile, and the people in the bottom quartile.

# In[ ]:


sns.pairplot(placed[(placed.salary>300000) | (placed.salary<240000)] [['ssc_p', 'hsc_p', 'degree_p', 'etest_p', 'mba_p', 'top_sal']], hue='top_sal')
plt.show()


# In[ ]:


# Correlation coefficients with top_sal and salary
placed[['ssc_p', 'hsc_p', 'degree_p', 'etest_p', 'mba_p', 'top_sal', 'salary']].corr()[['top_sal','salary']]


# We can see that even having top 25% salary does not correlate well with any of the numerical values, and salary even more so. This can be especially seen on pairplot graphs, as cadidates with top 25% salaries have their grades all over the place. Out of all grades, mba_p and etest seem to correlate the most with higher salaries.
# 
# Our first conclusion:
# ## Grades do not seem to influence the salary.
# Hence, we will not pursue further the topic of salary and focus more on 

# # Gender gap and job placement.

# In[ ]:


sns.pairplot(placed[['ssc_p', 'hsc_p', 'degree_p', 'etest_p', 'mba_p', 'gender', 'salary']], hue = 'gender')


# Again, it does seem that the data is all over the place, and as expected. We can see that female grades are slightly better at ssc_p and degree_p and mba_p (which is supposed to correlate higher salaries), but the salary distribution is slightly worse for the women. This might seem counterintuitive, but may be explained by categorical values. Let us look at those.

# In[ ]:


data.eval('male = gender=="M" ', inplace=True)
data.eval('placed = status=="Placed" ', inplace=True)


# In[ ]:


data.male.mean()


# In[ ]:


data.groupby('workex')['male', 'salary', 'placed'].mean().round(2)


# We can see that men are more likely to have work experience, which might skew their salary, and partially explain why their average salary is higher.

# In[ ]:


data.groupby('degree_t')['male', 'salary', 'placed'].mean().round(2)


# In[ ]:


data.groupby('specialisation')['male', 'salary', 'placed'].mean().round(2)


# In[ ]:


data.groupby('hsc_b')['male', 'salary', 'placed'].mean().round(2)


# What we see, that men are 65% of the data, in almost all above case, their percentage in higher paid group is higher than that.
# 
# Conclusion:
# ## Men are more represented in areas with higher average salaries.

# We might be tempted to explain the gender gap by uneven representation. It is true that uneven representation does not help women, but let us control for some of those factors.

# In[ ]:


placed = placed[placed.salary<400000]  # remove outlier salaries


# In[ ]:


placed.groupby(['workex', 'gender'])['salary'].aggregate(['mean','median', 'count']).round(0)


# In[ ]:


placed.groupby(['specialisation', 'gender'])['salary'].aggregate(['mean','median', 'count']).round(0)


# In[ ]:


placed.groupby(['degree_t', 'gender'])['salary'].aggregate(['mean','median', 'count']).round(0)


# It seems like both mean and median for females is lower in almost all cases once we control for some categorical values, and also after we removed outliers with salaries above 400K (which were 90% men)

# In[ ]:


data[data.salary>400000].male.mean()


# Categories degree_t and hsc_s seem to be very close, so I only chose one of them that seems more relevant for employmeny, ssc_b and hsc_b seem to not make much difference, so I ignored them.
# 
# Here is a breakdown of salaries by categories, that shows again that women are paid on average less in most categories:

# In[ ]:


placed.groupby(['workex', 'degree_t', 'specialisation',  'gender'])['salary'].aggregate(['mean','median', 'count']).round(0)


# Let us look at the most common category. 

# In[ ]:


most_common = data[(data.hsc_s=='Commerce') & (data.degree_t =='Comm&Mgmt')
                   & (data.workex=='No') & (data.specialisation=='Mkt&Fin') & (data.placed==True)]


# In[ ]:


most_common[['ssc_p', 'hsc_p', 'degree_p',  'etest_p', 'mba_p',
             'salary', 'male']].corr()[['salary', 'male']]


# We see that now, salary is more corellated to grades (which makes sense), and while being male correlates strongly negatively with grades, it only slightly negatively correlates with salary. (males are likely to have lower grades but not lower salaries)

# In[ ]:


most_common[most_common.male==1][['ssc_p', 'hsc_p', 'degree_p',  'etest_p', 'mba_p',
             'salary']].corr().salary


# In[ ]:


most_common[most_common.male==0][['ssc_p', 'hsc_p', 'degree_p',  'etest_p', 'mba_p',
             'salary']].corr().salary


# We see that correlations with grades are much stronger for females than for males. Let us see this on pairplot.

# In[ ]:


sns.pairplot(most_common[['ssc_p', 'hsc_p', 'degree_p','etest_p', 'mba_p', 'salary', 'gender']], hue='gender')
plt.show()


# From the graphs in the last row, we can see that for people with same salary, females have higher ssc_p, hsc_p, degree_p and mba_p grades, while etest_p grades are mixed. 
# ## Grades matter much more for women to get higher salary (and seems like they need higher grades to get the same salary as men) 
# 
# Of course, our most_common dataset is very small, so this could be all noise, but here is one observation:
# 
# ## Women seem to outperform men in all the grades except for etest. This raises a question whether something in the design of employability test favors men?
# 

# # Job placement

# In[ ]:


sns.pairplot(data[['ssc_p', 'hsc_p', 'degree_p','etest_p', 'mba_p', 'status']], hue='status')
plt.show()


# We see that grades have good separation for job placement. etest_p and mba_p which were more correlated with higher salaries do not look to predict job placement well. 

# In[ ]:


f, axes = plt.subplots(1, 3, sharey=True, figsize=(17,6))

sns.boxplot(x='status', y='degree_p', data=data[data.degree_t=='Comm&Mgmt'], ax=axes[0])
sns.boxplot(x='status', y='degree_p', data=data[data.degree_t=='Sci&Tech'], ax=axes[1])
sns.boxplot(x='status', y='degree_p', data=data[data.degree_t=='Others'], ax=axes[2])
axes[0].title.set_text('Commerce & Management')
axes[1].title.set_text('Science & Technology')
axes[2].title.set_text('Others')
plt.show()


# We see that picking different specialisation (which probably correlates with picking college specialisation) matter for job placement, as people who specialised in arts get placed with lower range of grades (probably because they choose to go to jobs demanding less skills on which schools focus)

# In[ ]:


f, axes = plt.subplots(1, 6, sharey=True, figsize=(17,6))

sns.boxplot(x='status', y='degree_p', data=data[data.gender=='M'], ax=axes[0])
sns.boxplot(x='status', y='degree_p', data=data[data.gender=='F'], ax=axes[1])
sns.boxplot(x='status', y='ssc_p', data=data[data.gender=='M'], ax=axes[2])
sns.boxplot(x='status', y='ssc_p', data=data[data.gender=='F'], ax=axes[3])
sns.boxplot(x='status', y='ssc_p', data=data[data.gender=='M'], ax=axes[4])
sns.boxplot(x='status', y='ssc_p', data=data[data.gender=='F'], ax=axes[5])
axes[0].title.set_text('Men')
axes[1].title.set_text('Women')
axes[2].title.set_text('Men')
axes[3].title.set_text('Women')
axes[4].title.set_text('Men')
axes[5].title.set_text('Women')
plt.show()


# Women with higher grades than men got rejected. Since women are more likely to be in areas that have lower pay, one would assume that they should have lower entry bar, but this is not the case. 

# ## The entry bar is higher for women

# # Machine Learning

# We will train a model to predict placement, based on the info. We will see the coefficients it assigns to the gender variable. 
# 
# Separately, we can train two ML models on datasets of men and women, and compare the coefficients they assign to different variables.

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


data_one = pd.get_dummies(data[['gender', 'ssc_p', 'hsc_p', 
                                'hsc_s', 'degree_p', 'degree_t',
                                  'workex', 'etest_p', 'specialisation', 'mba_p']], drop_first=True)


# In[ ]:


y_one = data.placed


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(data_one, y_one, test_size=0.3, random_state=42)


# In[ ]:


# Accuracy of base estimator, just guessing that everyone is placed.
data.placed.mean()


# In[ ]:


lr = LogisticRegression(penalty='l2',
    tol=0.001,
    C=50,
    random_state=42,
    solver='lbfgs',
    max_iter=1000,
    class_weight={1:1, 0:2})
lr.fit(X_train,y_train)
lr.score(X_train,y_train)


# In[ ]:


lr.score(X_test,y_test)


# In[ ]:


from sklearn.metrics import plot_confusion_matrix
plot_confusion_matrix(lr, X_test, y_test)
plt.show()


# Logistic regression has 83% accuracy, better than our base estimator. Our model seems to be slightly optimistic, with the more common error being predicting that a person got placed, while in reality they weren't. This is common in unbalanced dataset. I tried to correct for that setting class weights in regression, according to the sizes of the groups of placed/non-placed.

# In[ ]:


coeff = list(zip(data_one.columns, lr.coef_[0].round(2)))


# In[ ]:


pd.DataFrame(coeff, columns=['variable', 'coefficent']).set_index('variable')


# There is something that might seem counterintuitive. The coefficients for hsc_s Commerce and Science are negative, while these areas have higher employability than Arts. However, as we saw before, one is required higher grades to be placed there. 

# We see that gender_M coefficient is 1, while degee_p is 0.24. This means that according to our model
# # Being female while trying to find a job is equivalent to having less 4 points in degree_p.
# 
# We can make a small experiment. Let us create two borderline candidates with the same grades but different genders.

# In[ ]:


# A borderline candidate with barely passing grades
female_candidate = [60  , 55  , 60  , 60  , 55,  0.  ,  1.  ,  0.  ,  0.  ,
        0.  ,  1.  ,  1.  ]
male_candidate = [60  , 55  , 60  , 60  , 55,  1  ,  1.  ,  0.  ,  0.  ,
        0.  ,  1.  ,  1.  ]


# In[ ]:


lr.predict_proba([female_candidate])


# In[ ]:


lr.predict_proba([male_candidate])


# For borderline candidates with the same grades our model changes its prediction between being hired and not after changing the gender, with pretty big jump in probabilities. 

# # Conclusions
# 
# The dataset is very small, so all conclusions should be taken with caution. It is hard to predict the salary from the grades. We saw that once we specialise to certain areas, the grades seem to correlate with salary, but there are obviously other factors not seen in the data, which affect the salary more than just the grades.
# 
# Having said this, it does seem that women with the same salaries as men in the same areas of specialisation have higher grades. They also need higher grades to get placed. Machine learning models (eventhough not very accurate) do quantify the gender gap.
