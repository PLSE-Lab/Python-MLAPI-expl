#!/usr/bin/env python
# coding: utf-8

# Let's get started(I'm analyzing average student performance)

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as snb


# In[ ]:


df = pd.read_csv('../input/StudentsPerformance.csv')


# lets see a sample of 5 records.

# In[ ]:


df.sample(5)


# Based on the sample,
# What I understood was, 
# 1. data frame consists of 8 columns
# 2. Race/ethnicity column unable to understand. So I should plot later to see whether it influences scores.
# 3. I would like to rename columns for better understanding and accessing.

# In[ ]:


#Renaming Columns
df.columns = ['gender', 'race', 'parentDegree', 'lunch', 'course', 'mathScore', 'readingScore', 'writingScore'] 


# In[ ]:


#Checking whether there are any missing values
df.isna().sum()


# Let me create new column which calculates total percentage of student. I would like to generalize all scores and then see how these features influence scores of students. 
# Later using these features let's buiild model and predict the final percentage of students.

# In[ ]:


#Total Sore Percentage
df['total'] = (df['mathScore']+df['readingScore']+df['writingScore'])/3
df.sample()


# Let's start analyzing from now.
# 1. How Race/Group and Parent Degree is related to total scores??

# In[ ]:


#some stats..

df.groupby(['race','parentDegree']).mean()


# **Insight 1:**
# As we see, High school, some High school of parentDegree has less scores....
# **Insight 2**:
# Aso, As group Name is increasing, scores are increasing.
# Group A < Group B < Group C < Group D < Group E.
# 

# In[ ]:


df.groupby(['gender']).mean()


# **Insight 3:** As we can see, Female students perform better than Male students.
#    Of course, in Maths, Male students outperform Female Students.

# Let's see some plots

# In[ ]:


#relation between gender and course and total
course_gender = df.groupby(['gender','course']).mean().reset_index()


# In[ ]:


snb.factorplot(x='gender', y='total', hue='course', data=course_gender, kind='bar')


# In[ ]:


#Now we can observe that,Parents degree is also crucial in student's score. 
course_gender = df.groupby(['gender','parentDegree']).mean().reset_index()
snb.factorplot(x='gender', y='total', hue='parentDegree', data=course_gender, kind='bar')


# So, I would like to generalize "parentDegree" column, with 'has_degree' and 'no_degree'.<br>
# if parent Degree is in ("high school","some high school") then, I thought they don't have degree. or else they has degree
# 

# In[ ]:


df.parentDegree.unique()


# In[ ]:


for i in range(len(df)):
    if df.iloc[i,2] in ['high school', 'some high school']:
        df.iloc[i,2] = 'No_Degree'
    else:
        df.iloc[i,2] = 'has_Degree'
        


# In[ ]:


df.sample()


# Even Lunch affects student's scores

# In[ ]:


Lunch_course = df.groupby(['lunch','course']).mean().reset_index()
snb.factorplot(x='lunch', y='total', hue='course', data=Lunch_course, kind='bar')


# In[ ]:


df.parentDegree.value_counts()


# In[ ]:


#Now we can observe that,Parents degree is also crucial in student's score. 
course_gender = df.groupby(['gender','parentDegree']).mean().reset_index()
snb.factorplot(x='gender', y='total', hue='parentDegree', data=course_gender, kind='bar')


# In[ ]:


race_gender = df.groupby(['gender','race']).mean().reset_index()
snb.factorplot(x='gender', y='total', hue='race', data=race_gender, kind='bar')


# In[ ]:


final_df = df.groupby(['gender','parentDegree','course','lunch','race']).mean().reset_index()
after_sort = final_df.sort_values(by= ['total'],ascending = False)
after_sort.drop(columns=['mathScore','readingScore','writingScore'],inplace = True)
after_sort


# As you can see above, I have generalized all features <br>
# <ol>
# <li>  As you can see, Top students(mean) have completed their course, Took standard Lunch, Having parent_Degree is also + point.</li><li>Bottom students(mean) didn't complete course, Didn't take good lunch, parent has no degree.</li>
# <li>Out of Top 10(mean), 7 are female students</li>
# <li>Interestingly, Out of Bottom 10(mean), 7 are male students</li></ol>

# In[ ]:


#See, it's clear
print("Top students Performance \n",after_sort[:10])
#Simply, if you complete course, Have standard lunch, you   can score good grades. 


# In[ ]:


#See, it's clear
print("Bottom students Performance \n",after_sort[-10:][::-1])
#Simply, Lunch, Course are mandatory for scoring good. 


# In[ ]:



base = pd.get_dummies(final_df,columns=['gender','race','parentDegree','course','lunch'],dtype = int)
base.sample()
base.info()


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
train_x,test_x,train_y,test_y = train_test_split(base.iloc[:,4:],base.iloc[:,3],test_size = 0.05)
model = XGBRegressor(max_depth = 6)
model.fit(train_x,train_y)
target = model.predict(test_x)
mean_squared_error(target,test_y)


# In[ ]:


len(target)


# In[ ]:


test_y[:5].values


# In[ ]:


target[:5]


# Finally,
# What I observed is ,
# 1. Students percentage is correlated with Groups Average Percentage.
#             ex: Average student from Group E have scored more compared to Group A's average Student.
# 2. Parents Degree is also a bit correlated.
# 3. Female Students Percentage is higher than Male Students.
# 4. Students Who took courses have benefitted.
# 5. Students who took standard lunch has scored well.
# 
# Thanks,
# Forget about upvoting, at least you viewed my kernel,thanks.
# 

# I am a noobie. I would just like to explore this data, predict student total percentage based on the given features. I'm still editing this kernel. Please let me know if there any suggestions. Any tips/suggestions are highly encouraged. Please tell mistakes that I performed.

# In[ ]:




