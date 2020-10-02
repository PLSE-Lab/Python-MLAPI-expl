#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


#Import data
df = pd.read_csv('../input/StudentsPerformance.csv')


# In[ ]:


#shape of dataframe (Number of Rows and Columns)
df.shape


# In[ ]:


# Viewing sample of data set
df.head()


# In[ ]:


# Description of the data set
df.describe(include='all')


# In[ ]:


# Find any missing value (0-no missing value)
df.isnull().sum()


# There is no missing/null data present. That's good!

# In[ ]:


#Creating a new column of 'total score'
df['total score']=(df['math score']+df['reading score']+df['writing score'])/3

#New Column for Pass status, holds value as P > Pass and F > Fail
passing_marks=35
df['pass status'] = np.where(df['total score']<passing_marks,'F','P')


# ## Let us now see Visually which factor affects the scores(individually) 
# let us start with
# 
# 1. Gender
# 2. race/ethinicity
# 3. parental level of education
# 4. lunch
# 5. test preparation course

# In[ ]:


#Gender

gender_s=df.groupby('gender') ['math score','reading score', 'writing score'].mean()
tot=df.groupby('gender')['total score'].mean()
gender_s.plot(kind='bar')
tot.plot(kind='line', color='black') #total score line plot
plt.show()

#Plot Gender with respect to pass status to see which gender performed well.
gender_ps=df.groupby(['gender','pass status']).size().to_frame('count').reset_index()

"""Here i'm creating a second plot
to see how many males and female pass/failed the test
I should have have used just count but that could have been biased 
because no. of male/female differ and that is what not required to be done
so here i'm segregating in terms of percent which removes the bias
I'll do this same step for rest of the four variables
"""


gender_ps2=gender_ps.groupby('gender').agg('sum').reset_index()
result = pd.merge(gender_ps, gender_ps2, how='left', on=['gender'])
result['percentage'] = (result['count_x']/result['count_y'])*100
result['gender'] = result["gender"]+ " : " + result["pass status"].map(str)
sns.barplot(x='gender',y='percentage', data=result)
plt.xticks(rotation=90)
plt.show()


# * From above plot we can see that both male and female are quite not good in math
# but, they excel in both reading and writing.
# 

# In[ ]:


# RACE/ETHNICITY

race_s=df.groupby('race/ethnicity') ['math score','reading score', 'writing score'].mean()
tot=df.groupby('race/ethnicity')['total score'].mean()

#Plot race/ethnicity with average score in all three subjects and also averall total score average.
race_s.plot(kind='bar')
tot.plot(kind='line', color='black') #total score line plot
plt.show()

#Plot race/ethnicity with respect to pass status to see which race performed well.
race_ps=df.groupby(['race/ethnicity','pass status']).size().to_frame('count').reset_index()
race_ps2=race_ps.groupby('race/ethnicity').agg('sum').reset_index()
result = pd.merge(race_ps, race_ps2, how='left', on=['race/ethnicity'])
result['percentage'] = (result['count_x']/result['count_y'])*100
result['race/ethnicity'] = result["race/ethnicity"]+ " : " + result["pass status"].map(str)
sns.barplot(x='race/ethnicity',y='percentage', data=result)
plt.xticks(rotation=90)
plt.show()


# * Looking above in the first plot wee can see that there is not quiet a difference between the groups, as seen it shows a gradual increment as the down.
# * But in the second plot we can clearly see that "group C" has most number of student passed but it can be misleading because we do not know the ratio of P/F can varry and we can misinterpret the above 2nd plot.
# * I have already plotted with respect to ratio below where we can see that ratio is equal and hence race/ethnicity is not a factor which is correlated with pass status.

# In[ ]:


# PARENTAL LEVEL OF EDUCATION

parent_s=df.groupby('parental level of education') ['math score','reading score', 'writing score'].mean()
tot=df.groupby('parental level of education')['total score'].mean()
parent_s.plot(kind='bar')
tot.plot(kind='line') #total score line plot
plt.xticks(rotation=90)
plt.show()

#Plot parental level of education with respect to pass status to see which level make any difference.
ploe_ps=df.groupby(['parental level of education','pass status']).size().to_frame('count').reset_index()
ploe_ps2=ploe_ps.groupby('parental level of education').agg('sum').reset_index()
result = pd.merge(ploe_ps, ploe_ps2, how='left', on=['parental level of education'])
result['percentage'] = (result['count_x']/result['count_y'])*100
result['parental level of education'] = result["parental level of education"]+ " : " + result["pass status"].map(str)
sns.barplot(x='parental level of education',y='percentage', data=result)
plt.xticks(rotation=90)
plt.show()


# In above plot we can see a slight difference between different levels also in 2nd plot the pass status is almost equal in all levels. So certainly "parental level of education' does not make any difference.

# In[ ]:


# LUNCH

lunch_s=df.groupby('lunch') ['math score','reading score', 'writing score'].mean()
tot=df.groupby('lunch')['total score'].mean()
lunch_s.plot(kind='bar')
tot.plot(kind='line') #total score line plot
plt.show()

#Plot lunch with respect to pass status to see which category performed well.
lunch_ps=df.groupby(['lunch','pass status']).size().to_frame('count').reset_index()
lunch_ps2=lunch_ps.groupby('lunch').agg('sum').reset_index()
result = pd.merge(lunch_ps, lunch_ps2, how='left', on=['lunch'])
result['percentage'] = (result['count_x']/result['count_y'])*100
result['lunch'] = result["lunch"]+ " : " + result["pass status"].map(str)
sns.barplot(x='lunch',y='percentage', data=result)
plt.xticks(rotation=90)
plt.show()


# Average score for student who is in "free/reduced" have a lag than those of "standard". But in 2nd plot the ratio of pass in both is equal.

# In[ ]:


# TEST PREPARATION

testpre_s=df.groupby('test preparation course') ['math score','reading score', 'writing score'].mean()
tot=df.groupby('test preparation course')['total score'].mean()
testpre_s.plot(kind='bar')
tot.plot(kind='line') #total score line plot
plt.show()

#Plot 'test preparation course' with respect to pass status.
testpre_ps=df.groupby(['test preparation course','pass status']).size().to_frame('count').reset_index()
testpre_ps2=testpre_ps.groupby('test preparation course').agg('sum').reset_index()
result = pd.merge(testpre_ps, testpre_ps2, how='left', on=['test preparation course'])
result['percentage'] = (result['count_x']/result['count_y'])*100
result['test preparation course'] = result["test preparation course"]+ " : " + result["pass status"].map(str)
sns.barplot(x='test preparation course',y='percentage', data=result)
plt.xticks(rotation=90)
plt.show()


# * Looking at the about 2nd plot it is really a strange behaviour as we can see that students who were prepared and not-prepared
#   have equal.
# * Not only that, even in the 1st plot it says that the average does not have a large difference.
# * Owing to this we can say that it either the test was easy or this data might only say about preparation before test and not about how well versed the students are.

# ## Let us now see our data by sorting on different cases like
# 
# 1. Who scored the most in respective subjects(top20 and bottom20)?
# 2. Who score the most/least in term of total score(top20 and bottom20)?
# 

# In[ ]:


# Top20 Math Score

df.sort_values(by='math score', ascending=False).head(20)


# 1. Here by looking at the output we can see that its female who topped in math and male and female have approx equal ratio in top 20.
# 2. Also we can see at in this top 20 list based on top math scorer had a "standard" lunch type insted of "free/reduced", so here we can say that Diet is important to top in math.
# 3. There is one interesting thing here is that the top scorer in math did not prepared for it, I mean is that possible? Yes maybe because math is about how you know the concepts and remembering formulaes. It maybe the case that the data might be taken just before giving the test because we cannot say that they did not learn well, it just tell us about the preparation before the test.

# In[ ]:


# Top20 Reading Score

df.sort_values(by='reading score', ascending=False).head(20)


# 1. Now here we can see that there are more number of female who in top 20.(Refer the gender and score average in 1st plot)
# 2. Even here we can see the lunch is "standard" and it does help(confirmed in below analysis).

# In[ ]:


# Top20 Writing Score

df.sort_values(by='writing score', ascending=False).head(20)


# 1. In this too we can see more number of female who are in top 20, so as per this female are quite good in writing/reading and even math. (It is as per the data so please don't mind boys :P)
# 
# 2. Even here we can see diet helped to get better score, hence we can say diet is crucial in term of scoring better.

# In[ ]:


# Top20 Student (total score)

df.sort_values(by='total score', ascending=False).head(20)


# 1. This can be our overall interpretation to verify on bases of "total score" and here as well we can see that diet is important.
# 2. As seen in above individual top 20 scorer we have seen that female scored better and we can verify it here.
# 

# In[ ]:


# Bottom20 Student (total score)

df.sort_values(by='total score', ascending=True).head(20)


# 1. Now in term of least score, as per "total score" as we know diet is important so here it is obviously converse of top scorer and by looking above list many of them had "free/reduced" lunch and hence we can see the result..
# 
# 2. Also test preparation is important and many of them were not prepared.
# 
# 
#    * So now to interpret the overall result, diet and test preparation were the key to get better score and yes some of them obviously know that preparation is a must or they might don't even know about the lunch, that it does help and it is worth exploring such strategy.
#    
# <br>
# <br>
# * That's all folks, this is my 1st kaggle notebook so if there is any mistake or improvement to be done then please share it with me.
# 
#  **_Email : purohitvikram77@gmail.com_**
#  
# #### Thank you.
