#!/usr/bin/env python
# coding: utf-8

# ## Student Exam Performance Dataset

# #### The aim of this dataset is to analyse the performance of students based on the marks scored by the srudents in different exams and for finding out the number of students passed or failed and the grade recieved by the students
# 
# #### There are 3 stages used for evaluation
# #### > Data Cleaning
# #### > Data Visualization
# #### > Applying Machine Learning for prediction

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df=pd.read_csv('../input/StudentsPerformance.csv')
df.head()


# In[ ]:


new_df=df
new_df.head()


# ### Data Cleaning

# In[ ]:


df.info()


# In[ ]:


df.describe()


# In[ ]:


df=df.rename(columns={'parental level of education':'parental_level_of_education',
                      'test preparation course':'test_preparation_course',
                     'math score':'math_score','reading score':'reading_score','writing score':'writing_score'})
df.head()


# In[ ]:


df.parental_level_of_education.unique()


# In[ ]:


df.lunch.unique()


# In[ ]:


df.test_preparation_course.unique()


# In[ ]:


df=df.replace(['group A','group B','group C','group D','group E'],[0,1,2,3,4])
df=df.replace(["bachelor's degree", 'some college', "master's degree","associate's degree", 'high school', 'some high school'],
             [0,1,2,3,4,5])
df=df.replace(['standard', 'free/reduced'],[0,1])
df=df.replace(['none', 'completed'],[0,1])
df=df.replace(['male','female'],[0,1])
df.head()


# In[ ]:


df['total_score']=(df['math_score']+df['reading_score']+df['writing_score'])/3 
df['total_score']=df['total_score'].astype(int)
df.head()


# ### Data Visualization

# In[ ]:


df.head()


# In[ ]:


x=sns.PairGrid(df,palette='coolwarm')
x=x.map_diag(plt.hist)
x=x.map_offdiag(plt.scatter,color='red',edgecolor='black')


# #### The above plot descibes about the importance of different features and its effect on other features

# In[ ]:


plt.figure(figsize=(10,8))
sns.barplot(x='gender',y='total_score',data=df,hue='race/ethnicity',ci=0,palette='spring')

#(['group A','group B','group C','group D','group E'],[0,1,2,3,4])
#(['male','female'],[0,1])


# ####  Male and female of group E has the highest score while male and female of group A has the lowest total marks 

# In[ ]:


plt.figure(figsize=(10,8))
sns.barplot(x='gender',y='total_score',data=df,hue='parental_level_of_education',ci=0)


#(["bachelor's degree", 'some college', "master's degree","associate's degree", 'high school', 'some high school'],[0,1,2,3,4,5])
#(['male','female'],[0,1])


# #### It can be found that a male with master's degree earned highest total_score while male in high school earned the least total_score. In second case, female with bachelor's degree highest total_score and female in high school earned least total_score

# In[ ]:


plt.figure(figsize=(10,8))
sns.barplot(x='test_preparation_course',y='total_score',data=df,ci=0)


# #### It can be seen that students with complete preparation scored better as compared to without preparation

# In[ ]:


def func(x):
    if 0<x<40:
        return False
    else:
        return True

df['performance']=df['total_score'].apply(func)
df.head()


# In[ ]:


df.performance.value_counts()


# #### Hence it can be seen that in total combined average score 30 students failed the exam while 970 students have passed the exam

# In[ ]:


def func(x):
   if x>90:
       return 'A+'
   elif 80<x<=90:
       return 'A'
   elif 70<x<=80:
       return 'A-'
   elif 60<x<=70:
       return 'B+'
   elif 50<x<=60:
       return 'B'
   elif 40<=x<=50:
       return 'B-'
   else:
       return 'Fail'
df['grade']=df['total_score'].apply(func)
df.head()


# In[ ]:


df.grade.value_counts()


# #### Above shows the description of the grades obtained by different students

# ### Linear Regression

# In[ ]:


x=df[['gender','race/ethnicity','parental_level_of_education','lunch','test_preparation_course','math_score','reading_score','writing_score']]
y=df['total_score']


# In[ ]:


x.shape,y.shape


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=7)


# In[ ]:


print(x_train.shape,y_train.shape)


# In[ ]:


print(x_test.shape,y_test.shape)


# In[ ]:


from sklearn.linear_model import LinearRegression


# In[ ]:


model=LinearRegression()


# In[ ]:


model.fit(x_train,y_train)


# In[ ]:


prediction=model.predict(x_test)


# In[ ]:


plt.figure(figsize=(11,6))
plt.scatter(y_test,prediction,edgecolors='black',c='red',vmin=30,vmax=70)
#x.set_yticklabels([30,35,40,45,50])


# In[ ]:


model.coef_


# In[ ]:


model.intercept_


# In[ ]:


from sklearn import metrics
mean_sq=metrics.mean_squared_error(y_test,prediction)
RMSE=np.sqrt(mean_sq)
RMSE


# #### Lower the root mean square error better is the model as the difference between the true value and predicted value decreases which leads to better output prediction

# In[ ]:


from sklearn.metrics import r2_score
r2_score(y_test,prediction)


# #### Higher the r2 score better is the model and here it is 0.9 which is closer the 1 hence the model predicted the most of the values correctly
