#!/usr/bin/env python
# coding: utf-8

# **This is  basic kernal in which I am going to analyse the students perfomence in exams..In this I will do only basic analysis..First will do data exploration and I will check wether there is some false values or data that needs to be cleaned and after this we will move furthur and analysis students perfomence in the exams on the basis of their Gender , type of lunch they had , level of education of of their parents and so on and during this whole analysis we will use pandas only.**

# In[ ]:



import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# 1. * **First We will Read our dataset which is in a csv file format.**

# In[ ]:


df = pd.read_csv('../input/StudentsPerformance.csv') #Read_csv function is used to read our csv file
df.head(10)


# **2. Now let's explore our dataset.**

# In[ ]:


df.info() #this info() function will describe all the columns with their data types and total number of values in eac column.


# **3. Now let's check our are there any values in our data ? **

# In[ ]:


df.isnull().any()  #false means there is no null values in each column


# > **4. now let's check that is all of the values in our data is making sense or are there some ambiguos vaues in it ? for this  let's check all the unique values in each column. **

# In[ ]:


df['gender'].unique() # so there are only two unique values 'male' and 'female' so there is no ambigous values in this column


# In[ ]:


df['race/ethnicity'].unique() # the race/ethnicity is divided in 5 diffrent groups.


# In[ ]:


df['parental level of education'].unique()


# In[ ]:


df = df.replace('some college',"bachelor's degree")
df = df.replace('some high school','high school')
df['parental level of education'].unique()


# In[ ]:


df['lunch'].unique()


# In[ ]:


df['test preparation course'].unique()


# In[ ]:


print("All the unique values in column 'math score' : \n",df['math score'].unique())

print("\nhighest value in column : ",max(df['math score']))

print("\nlowest value in column : ",min(df['math score']))

# so the values are from 0 to 100 and there are no false values.


# In[ ]:


print("All the unique values in column 'reading score' : \n",df['reading score'].unique())

print("\nhighest value in column : ",max(df['reading score']))

print("\nlowest value in column : ",min(df['reading score']))

# so the values are from 17 to 100 and there are no false values.


# In[ ]:


print("All the unique values in column 'writing score' : \n",df['writing score'].unique())

print("\nhighest value in column : ",max(df['writing score']))

print("\nlowest value in column : ",min(df['writing score']))

# so the values are from 10 to 100 and there are no false values.


# **5. now we have explored our data and found no null values and false values and each column has 1000 values and there is no problem with data types also... so this is a cleaned data we did'nt have to do any cleaning..now we can move to do analysis and can get insights from this data.****

#  **Q.1 according to gender who has more marks ?**

# In[ ]:


marks = df.groupby(df['gender']).describe()
marks


# ***for conviniance in reading lets check marks seperatly for each subject.**

# In[ ]:


marks_according_gender = df[['math score','reading score','writing score']].groupby(df['gender'])
print('marks in maths according to gender : \n',marks_according_gender.sum()) #so boys are little ahead in marks in maths
# but in reading and writing girls are ahead of boys.


# In[ ]:


print("average marks according to gender : \n",marks_according_gender.mean())


# **Now let's check wehter the marks of students depends on the level of education of their parents ?**

# In[ ]:


marks_parent_edu = df[['math score','reading score','writing score']].groupby(df['parental level of education'])
marks_parent_edu.sum()


# **Now let's check wether the marks of students changes according to the type of lunch.**

# In[ ]:


marks_according_lunch = df[['math score','reading score','writing score']].groupby(df['lunch']).sum()

marks_according_lunch 


# **So , it is clear that those students who had standard lunch got more marks then those who had free/reduced lunch.**

# **now at last let's check marks on the basis of test preparation course.**
# 

# In[ ]:


marks_test_pre = df[['math score','reading score','writing score']].groupby(df['test preparation course']).sum()

marks_test_pre   


# **So unexpectedly those who completed test preparation course has less marks overall in the comparison of those who has'nt completed it.**

# In[ ]:





# In[ ]:





# In[ ]:




