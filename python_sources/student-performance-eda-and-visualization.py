#!/usr/bin/env python
# coding: utf-8

# ## **AIM: To Study the effect of various factors on students reading, writing and maths score**

# I hope you find this kernel helpful and some<font color="red"><b> UPVOTES</b></font>  would be very much appreciated
# 
# 

# In[ ]:


import warnings                       # to hide warnings if any
warnings.filterwarnings('ignore')

import os
print(os.listdir("../input"))


# ### ** Importing Required Libraries**

# In[ ]:


import pandas as pd                #Data Processing
import numpy as np                 # Linear Algebra
import matplotlib.pyplot as plt    # Data Visualization
import seaborn as sns              # Data Visualization

get_ipython().run_line_magic('matplotlib', 'inline')


# ### **Reading the data**

# In[ ]:


df = pd.read_csv('../input/StudentsPerformance.csv')
df.head(3)


# ### **Dataset Summary**

# In[ ]:


df.info()


# #### **Checking the number of Null Values**

# In[ ]:


df.isnull().sum()


# **There are no Null values in any of the columns**

# In[ ]:


df.shape


# In[ ]:


df.describe()


# **The features described in the above data set are:**
# 
# **1. Count** tells us the number of NoN-empty rows in a feature.
# 
# **2. Mean** tells us the mean value of that feature.
# 
# **3. Std** tells us the Standard Deviation Value of that feature.
# 
# **4. Min** tells us the minimum value of that feature.
# 
# **5. 25%**, **50%**, and **75%** are the percentile/quartile of each features.
# 
# **6. Max** tells us the maximum value of that feature.

# ### **Checking each columns features**

# #### **1. Gender**

# In[ ]:


count  = 0
for i in df['gender'].unique():
    count = count + 1
    print(count,'. ',i)


# #### **2. Race/Ehtnicity of People**

# In[ ]:


count = 0
for i in sorted(df['race/ethnicity'].unique()):
    count = count + 1
    print(count, '. ',i)
print('Number of different races/ethnicity of people: ', df['race/ethnicity'].nunique())


# #### **3. Parent's level of Education**

# In[ ]:


count = 0
for i in df['parental level of education'].unique():
    count = count + 1
    print(count, '. ', i)


# #### **4. Different types of lunches**

# In[ ]:


count  = 0
for i in df['lunch'].unique():
    count = count + 1
    print(count,'. ',i)


# #### **5. Types of Test Prepration Course**
# 

# In[ ]:


count  = 0
for i in df['test preparation course'].unique():
    count = count + 1
    print(count,'.',i)


# ## **Exploratory Data Analysis**
# 

# In[ ]:


sns.set_style('darkgrid')


# #### **1. Pairplot**

# In[ ]:


sns.pairplot(df, hue = 'gender')
plt.show()


# #### **2. Heatmap**

# In[ ]:


sns.heatmap(df.corr(), annot = True, cmap='inferno')
plt.show()


# **There is strong correlation between a student's reading score & writing score, reading score & math score and writing score & math score**

# ### **Plotting the distribution of students marks**

# #### **1. Math Score**

# In[ ]:


plt.figure(figsize=(8,5))
sns.distplot(df['math score'], kde = False, color='m', bins = 30)
plt.ylabel('Frequency')
plt.title('Math Score Distribution')
plt.show()


# **Most students have their math score in the range of 60 to 80**

# #### **2. Reading Score**

# In[ ]:


plt.figure(figsize=(8,5))
sns.distplot(df['reading score'], kde = False, color='r', bins = 30)
plt.ylabel('Frequency')
plt.title('Reading Score Distribution')
plt.show()


# **Most students have their reading score in the range of 60 to 80**

# #### ** 3. Writing Score**

# In[ ]:


plt.figure(figsize=(8,5))
sns.distplot(df['writing score'], kde = False, color='blue', bins = 30)
plt.ylabel('Frequency')
plt.title('Writing Score Distribution')
plt.show()


# **Most students have their writing score in the range of 60 to 80**

# ### **Analyzing Maximum and Minimum marks of Students**
# 

# #### **1. Maximum & Minimum score in Math**

# In[ ]:


print('Maximum score in Maths is: ',max(df['math score']))
print('Minimum score in Maths is: ',min(df['math score']))


# #### **2. Maximum & Minimum score in Reading**

# In[ ]:


print('Maximum score in Reading is: ',max(df['reading score']))
print('Minimum score in Reading is: ',min(df['reading score']))


# #### **3. Maximum & Minimum score in Writing**

# In[ ]:


print('Maximum score in Writing is: ',max(df['writing score']))
print('Mimimum score in Writing is: ',min(df['writing score']))


# #### **4. Number of students having maximum score in Math**
# 

# In[ ]:


print('No. of students having maximum score in math: ', len(df[df['math score'] == 100]))


# #### ** 5. Number of students having maximum score in Reading**
# 

# In[ ]:


print('No. of students having maximum score in reading: ', len(df[df['reading score'] == 100]))


# #### ** 6. Number of students having maximum score in Writing**

# In[ ]:


print('No. of students having maximum score in writing: ', len(df[df['writing score'] == 100]))


# #### **7. Number of Students having maximum marks in all three categories**

# In[ ]:


perfect_writing = df['writing score'] == 100
perfect_reading = df['reading score'] == 100
perfect_math = df['math score'] == 100

perfect_score = df[(perfect_math) & (perfect_reading) & (perfect_writing)]
perfect_score


# In[ ]:


print('Number of students having maximum marks in all three subjects: ',len(perfect_score))


# #### **7. Number of Students having minimum marks in all three categories**

# In[ ]:


minimum_math = df['math score'] == 0
minimum_reading = df['reading score'] == 17
minimum_writing = df['writing score'] == 10



minimum_score = df[(minimum_math) & (minimum_reading) & (minimum_writing)]
minimum_score


# In[ ]:


print('No. of students having minimum marks in all three subjects: ', len(minimum_score))


# ### **Data Visualization and Interpretation**

# #### **1. Bar Plot of Scores according to gender**
# 

# In[ ]:


plt.figure(figsize=(10,4))

plt.subplot(1,3,1)
sns.barplot(x = 'gender', y = 'reading score', data = df)

plt.subplot(1,3,2)
sns.barplot(x = 'gender', y = 'writing score', data = df)

plt.subplot(1,3,3)
sns.barplot(x = 'gender', y = 'math score', data = df)

plt.tight_layout()


# **Males have higher math score than Females, whereas Females have higher scores in reading and writing than Males**

# #### ** 2. Bar plot of Scores on the basis of Race/Ethnicity**

# In[ ]:


plt.figure(figsize=(14,4))

plt.subplot(1,3,1)
sns.barplot(x = 'race/ethnicity', y = 'reading score', data = df)
plt.xticks(rotation = 90)

plt.subplot(1,3,2)
sns.barplot(x = 'race/ethnicity', y = 'writing score', data = df)
plt.xticks(rotation = 90)

plt.subplot(1,3,3)
sns.barplot(x = 'race/ethnicity', y = 'math score', data = df)
plt.xticks(rotation = 90)

plt.tight_layout()


# **People from group E have higher score in all three categories, where as people from group A have the lowest score in all three categories**

# #### **3. Bar plots of Scores on the basis of Test Prepration Course**

# In[ ]:


plt.figure(figsize=(14,4))

plt.subplot(1,3,1)
sns.barplot(x = 'test preparation course', y = 'reading score', hue = 'gender', data = df)

plt.subplot(1,3,2)
sns.barplot(x = 'test preparation course', y = 'writing score',hue = 'gender', data = df)

plt.subplot(1,3,3)
sns.barplot(x = 'test preparation course', y = 'math score',hue = 'gender', data = df)

plt.tight_layout()


# **Students who have completed the Test Prepration Course have scores higher in all three categories than those who haven't taken the course**

# #### **4. Bar Plots of Scores on the basis of Parent's Education Level**

# In[ ]:


plt.figure(figsize=(13,5))

plt.subplot(1,3,1)
sns.barplot(x = 'parental level of education', y = 'reading score', data = df)
plt.xticks(rotation = 90)

plt.subplot(1,3,2)
sns.barplot(x = 'parental level of education', y = 'writing score', data = df)
plt.xticks(rotation = 90)

plt.subplot(1,3,3)
sns.barplot(x = 'parental level of education', y = 'math score', data = df)
plt.xticks(rotation = 90)

plt.tight_layout()


# **Student's whose parents have a Master's degree have scored higher compared to others whereas Student's whose parent's went to high school have obtained low marks compared to others**

# #### **5. Bar Plots of Scores on the basis of Types of Luch**

# In[ ]:


plt.figure(figsize=(14,4))

plt.subplot(1,3,1)
sns.barplot(x = 'lunch', y = 'reading score', data = df)

plt.subplot(1,3,2)
sns.barplot(x = 'lunch', y = 'writing score', data = df)

plt.subplot(1,3,3)
sns.barplot(x = 'lunch', y = 'math score', data = df)


plt.tight_layout()


# **Students who availed standard luch have scored higher in all the three categories compared to students who have taken free/ reduced lunch.**

#  #### **6. Marks break down according to Gender**

# #### **i. Math Score**

# In[ ]:


print('----Females----')
print('Max. math Score: ', df[df['gender'] == 'female']['math score'].max())
print('Min. math Score: ', df[df['gender'] == 'female']['math score'].min())
print('Average math Score: ', df[df['gender'] == 'female']['math score'].mean())
print('----Males----')
print('Max. math Score: ', df[df['gender'] == 'male']['math score'].max())
print('Min. math Score: ', df[df['gender'] == 'male']['math score'].min())
print('Average math Score: ', df[df['gender'] == 'male']['math score'].mean())


# #### ** ii. Reading Score**

# In[ ]:


print('----Females----')
print('Max. reading Score: ', df[df['gender'] == 'female']['reading score'].max())
print('Min. reading Score: ', df[df['gender'] == 'female']['reading score'].min())
print('Average reading Score: ', df[df['gender'] == 'female']['reading score'].mean())
print('----Males----')
print('Max. reading Score: ', df[df['gender'] == 'male']['reading score'].max())
print('Min. reading Score: ', df[df['gender'] == 'male']['reading score'].min())
print('Average reading Score: ', df[df['gender'] == 'male']['reading score'].mean())


# #### ** iii. Writing Score**

# In[ ]:


print('----Females----')
print('Max. writing Score: ', df[df['gender'] == 'female']['writing score'].max())
print('Min. writing Score: ', df[df['gender'] == 'female']['writing score'].min())
print('Average writing Score: ', df[df['gender'] == 'female']['writing score'].mean())
print('----Males----')
print('Max. writing Score: ', df[df['gender'] == 'male']['writing score'].max())
print('Min. writing Score: ', df[df['gender'] == 'male']['writing score'].min())
print('Average writing Score: ', df[df['gender'] == 'male']['writing score'].mean())


# In[ ]:


plt.figure(figsize=(12,5))

plt.subplot(1,3,1)
sns.boxplot(x = 'gender', y = 'math score', data = df,palette = ['coral', 'lawngreen'])

plt.subplot(1,3,2)
sns.boxplot(x = 'gender', y = 'reading score', data = df,palette = ['coral', 'lawngreen'])

plt.subplot(1,3,3)
sns.boxplot(x = 'gender', y = 'writing score', data = df,palette = ['coral', 'lawngreen'])

plt.tight_layout()


# #### ** 7. Marks breakdown according to Race/Ethnicity**

# #### ** i. Math Score**

# In[ ]:


for i in sorted(df['race/ethnicity'].unique()):
    print('-----',i,'-----')
    print('Max. marks: ', df[df['race/ethnicity'] == i]['math score'].max())
    print('Min. marks: ', df[df['race/ethnicity'] == i]['math score'].min())
    print('Average marks: ', df[df['race/ethnicity'] == i]['math score'].mean())


# #### **ii. Reading Score**

# In[ ]:


for i in sorted(df['race/ethnicity'].unique()):
    print('-----',i,'-----')
    print('Max. marks: ', df[df['race/ethnicity'] == i]['reading score'].max())
    print('Min. marks: ', df[df['race/ethnicity'] == i]['reading score'].min())
    print('Average marks: ', df[df['race/ethnicity'] == i]['reading score'].mean())


# #### **iii. Writing Score**

# In[ ]:


for i in sorted(df['race/ethnicity'].unique()):
    print('-----',i,'-----')
    print('Max. marks: ', df[df['race/ethnicity'] == i]['writing score'].max())
    print('Min. marks: ', df[df['race/ethnicity'] == i]['writing score'].min())
    print('Average marks: ', df[df['race/ethnicity'] == i]['writing score'].mean())


# In[ ]:


plt.figure(figsize=(14,5))
plt.subplot(1,3,1)
sns.boxplot(x = 'race/ethnicity', y = 'math score', data = df)

plt.subplot(1,3,2)
sns.boxplot(x = 'race/ethnicity', y = 'reading score', data = df)

plt.subplot(1,3,3)
sns.boxplot(x = 'race/ethnicity', y = 'writing score', data = df)

plt.tight_layout()


# ####  **8. Marks breakdown on the basis of Parent's Education Level**

# #### **i. Math Score**

# In[ ]:


for i in df['parental level of education'].unique():
    print('-----',i,'-----')
    print('Max. marks: ', df[df['parental level of education'] == i]['math score'].max())
    print('Min. marks: ', df[df['parental level of education'] == i]['math score'].min())
    print('Average. marks: ', df[df['parental level of education'] == i]['math score'].mean())
    


# #### ** ii. Reading Score**
# 

# In[ ]:


for i in df['parental level of education'].unique():
    print('-----',i,'-----')
    print('Max. marks: ', df[df['parental level of education'] == i]['reading score'].max())
    print('Min. marks: ', df[df['parental level of education'] == i]['reading score'].min())
    print('Average. marks: ', df[df['parental level of education'] == i]['reading score'].mean())
    


# #### ** iii. Writing Score**

# In[ ]:


for i in df['parental level of education'].unique():
    print('-----',i,'-----')
    print('Max. marks: ', df[df['parental level of education'] == i]['writing score'].max())
    print('Min. marks: ', df[df['parental level of education'] == i]['writing score'].min())
    print('Average. marks: ', df[df['parental level of education'] == i]['writing score'].mean())
    


# In[ ]:


sns.set_style('whitegrid')
plt.figure(figsize=(16,7))
plt.subplot(1,3,1)
sns.boxplot(x ='parental level of education' , y = 'math score', data = df)
plt.xticks(rotation = 90)

plt.subplot(1,3,2)
sns.boxplot(x ='parental level of education' , y = 'reading score', data = df)
plt.xticks(rotation = 90)

plt.subplot(1,3,3)
sns.boxplot(x ='parental level of education' , y = 'writing score', data = df)
plt.xticks(rotation = 90)

plt.tight_layout()


# ### **Implementing a Grading system for marks obtained**

# **The grading system is described as follows:**

# **1. O (Outstanding)**: Student who scores 91 marks or higher in a subject<br>
# **2. A+ (Excellent)**: Student who scores 82 marks or higher in a subject<br>
# **3. A (Very Good)**: Student who scores 73 marks or higher in a subject<br>
# **4. B+ (Good)**: Student who scores 64 marks or higher in a subject<br>
# **5. B (Above Average)**: Student who scores 55 marks or higher in a subject<br>
# **6. C (Average)**: Student who scores 46 marks or higher in a subject<br>
# **7. P (Pass)**: Student who scores 35 marks or higher in a subject<br>
# **8. F (Fail)**: Student who scores less than 35 marks in a subject<br>
# 

# In[ ]:


# Function to assign grades

def get_grade(marks):
    if marks >= 91:
        return 'O'
    elif marks >= 82 and marks < 91:
        return 'A+'
    elif marks >=73 and marks < 82:
        return 'A'
    elif marks >=64 and marks < 73:
        return 'B+'
    elif marks >= 55 and marks < 64:
        return 'B'
    elif marks >=46 and marks < 55:
        return 'C'
    elif marks >= 35 and marks < 46:
        return 'P'
    elif marks < 35:
        return 'F'


# In[ ]:


df['reading_grade'] = df['reading score'].apply(get_grade)
df['writing_grade'] = df['writing score'].apply(get_grade)
df['math_grade'] = df['math score'].apply(get_grade)


# #### **Plotting Grade Statistics**

# In[ ]:


sns.set_style('whitegrid')
plt.figure(figsize=(16,5))
plt.subplot(1,3,1)
sns.countplot(x ='math_grade', data = df,order = ['O','A+','A','B+','B','C','P','F'],palette='magma')
plt.title('Grade Count in Math')


plt.subplot(1,3,2)
sns.countplot(x ='reading_grade', data = df,order = ['O','A+','A','B+','B','C','P','F'],palette='magma')
plt.title('Grade Count in Reading')

plt.subplot(1,3,3)
sns.countplot(x ='writing_grade', data = df,order = ['O','A+','A','B+','B','C','P','F'],palette='magma')
plt.title('Grade Count in Writing')

plt.tight_layout()


# **Plot Summary**

# In[ ]:


print('-------- GRADE STATISTICS --------')
print('==== MATH GRADE ====')
print(df['math_grade'].value_counts())
print('==== READING GRADE ====')
print(df['reading_grade'].value_counts())
print('==== WRITING GRADE ====')
print(df['writing_grade'].value_counts())


# Most of the students have got a B+ in Maths and Reading Section whereas in Writing Section almost equal number of students have got A and B+ grade.<br>
# The number of candidates who just got a qualifying grade(P) and passed is the highest in maths.

# #### **Number of students having maximum grade in Maths**

# In[ ]:


print('No. of students having maximum grade in math: ', len(df[df['math_grade'] == 'O']))


# #### **Number of students having maximum grade in Reading**

# In[ ]:


print('No. of students having maximum grade in reading: ', len(df[df['reading_grade'] == 'O']))


# #### **Number of students having maximum grade in Writing**

# In[ ]:


print('No. of students having maximum grade in writing: ', len(df[df['writing_grade'] == 'O']))


# #### **Number of students having maximum grade in all three categories**

# In[ ]:


perfect_writing = df['writing_grade'] == 'O'
perfect_reading = df['reading_grade'] == 'O'
perfect_math = df['math_grade'] == 'O'

perfect_grade = df[(perfect_math) & (perfect_reading) & (perfect_writing)]
print('Number of students having maximum grade(O) in all three subjects: ',len(perfect_grade))


# #### **Number of students having minimum grade in all three categories**

# In[ ]:


minimum_math = df['math_grade'] == 'F'
minimum_reading = df['reading_grade'] == 'F'
minimum_writing = df['writing_grade'] == 'F'



minimum_grade = df[(minimum_math) & (minimum_reading) & (minimum_writing)]
print('Number of students having minimum grade(F) in all three subjects: ',len(minimum_grade))


# ### **Classifying Students as Passed or Failed**
# A student is classified failed if he/she has failed in any one of three subjects otherwise he/she is classified as passed.

# In[ ]:


#Failed Students
failed_students = df[(minimum_math) | (minimum_reading)|(minimum_writing)]
failed = len(failed_students)
print('Total Number of students who failed are: {}' .format(len(failed_students)))


# In[ ]:


#Passed Students
passed_students = len(df) - len(failed_students)
print('Total Number of students who passed are: {}' .format(passed_students))


# In[ ]:


plt.figure(figsize=(8,6))

#Data to plot
labels = 'Passed Students', 'Failed Students'
sizes = [passed_students,failed]
colors = ['skyblue','yellowgreen']
explode = (.2,0)

#Plot
plt.pie(sizes,explode = explode, labels = labels,colors = colors,
       autopct='%1.1f%%',shadow = True, startangle=360)
plt.axis('equal')
plt.title('Percentage of Students who passed/failed in Exams')
plt.show()


# Majority(97.1%) of students passed in all the three subjects.Only 2.9% students failed in atleast one of the three subjects

# **Suggestions are welcome**

# In[ ]:




