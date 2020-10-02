#!/usr/bin/env python
# coding: utf-8

# # Performing Data Analysis and Visualization with Student Alcohol Dataset

# Introduction
# I have performed the data analysis to answer the  following questions
# <ol>
#     <li>How the data is correlated?</li>
#     <li>How previous grade affects the upcoming one</li>
#     <li>How Weekly alcohol consumption is distributed</li>
#     <li>How the grade and walc related(basic level)</li>
#     <li>How the parenting and walc affects the grade</li>
#     <li>Finally how the grade and walc related</li>
#    </ol>

# In[ ]:


import numpy as np#Linear Algebra
import pandas as pd#data processing
import seaborn as sns#visualization
import matplotlib.pyplot as plt#visualization


# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


data = pd.read_csv('/kaggle/input/student-alcohol-consumption/student-mat.csv') #loading the data


# <h2>Displaying the columns</h2>[](http://)

# In[ ]:


data.columns


# <h2>Displaying the info</h2>

# In[ ]:


data.info()#Getting the info about the dataset


# # Displaying the categorical data and numerical data

# In[ ]:


cat =[]
num =[]
for i in data.columns:
    if data[i].dtype==object:
        cat.append(i)
    else:
        num.append(i)


# In[ ]:


print("Categorical Features:{}".format(cat),end="\n\n")
print("Numerical Features:{}".format(num))


# # What about correlation?

# In[ ]:


plt.figure(figsize=(10,10))
sns.heatmap(data[num].corr(),annot = True,cbar = True)


# # Finding out the correlated columns using the heatmap 

# In[ ]:


correlated =[]
corr = data[num].corr()
for i in num:
    for j in num:
        if corr[i][j]>0.80 and i!=j:
            correlated.append(i)
            correlated.append(j)
correlated = list(set(correlated))
    


# In[ ]:


print('correlated columns:{}'.format(correlated))


# # Plotting the relationship between G1 and G2

# In[ ]:


plt.figure(figsize=(5,5))
sns.scatterplot(x = data['G1'],y = data['G2'],hue =data['sex'])


# From this plot i can  infer the following details:
# <ul>
#     <li>Those you get good grades(>12.5) in G1 no matter what gender she belongs, no matter how much she consumes alcohol she will get good grades in the G2</li>
#     <li>Those you get less grade in G1 no matter what gender she belongs, no matter how much she consumes alcohol she will get less grades in the G2</li>
#     <li>Also there lies some outliers which doesn't follow this pattern</li>
#         </ul>

# # Plotting the relationship between G2 and G3

# In[ ]:


sns.scatterplot(x = data['G2'],y = data['G3'],hue = data['sex'])


# From this plot i can  infer the following details:
# <ul>
#     <li>Those you get good grades(>12.5) in G2 no matter what gender she belongs, no matter how much she consumes alcohol she will get good grades in the G2</li>
#     <li>Those you get less grade in G2 no matter what gender she belongs, no matter how much she consumes alcohol she will get less grades in the G2</li>
#     <li>Also there lies some outliers which doesn't follow this pattern</li>
#         </ul>

# # How is  walc distributed?

# In[ ]:


sns.distplot(data['Walc'],kde =True)


# I can Infer the follow details:
# <ul>
# <li>There is no person in the sample who didn't drink at all</li>
# <li>Most of them drink once in the week</li>
# <li>The student who drink 5 times a week is very less</li>
# </ul>

# # How the grade and walc related(basic level)

# In[ ]:


sns.boxplot(x='Walc',y='G1',data =data)


# From this plot i can  infer the following details:
# <ul>
#     <li>The Range of grade for the people who drink once in a week is 15</li>
#     <li>The Range of grade for the people who drink twice in a week is 13</li>
#     <li>The Range of grade for the people who drink thrice in a week is 16</li>
#     <li>The Range of grade for the people who drink four times  a week is 12</li>
#     <li>The Range of grade for the people who drink once in a week is 13</li>
#     <li>The highest Rank for scored by the persons who drink once in a week</li>
#     <li>The lowest Rank was scored by the person who drink thrice a week</li>
#     </ul>

# # How the parenting and walc affects the grade

# In[ ]:


plt.figure(figsize=(10,10))
sns.boxplot(x='Walc',y='G1',data =data,hue='guardian')


# From this plot I can infer the following details:
# <ul>
#     <li>The maximum grade is scored by the student who drinks once in a week and guarded by mother or father</li>
#     <li>Even if the student drinks five times a day if he is guarded by mother he can score more grade</li>
#     <li>The minimum grade is scored by the student who drinks thrice in a week and guarded by mother</li>

# # Finally how the grade and walc related

# In[ ]:


ave = sum(data.G1)/float(len(data))
data['ave_line'] = ave
data['average'] = ['above average' if i > ave else 'under average' for i in data.G1]
sns.swarmplot(x='Walc', y = 'G1', hue = 'average',data= data,palette={'above average':'violet', 'under average': 'red'})


# Finally from this plot i can infer the following details:
# <ul>
#     <li>If student drink once a week it doesn't much affect his grade</li>
#     <li>If student drink more than 3 times a week it surely afects his grade</li>
#     <li>The students who drink once a week are the topper with high grade</li>
#     <li>The student who drink thrice a week is the low grade scorer</li>
#     </ul>
