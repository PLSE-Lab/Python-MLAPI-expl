#!/usr/bin/env python
# coding: utf-8

# 
# # Inspiration :
#     - Overall lifters distribution by Gender
#     - Does age have an impact on lifting capacity
#     - How big of a difference does gender make?
#     - How much influence does overall weight have on lifting capacity?
#     
#     
# # Pipeline
#     - Importing Libraries
#     - Data Cleaning
#             -Dropping Uncessary Columns
#             -Exploring Missing Values
#             -Filling Missing values
#         
#     

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


path = "../input/openpowerlifting.csv"
data = pd.read_csv(path)
data = data[1:147149:]
data.shape


# In[ ]:


data.head(5)


# In[ ]:


data_new = data[['Name','Sex']]
data_new = data_new.drop_duplicates()


# # Data Cleaning

# Dropping Unecessary Columns

# In[ ]:


data.columns
data = data.drop(labels = ['Squat4Kg','Bench4Kg','Deadlift4Kg','Wilks'], axis = 1)


# In[ ]:


data.shape


# In[ ]:


data.tail(5)


# ### Missing Values by Type

# In[ ]:


print(data.isnull().sum())


# ### Filling the missing values using ffill

# In[ ]:


data.fillna(method ='ffill',inplace = True)
data.isnull().sum()


# In[ ]:


data.tail(5)


# # Q1 - Distribution of Powerlifters by Gender in the Sample

# In[ ]:



gender_size = data_new.Sex.value_counts().sort_index().tolist()
gender_names = ['Female','Male']
col = ['#c973d0','#4a73ab']
fig, ax = plt.subplots()
ax.axis('equal')
mypie, _ = ax.pie(gender_size, radius=3.3, labels=gender_names, colors = ['#e54370','#0093b7']) 
plt.setp( mypie, width=0.9, edgecolor='white')


# # Intuition : 
#     
#   Based on the distinct athletes by gender, the sample chosen  had ~70% males  and ~30% females.
#   
# NOTE : The original dataset had majority of the values missing for rows, so forward/backward fill wouldn't be intuitive to apply in that scenario. Considering this those rows were dropped

# # Q2 - Does age have an impact on lifting capacity

# ### Catgorizing by Age

# In[ ]:


def squat_calculate(x):
    if(x < 10.0):
        return "05-10"
    if(x >= 10.0 and x < 20.0):
        return "10-20"
    if(x >= 20.0 and x < 30.0):
        return "20-30"
    if(x >= 30.0 and x < 40.0):
        return "30-40"
    if(x >= 40.0 and x < 50.0):
        return "40-50"
    if(x >= 50.0 and x < 60.0):
        return "50-60"
    if(x >= 60.0 and x < 70.0):
        return "60-70"
    if(x >= 70.0 and x < 80.0):
        return "70-80"
    if(x >= 80.0 and x < 90.0):
        return "80-90"
    else:
        return "90-100"
    


data['Agecategory'] = pd.DataFrame(data.Age.apply(lambda x : squat_calculate(x)))


# In[ ]:


data.head(20)


# ### Calculating Average Best [Squat, Bench,Deadligt] by Age Category for Male/Female Athletes

# In[ ]:


data_male = pd.DataFrame(data[data['Sex'] == 'M'])
data_female = pd.DataFrame(data[data['Sex'] == 'F'])
lifting_capacity_m = pd.DataFrame(data_male.groupby('Agecategory')[['BestSquatKg','BestBenchKg','BestDeadliftKg']].mean()).reset_index()
lifting_capacity_f = pd.DataFrame(data_female.groupby('Agecategory')[['BestSquatKg','BestBenchKg','BestDeadliftKg']].mean()).reset_index()


# ### Resetting the Indexes

# In[ ]:


#plt.figure(figsize = (20,10))
lifting_capacity_m.plot(kind = 'bar', color = ['#63cdd7','#0093b7','#005f89'], figsize = (15,10), x = 'Agecategory', rot = 30)


# # Intuition : 
# 
# For Male Atheletes
# age group 30-40 lifted the maximum in either of the categories, followed by age groups 20-30 and 40-50.

# In[ ]:


lifting_capacity_f.plot(kind = 'bar', color = ['#f9dff0','#f0acc3','#e54370'], figsize = (15,10), x = 'Agecategory', rot = 30)


# # Intuition :
# 
# For Female Atheletes the intuition holds true as age group 30-40 lifted the maximum in either of the categories, followed by age groups 20-30.

# # Q3 - How big of a difference does gender make?

# In[ ]:


import seaborn as sns
plt.figure(figsize = (20,15))

plt.subplot(1,3,1)

plt.ylim(0,600)
sns.violinplot(data = data, x = 'Sex', y = 'BestSquatKg',hue = 'Sex', scale = 'count',dodge = True, palette = ['#e54370','#0093b7'])
plt.style.use("fast")
plt.title('Squat Capacity by Gender')
plt.xlabel('Gender')
plt.ylabel('Squat Lifting Capacity')


plt.subplot(1,3,2)
plt.ylim(0,500)
plt.style.use("fast")
sns.violinplot(data = data, x = 'Sex', y = 'BestBenchKg',hue = 'Sex',scale = 'count',dodge = True, palette = ['#e54370','#0093b7'])
plt.xlabel('Gender')
plt.ylabel('Bench Lifting Capacity')
plt.title('Bench Capacity by Gender')


plt.subplot(1,3,3)
plt.ylim(0,500)
plt.style.use("fast")
sns.violinplot(data = data, x = 'Sex', y = 'BestDeadliftKg',hue = 'Sex',scale = 'count',dodge = True, palette = ['#e54370','#0093b7'])
plt.xlabel('Gender')
plt.ylabel('Deadlift Lifting Capacity')
plt.title('Deadlift Lifting Capacity by Gender')


plt.show()


# # Intuition
# 
# Gender played a key role in determining the lifting capacity of the atheletes,
# in all the categories i.e. bench, squat and deadlift male atheletes lifted higher 
# as compared to their female counterparts.

# # Q4 How much influence does overall weight have on lifting capacity?

# In[ ]:


data_male = pd.DataFrame(data[data['Sex'] == 'M'])
data_female = pd.DataFrame(data[data['Sex'] == 'F'])
bodyw_lcm = pd.DataFrame(data_male.groupby('Agecategory')[['BodyweightKg','BestSquatKg','BestBenchKg','BestDeadliftKg']].mean()).reset_index()
bodyw_lcf = pd.DataFrame(data_female.groupby('Agecategory')[['BodyweightKg','BestSquatKg','BestBenchKg','BestDeadliftKg']].mean()).reset_index()


# In[ ]:


bodyw_lcm 


# In[ ]:


bodyw_lcf


# ### Total Lifting Capacity

# In[ ]:


bodyw_lcm['Total'] = bodyw_lcm['BestSquatKg'] + bodyw_lcm['BestBenchKg']+bodyw_lcm['BestDeadliftKg']
bodyw_lcf['Total'] = bodyw_lcf['BestSquatKg'] + bodyw_lcf['BestBenchKg']+bodyw_lcf['BestDeadliftKg']


# In[ ]:


bodyw_lcm['wRatio'] = bodyw_lcm['Total']/bodyw_lcm['BodyweightKg']
bodyw_lcf['wRatio'] = bodyw_lcf['Total']/bodyw_lcf['BodyweightKg']


# In[ ]:


bodyw_lcm


# In[ ]:


bodyw_lcf


# In[ ]:


plt.figure(figsize = (20,10))
plt.plot(bodyw_lcm.Agecategory,bodyw_lcm.wRatio, color = '#0093b7')
plt.plot(bodyw_lcf.Agecategory,bodyw_lcf.wRatio, color = '#e54370')
#plt.plot(bodyw_lcf.Agecategory, y = bodyW_lcf.wRatio, kind = 'line')


# # Intuition
# 
# The idea is to follow the same analogy as ants. Each ant has the ability to lift significantly higher weight in comparison to its own weight.  For the male and female atheletes of all age groups, their lifting capacity as a rtio of their own weight was computed. 
# 
# 
# Results showed that for age group 20-30, the ratio of Total Weight lifting capacity to avg weight of atheletes was highest in this category, implying that atletes belonging to this age group could lift the highest weight in comparison to their own weight.

# # Fin!
