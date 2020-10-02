#!/usr/bin/env python
# coding: utf-8

# **Problem Scenario**
# 
# You are a faculty member in the Industrial Engineeering department at a university, & taking the course *Modeling & Simulation*. Now, the **HEAD** of the department is asking about the performance of the students in that course. Lets say the blackbox below is the dataset (the mark obtained by students in the exam).

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
marks_in_exam = np.random.randint(10, 100, 50) # randomly generated dataset
print('Marks in the exam:', marks_in_exam) 


# Now, if you tell your Head about the **average performance** of the students, is it meaningful? I mean, does Head can get an idea of the *academic excellence* of the students  just from the MEAN? 
# 
# * **Hell NO! **
# 
# Instead, if we tell that **25%** of the students are performing superb in Modeling & Simulation, next 50% students are good enough to survive, and bottom 25% are poorly performing (may be getting below 40 out of 100 marks), the Head would understand the situation of the class. This percentile can be represented by **Box Plot**.

# In[ ]:


print('Mean of the Marks:',np.mean(marks_in_exam)) # calculate the mean
print()  
plt.boxplot(marks_in_exam) #, meanline=True, showmeans=True 
Q1 = np.percentile(marks_in_exam, 25)
print('The 25 percentile is:', Q1)
Q3 = np.percentile(marks_in_exam, 75)
print('The 75 percentile is:', Q3)

