#!/usr/bin/env python
# coding: utf-8

# **Guideline for Take Home Exam 1 of IPE 205:--**
# 1. Roll number 1-30 will work on the dataset "Grading of the students in the exam (IPE101) raw" 
# & the remaining students will work on "Marking of the students in the exam (IPE101) raw". The file name will be "Take home exam 1 2017032\**"
# 2. You will FORK this code & work on that
# 3. In the dataset given, most data is in "String" format, convert them into int/float
# Replace the values of the entire row that match with your roll number with your roll number programmatically. For instance,
# your roll number is 05, replace all values in 5th row with 5
# [[Example of Replacing values](https://github.com/tanmoyie/Applied-Statistics/blob/master/Projects%20%26%20Dataset/Performance%20of%20the%20students%20of%20IPE%2C%20MIST/Replacing%20values.png)] 
# 
# 4. 
# variable_x1 = column number equal to the last digit of your roll number. If your roll is 21, you will choose 1st column; 
# variable_y1 = column of total mark in class tests; 
# variable_x2 = any column u wish; 
# variable_y2 = total mark in final exam (the column "Mark total 300")
# Since Python is a 0 indexing language, if your roll number is 201703110 (last digit is 0), 
# you will choose column 1, variable_x1 = dataframe(:,0) 
# 
# 5. Draw a Scatter plot (variable_x1 vs variable_y1) 
# 6. Draw a Scatter plot (variable_x2 vs variable_y2)
# 7. Explan the Scatter plots you obtain
# 8. Draw a Histogram based on your dataset & interpret it
# 9. Draw a Box Plot of your dataset & interpret it.
# 
# 10. You would find help in the link 
# https://www.kaggle.com/tanmoyie/exploratory-data-analysis-eda
# https://www.kaggle.com/tanmoyie/python-introduction

# # Roll Number: 201736044
# # Name: Sohag Das Sourav
# 

# 

# In[ ]:


import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))
import matplotlib.pyplot as plt


# In[ ]:


# load the EXCEL file & read the data 
dataframe1 = pd.read_csv("../input/students mark.csv")
print(dataframe1) # printing original dataset
dataframe2 =dataframe1.dropna() # eliminating NaN & extra strings
print("\n\n\n\n-------------------------------------neat and clean dataset---------------------\n\n")
print(dataframe2) 
print(dataframe2.dtypes) # data types
input_data = dataframe2.values


# In[ ]:


#roll : 201736044 
# Scatter plot of CT-3 vs Total mark in CT
import matplotlib.pyplot as plt
print("-------------Scatter Plot: 'CT-3' vs Total mark in CT -----------")
variable_x1 = input_data[:,3] # column 4 ( CT-3 )
variable_y1 = input_data[:,5] # column 6 ( Class Test Total of best 3, Marks: 60 ) 
plt.scatter(variable_x1, variable_y1)# Scatter plot for CT-3 vs Total mark in CT
plt.xlabel('column 4 ( CT-3 )')
plt.ylabel('column 6 ( Class Test Total of best 3, Marks: 60 )')
plt.title("Scatter Plot: 'CT-3' vs Total mark in CT")
plt.show() 
# Scatter plot of 'Roll' vs 'Mark total 300'
print("------------- Scatter Plot:'Roll' vs 'Mark total 300' -----------")

variable_x2 = input_data[:, 5]  # column 6  ( Class test total mark )
variable_y2 = input_data[:, 11] # column 12 (Mark total 300)
plt.scatter(variable_x2, variable_y2) # Scatter plot of 'Roll' vs 'Mark total 300'
plt.xlabel('Roll')
plt.ylabel('Mark total 300')
plt.title("Scatter Plot:'Roll' vs 'Mark total 300'")
plt.show() 


# In[ ]:


dataframe2.boxplot()


# In[ ]:


dataframe2.hist()

