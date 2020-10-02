#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# # A Pandas Tutorial
# 
# This Tutorial will include:
#     1. Importing Data
#     2. Reading the data
#     3. Selecting Data
#     4. Analyzing Data
#     5. Graphing Data
#     
# 
#         > This dataset lists IBM employees and categorizes them into two groups, Those who have left their jobs ("Attrition" = 
#         Yes") and those who have not left their jobs at IBM ("Attrition" = "No"). The data also provides other information on
#         the employee, such as, how long they have worked at IBM, their daily pay rate, how far away from work they live. By
#         analyzing these factors, we can determine which contribute to the an employee staying or leaving their job at IBM.

# # 1. Importing the Data
# 
# The first step in analyzing data is to upload the data into python. This can be done using the pandas read_csv function.
# 
# 
# Note: 
# > 1. Ensure quotes (" ") are put around the file name within the read_csv function (i.e. pd.read_csv("file_name.csv"))
#         Within the quotes, you want to put the path to your file. (Ex: C:/Documents/DSO499/FinalProject/final.csv) 
#         In this case, the directory path is provided by running the code above.
# > 2. CSV = Comma Separated Value
# > 3. You can set the index of the dataframe when using read_csv by doing pd.read_csv("file.csv", index_col="ColumnName"). You can also set the index after reading in the file. This will be covered in Step 2.
# 

# In[ ]:


employees = pd.read_csv("/kaggle/input/ibm-hr-analytics-attrition-dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv")


# # 2. Reading the data
# 
# After uploading the data, you want to take a look at it. The dataset is very large, so we can just look at the first couple of rows to get an idea of how the data is structured.
# 
# Use, employees.head() to see the first 5 (5 is the default) rows of the dataframe.
# > Note: To see a different number of rows, put desired number of rows in the functions input (i.e. employees.head(20) will show the first 20 rows)

# In[ ]:


employees.head()


# Based on the first 5 rows, we can see that each row is an employee at IBM. Each employee has many characteristics to define them including; age, how often they travel on business, how far work is from home, etc. 
# 
# A thing to note is the first column in the output (the one that has no header and starts counting from 0 and incrementing by one), otherwise known as the index column. If you do not specify an index column when reading in the csv file (Step 1), you can do so using the function DataFrame.set_index()
# 
# An important column is attrition. The purpose of this dataset is to see if there are any factors that contribute to a reduction in attrition. So lets practice setting the dataframe index to "Attrition".
# 
# Note:
# > If you want to save the dataframe with the new column, you can't just call the function. You have to save the new version to a variable. For example:
# * < employees.set_index("Attrition") > **WOULD NOT** result in the dataframe employees being overriden with the new dataframe
# * < employees = employees.set_index("Attrition") > **WOULD** result in the dataframe employees having the new version of the dataframe with "Attrition" as the index column

# In[ ]:


#Try it out.
# 1. Run the first line of code. 
# 2. Then run only the second line to see that the employees dataframe hasn't changed.
# 3. Then run the third and fourth line of code to see the new dataframe

employees.set_index("Attrition")
employees
employees = employees.set_index("Attrition")
employees


# In[ ]:


# Resetting the index is similar to changing the index
# Make sure you save the new dataframe to its own variable and use the function reset_index() to revert the index back to the original
employees = employees.reset_index()


# Lets take a look at how many rows (employees) and how many columns are in the dataset using employees.shape

# In[ ]:


# employees.shape provides how many rows and columns are in the dataframe in the format (rows, columns)
employees.shape


# There are 1470 employees in the dataframe and 35 columns of characteristics
# 
# 35 columns is a lot to work with, so lets choose a few that look interesting!

# # 3. Selection
# There are a lot of columns, and not all of them seem useful in our analysis, so we will par it down to a few columns that seem to be important.
# 
# 
# We can select specific parts of the dataframe using the following two methods:
# 
# 
# 1. employees.iloc[  ] --> Index Slicing
# 2. employees[employees.Age == 20]  --> Conditional Selection

# **Selection Method #1: Index Slicing**
# 
# iloc[ ] is a great way to slice a dataframe into more digestible slices
# > The easiest way to remember the use of iloc[ ] is to think of the i in iloc to stand for index
# 
# iloc slices based on the index of the dataframe. You can slice by both row and columns. This is the format:
# 
# employees.iloc[start_row : end_row , start_column : end_column]
# 
# ***Example***
# > employees.iloc[0:3, 0:4]
# 
#                 
# > 0:3 is the indices for rows. It will show rows 0, 1, and 2
# 
# > 0:4 is the indices fro columns. It will show columns 0, 1, 2, and 3
# 
#         Note:
#             Index slicing always results in the the end_row or end_column being exclusive (slice up to but not including 
#             the end_row or end_column value       

# In[ ]:


# Test it out
employees.iloc[0:3,0:4]


# **Selection Method #2: Conditional Selection**
# 
# Conditional Selection is other method of slicing a dataframe. This approach allows you to slice by the values within the dataframe rather than just the index.
# 
# > For example:
# > We can slice the employees dataframe and have it show us only the rows where the value of the attrition column is "No"
# 
# By selecting only those rows where the employee Attrition value is "No", we can see if any columns might be a contributing factor to the individual not leaving the company.

# In[ ]:


employees[employees.Attrition == "No"]


# We are going to look at DistanceFromHome, WorkLifeBalance, Age, EnvironmentSatisfaction, DailyRate, YearsAtCompany, and YearsSinceLastPromotion in our analysis. We will determine if any of these factors can account for an employee choosing to leave or not to leave IBM.
# 
# We are going to create a new dataframe called **employees_select** and use that dataframe for our analysis.
# 
#     Note:
#     If you want to select more othan 1 column, you must use double brackets to when listing the columns.
#     
#     Example:
#     employees[ [ "Attrition", "Age'] ]

# In[ ]:


employees_select = employees[["Attrition", "Age", "DistanceFromHome", "WorkLifeBalance", "EnvironmentSatisfaction", "DailyRate",                              "YearsAtCompany", "YearsSinceLastPromotion"]]
employees_select


# # 4. Analyzing Data
# 
# After selecting a few columns, now can analyze them to determine which of them are truly important in determining Attrition. We will do this using several of the following functions:
# 
# 1. groupby()
# 2. mean()
# 3. groupby().size()

# **Groupby and mean**
# 
# --- Groupby ---
# 
# Groupby involves three steps:
# 1. Split - Choose a column to split the dataframe by 
# > In this case, we use Attrition which has 2 values ("Yes", "No")
# > The dataframe will split the rows based on the value in Attrition (Yes's go together, No's go together)
# 2. Apply - Choose a function to apply to each group created in the split (mean, std, median, etc.)
# > In this case, we use mean() to find the average values for each column in the employees_select dataframe
# 3. Combine - Combine the results into a dataframe 
# 
# 
# > Note: If you don't want to include all columns in the Split-Apply-Combine process, you can select which columns to include.
#     
#     > Example:
#     > employees_select.groupby("Attrition")[["Age", "DistanceFromHome"]].mean()
#     
#     
# >> Output:
#     
#           	    Age	        DistanceFromHome
#     Attrition		
#     No	         37.561233	8.915653
#     Yes	        33.607595	10.632911

# In[ ]:


employees_select.groupby("Attrition").mean()


# Based on the averages calculated above, we can see if any of them are useful in determining a pattern in an employees Attrition status. We will do this by conditionally selecting rows that are larger (or smaller) to the mean for the Attrition value.
# 
# For example, if we want to see what percentage of employees with the Attrition value of "No" can be expalined by their DistanceFromHome. 
# 
# 1. We will select all employees whose DistanceFromHome is less than the mean for DistanceFromHome for the Attrition value (10.632911)
# 2. Save that dataframe to the variable employees_attrition
# 3. Group by Attrition and then select only the attrition column
# 4. Select the column for "No"
# 
# Divide by the number of No's in the original dataset
# 1. Split the employees database by Attrition
# 2. Select Attrition column
# 3. Apply the size()["No"] function to get the number of "No"s
# 

# In[ ]:


employees_attrition = employees[employees.DistanceFromHome <= 10.632911]
employees_attrition.groupby("Attrition").Attrition.size()["No"]/employees.groupby("Attrition").Attrition.size()["No"]


# Around 71% of employees that live less than 10.632911 miles away from work have the Attrition status "No". This indicates that employees that live close to work are likely to stay at work. 
# 
# We can also check it the other way. Are employees that live further away more likely to have the Attrition status "Yes"
# 

# In[ ]:


employees_attrition = employees[employees.DistanceFromHome >= 10.632911]
employees_attrition.groupby("Attrition").Attrition.size()["Yes"]/employees.groupby("Attrition").Attrition.size()["Yes"]


# Around 39% of employees that live more than 10.632911 miles away from work have the Attrition status "No". This indicates that employees that live further from work are not as likely to have the Attrition status "Yes".

# In[ ]:


# Try this with another factor!


# # 5. Graphing the Data
# 
# Sometimes, it is easier to understand data when you can see it. Pandas allows you to visualize the data by plotting it. Pandas also allows you to combine different functions.
# 
# You can groupby, get the mean, select specific columns, and plot data all in one line. We can see the means for the first four columns and plot a horizontal bar chart based the average based on attribtion ("Yes" or "No")
# 
# Another function that can be used is sort_values(). This allows for sorting based on a value, in this case Attrition.
# > The default direction for sort_values is ascending
# 
# The plot function allows you pick what type of plot you want to use. plot.scatter(), plot.bar(), plot.pie(), etc. 
# 
# Play around with the different types of plots to see the results.
#     

# In[ ]:


# Here is one example. plot.barh() is a horizontal bar plot
employees_select.groupby("Attrition").mean().iloc[:,0:4].sort_values("Attrition", ascending=False).plot.barh()


# # 6. Conclusion
# 
# Employees that live closer to work at more likely to continue their jobs at IBM. The DistanceFromHome factor illustrates that if IBM can determine which factors truly matter to their employees, they can reduce their turnover rate resulting in employees that stay longer and have highly skilled and committed workforce.
