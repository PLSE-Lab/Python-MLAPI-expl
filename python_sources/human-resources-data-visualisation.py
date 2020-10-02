#!/usr/bin/env python
# coding: utf-8

# In this notebook, you will find a visualization for Human Resources data set. The following cell is for setup code.

# In[ ]:


from collections import Counter
import pandas as pd
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import numpy as np
print("Setup Complete")


# The following code is for reading the data from the file.

# In[ ]:


human_filepath = '../input/human-resources-data-set/HRDataset_v13.csv'
human_data = pd.read_csv(human_filepath)


# # Data Exploration

# In[ ]:


human_data


# In[ ]:


human_data.columns


# In[ ]:


human_data.shape


# The following code is for dropping the empty rows from the data frame.
# 

# In[ ]:


nan_value = float("NaN") #Convert NaN values to empty string
human_data.replace("", nan_value, inplace=True)
human_data.dropna(subset = ["Employee_Name"], inplace=True)


# In[ ]:


print(human_data.shape)


# In[ ]:


human_data


# In[ ]:


human_data.describe()


# In[ ]:


print(human_data['Sex'].value_counts())
human_data['Sex'].value_counts().plot(kind='bar')


# It is clear that the majority of the employees are women.
# Now we want to know which departments, they work in?

# In[ ]:


plt.figure(figsize=(16,5))
sns.countplot(x=human_data['Department'],hue=human_data['Sex'])


# Women work in all departments by close proportion of men, except in production department the women are the majority of the employees. 

# Is there any relationship between who a person works for and their performance score?

# In[ ]:


plt.figure(figsize=(16,5))
sns.countplot(x="ManagerName", data=human_data)
plt.yticks(np.arange(0, 25,2))
plt.xticks(rotation = 80)


# From the previous chart, we notice that the employees distributed under nearly 14 managers, so what are their departments?

# In[ ]:


plt.figure(figsize=(16,5))
sns.countplot(x="ManagerName", hue="Department", data=human_data)
import numpy as np
plt.yticks(np.arange(0, 22, 2))
plt.xticks(rotation = 80)


# The managers from Production, IT/IS, and Sales. Moreover, most of the employee and managers are in the Production department, following it the IT/IS, then the Sales department comes.

# In[ ]:


plt.figure(figsize=(16,5))
sns.countplot(x="ManagerName", hue="PerformanceScore", data=human_data)
import numpy as np
plt.yticks(np.arange(0, 22, 2))
plt.xticks(rotation = 80)


# The previous chart is between the manager name and its employees classified on Performance Scores.
# 
# Eor example, The Manager Brannon Miller has the worest Performance in the data frame, whereas David Stanley has the best data set in the dtat frame.

# Is there any relation between Employees performances and their employee satisfaction?

# In[ ]:


plt.figure(figsize=(16,5))
sns.countplot(x="ManagerName", hue="EmpSatisfaction", data=human_data)
import numpy as np
plt.yticks(np.arange(0, 12))
plt.xticks(rotation = 80)


# It is noticed that the manager Brannon Miller has the worst employees satisfaction, and this may be because the employees have the worst evaluation in the Preference Scores.

# In[ ]:


plt.figure(figsize=(16,6))
#sns.distplot(a=human_data['PositionID'], kde=False)
sns.countplot(x="Position", data=human_data)
plt.yticks(np.arange(0, 150, 10))
plt.xticks(rotation = 80)


# The previous chart shows the distribution of the employees on positions in the company. It is clear that the Production Technician I and Production Technician II, represent two-thirds of positions for the employees.

# In[ ]:


plt.figure(figsize=(16,5))
sns.swarmplot(y=human_data['PayRate'],
              x=human_data['Position'])
plt.xticks(rotation = 80)


# As the chart shows the Production Technician I and Production Technician II, represent the lowest part of employees Pay Rate in the company.

# In[ ]:


plt.figure(figsize=(16,5))
sns.kdeplot(data=human_data['PayRate'],shade=True)
plt.xticks(np.arange(0, 100,5))


# As it is clear from the chart, two-third of the employees, their Pay Rate cannot exceed 40 $/hr.

# Is there discrimination based on race in the company?

# In[ ]:


plt.figure(figsize=(12,5))
sns.countplot(y=human_data['RaceDesc'])


# The majority of the company employees are white people, but we can't determine if there is discrimination based on race in the company? or not. We should test another factor for more accuracy.

# In[ ]:


plt.figure(figsize=(16,8))
sns.swarmplot(x=human_data['PayRate'],
              y=human_data['RaceDesc'])
#plt.yticks(rotation = 80)


# The chart shows that white people have the highest Pay Rates at the company, but this is because of their jobs or for another reason we don't know.

# # Is there any relation between Performance Score and Special Projects that you worked?

# In[ ]:


plt.figure(figsize=(12,6))
sns.regplot(x=human_data['PerfScoreID'],
            y=human_data['SpecialProjectsCount'])


# Yes, there is a slight positive relationship between Performance Score and Special Projects the employee has worked.

# In[ ]:


plt.figure(figsize=(12,6))
sns.regplot(x=human_data['SpecialProjectsCount'],
            y=human_data['PayRate'])


# From the previous chart it is clear that there is a positive relationship between the Projects the employee has worked and his Pay Rate.

# In[ ]:


plt.figure(figsize=(12,6))
sns.regplot(x=human_data['PayRate'],
            y=human_data['PerfScoreID'])


# But there isn't a relationship between a Preference Score of the employee and his/her Pay Rate.

# # Does the Preference Score of the employee-related of his/her Pay Rate and based on his/her gender?

# In[ ]:


plt.figure(figsize=(16,6))
sns.lmplot(x='PayRate', y='PerfScoreID', hue='Sex', data=human_data)


# Not really related, but it seems larger for Females.

# 

# In[ ]:


plt.figure(figsize=(16,5))
sns.countplot(x="Position", hue="PerfScoreID", data=human_data)
import numpy as np
plt.yticks(np.arange(0, 120, 10))
plt.xticks(rotation = 80)


# In[ ]:


plt.figure(figsize=(16,4))
sns.countplot(x="Position", hue="EmpSatisfaction", data=human_data)
import numpy as np
plt.yticks(np.arange(0, 50, 5))
plt.xticks(rotation = 80)


# From the previous 2 charts, it seems most of the employees satisfy their positions and departments especially in production department.

# In[ ]:


plt.figure(figsize=(14,8))
sns.regplot(x=human_data['PayRate'],y=human_data['EmpSatisfaction'])


# The chart shows that Employees Satisfaction increases when they Pay Rate increase.

# # What is the variety of Marital status of the company's employees?

# In[ ]:


human_data['MaritalDesc'].value_counts().plot(kind='bar')


# It seems that the majority of employees are single or married, but how many of them are still active?

# In[ ]:


plt.figure(figsize=(16,6))
sns.countplot(x=human_data['EmploymentStatus'],hue=human_data['MaritalDesc'])


# It seems that two-thirds of them still active and the majority of active employees are single, and most of whom left the work were married.

# # What the best resource to have new employees from it?

# In[ ]:


plt.figure(figsize=(16,5))
sns.countplot(x=human_data['RecruitmentSource'],hue=human_data['PerformanceScore'])
plt.xticks(rotation = 80)


# It is clear that the best sources are Employee Referral and Search Engine - Google Bing Yahoo, because they were the most performance employees in the company.
