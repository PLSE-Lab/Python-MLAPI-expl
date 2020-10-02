#!/usr/bin/env python
# coding: utf-8

# **Introduction**
# 
# In this dataset we have details of every employee.We have to find out who are in the condition of leaving prematurely.
# To do this first we have to analyse our data.

# In[2]:



import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

hr = pd.read_csv("../input/HR_comma_sep.csv")
print(hr.head())


# Following are the details in the column accordingly:
# 
# Satisfaction level
# 
# Last evaluation
# 
# Number of projects
# 
# Average monthly hours
# 
# Time spent at the company
# 
# Whether they have had a work accident
# 
# Whether the employee has left
# 
# Whether they have had a promotion in the last 5 years
# 
# Department
# 
# Salary
# 
# **Objective**:-
# In this last two columns , they are type string i.e Department and salary:
# 
# Our first objective is to convert department and salary into integer.

# In[3]:


print('dept:')
print(set(hr['sales']))
print('salary:')
print(set(hr['salary']))


# In[4]:


sales = ['management','RandD','hr','accounting','marketing','product_mng','IT','support','technical','sales']
salary = set(hr['salary'])
for i in range(hr.shape[0]):
    for j in range(len(sales)):
        if(hr['sales'][i]==sales[j]):
            hr.set_value(i,'sales',int(j))
    if(hr['salary'][i]=='low'):
        hr.set_value(i,'salary',int(3))
    if(hr['salary'][i]=='medium'):
        hr.set_value(i,'salary',int(2))
    if(hr['salary'][i]=='high'):
        hr.set_value(i,'salary',int(1))
hr[['sales']]=hr[['sales']].apply(pd.to_numeric)
hr[['salary']]=hr[['salary']].apply(pd.to_numeric)
print(hr.head())


# **Objective**:To convert float data into integer of column  satisfaction_level and last_evaluation

# In[5]:


hr['satisfaction_level']=100*hr['satisfaction_level']
hr['last_evaluation']=100*hr['last_evaluation']
hr['satisfaction_level']=hr['satisfaction_level'].astype(np.int32)
hr['last_evaluation']=hr['last_evaluation'].astype(np.int32)
print(hr.head())


# **Objective**: To find out how much the data is correlatable with the label('left' column)

# In[6]:


plt.figure(figsize=(10,10))
h = sns.heatmap(hr.corr(),annot=True,square=True,annot_kws={"size":9})
plt.title('Correlation between features')
plt.show()


# 
# 
# -Correlation heatmap gives you the idea about how the two independent variables relates with each other.
# 
# -Means if both the variables are increasing then it goes towards 1 and vice versa. If correlation is towards 0 then there is no relation between them. 
# 
# -If the correlation between the label('left') and the variable is towards 0 then we do not get much confidence level in our algorithm .
# 
# -we can clearly see that columns like :
# 
# 1. last_evaluation
# 
# 2.  number_of_project
# 
# 3.average_monthly_hours
# 
# 4.promotion_last_5years
# 
# 5.sales
# 
#    
# 
# 
# 
#  do not have much correlation with left column
# 
# - To increase our confidence we have to add more features in data frame
# 
# **Objective**:Analyzing  columns using graphs

# In[7]:


left = hr[hr['left']==1]
non_left = hr[hr['left']==0]
sns.kdeplot(left['last_evaluation'])
sns.kdeplot(non_left['last_evaluation'],shade=True)
plt.show()


# -As we can see there is two peak in the left person's last evaluation .
# 
# -But there a drop in between 60 to 80 and inappropriate constant line of non left people in between 55 to 82.
# 
# ----------
# Now coming towards number of projects.
# Since number of project column contains discrete numbers therefore:

# In[8]:



sns.countplot('number_project',hue='left',data=hr)
plt.show()


# We can see majority of people who had done projects between 3 to 5 do not left.
# 
#  Now for Average working hours in a month:

# In[9]:


sns.kdeplot(left['average_montly_hours'])
sns.kdeplot(non_left['average_montly_hours'],shade=True)
plt.show()


# 
# Comming to promotion of last 5 years.Since it is like sparse row we do not expect too much in affecting the coinfidence level

# In[10]:


print(set(hr['promotion_last_5years']))


# In[11]:


sns.countplot('promotion_last_5years',hue='left',data=hr)
plt.show()


# In[12]:


print(hr.shape[0])
print(hr[hr['promotion_last_5years']==1].shape[0])
print(non_left[non_left['promotion_last_5years']==1].shape[0])
print(left[left['promotion_last_5years']==1].shape[0])


# OK! Out of **14999** employees there are only **319** employees who got the promotion in last 5 years in which **300** do not left but **19** people left. Almost 2% people got promotion which do not affect much in the confidence level.

# Last but not the least column i.e. department .
# 
# Unlike other columns it can give information about the other variables like expected working hours and salary for every each department.
# 
# let a see the data:

# In[13]:


sns.countplot('sales',hue='left',data=hr)


# As we can see the number of remains is greater then the number of left.
# 
# But there is a difference between the ratio of the left:remain in all the departments

# In[14]:


m = []
for i in range(0,10):
    print(left[left['sales']==i].shape[0]/hr[hr['sales']==i].shape[0])
    m.append(left[left['sales']==i].shape[0]/hr[hr['sales']==i].shape[0])


# In[15]:


n = m
n = sorted(n)
rename = []
for i in range(0,10):
    for j in range(0,10):
        if(m[i]==n[j]):
            rename.append(j)
hr['sales'].replace(range(0,10),rename,inplace=True)
sns.countplot('sales',hue='left',data=hr)


# Now we will see the correlation between the sales and other groups by graphs. 

# In[16]:


left = hr[hr['left']==1]
non_left = hr[hr['left']==0]
sns.countplot('sales',hue='salary',data=left)


# In[17]:


sns.violinplot(x='sales',y='satisfaction_level',hue='left',data=hr,split=True)


# In[18]:


sns.violinplot(x='sales',y='last_evaluation',hue='left',data=hr,split=True)


# In[19]:


sns.countplot('sales',hue='number_project',data=left)


# In[20]:


sns.violinplot(x='sales',y='average_montly_hours',hue='left',data=hr,split=True)


# In[21]:


sns.countplot('sales',hue='time_spend_company',data=non_left,palette=sns.light_palette("green"))
sns.countplot('sales',hue='time_spend_company',data=left,palette=sns.cubehelix_palette(8))#violet


# In[22]:



sns.countplot('Work_accident',hue='sales',data=non_left,palette=sns.cubehelix_palette(8))#violet
sns.countplot('Work_accident',hue='sales',data=left,palette=sns.light_palette("green"))


# In[23]:


sns.countplot('promotion_last_5years',hue='sales',data=non_left,palette=sns.cubehelix_palette(8))#violet
sns.countplot('promotion_last_5years',hue='sales',data=left,palette=sns.light_palette("green"))


# **Graph Conclusion**:
# 
# Based on all the graphs mentioned above we conclude followings:
# 
# 1.people who left jobs are in scarcity in between 55 to 80 in last evaluation
# 
# 2.people who have average monthly hours in between 160 to 260 rarely left jobs
# 
# 3.people who had worked for 2,3 or 4 years mostly don't left their jobs 
# 
# 4.people who have done 3,4 or 5 projects normally do not left jobs
# 
# 5.In Sales(Department):
# 
#    a.Those who are in pay grade 3 mostly left their jobs after 2.
# 
#    b.By visualizing Last evaluation vs department violin graph we can say that people who are below or above average              mostly left their jobs.
# 
#    c.Time spend in the company , Work accident , promotion in last 5 years do not affect much in job left which is expected.
# 

# **Introduction of new features**
#         On the basis of above visualization we add new features in order to gain more confidence level .
#  

# In[24]:


d_s=[]
d_a=[]
d_b=[]
d_c=[]
for i in range(hr.shape[0]):
    if(hr['last_evaluation'][i] in range(55,80)):
        d_s.append(1)
    else:
        d_s.append(0)
    if(hr['average_montly_hours'][i] in range(160,260)):
        d_a.append(1) 
    else:
        d_a.append(0)
    if(hr['time_spend_company'][i] in [2,3,4]):
        d_b.append(1)
    else:
        d_b.append(0)
    if(hr['number_project'][i] in [3,4,5]):
        d_c.append(0)
    else:
        d_c.append(1)
hr['discrete_last_evaluation']=pd.Series(d_s)
hr['expected_avg_hr']=pd.Series(d_a)
hr['shouldbe_time_spend_company']=pd.Series(d_b)
hr['expected_no_of_projects']=pd.Series(d_c)
hr.head()


# I made my effort to make this notebook more understandable .Its my first notebook so I request you to give me feedback and suggestions in comments for analyzing the data and introducing more features for greater confidence level.
