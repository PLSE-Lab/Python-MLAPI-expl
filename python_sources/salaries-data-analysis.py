#!/usr/bin/env python
# coding: utf-8

# # Salaries_Data_Analysis

# ## Libraries Import -

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sal = pd.read_csv("../input/Salaries.csv")

sal.head()


# ## Before to do anything with dataframe, simply explore the data a bit. Lets, have a look at the datatypes of each columns-

# In[ ]:


sal.dtypes


# ## Let's take a look at other columns using 'value_counts()' method-

# In[ ]:


sal['Year'].value_counts()


# In[ ]:


sal['Notes'].value_counts()


# In[ ]:


sal['Agency'].value_counts()


# In[ ]:


sal['Status'].value_counts()


# ## Dealing with non numeric value
# 
# ### Dropping down the rows which has no information. drop by index method is used

# In[ ]:


sal = sal.drop([148646, 148650 , 148651 , 148652])  ## All rows which contain 'NOT Provided' will be droped 


# ### Lastly, let's drop the Notes columns as it does not provide any information.

# In[ ]:


sal = sal.drop(columns = ['Notes'])


# ### Not everyone will recieve benefits for their job, so it makes more sense to fill in the null values for Benefits with zeroes.

# In[ ]:


sal['Benefits'] = sal['Benefits'].fillna(0)  


# # Now, Task is to fill up the NaN in Status column by PT and FT appropriately.
# 
# ## Lets consider, Job Title = Transit Operator. 

# In[ ]:


## JTTO = Job Title Transit Operator

JTTO = sal.loc[sal['JobTitle'] == 'Transit Operator']

## Total no. FT status for transit operator

Status_FT = JTTO.loc[JTTO['Status'] == 'FT']

## Mean value of Base Pay associated with FT 
Mean_FT= Status_FT['BasePay'].astype('Float64').mean()
print("Mean_of_FT=", Mean_FT)

## Total no. NaN status for transit operator
Status_NaN = JTTO[JTTO['Status'].isna()]

## Variable a contains basepay associated with NaN Status 
a = Status_NaN['BasePay'].astype('Float64')

## Variable b contain NaN Status
b = Status_NaN['Status']


# ##  len(a.index) = it will show a length of a's Index ('a' contains data of basepay associated with NaN Status) and ('b' contain NaN Status) 
# ## here for instance, if 0th location of 'a' is greater than mean of basepay for FT then 0th location of 'b' will be filled by FT else PT
# 

# In[ ]:


for i in range(0, len(a.index)):
    
    if (a.iloc[i] > Mean_FT):
        b.iloc[i] = 'FT'
    else:
        b.iloc[i] = 'PT'


# ### Create a New Dataframe called New_df which has updated NaN_Status.
# ### Lastly, update a JTTO column 

# In[ ]:


New_df= pd.DataFrame(Status_NaN['Status'])

JTTO.update(New_df)


# In[ ]:


JTTO['Status'].value_counts()


# In[ ]:


JTTO.head()


# # Data Visualization
# 
# ## Full Time vs Part Time employee

# In[ ]:


data_FT = JTTO[JTTO['Status'] == 'FT']
data_PT = JTTO[JTTO['Status'] == 'PT']

fig = plt.figure(figsize=(8, 8))
ax = sns.kdeplot(data_PT['TotalPayBenefits'], color = 'Orange', label='Part Time Employees', shade=True)
ax = sns.kdeplot(data_FT['TotalPayBenefits'], color = 'Green', label='Full Time Employees', shade=True)
plt.yticks([])

plt.title('Part Time Employees vs. Full Time Employees')
plt.ylabel('Density of Employees')
plt.xlabel('Total Pay + Benefits ($)')
plt.xlim(0, 200000)
plt.show()


# ##  We can visualize relationship between BasePay and Benefits column using a scatter plot.

# In[ ]:


ax = plt.scatter(JTTO['BasePay'], JTTO['Benefits'])
plt.ylabel('Benefits')
plt.xlabel('BasePay')


# Base pay and benefits are positively corelated, because an employee's benefits is based on a percentage of their base pay 

# # For All Job Titles: 
# 
# ## Method:1 = using for loop

# In[ ]:


def fill_status(X):
    Status_FT = sal.loc[sal['Status'] == 'FT']
    Mean_FT= Status_FT['BasePay'].astype('Float64').mean()              ## Mean value of Base Pay associated with FT
    print("Mean_of_FT=", Mean_FT)
    Status_NaN = sal[sal['Status'].isna()]                                ## NaN status
    a = Status_NaN['BasePay'].astype('Float64')                         ## Variable a contains basepay associated with NaN Status 
    b = Status_NaN['Status']                                            ## Variable b contain NaN Status
    for i in range(0, len(a.index)):
        if (a.iloc[i] > Mean_FT):
            b.iloc[i] = 'FT'
        else:
            b.iloc[i] = 'PT'


#  This method is very time consuming. Let's do it by other method 
#  
#  ## Method 2 

# In[ ]:


JobTitle = sal.groupby('JobTitle')       ## Group by job title 


# In[ ]:


TotalPayMean = JobTitle.mean()['TotalPay']  ## take a mean of grouped job titles for TotalPay columns
TotalPayMean.head()


# In[ ]:


NaN_Status = sal[sal['Status'].isna()]      ## Get a NaN status by isna() method
NaN_Status['Status']


# In[ ]:


def fill_status(X):
    job_title = X[2]                             ## X[2] is index column 2 which is job title
    totle_Pay = X[7]                             ## X[7] is index column 7 which is Total Pay
    mean = TotalPayMean[job_title]               ## mean of perticular job title (mean will change as job title changes)
    
    if (totle_Pay > mean):                       ## Value in total pay is being comparing with mean value of perticular job title
        return "FT"                              ## if greater it will retuns FT in Status column
    else:
        return "PT"
    
NaN_Status.iloc[:110531,-1] = NaN_Status.iloc[:110531,].apply(fill_status, axis = 1)     ## 110531 is the lenght, -1 is last column
                                                                                         ## here we pass a function fill_status


#  Lets check NaN_Status are filled --

# In[ ]:


NaN_Status['Status']


# Now need to update existed dataframe of sal with new data frame which has filled NaN_Status 

# In[ ]:


new_df = pd.DataFrame(NaN_Status['Status'])


# In[ ]:


sal.update(new_df )
sal['Status'].value_counts()


# In[ ]:


sal


#  ## Full Time Employee vs Part Time Employee for all Job Titles

# In[ ]:


data_FT = sal[sal['Status'] == 'FT']
data_PT = sal[sal['Status'] == 'PT']

fig = plt.figure(figsize=(8, 8))
ax = sns.kdeplot(data_PT['TotalPayBenefits'], color = 'Orange', label='Part Time Employees', shade=True)
ax = sns.kdeplot(data_FT['TotalPayBenefits'], color = 'Green', label='Full Time Employees', shade=True)
plt.yticks([])

plt.title('Part Time Employees vs. Full Time Employees')
plt.ylabel('Density of Employees')
plt.xlabel('Total Pay + Benefits ($)')
plt.xlim(0, 600000)
plt.show()


# Lets check correlation of sal dataframe

# In[ ]:


sal.corr()


# We can see that total pay and totalpaybenefits columns are nearly correlated that we can visualize over scatter plot.

# In[ ]:


ax = plt.scatter(sal['TotalPay'], sal['TotalPayBenefits'])
plt.ylabel('TotalPayBenefits')
plt.xlabel('TotalPay')
plt.show()


# From the scatter plot above it observes that Total Pay is possitively correlated with total pay benefits.(Fair enough employess total pay benefits are based on percentage of Total Pay). 

# In[ ]:




