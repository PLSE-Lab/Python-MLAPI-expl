#!/usr/bin/env python
# coding: utf-8

# # Python Assignment 

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# In[ ]:


data = pd.read_csv("../input/tipping/tips.csv")


# ## Sample Data

# In[ ]:


data.head()


# ### 1.What is the overall average tip?

# In[ ]:


data['tip'].mean()


# ### 2.Get a numerical summary for 'tip' - are the median and mean very different? What does this tell you about the field?

# In[ ]:


data['tip'].median()


# Hence, Median & mean are almost same.

# If the distribution is symmetric then the mean is equal to the median and the distribution will have zero skewness

# ### 3.Prepare a boxplot for 'tip', are there any outliers?  

# In[ ]:


sns.boxplot(x="tip", data=data)


# In[ ]:


q3, q1 = np.percentile(data.tip, [75,25])

iqr = q3 - q1
iqr = round(iqr,2)

print ("Lower Quatile:- ", q1 )
print ("Lower Quatile:- ", q3 )
print ("IQR:- ", iqr )
l = q1 - (1.5*iqr)
u = q1 + (1.5*iqr)
l = round(l,2)
u = round(u,2)
print("Lower range in boxplot is {}, & the upper range is, {}".format(l,u))


# ### 4.Prepare a boxplot for 'total_bill', are there any outliers?

# In[ ]:


sns.boxplot(x="total_bill", data = data)


# In[ ]:


q3, q1 = np.percentile(data.total_bill, [75,25])

iqr = q3 - q1
iqr = round(iqr,2)

print ("Lower Quatile:- ", q1 )
print ("Lower Quatile:- ", q3 )
print ("IQR:- ", iqr )
l = q1 - (1.5*iqr)
u = q1 + (1.5*iqr)
l = round(l,2)
u = round(u,2)
print("Lower range in boxplot is {}, & the upper range is, {}".format(l,u))


# ### 5. Gender: what is the percent of females in the data?

# In[ ]:


data.groupby('sex').size()


# In[ ]:


x = data.groupby("sex").size()
t = data["sex"].count()
p = x/t * 100
p[0]


# ### 6.Prepare a bar plot with the bars representing the percentage of records for each gender.

# In[ ]:


cnt = data.groupby(['sex']).count().reset_index()
cnt


# In[ ]:


cnt['count_perc'] = (cnt['total_bill']/ len(data)) *100
cnt


# In[ ]:


sns.barplot(x="sex",y='count_perc',
            hue = 'count_perc'
            ,data = cnt)


# In[ ]:


cnt = data.groupby(['sex']).count().reset_index()
cnt
cnt['count_perc'] = (cnt['total_bill']/ len(data)) *100 

plt.pie(x='count_perc',data=cnt,labels=['Female', 'Male'], autopct='%1.1f%%',
       shadow=True, startangle=90)


# ### 7.Does the average tip differ by gender? Does one gender tip more than the other?

# In[ ]:


data.groupby(["sex"]).mean()['tip']


# yes, the avg value of male tip is greater than female tip

# ### 8. Does the average tip differ by the time of day?

# In[ ]:


data.groupby(["day","time"]).mean()['tip']


# In[ ]:


data.groupby(["day"]).mean()['tip']


# In[ ]:


data.groupby(["time"]).mean()['tip']


# ### 9. Does the average tip differ by size (number of people at the table)?

# In[ ]:


data.groupby('size').mean()['tip']


# ### 10. Do smokers tip more than non-smokers?

# In[ ]:


data.groupby('smoker').sum()['tip']


# ### 11. Gender vs. smoker/non-smoker and tip size - create a 2 by 2 and get the average tip size. Which group tips the most?

# In[ ]:


data.groupby(['sex','smoker']).mean()['tip']


# ### 12.Create a new metric called 'pct_tip' = tip/ total_bill - this would be percent tip give, and should be a better measure of the tipping behaviour.

# In[ ]:


data['pct_tip'] = data['tip']/data['total_bill']


# ### 13. Does pct_tip differ by gender? Does one gender tip more than the other?

# In[ ]:


data.groupby(["sex"]).sum()['pct_tip']


# ### 14. Does pct_tip differ by size (number of people at the table)?

# In[ ]:


data.groupby(["size"]).sum()['pct_tip']


# ### 15. Make the gender vs. smoker view using pct_tip - does your inference change?

# In[ ]:


data['sex'].groupby(data["smoker"]).value_counts(normalize=True).rename('pct_tip').reset_index()


x,y,hue = 'sex','pct_tip','smoker'

sns.barplot(x,y,hue,data=data)


# ### 16. Make a scatter plot of total_bill vs. tip.

# In[ ]:


sns.scatterplot(x="total_bill", y = "tip",
               data = data)


# ### 17. Make a scatter plot of total_bill vs. pct_tip.

# In[ ]:


sns.scatterplot(x="total_bill", y = "pct_tip",
               data = data)


# In[ ]:


data["smoker"].count()

