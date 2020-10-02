#!/usr/bin/env python
# coding: utf-8

# # > **Hello Friends This is the Assighment 1 of Python on Tips Data Analysis.**
# > Author: Rahul Padwani

# * First of all we need to upload our dataset of tips, the link given is of Tips Data Analysis and problem statements is also given . (https://www.kaggle.com/therahulpadwani/tipsdataset)-> Sheet1(Datasets) , Sheet2(Problems)
# * Than we need to import our datasets and Follow the steps.

# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname,filename))


# * **Import Libabries**

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


tips = pd.read_excel("/kaggle/input/Tips.xlsx")


# In[ ]:


tips.head()


# In[ ]:


tips.corr()


# # **Que1: What is the overall average tip?**

# In[ ]:


Avg_tip = tips["tip"].mean()
print('The overall average tip is ', Avg_tip)

Therefor Overall Average tip is Approx 3 
# # **Que2: Get a numerical summary for 'tip' - are the median and mean very different? What does this tell you about the field?**

# **Information Only**   :  You can find all the mathmetical operations from below Code ("If You Want to go for Shortcut")

# In[ ]:


tips.tip.describe()


# In[ ]:


import statistics

Median = statistics.median(tips["tip"])
print("The mean and median are  " , Avg_tip, "&", Median, "respectively")


# > #  -> As Mean and Median are close enough to each other, Hence the dataset has symmetrical distribution

# # Que3: Prepare a boxplot for 'tip', are there any outliers? 

# In[ ]:


sns.boxplot(tips["tip"] , orient = 'vertical')


# > Yes ,Outlinears are Present .

# # Que4: Prepare a boxplot for 'total_bill', are there any outliers?

# In[ ]:


sns.boxplot(tips["total_bill"] , orient = 'vertical')


# In[ ]:


Yes, total_bill field has outliners.


# # Que5: Gender: what is the percent of females in the data? 

# In[ ]:


Female_pop = tips["sex"].value_counts('Female') * 100

Female_pop_percentage = Female_pop[1:]

Female_pop_percentage


# > Their are 35.65% Females

# # Que6: Prepare a bar plot with the bars representing the percentage of records for each gender?

# In[ ]:


New = tips["sex"].value_counts()
z = New/tips["sex"].count()*100
r = ['Male', 'Female']
sns.barplot(x = r, y = z,)
plt.ylabel("Percenatge share")
plt.xlabel("Sex")
plt.title("Gender vs. Percentage share")


# # Que7: Does the average tip differ by gender? Does one gender tip more than the other?

# In[ ]:


gender_avg_tip = tips.groupby("sex").tip.mean()
gender_avg_tip


# > Male Tip is More!

# # Que8: Does the average tip differ by the time of day?

# In[ ]:


time_avg_tip = tips.groupby("time").tip.mean()
time_avg_tip


# > Yes average tip differ by time ; dinner tip is more.

# # Que9: Does the average tip differ by size (number of people at the table)? 

# In[ ]:


size_avg_tip = tips.groupby("size").tip.mean()
size_avg_tip


# > Yes average time is differed by size.

# # Que10: Do smokers tip more than non-smokers?

# In[ ]:


smoker_avg_tip = tips.groupby("smoker").tip.mean()
smoker_avg_tip


# > Yes smokers tip more than non-smokers on an average.

# # Que11:Gender vs. smoker/non-smoker and tip size - create a 2 by 2 and get the average tip size. Which group tips the most?

# In[ ]:


tips.groupby(["sex" , "smoker"]).tip.mean()


# > Thus, non-smoker males tips the most.

# # Que12: Create a new metric called 'pct_tip' = tip/ total_bill.
# -this would be percent tip give, and should be a better measure of the tipping behaviour

# In[ ]:


tips.insert(2,"pct_tip",tips.tip/tips.total_bill)
tips.head()


# # Que13: Does pct_tip differ by gender? Does one gender tip more than the other?

# In[ ]:


tips.groupby(['sex']) ['pct_tip'].count().plot.bar(color="darkblue")
plt.show()


# In[ ]:


pd.pivot_table(data=tips, index='sex',columns='smoker',values='pct_tip')


# > Yes,percentage_tip differs by gender.Average percentage tip of male is more

# # Que14: Does pct_tip differ by size (number of people at the table)?

# In[ ]:


tips.groupby(['size']) ['pct_tip'].count().plot.bar(color="darkblue")
plt.show()


# > Yes, average percentage tip do differ by size and the table with size 2 give max pct_tip

# # Que15: Make the gender vs. smoker view using pct_tip  - does your inference change?

# In[ ]:


fig , ax = plt.subplots(figsize=(10,7))
tips.groupby(['smoker','sex']) ['pct_tip'].mean().plot.bar(color='b')
ax.set_xlabel("sex")
ax.set_ylabel("Average tip percentage")
plt.show()
plt.clf()


# > Yes,group with femalesmoker gives more pct_tip.

# # Que16: Make a scatter plot of total_bill vs. tip?

# In[ ]:


plt.scatter(x='total_bill', y='tip' , data=tips)
plt.xlabel('Total Bill')   #name_of_lable
plt.ylabel('Tip')
plt.title('Total Bill Vs Tip Scatter Plot')  #Title of Graph
plt.show()


# In[ ]:


tips.plot.scatter(x='total_bill',y='tip')


# In[ ]:


sns.scatterplot(x='total_bill', y='tip', data = tips)


# > So , Higher the amount of the bill, higher is the tip.

# # Que17:  Make a scatter plot of total_bill vs. pct_tip?

# In[ ]:


plt.scatter(tips['total_bill'], tips['pct_tip']) 
plt.xlabel("Total bill")
plt.ylabel("Percentage of tip")
plt.title("Total bill vs. Percentage of tip")
plt.show()


# 
