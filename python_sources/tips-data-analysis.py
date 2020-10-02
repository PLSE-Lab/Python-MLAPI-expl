#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns


# In[ ]:


tips = pd.read_excel("/kaggle/input/Tips.xlsx")


# In[ ]:


tips.head()


# In[ ]:


tips.corr()


# In[ ]:


sum(tips.tip,tips.size)


# In[ ]:


get_ipython().run_line_magic('pinfo', 'sum')


# #### Ques:1 What is overall average tip

# In[ ]:


tips.tip.mean()


# #### Therefore overall average tip is approximately 3

# #### Ques:2 Get a numerical summary for 'tip' - are the median and mean very different? What does this tell you about the field?
# 

# In[ ]:


tips.tip.describe()


# #### Since mean and median doesn't have significant difference we can say that tip field probably doesn't have any outliers.

# #### Ques:3 Prepare a boxplot for 'tip', are there any outliers?
# 

# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


plt.boxplot(x=tips['tip'])
plt.show()


# In[ ]:


tips[tips.tip>=8]


# #### Yes, tip field contains Outliers

# #### Ques:4 Prepare a boxplot for 'total_bill', are there any outliers?
# 

# In[ ]:


plt.boxplot(x=tips['total_bill'])
plt.show()


# In[ ]:


tips.total_bill.plot.box()
plt.show()


# #### Yes, total_bill field contains Outliers

# #### Ques:5 Gender: what is the percent of females in the data? 
# 

# In[ ]:


tips.sex.value_counts(normalize = True)


# #### Their are 35.65% Females

# #### Ques:6 Prepare a bar plot with the bars representing the percentage of records for each gender.

# In[ ]:


freqs = tips.sex.value_counts(normalize = True )
freqs


# In[ ]:


plt.bar(freqs.index, freqs.values, color="green")
plt.show()


# In[ ]:


tips.sex.value_counts(normalize=True).plot.bar()
plt.show()


# #### Ques:7 Does the average tip differ by gender? Does one gender tip more than the other?

# In[ ]:


tips.groupby(['sex'])['tip'].mean().plot.bar(color="darkgreen")
plt.show()


# #### Yes, the average tip slightly differs by gender. Male tip slightly more than female

# 
# #### Ques:8 Does the average tip differ by the time of day?
# 

# In[ ]:


tips.groupby(['time'])['tip'].mean().plot.bar(color="darkgreen")
plt.show()


# #### Yes,people give more tip during Dinner compared to Lunch

# 
# #### Ques:9 Does the average tip differ by size (number of people at the table)? 
# 

# In[ ]:


tips.groupby(['size'])['tip'].mean().plot.bar(color="darkgreen")
plt.show()


# #### Yes, more the number of people, higher the tip.

# #### Ques10: Do smokers tip more than non-smokers?
# 

# In[ ]:


tips.groupby(['smoker'])['tip'].count().plot.bar(color="darkgreen")
plt.show()


# #### No, smokers tip less than non-smoker

# #### Ques:11 Gender vs. smoker/non-smoker and tip size - create a 2 by 2 and get the average tip size. Which group tips the most?
# 

# In[ ]:


tips.groupby(['smoker','sex'])['tip'].mean().plot.bar(color="darkgreen")
plt.show()


# In[ ]:


tips.groupby(['smoker','sex'])['tip'].mean().unstack()


# In[ ]:


pd.pivot_table(data=tips, index='sex',columns='smoker',values='tip')


# In[ ]:


pd.pivot_table(data=tips, index='sex',columns='smoker',values='tip',aggfunc=np.median)


# #### 20th qunatile

# In[ ]:


pd.pivot_table(data=tips, index='sex',columns='smoker',values='tip',aggfunc= lambda x:np.quantile(x,0.2))


# #### The group with Male and Non-smoker tips the most. Also group with Male and smoker tips slightly less than group with Male non-smoker

# #### Ques:12 Create a new metric called 'pct_tip' = tip/ total_bill - this would be percent tip give, and should be a better measure of the tipping behaviour.
# 

# In[ ]:


tips.insert(2,"pct_tip",tips.tip/tips.total_bill)


# In[ ]:


tips.head()


# #### Ques:13 Does pct_tip differ by gender? Does one gender tip more than the other?
# 

# In[ ]:


tips.groupby(['sex'])['pct_tip'].count().plot.bar(color="darkgreen")
plt.show()


# In[ ]:


pd.pivot_table(data=tips, index='sex',columns='smoker',values='pct_tip')


# #### Yes, the pct_tip differs by gender. Male tip more than female.

# #### Ques:14 Does pct_tip differ by size (number of people at the table)? 
# 

# In[ ]:


tips.groupby(['size'])['pct_tip'].count().plot.bar(color="darkgreen")
plt.show()


# #### Yes, pct_tip do differ by size. The table with size 2 gives maximum pct_tip

# #### Ques:15 Make the gender vs. smoker view using pct_tip  - does your inference change?
# 

# In[ ]:


tips.groupby(['smoker','sex'])['pct_tip'].mean().plot.bar(color="darkgreen")
plt.show()


# #### Yes, group with Female smoker gives more average pct_tip

# #### Ques:16 Make a scatter plot of total_bill vs. tip.
# 

# In[ ]:


plt.scatter(x='total_bill', y='tip', data = tips)
plt.xlabel('Total Bill')              # label = name of label
plt.ylabel('Tip')
plt.title('Total Bill vs Tip Scatter Plot')            # title = title of plot
plt.show()


# In[ ]:


tips.plot.scatter(x='total_bill', y='tip')


# In[ ]:


sns.scatterplot(x='total_bill', y='tip', data = tips)


# #### Thus, higher the amount of bill, higher the tip

# #### Ques:17 Make a scatter plot of total_bill vs. pct_tip.
# 

# In[ ]:


plt.scatter(x='total_bill', y='pct_tip', data = tips)
plt.xlabel('Total Bill')              # label = name of label
plt.ylabel('Percentage Tip')
plt.title('Total Bill vs Percentage Tip Scatter Plot')            # title = title of plot
plt.show()


# In[ ]:


sns.scatterplot(x='total_bill', y='pct_tip', data = tips)


# In[ ]:


tips[tips.pct_tip<=0.3].plot.scatter(x='total_bill', y='pct_tip')


# #### Thus, higher the amount of bill doesn't make much difference to pct_bill

# #### Heat Map

# In[ ]:


#correlation map
f,ax = plt.subplots(figsize=(5, 5))
sns.heatmap(tips.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()


# #### Line Plot

# In[ ]:


# Line Plot
# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line
tips.tip.plot(kind = 'line', color = 'g',label = 'Tip',linewidth=1,alpha = 1,grid = True,linestyle = ':')
tips.total_bill.plot(color = 'r',label = 'Total Bill',linewidth=1, alpha = 1,grid = True,linestyle = '-.')
plt.legend(loc='best')     # legend = puts label into plot
plt.xlabel('x axis')              # label = name of label
plt.ylabel('y axis')
plt.title('Line Plot')         # title = title of plot
plt.show()


# #### Filtering dataframe

# In[ ]:


x = tips['total_bill']>45     # There are only 5 fiels who have higher total_bill value than 45
tips[x]


# In[ ]:


(tips.total_bill>48) & (tips.tip > 5)


# In[ ]:


tips.loc[(tips.total_bill>48) & (tips.tip > 5)]


# #### Ploting all data

# In[ ]:


# Plotting all data 
tips1=tips.loc[:,["total_bill","tip","pct_tip"]]
tips1.plot()
plt.show()


# #### Sub-Plots

# In[ ]:


# subplots
tips1.plot(subplots = True)
plt.show()


# In[ ]:


# histogram subplot with non cumulative and cumulative
fig, axes = plt.subplots(nrows=2,ncols=1)
tips1.plot(kind = "hist",y = "total_bill",bins = 60,range= (0,50),normed = True,ax = axes[0])
tips1.plot(kind = "hist",y = "total_bill",bins = 60,range= (0,50),normed = True,ax = axes[1],cumulative = True)
plt.savefig('graph.png')
plt.show()


# #### Use SEABORN

# In[ ]:


import warnings
warnings.filterwarnings("ignore")


# In[ ]:


sns.distplot(tips.tip)


# In[ ]:


sns.jointplot(data=tips, x='total_bill', y='tip', kind="reg")
plt.show()


# In[ ]:


tips.columns


# In[ ]:


sns.pairplot(data=tips,kind="reg", size=2)
plt.show()


# In[ ]:


sns.pairplot(data=tips,kind="reg", size=1.8, hue="smoker")
plt.show()


# In[ ]:


sns.pairplot(data=tips,kind="reg", size=2, hue="sex")
plt.show()


# In[ ]:


res=pd.pivot_table(data=tips,index="day",columns="size",values="tip")


# In[ ]:


sns.heatmap(res, annot=True, cmap="RdYlGn")
plt.show()


# In[ ]:





# In[ ]:




