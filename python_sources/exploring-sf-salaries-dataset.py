#!/usr/bin/env python
# coding: utf-8

# # A Self-Exercise on the San Francisco Salaries Dataset

# Note that you might need to download the dataset if you want to run the code in this notebook. Don't forget to check that the dataset file is in the same folder as you are working in. 
# 
# You can download the dataset here: https://www.kaggle.com/kaggle/sf-salaries
# 
# Let's start.

# __Importing the Libraries we'll be using.__

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# __Reading the data from the CSV file.__

# In[ ]:


sal = pd.read_csv('Salaries.csv', low_memory = False)


# Note that my CSV file is in the same folder as my jupyter notebook file, therefore I am able to just give the CSV file's name.
# If your CSV file is not in the same folder as the location you are currently working in, then you might need to give in the file's full address.

# __Checking the data.__

# In[ ]:


# It is always recommended to check the head of the data file you are working with.
sal.head()


# In[ ]:


# Note that the default parameter for the head() function is 5, but you can change it as such:
sal.head(3)


# In[ ]:


# Or:
sal.head(10)


# __Accessing the information about the dataset.__

# In[ ]:


sal.info()


# Notice that we have different amount of values for the columns of __BasePay__ and __Benefits__. It is important to notice these details early on.

# __Converting the payment-related columns to numeric values.__

# If you had 'played' with the dataset already, you probably have seen that there are some columns that have no data, or some with missing data. And there are also columns that are not in numeric form but rather in string forms. For examply, try to get the BasePay value of the __148646th__ entry in the dataset.

# In[ ]:


sal['BasePay'][148646]


# These kind of rows are posing numerous problems for us, of which is that we cannot deal with multiple type of objects (integer, string etc.) at the same time. Therefore, we need to convert everything to numeric values.

# In[ ]:


sal = sal.convert_objects(convert_numeric = True)


# We see that it gives us a FutureWarning, saying that the function we are trying to use here is deprecated, meaning that it will not be in use in the future. So, it is of our best interest that we write our own function to convert the payment related columns into numeric values.

# In[ ]:


for column in ['BasePay', 'OvertimePay', 'OtherPay', 'Benefits']:
    sal[column] = pd.to_numeric(sal[column], errors = 'coerce')


# Now, let's try to get the __average BasePay value.__

# In[ ]:


sal['BasePay'].mean() # See that it is infact working.


# When we used the __info()__ function, we saw that the __Notes__ column is missing a lot of values. Therefore, it is of our best interest to drop that column for good.

# In[ ]:


# The 'axis' parameter is by-default set to 0, which indicates the rows.
# Since our dataset has no row label with 'Notes', it would throw and error.
# And since we want to drop a COLUMN, but not a row, we need to specify it
# by saying axis = 1
sal = sal.drop('Notes', axis = 1)


# __What is the highest amount of OvertimePay in the dataset?__

# In[ ]:


sal['OvertimePay'].max()


# __Who is getting the highest amount of OvertimePay?__

# In[ ]:


sal[sal['OvertimePay'] == sal['OvertimePay'].max()]


# __Who is getting an OvertimePay between 150.000 and 200.000?__

# In[ ]:


sal[(sal['OvertimePay'] >= 150000) & (sal['OvertimePay'] <= 200000)]


# __What is the job title of Gary L Altenberg?__

# In[ ]:


sal[sal['EmployeeName'] == 'Gary L Altenberg']['JobTitle']


# In[ ]:


# So it seems that we have two people with the name Gary L Altenberg. Let's try another name:
sal[sal['EmployeeName'] == 'Scott Scholzen']['JobTitle']


# __How much does the person with the name Scott Scholzen make including benefits?__

# In[ ]:


sal[sal['EmployeeName'] == 'Scott Scholzen']['TotalPayBenefits']


# __What is the name of the highest paying person including benefits?__

# In[ ]:


sal[sal['TotalPayBenefits'] == sal['TotalPayBenefits'].max()]['EmployeeName']


# Or if we want to see the whole row for that person:

# In[ ]:


sal[sal['TotalPayBenefits'] == sal['TotalPayBenefits'].max()]


# __What is the name of the lowest paying person including benefits?__

# In[ ]:


sal[sal['TotalPayBenefits'] == sal['TotalPayBenefits'].min()]


# Do you notice something? This person has a TotalPay of -618 dollars. A negative value. Interesting.

# __What is the average values of all employees per year?__

# In[ ]:


sal.groupby('Year').mean()


# What is the average __TotalPayBenefits__ of all employees per year?

# In[ ]:


sal.groupby('Year').mean()['TotalPayBenefits']


# __How many unique job titles are there in the dataset?__

# In[ ]:


sal['JobTitle'].nunique()


# __What are the highest paying jobs?__

# In[ ]:


sal.groupby('JobTitle')['BasePay'].median().


# Let's get the top 10 highest paying jobs.

# In[ ]:


(sal.groupby(['JobTitle'])['BasePay'].median()).nlargest(10)


# __Top 10 highest Overtime paying jobs.__

# In[ ]:


(sal.groupby(['JobTitle'])['OvertimePay'].median()).nlargest(10)


# __What are the top 10 most common jobs?__

# In[ ]:


sal['JobTitle'].value_counts().head(10)


# __How many people have the word "officer" in their title?__

# In[ ]:


def officerWordCount(title):
    if 'officer' in title.lower():
        return True
    else:
        return False
    
sum(sal['JobTitle'].apply(lambda title: officerWordCount(title)))


# One last thing I want to do before plotting is to delete the rows of people with negative payments, just to clear my plottings.

# In[ ]:


sal[sal['TotalPay'] < 0]


# Just one person... How about people with 0 salaries?

# In[ ]:


sal[sal['TotalPay'] == 0]


# That's ... more than I expected.

# In[ ]:


len(sal[sal['TotalPay'] == 0])


# Okay ... yeah. 368 people is definitely much. I am definitely going to get rid of all of them.

# In[ ]:


len(sal)


# In[ ]:


sal = sal[sal['TotalPay'] > 0]


# In[ ]:


len(sal)


# Now that we got rid of that 'bad' data, let's start plotting.

# # Plotting

# Let's work on TotalPay column.

# In[ ]:


graph = sns.FacetGrid(sal, col="Year", col_wrap=2, height=6, dropna=True)
graph.map(sns.kdeplot, 'TotalPay', shade=True)


# There are __two__ things to note here:
# 
# 1) All four plots looks similar. This indicates of a stable economy or a stable job market.
# 
# 2) We seem to have __two spikes__. Let's try to understand what they represent.

# In[ ]:


# Remember that we have 2 types of job status values: Part-time and Full-time.
# Let's separate them.
full_time = sal[sal['Status'] == 'FT']
part_time = sal[sal['Status'] == 'PT']

# Using subplots() function to plot different datas on top of each other.
fig, ax = plt.subplots(figsize=(15,6))

# Plotting the separated data.
sns.kdeplot(full_time['TotalPay'].dropna(), label="Full Time", shade=True, ax=ax)
sns.kdeplot(part_time['TotalPay'].dropna(), label="Part Time", shade=True, ax=ax)

# Setting the titles.
plt.xlabel('Total Pay', fontsize=16)
plt.ylabel('Density', fontsize=16)
plt.title('Total Pay Distribution: Full-Time vs. Part-Time', fontsize=20)


# So it seems that those two pikes are basically indicating the different types of job status.

# __Now let's plot the data for 'BasePay' instead of 'TotalPay'.__

# In[ ]:


fig, ax = plt.subplots(figsize=(15,6))

# Plotting the separated data.
sns.kdeplot(full_time['BasePay'].dropna(), label="Full Time", shade=True, ax=ax)
sns.kdeplot(part_time['BasePay'].dropna(), label="Part Time", shade=True, ax=ax)

# Setting the titles.
plt.xlabel('Base Pay', fontsize=16)
plt.ylabel('Density', fontsize=16)
plt.title('Base Pay Distribution: Full-Time vs. Part-Time', fontsize=20)


# Although this BasePay distribution looks similar to the previous TotalPay distribution, we have different spikes in this one.

# __Now let's plot the data for OvertimePay.__

# In[ ]:


fig, ax = plt.subplots(figsize=(15,6))

# Plotting the separated data.
sns.kdeplot(full_time['OvertimePay'].dropna(), label="Full Time", shade=True, ax=ax)
sns.kdeplot(part_time['OvertimePay'].dropna(), label="Part Time", shade=True, ax=ax)

# Setting the titles.
plt.xlabel('Overtime Pay', fontsize=16)
plt.ylabel('Density', fontsize=16)
plt.title('Overtime Pay Distribution: Full-Time vs. Part-Time', fontsize=20)


# Two things to note here:
# 
# 1) It is not that surprising that most people don't get paid for their overtimes, that is a common thing unfortunately.
# 
# 2) Looking at the tail of the graph, we can see that someone is getting paid __more than 175.000 dollars for overtime__... Now, that's impressive!

# __Now, let's plot the data for Benefits.__

# In[ ]:


fig, ax = plt.subplots(figsize=(15,6))

# Plotting the separated data.
sns.kdeplot(full_time['Benefits'].dropna(), label="Full Time", shade=True, ax=ax)
sns.kdeplot(part_time['Benefits'].dropna(), label="Part Time", shade=True, ax=ax)

# Setting the titles.
plt.xlabel('Benefits', fontsize=16)
plt.ylabel('Density', fontsize=16)
plt.title('Benefits Distribution: Full-Time vs. Part-Time', fontsize=20)


# This graph basically indicates that full-time employees are usually getting more benefits than part-time employees.

# __Let's draw the heatmap of our dataset.__

# In[ ]:


f, ax = plt.subplots(figsize=(8, 8))
sns.heatmap(sal.corr(), annot=True, linewidths=0.3, fmt= '.3f', ax=ax)


# __Let's draw the bar plot of the most popular 30 jobs.__

# In[ ]:


plt.figure(figsize=(15,10))
jobs = sal['JobTitle'].value_counts()[0:30]
sns.barplot(x=jobs.values, y=jobs.index, alpha=0.5)
plt.xlabel("Number of Jobs", fontsize=16)
plt.ylabel("Job Title", fontsize=16)
plt.title("Most Popular 30 Jobs and Their Distribution", fontsize=20)


# ### Now let's use Seaborn's JointPlot class.

# In[ ]:


sns.set(style="darkgrid", color_codes=True)
g = sns.jointplot(x="BasePay", y="Benefits", data=sal)


# __Density Plot.__

# In[ ]:


g = sns.jointplot(x="BasePay", y='Benefits', kind="kde", data=sal)


# Now, this is not a good graph. Because the representation of the data could me more clearer. But since the values of the dataset are too broad (that there are people getting 100.000 dollars of benefits or that there are people getting 300.000 dollars of BasePay) it would be better if we trim it a bit.

# In[ ]:


# Restricting my data down to the people who get less than 100.000 dollars of BasePay
# and down to the people who get less than 40.000 dollars of Benefits.
sal = sal[sal['BasePay'] < 100000]
sal = sal[sal['Benefits'] < 40000]


# In[ ]:


# Now, let's plot the graph again.
g = sns.jointplot(x="BasePay", y='Benefits', kind="kde", data=sal, space=0)


# In[ ]:


g = sns.jointplot(x="BasePay", y='Benefits', kind="kde", data=sal, color="g", space=0)


# __Regression and kernel density fits.__

# In[ ]:


g = sns.jointplot(x="BasePay", y='Benefits', kind="reg", data=sal, space=0)


# Using the type __hex__.

# In[ ]:


g = sns.jointplot(x="BasePay", y='Benefits', kind="scatter", data=sal, space=0)


# End.
