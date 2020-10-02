#!/usr/bin/env python
# coding: utf-8

# # Hello Explorer.
# 
# ### I wrote this kernel long back when I was trying to do some good for the society by exploring the crime data on Women of our Country "INDIA". But recently due to some deep worrying events happening, I feel that this kernel needs to be explored more, and give more usefull insights. My main motive is to drill down the given data, so that some preventive inferences can be deduced which could help in saving this country's mothers and sister.!
# 
# ## Let's Explore!

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# ## We are going to explore lot of crimes, but starting with ***Rape Crimes*** as this has to be adressed first!

# In[ ]:


df_victim_of_rape = pd.read_csv("../input/20_Victims_of_rape.csv")


# In[ ]:


# Looking at the shape of data
df_victim_of_rape.shape


# In[ ]:


# Let's understand the dtypes for this data set!
df_victim_of_rape.dtypes


# In[ ]:


df_victim_of_rape.head()


# In[ ]:


# Let's see what are the AREAs, provided to us for exploration!
df_victim_of_rape.Area_Name.value_counts()


# In[ ]:


df_victim_of_rape.groupby(['Year','Subgroup']).Rape_Cases_Reported.sum()


# In[ ]:


plt.figure(figsize=(8,8))
df_victim_of_rape.groupby(['Year','Subgroup']).Rape_Cases_Reported.sum().plot(kind='bar');
#We will look at each subgroup of rape over the years and will infer!


# **One thing is clear that with time increase, the rapes performed are increasing!** This is a very concerning look as with time, the people should be educated, maybe they are, and if they are, then it is more concerning as what type of education are they recieving!

# In[ ]:


df_victim_of_rape.Rape_Cases_Reported.plot(kind='hist',bins=20);


# # We will Explore the Crimes conducted for Uttar Pradesh
# 
# ## Total Rapes and Incest Rapes..!!

# In[ ]:


up_total_rape = df_victim_of_rape.loc[df_victim_of_rape['Area_Name']=='Uttar Pradesh']
up_victim_2010_total = up_total_rape [(up_total_rape['Year']==2010) & (up_total_rape['Subgroup']=='Total Rape Victims')]
up_victim_2010_total_incest_rape = up_total_rape [(up_total_rape['Year']==2010) & (up_total_rape['Subgroup']=='Victims of Incest Rape')]

#Plotting age breakup of victims
ax = up_victim_2010_total[['Victims_Upto_10_Yrs','Victims_Between_10-14_Yrs','Victims_Between_14-18_Yrs','Victims_Between_18-30_Yrs','Victims_Between_30-50_Yrs']].plot(kind='bar',legend=True, title = 'Age Breakup of rape victims (Uttar Pradesh..!!)')
ax.set_ylabel("No of Victims", fontsize=12)
ax.set_xticklabels([]);
ax = up_victim_2010_total_incest_rape[['Victims_Upto_10_Yrs','Victims_Between_10-14_Yrs','Victims_Between_14-18_Yrs','Victims_Between_18-30_Yrs','Victims_Between_30-50_Yrs']].plot(kind='bar',legend=True, title = 'Age Breakup of Incest rape victims (Uttar Pradesh..!!)')
ax.set_ylabel("No of Victims", fontsize=12)
ax.set_xticklabels([]);


# Now we can see that over **800** such cases are there in **2010** where the age of the **Victim** is between **18

# # Having a Look on Total No. of Rapes performed in the year 2010

# In[ ]:


victims_rape_2010_total = df_victim_of_rape[(df_victim_of_rape['Year']==2010) & (df_victim_of_rape['Subgroup']== 'Total Rape Victims')]
ax1 = victims_rape_2010_total['Victims_of_Rape_Total'].plot(kind='barh',figsize=(20, 15))
ax1.set_xlabel("Number of rape victims (2010)", fontsize=25)
ax1.set_yticklabels(victims_rape_2010_total['Area_Name']);


# We can see that **Madhya Pradesh** had the most number of rape incidents for the year 2010! What could be the reason?
# Can we find it theoretically, we can do it using COrrelation as then we can see which **Attribute** is dependent on which **Cause!!**
# 
# # Plotting a Correlation between all kinds of Age Groups with Year

# In[ ]:


import seaborn as sns
plt.figure(figsize=(8,8))
df_corr=df_victim_of_rape.corr()
sns.heatmap(df_corr, xticklabels = df_corr.columns.values, yticklabels = df_corr.columns.values,annot=True);


# **Damn! 97% of the Rape Cases reported are for Victims between 18 to 30!**
# We can see that **Year** is least correlated, which tells us that the Age of the Victim doesn't matters, people out there are getting more and more **EVIL**!

# # We find the Areas where Crimes are greater than a Number and shown Back at First the Number. Then the Particular State having that Index number is shown to the Output..!!

# In[ ]:


# What is the mean of the Total Rapes:
df_mean = df_victim_of_rape.Victims_of_Rape_Total.mean()
print(df_mean)


# In[ ]:


# We have seen the MEAN value, now we will try to see the states where the Total Rapes are greater than MEAN
df_total = df_victim_of_rape[df_victim_of_rape.Victims_of_Rape_Total>362]
df_total.head()


# In[ ]:


# Let's find out those States! where the Rapes are more than Mean!
plt.figure(figsize=(24,5))
plt.title('Count of States, where the Rapes are more than Mean Value of Rape!');
sns.countplot(df_total.Area_Name);
plt.xticks(rotation = 60);


# In[ ]:


# Let's explore the recieved Data Frame and perform drilling analysis on that!
data_frame_to_drill = df_total.copy()


# In[ ]:


data_frame_to_drill.describe()


# In[ ]:


data_frame_to_drill.shape
# We have only 326 Rows to Analyze, so let's break it down!


# We have seen the Age group being hampered and the year along with the States, but what about the Kids, the small lovely children of our home?
# Are they safe?
# 
# There's only one way to find out!

# In[ ]:


data_frame_to_drill.Victims_Upto_10_Yrs.sum()
# This is the sum of Victims which are upto 10 Years in age only!, but over the due course of time!


# In[ ]:


plt.figure(figsize=(24,8))
sns.barplot(x = data_frame_to_drill.Area_Name, y = data_frame_to_drill['Rape_Cases_Reported'], color = "red")
upto_18 = sns.barplot(x = data_frame_to_drill.Area_Name, y = data_frame_to_drill['Victims_Between_14-18_Yrs'], color = "#a8ddb5")
topbar = plt.Rectangle((0,0),1,1,fc="red", edgecolor = 'none')
upto_18 = plt.Rectangle((0,0),1,1,fc='#a8ddb5',  edgecolor = 'none')
l = plt.legend([upto_18, topbar], ['Victims between 14 to 18 years of Age', 'Total Rape Cases'], loc=1, ncol = 2, prop={'size':16})
l.draw_frame(False)
sns.despine(left=True)


# In[ ]:


plt.figure(figsize=(24,8))
sns.barplot(x = data_frame_to_drill.Area_Name, y = data_frame_to_drill['Rape_Cases_Reported'], color = "red")
upto_30 = sns.barplot(x = data_frame_to_drill.Area_Name, y = data_frame_to_drill['Victims_Between_18-30_Yrs'], color = "#fe9929")
topbar = plt.Rectangle((0,0),1,1,fc="red", edgecolor = 'none')
upto_30 = plt.Rectangle((0,0),1,1,fc='#fe9929',  edgecolor = 'none')
l = plt.legend([upto_30, topbar], ['Victims between 18 to 30', 'Total Rape Cases'], loc=1, ncol = 2, prop={'size':16})
l.draw_frame(False)
sns.despine(left=True)
#upto_30.set_ylabel("Y-axis label");
#upto_30.set_xlabel("X-axis label");


# In[ ]:


plt.figure(figsize=(24,8))
sns.boxplot(data_frame_to_drill.Victims_Upto_10_Yrs,data_frame_to_drill.Area_Name,data = data_frame_to_drill);
plt.title('Spread of Rapes performed on Children upto 10 Years!');


# **We can see that the States Madhya Pradesh and Maharashtra** are in some kind of competition for this crime!
# Please be informed that this is not helping at all. The **State Government** should do something about it! The **Central Government** should bring some hard rules for this kind of Crimes!
# 
# It is very shocking to see **Maharashtra** to be leading in this kind of Crime! Anyone reading this kernel from any part of the State -- Maharashtra, this is happening in your city.
# 
# # Now we will Explore an Another Data Set..!!
# ## Let us pick **"31 Serious Fraud"** Data Set and begin *Exploration..!!*

# In[ ]:


df_31s_f = pd.read_csv("../input/31_Serious_fraud.csv")


# In[ ]:


df_31s_f.describe()


# In[ ]:


df_31s_f.dtypes


# In[ ]:


df_31s_f.shape
#We have 448 rows and 9 Columns


# **Now Let us see some data from the Data Set**

# In[ ]:


df_31s_f.head(10)


# In[ ]:


df_31s_f.isnull().sum()


# In[ ]:


df_31s_f.Loss_of_Property_1_10_Crores.mean()


# In[ ]:


df_31s_f.groupby(['Area_Name','Year']).Loss_of_Property_10_25_Crores.plot(kind='barh');
plt.xlabel('Bins')
plt.ylabel('Loss of Property')
plt.show()


# # We perform a Break Down of Frauds performed in Some Cities to Get an Idea as Which Category of Fraud is Hurting the Country More..!!
# ## Cities we'll be taking are: 
# ###1. Gujarat
# ###2. Delhi
# ###3. Andhra Pradesh as a State

# In[ ]:


#1. Gujarat
df_plot_fraud = df_31s_f.loc[df_31s_f['Area_Name'] == 'Gujarat']
df_year_fraud = df_plot_fraud[(df_plot_fraud['Year'] == 2001) & (df_plot_fraud['Group_Name'] == 'Serious Fraud - Cheating')]
ax = df_year_fraud[['Loss_of_Property_1_10_Crores','Loss_of_Property_10_25_Crores','Loss_of_Property_25_50_Crores','Loss_of_Property_50_100_Crores','Loss_of_Property_Above_100_Crores']].plot(kind='bar',legend=True, title = 'Breakup of Frauds Performed in Gujrat')
ax.set_ylabel("Loss of Property", fontsize=12)
ax.set_xticklabels([]);


# In[ ]:


#2. Delhi
df_plot_fraud = df_31s_f.loc[df_31s_f['Area_Name'] == 'Delhi']
df_year_fraud = df_plot_fraud[(df_plot_fraud['Year'] == 2001) & (df_plot_fraud['Group_Name'] == 'Serious Fraud - Cheating')]
ax = df_year_fraud[['Loss_of_Property_1_10_Crores','Loss_of_Property_10_25_Crores','Loss_of_Property_25_50_Crores','Loss_of_Property_50_100_Crores','Loss_of_Property_Above_100_Crores']].plot(kind='bar',legend=True, title = 'Breakup of Frauds Performed in Gujrat')
ax.set_ylabel("Loss of Property", fontsize=15)
ax.set_xticklabels([]);


# In[ ]:


#3. Andhra Pradesh
df_plot_fraud = df_31s_f.loc[df_31s_f['Area_Name'] == 'Andhra Pradesh']
df_year_fraud = df_plot_fraud[(df_plot_fraud['Year'] == 2001) & (df_plot_fraud['Group_Name'] == 'Serious Fraud - Cheating')]
ax = df_year_fraud[['Loss_of_Property_1_10_Crores','Loss_of_Property_10_25_Crores','Loss_of_Property_25_50_Crores','Loss_of_Property_50_100_Crores','Loss_of_Property_Above_100_Crores']].plot(kind='bar',legend=True, title = 'Breakup of Frauds Performed in Gujrat')
ax.set_ylabel("Loss of Property", fontsize=12)
ax.set_xticklabels([]);


# ***Here we look at the Areas where the frauds under 1 to 10 Crores are performed more than 40.0 times.!!***

# In[ ]:


df_frauds_1_10_crores  = np.where(df_31s_f['Loss_of_Property_1_10_Crores']>40.0)
print(df_frauds_1_10_crores)
for i in df_frauds_1_10_crores:
    print(df_31s_f['Area_Name'][i])


# In[ ]:


sns.kdeplot(df_31s_f['Loss_of_Property_1_10_Crores'],shade = True,color="red",alpha = 0.4)
sns.kdeplot(df_31s_f['Loss_of_Property_10_25_Crores'],shade = True,color="blue",alpha = 0.3)
sns.kdeplot(df_31s_f['Loss_of_Property_25_50_Crores'],shade = True,color="orange",alpha = 0.2)
plt.show();


# In[ ]:


p = sns.countplot(x="Loss_of_Property_1_10_Crores" , data=df_31s_f , palette = "bright")
_ = plt.setp(p.get_xticklabels(),rotation = 90)


# In[ ]:


p = sns.countplot(x="Loss_of_Property_10_25_Crores" , data=df_31s_f , palette = "bright")
_ = plt.setp(p.get_xticklabels(),rotation = 90)


# **We will further explore this Data Set..!! By the Mean Time..!!**
# ***Like it...Wanna Suggest...Fork IT...UpVote It and More Version for other Files are Also on the Way..!!***
# Live Logically....Don't Commit a Crime..!!
# # Commit a CODE...!!

# ## Hi Everyone! I am back and intend to explore the Data Set of Court Trials as i wonder about the stats of our courts, number of cases pending, what is the mean/average duration in which a trial is completed, which location faces the most difficulty while performing a trial and the ratio of guilty vs non guilty state.
# 
# ### Let's look at the Data Frame and make some inferences! It is good to be back!

# In[ ]:


trials_by_court = pd.read_csv('../input/29_Period_of_trials_by_courts.csv')
trials_by_court.head(10)


# In[ ]:


print('Shape of our Data frame: ',trials_by_court.shape)
print("-----------------")
print(trials_by_court.info())


# In[ ]:


# Let's find the Missing values:
print(round(100*(trials_by_court.isnull().sum()/len(trials_by_court)),2))


# In[ ]:


# Let's plot a HeatMap for visualization of Missing values;
sns.heatmap(trials_by_court.isnull(),cbar=False);


# In[ ]:


# So we see that maximum of 8% is the missing value faced, hence we can drop the rows with missing values, and this will help us in obtaining more precise data!
trials_by_court = trials_by_court.dropna(axis=0)
trials_by_court.info()


# In[ ]:


# So approximately 200 rows are dropped! and We feel confident enough to move onto!
sns.heatmap(trials_by_court.isnull(),cbar=False);


# ## Now let's see the unique states and try to compare the trial periods respectively.

# In[ ]:


trials_by_court.Area_Name.value_counts()


# In[ ]:


trials_by_court.Group_Name.value_counts()


# In[ ]:


trials_by_court.Sub_Group_Name.value_counts()


# In[ ]:


trials_by_court.Year.value_counts()
# We will group the Years into batch of 3 Years: 2004 to 2007 and 2008 to 2010


# In[ ]:


# We also see that we have same Sub Group Name and Group Name, reflecting the same variable, hence we will drop one of this variable.
# Let's bin first:
mapping = {2004 : '2004 to 2007',2005 : '2004 to 2007',2006 : '2004 to 2007',2007 : '2004 to 2007',
           2008 : '2008 to 2010',2009 : '2008 to 2010',2010 : '2008 to 2010'}
trials_by_court['Year'] = trials_by_court['Year'].apply(mapping.get)
trials_by_court.head(50)


# In[ ]:


trials_by_court.Year.value_counts()


# In[ ]:


# Let's look at the distribution of the Year categories on the period of trials as TOTAL!
sns.barplot(x = 'Year' , y = 'PT_Total',data = trials_by_court);


# In[ ]:


# Pairplot
plt.figure(figsize=(20,20));
sns.pairplot(trials_by_court);


# In[ ]:


plt.figure(figsize=(12,5));
# Let's see the affect of the Trial period, less than 6 months. This will give the duration where the cases have been solved quickly
sns.barplot(x = trials_by_court.PT_Less_than_6_Months,y = trials_by_court.Year,data=trials_by_court);


# In[ ]:


# next is we can see the Areas where Trial period of less than 6 months has happened. we will do so by taking the mean of that variables and then returning out the true dataframe
mean_pt_less_than_6_months = trials_by_court.PT_Less_than_6_Months.mean()
mean_pt_less_than_6_months


# In[ ]:


df_less_than_6_months = trials_by_court.loc[trials_by_court.PT_Less_than_6_Months >= mean_pt_less_than_6_months]
df_less_than_6_months.head()


# In[ ]:


print(df_less_than_6_months.shape)
print(df_less_than_6_months.Area_Name.value_counts())


# In[ ]:


# Let's try plotting for these areas:
plt.figure(figsize=(8,8));
plt.title('Plot of Area vs Number of Trials!');
plt.rcParams["axes.labelsize"] = 20
plt.yticks(rotation=15)
sns.barplot(x = trials_by_court.index,y=df_less_than_6_months.Area_Name,data = df_less_than_6_months,ci=None);
plt.xlabel('Frequency!');
# Here we have used the count of our original dataset, as we want to see the frequency of the count of the states where Trials have been executed within 6 months.
# This was tricky to think, as i spent around 25 minutes to see this in a way!


# ## So we have observd the Year groups, compared the frequency as in which bucket most trials have been placed/excuted.
# ## next we went onto explore the Areas where trials were executed/finished in less thatn 6 months!
# 
# ### Our next task will be to Drop one of the Group/Sub group Column and see which of the Group has executed trials successfully in less than 6 Months.

# In[ ]:


trials_by_court = trials_by_court.drop('Sub_Group_Name',axis=1)
trials_by_court.info()


# In[ ]:


plt.figure(figsize=(8,8))
sns.barplot(x = 'Year' , y='PT_Less_than_6_Months' , hue = 'Group_Name' , data = trials_by_court,ci=None);


# ## Comparitively we see that all the stats for 2008 to 2010 is higher than 2004 to 2007 duration except for Other OCurts which may include Panchayat Decsions and Compro decisions.
# 
# ## Upvote it if you like it.!
# Till then! Be aware and stay away from the Negative people. We should be respecting Women!
# 
# ## Keep Kaggling!
