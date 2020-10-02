#!/usr/bin/env python
# coding: utf-8

# # This is my first Exploratory Data Analysis only for the battles.csv data file 

# In[ ]:


#import requried packages/modules
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


#read csv file and print first 10 data sets and 5 is default value for head
df=pd.read_csv('../input/game-of-thrones/battles.csv')
df.head(5)


# In[ ]:


#Print available columns and shape of the data set like rows and columns
print(df.columns)
df.shape


# In[ ]:


#Check information of data frame, for initial data analysis
df.info()


# # we can deduce some information from the data
# # defender_3 and defender_4 columns have NAN values for the entire column
# # attacker_3, attacker_4 and notes columns have very minimal data available

# In[ ]:


# Dropping columns that does not give any gain in maintaining in data set.
# Doing so can improve performance for large data sets


# In[ ]:


# Drop columns and update the data in df object itself
df.drop(columns=['defender_3','defender_4','note', 'battle_number'],inplace=True)


# In[ ]:


# Below line provides details of unique values in the data set, it helps derive the categorical vs numerical information 
df.T.nunique(axis=1)
# Example, attacker_king has data for 36 columns, but it contains only 4 unique values
# Year has data for all 38 rows, but only 3 unique values. Even though it is numeric, it is rather considered as categorical


# In[ ]:


# Describe the data for the primary analysis, (all numerical data is described by Python)
df.describe()


# # The min and max value difference for attacker size is huge, hence it might contain outliers
# # The difference in size of defenders also seems high, hence it might also contain outliers

# In[ ]:


# Let's draw box plot to identify the Outliers existence, it is also considered as 5 point summary plot
sns.boxplot(df.attacker_size,data=df)
plt.show()


# # The attacker size plot, shows clearely there are couple of outliers to the higer side of data
# # Print outlier data sets i.e., the data outside of Interquartile Range (IQR)

# In[ ]:


df[df['attacker_size']>(df.attacker_size.quantile(.75)+(df.attacker_size.quantile(.75)-df.attacker_size.quantile(.25))*1.5)]


# # There are 3 data sets that are outside of 1.5 times IQR, and thus can be considered as Outliers

# In[ ]:


# Draw a similar plot for defender size
sns.boxplot(df.defender_size,data=df);


# # The boxplot for defender size, does not show any outliers i.e, points outside the whiskers, thus our assumption earlier is not appropriate. 
# # Need more analysis to prove there are outliers or not

# In[ ]:


# Let us visualize the data for relation between variables
sns.pairplot(df);


# # Following points could be drawn from the pair plot above
# ## The attackers side has captured more number of times than the defenders
# ## The sizes of atackers and defenders has not much impact on major death

# In[ ]:


# Let's draw a heatmap for better understanding of correlation between variables in the data
plt.figure(figsize=(10,5));
sns.heatmap(df.corr(),annot=True, cmap="Blues");


# # The attacker has higer chances to win than the defender
# # attacker size is also highly corelated to the major death

# # Let us draw few more plots for additional details
# # *Number of battles per year*

# In[ ]:


dfg=pd.DataFrame(df['year'].value_counts())
dfg


# In[ ]:


sns.barplot(x=dfg.index,y=dfg.year);
plt.title('Battles per year', fontsize=20)
plt.xlabel('year',fontsize=15)
plt.ylabel('count',fontsize=15)


# # There are more battles fought in year 299

# # *Most attacker commander*

# In[ ]:


# Removing null values from the data, concatenates strings with comma seperate and finally splits to a list
ac=df[df['attacker_commander'].notna()]['attacker_commander'].str.cat(sep=', ').split(', ')
# Count the values in the list
acs=pd.Series(ac).value_counts()
# plot graph in bigger size due to large list of attackers
plt.figure(figsize=(16,9))
sns.barplot(x=acs.index,y=acs.values)
plt.title('Attacker commanders',fontsize=20)
plt.xlabel('Attacker commanders',fontsize=15)
plt.ylabel('count',fontsize=15)
plt.xticks(rotation=90)
plt.show()


# # *Most Defender commander*

# In[ ]:


dc=df[df['defender_commander'].notna()]['defender_commander'].str.cat(sep=', ').split(', ')
dcs=pd.Series(dc).value_counts()
plt.figure(figsize=(16,9))
sns.barplot(x=dcs.index,y=dcs.values)
plt.title('Defender commanders',fontsize=20)
plt.xlabel('Defender commanders',fontsize=15)
plt.ylabel('count',fontsize=15)
plt.xticks(rotation=90)
plt.show()


# # *Number of battles per region*
# 

# In[ ]:


plt.figure(figsize=(10,4))
plt.xticks(rotation=90)
# No of battles for region
sns.countplot(x='region',data=df)
plt.show()


# # Most battles took place in The Riverlands regions
# # *Number of battles per location*

# In[ ]:


plt.figure(figsize=(10,4))
plt.xticks(rotation=90)
# No of battles for location
sns.countplot(x='location',data=df)
plt.show()


# # Most battles took place in two location
# # Number of battles per region and location

# In[ ]:


plt.figure(figsize=(10,4))
plt.xticks(rotation=90)

# No of battles for region and location
cp=sns.countplot(x='region',data=df, hue='location')
plt.title('Number of battles per region and location',fontsize=23)
plt.xlabel('Region',fontsize=13)
plt.xticks(rotation=45)
plt.ylabel('count',fontsize=13)
# place the legend of locations to the right and outside the plot
plt.legend(title ='Locations',bbox_to_anchor=(1.04,1), loc="upper left")
plt.show()
# Is there a way to hide the empty cells so the graph looks nicer?


# # Draw different types of battles

# In[ ]:


sns.countplot(x='battle_type',data=df)
plt.title('Types of battle', fontsize=20)
plt.xlabel('Battle Type',fontsize=13)
plt.ylabel('count', fontsize=13)
plt.show()


# # Identify if attacker or defender has captured the battle

# In[ ]:


# Print info to detect if na or null or NAN data values exist for major capture, attacker king and drop them from the data set
df.info()


# # The attacker king has 2 null values

# In[ ]:


# Drop null from data set for major capture and attacker into a differnt data frame
df1=df.dropna(subset=['major_capture','attacker_king'])[['attacker_king','major_capture']]
df1.shape


# In[ ]:


df2=df1.groupby('major_capture').count().reset_index().replace({0.0:'Lost',1.0:'Won'})
df2


# # Attacker has won 11 battles out of 35, indicating atackers have less chances

# In[ ]:


sns.barplot(x='major_capture',data=df2,y='attacker_king')
plt.title('Attacker battle result',fontsize=15)
plt.xlabel('capture')
plt.ylabel('attacker')
plt.show()


# # We had just explored the data and did not perform any prediction. It is just an observation from the data

# # Thanks for reading and providing your invaluable inputs to help me improve and many other data scientist aspirants
