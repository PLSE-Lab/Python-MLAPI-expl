#!/usr/bin/env python
# coding: utf-8

# ### This notebook goes through the besic steps of inspecting, cleaning and analyzing this beginner dataset.  
# ### Content:
# 1. Inspection of the data
# 2. Data cleaning
# 3. Exploratory Data Analysis (EDA)
# 4. Quick comparison with Google Play Store Apps
# 5. Summary

# In[ ]:


# Setup
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

file = "../input/microsoft-common-apps/common_apps.csv"
df = pd.read_csv(file, index_col="App_Order")


# # 1. Inspection of the data

# In[ ]:


df.head()


# In[ ]:


df.shape


# In[ ]:


df.describe()


# In[ ]:


df.dtypes


# In[ ]:


df.isnull().sum()


# In[ ]:


df.loc[df["App_Star"].isnull()]


# -> 25 rows have only the price info
# 
# Check for duplicates:

# In[ ]:


df.loc[df.duplicated()].shape


# -> 90 Entries are duplicates

# # 2. Data cleaning

# Rename columns to make the following work a bit easier:

# In[ ]:


df = df.rename(columns={"App_Name": "Name", "App_Star": "Stars", "App_Price": "Price ($)","App_Review": "Views"})


# Remove Rows with missing values and duplicates:

# In[ ]:


df = df.dropna().drop_duplicates(subset="Name")


# Convert price column to float for analysis:

# In[ ]:


df["Price ($)"] = df["Price ($)"].replace("Free", "0")
df["Price ($)"] = df["Price ($)"].str.lstrip("$")
df["Price ($)"] = pd.to_numeric(df["Price ($)"])


# Views should be whole numbers:

# In[ ]:


df["Views"] = df["Views"].astype(int)


# Now that there are no duplicates anymore, the Name column would be a good index ("App_Order" doesn't tell us anything)

# In[ ]:


df = df.set_index("Name")


# Check that everything is ok now:

# In[ ]:


df.head()


# In[ ]:


df.describe()


# In[ ]:


df.dtypes


# In[ ]:


df.isnull().sum()


# -> Everything looks ok now.

# # 3. Exploratory Data Analysis

# ## Top 10 most viewed apps

# In[ ]:


top10 = df.sort_values(by="Views", ascending=False).head(10)
top10


# In[ ]:


plt.figure(figsize=(10,6))
sns.barplot(x = top10.index, y=top10["Views"]/1000)
plt.title("Most viewed apps", fontsize=16)
plt.xlabel("App", fontsize=12)
plt.xticks(rotation=90)
plt.ylabel("Number of views in thousands", fontsize=12)


# - Top 10 of the most viewed app are viewed ~ 200-650k times.
# - All of them are free.
# - Ratings are from 3.5 - 4.5.

# ## Top 10 most often viewed 5.0 Star apps:

# In[ ]:


top10_5star = df.sort_values(by=["Stars","Views"], ascending=False).head(10)
top10_5star


# In[ ]:


plt.figure(figsize=(10,6))
sns.barplot(x = top10_5star.index, y=top10_5star["Views"]/1000)
plt.title("Most viewed 5-Star apps", fontsize=16)
plt.xlabel("App", fontsize=12)
plt.xticks(rotation=90)
plt.ylabel("Number of views in thousands", fontsize=12)


# - Only three of the Top10 5-Star apps have a view count above average (11497). Eight of them are free.

# ## Top 10 most often viewed non_fee_apps:

# In[ ]:


top10_non_free = df.loc[df["Price ($)"] > 0].sort_values(by=["Views"], ascending=False).head(10)
top10_non_free


# In[ ]:


plt.figure(figsize=(10,6))
sns.barplot(x = top10_non_free.index, y=top10_non_free["Views"]/1000)
plt.title("Most viewed non_free apps", fontsize=16)
plt.xlabel("App", fontsize=12)
plt.xticks(rotation=90)
plt.ylabel("Number of views in thousands", fontsize=12)


# ## Star rating distribution

# In[ ]:


df["Stars"].value_counts()


# In[ ]:


plt.figure(figsize=(10,6))
sns.barplot(x=df["Stars"].value_counts().index, y=df["Stars"].value_counts().values, palette="Blues")
plt.title("Star rating of apps", fontsize=16)
plt.xlabel("Stars", fontsize=12)
plt.ylabel("Number of ratings", fontsize=12)


# - 4.0 is the most often given rating, followed by 4.5.
# - Only 6 % of all apps are rated with 5.0 Stars.
# - Apps with >= 4.0 Stars account for 62.9 % of all apps.
# - 5.8 % of all apps are rated with <= 2.0 Stars.

# Differences between free and non free apps:

# In[ ]:


free_apps = df.loc[df["Price ($)"] == 0]
non_free_apps = df.loc[df["Price ($)"] > 0]


# In[ ]:


plt.figure(figsize=(10,6))
sns.barplot(x=free_apps["Stars"].value_counts().index, y=free_apps["Stars"].value_counts().values, palette="Blues")
plt.title("Star rating of free apps", fontsize=16)
plt.xlabel("Stars", fontsize=12)
plt.ylabel("Number of ratings", fontsize=12)


# In[ ]:


plt.figure(figsize=(10,6))
sns.barplot(x=non_free_apps["Stars"].value_counts().index, y=non_free_apps["Stars"].value_counts().values, palette="Blues")
plt.title("Star rating of non-free apps", fontsize=16)
plt.xlabel("Stars", fontsize=12)
plt.ylabel("Number of ratings", fontsize=12)


# In[ ]:


free_apps.describe()


# In[ ]:


non_free_apps.describe()


# In[ ]:


free_apps["Views"].sum()


# In[ ]:


non_free_apps["Views"].sum()


# - Average rating of free and non free apps is almost identical (3.8).
# - Apps are eighter free or cost 5.29 USD - no price variation in non free apps.
# - 90 % of apps are free.
# - On average, free apps are viewed 4.6 times as often as non_free.
# - Free apps account for 97.67 % of all views.

# ### Correlation between stars and views

# In[ ]:


plt.figure(figsize=(10,6))
sns.regplot(x=df["Stars"], y=df["Views"])


# Range of views to large to see the slope of the regression line. Focus on lower part of the graph:

# In[ ]:


plt.figure(figsize=(10,6))
sns.regplot(x=df["Stars"], y=df["Views"])
plt.gca().set_ylim(0, 50000)


# - General trend: Higher rated apps are viewed more often.
# - Exception: 5 Star apps do not fit into this pattern.

# Is there a price dependence?

# In[ ]:


sns.lmplot(x="Stars", y="Views", hue="Price ($)", data=df)
plt.gca().set_ylim(0, 50000)


# - Correlation is higher for free apps. 

# # 4. Quick comparison with Google Play Store Apps
# Data for coparison: https://www.kaggle.com/lava18/google-play-store-apps
# 
# - Number of available apps is less than 1/10 of Google Play (9660)
# - Average rating is slightly lower (3.8 vs. 4.2)
# - Similar percentage of free apps (90 vs. 93 %)
# 
# - Views are not directly compareable, but for a rough estimate:
#   Max views for Windows app is 671k.
#   Several Android apps have over 1B installs.
#   

# # 5. Summary
# - The dataset shows the price, views and stars of Windows apps.
# - There are 881 unique apps in the dataset.
# - Apps are eighter free or cost 5.29 USD - no price variation in non free apps.
# - 90 % of apps are free.
# - On average, free apps are viewed 4.6 times as often as non_free.
# - Average rating of free and non free apps is almost identical (3.8).
# - Free apps account for 97.67 % of all views.
# - Top 10 of the most viewed app are viewed ~ 210-650k times. All of them are free.

# This is my first kernel. If you like it or learned something plese upvote.  
# Comments/suggestions are welcome, too.
