#!/usr/bin/env python
# coding: utf-8

# In this kernel I have performed Exploratory Data Analysis on the Google Play Store App Dataset and tried to identify relationsips between  various app featues. Also I have tried to figure out ongoing trends in the playstore.

# I hope you find this kernel helpful and some **<font color='red'>UPVOTES</font>** would be very much appreciated

# In[ ]:


import os
#print(os.listdir("../input"))
import warnings
warnings.filterwarnings('ignore')


# ### **Importing Required Libraries**

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# #### **Reading the dataset**

# In[ ]:


df = pd.read_csv('../input/googleplaystore.csv')
df.head(3)


# The data set contains following featues

# **1. App:** Application name<br>
# **2. Category:** Category the app belongs to<br>
# **3. Rating:** Overall user rating of the app (as when scraped)<br>
# **4. Reviews:** Number of user reviews for the app (as when scraped)<br>
# **5. Size:** Size of the app (as when scraped)<br>
# **6. Installs**: Number of user downloads/installs for the app (as when scraped)<br>
# **7. Type:** Paid or Free<br>
# **8. Price:** Price of the app (as when scraped)<br>
# **9. Content Rating**: Age group the app is targeted at - Children / Mature 21+ / Adult<br>
# **10.Genres:** An app can belong to multiple genres (apart from its main category). For eg, a musical family game will belong to Music, Game, Family genres.<br>
# **11. Last Updated**: Date when the app was last updated on Play Store (as when scraped)<br>
# **12. Current Ver**: Current version of the app available on Play Store (as when scraped)<br>
# **13. Android Ver:** Min required Android version (as when scraped)<br>

# ### **Checking features of each column given in the dataset**

# #### **1. Category**

# In[ ]:


print('Different types of App Categories as present in the dataset are: ')
print('--------------------------------------------------------------------')

count = 1
for i in df['Category'].unique():
    print(count,': ',i)
    count = count + 1


# There are 34 different app categories present in the Google Playstore.<br>
# The 34th category(1.9) is some what different from other app categories.Checking it

# In[ ]:


df[df['Category'] == '1.9']


# It seems that data was not correctly entered for this app.I will remove this app from the data set, removing one app will not affect the dataset much.

# In[ ]:


df.drop(df.index[[10472]],inplace = True)     #Removing the app on row 10472


# ####  **Countplot of Number of Apps on the basis of category**

# In[ ]:


sns.set_style('whitegrid')
plt.figure(figsize=(16,8))
plt.title('Number of apps on the basis of category')
sns.countplot(x='Category',data = df)
plt.xticks(rotation=90)
plt.show()


# Most of the apps in the playstore belong to the **Family** category followed by **Games** category.

# #### **Top 10 App Categories**

# In[ ]:


category = pd.DataFrame(df['Category'].value_counts())        #Dataframe of apps on the basis of category
category.rename(columns = {'Category':'Count'},inplace=True)


# In[ ]:


plt.figure(figsize=(15,6))
sns.barplot(x=category.index[:10], y ='Count',data = category[:10],palette='hls')
plt.title('Top 10 App categories')
plt.xticks(rotation=90)
plt.show()


# In[ ]:


family_category = len(df[df['Category'] == 'FAMILY'])/len(df)*100
games_category = len(df[df['Category'] == 'GAME'])/len(df)*100
beauty_category = len(df[df['Category'] == 'BEAUTY'])/len(df)*100
print('Percentage of Apps in the family category: {}%'.format(round(family_category,2)))
print('Percentage of Apps in the games category: {}%'.format(round(games_category,2)))
print('Percentage of Apps in the beauty category: {}%'.format(round(beauty_category,2)))


# **Family** category has the most number of apps with 18% of apps belonging to it followed by **Games** category which has 11% of the apps.Least number of apps belong to the **Beauty** category with less than 1% of the total apps belonging to it.

# #### **2. Rating**

# In[ ]:


plt.figure(figsize=(15,8))
sns.countplot(x='Rating',data = df)
plt.xticks(rotation =90)
plt.title('Countplot for ratings')             
plt.show()


# In[ ]:


rating_greater_4 = len(df[df['Rating'] >= 4])/len(df)*100
print('Percentage of Apps having ratings of 4 or greater: {}%'.format(round(rating_greater_4,2)))


# Majority of apps in the playstore have a rating 4 or above

# #### **3.Size**

# In the Size columns either the size is given in MB or kB or it's written that the size varies with the device.<br>
# I have converted all the app sizes given in kB and MB to MB. Also 1 have used 1kB = 1000MB

# In[ ]:


df['Size'] = df['Size'].apply(lambda x: str(x).replace('M',''))
df['Size'] = df['Size'].apply(lambda x: str(x).replace('k','e-3'))


# In[ ]:


#Converting the data type of Size category to float wherever possible
def convert(val):
    try:
        return float(val)
    except:
        return val
df['Size'] = df['Size'].apply(lambda x: convert(x))


# In[ ]:


#Seperate the apps whose size is given from those whose size varies with the device.
sized = df[df['Size'] != 'Varies with device'].copy()


# In[ ]:


sized['Size'] = pd.to_numeric(sized['Size'])


# In[ ]:


plt.figure(figsize=(12,6))
plt.title('Distribution of App Sizes')
sns.distplot(sized['Size'],bins = 30,rug=True)
plt.show()


# In[ ]:


size_less_20 = len(sized[sized['Size'] <= 50 ])/len(sized)*100
print('Percentage of Apps in the beauty category: {}%'.format(round(size_less_20,2)))


# In the dataset majority(88%) of the apps whose size are given have app size less than or equal to 50MB

# #### **4. Installs**

# In[ ]:


order = ['0','0+','1+','5+','10+','50+','100+','500+','1,000+','5,000+','10,000+','50,000+','100,000+','500,000+','1,000,000+',
         '5,000,000+','10,000,000+',
         '50,000,000+','100,000,000+','500,000,000+','1,000,000,000+']
sns.set_style('whitegrid')
plt.figure(figsize=(22,8))
plt.title('Number of apps on the basis of Installs')
sns.countplot(x='Installs',data = df,palette='hls',order = order)
plt.xticks(rotation = 90)

plt.show()


# In[ ]:


print('{}% apps in the play store having more than 1,000,000 installs and {}% apps have more than 10,000,000+ downloads' .format(round(len(df[df['Installs'] == '1,000,000+'])/len(df)*100,2),round(len(df[df['Installs'] == '10,000,000+'])/len(df)*100,2)))


# #### **5. Type**

# In[ ]:


print('Apps on the basis of Type are classified as')
print('--------------------------------------------------------------------')

count = 1
for i in df['Type'].unique():
    print(count,': ',i)
    count = count + 1


# In[ ]:


plt.figure(figsize=(10,6))

# Data to plot
labels = ['Free','Paid']
sizes = [len(df[df['Type'] == 'Free']),len(df[df['Type'] == 'Paid'])]
colors = ['skyblue', 'yellowgreen','orange','gold']
explode = (0.1, 0)  # explode 1st slice
 
# Plot
plt.title('Percentage of Free and paid apps in playstore')
plt.pie(sizes, labels=labels,
autopct='%1.1f%%', startangle=380,colors=colors,explode=explode)

plt.axis('equal')
plt.show()


# Most of the apps in the app store are free.Only a small percentage of apps are paid apps.

# #### **6. Price**

# In[ ]:


df['Price'] = df['Price'].apply(lambda x: str(x).replace('$',''))
df['Price'] = pd.to_numeric(df['Price'])


# **Seperating paid apps from free apps**

# I have created a seperate dataset for paid apps from the whole playstore dataset

# In[ ]:


paid_apps = df[df['Price'] != 0]


# In[ ]:


plt.figure(figsize=(8,6))
plt.title('Distribution of Paid App Prices')
sns.distplot(paid_apps['Price'],bins=50)
plt.show()


# In[ ]:


price_less_10 = len(paid_apps[paid_apps['Price'] <= 10])/len(paid_apps)*100
print('Percentage of Apps having price less than 10$: {}%'.format(round(price_less_10,2)))


# 89% apps in the play store have a price tag of 10\\$ or less. Although some apps have price greater than 350\\$

# Checking apps having price greater than 350$

# In[ ]:


paid_apps[paid_apps['Price'] >= 350]


# We found that 16 apps by the name **I am rich** or having similar names have a price tage of 399$ and most of them even have 10,000+ downloads.

# ####  **7. Content Rating**

# In[ ]:


print('Apps on the basis of Content Rating are classified as')
print('-------------------------------------------------------------------')

count = 1
for i in df['Content Rating'].unique():
    print(count,': ',i)
    count = count + 1


# In[ ]:


plt.figure(figsize=(12,6))
sns.countplot(x=df['Content Rating'],palette='hls')
plt.show()


# In[ ]:


print('Percentage of Apps having content rating as everyone: {}%'.format(round(len(df[df['Content Rating'] == 'Everyone'])/len(df)*100,2)))


# ####  **8. Genres**

# In[ ]:


plt.figure(figsize=(22,8))
plt.title('Number of Apps on the basis of Genre')
sns.countplot(x='Genres',data = df,palette='hls')
plt.xticks(rotation = 90)
plt.show()


# In[ ]:


print('Total Number of Genres: ',df['Genres'].nunique())


# There are 119 Genres in the app store with the Tools Genre having the highest number of apps followed by Entertainment.

# 

# #### **10. Current Version**

# In[ ]:


plt.figure(figsize=(22,8))
plt.title('Number of Apps on the basis of Android version required to run them')
sns.countplot(x='Android Ver',data = df.sort_values(by = 'Android Ver'),palette='hls')
plt.xticks(rotation = 90)

plt.show()


# 22.6% of the apps out of 10840 apps require android version 4.1 or greater version to work.

# In[ ]:


#function to convert columns to numeric data type from object data type
for i in df.columns:
    try:
        df[i] = pd.to_numeric(df[i])
    except:
        pass


# ### **Relationships between different features**

# ####  **1. Rating vs. Category**

# In[ ]:


plt.figure(figsize=(20,6))
sns.boxplot(x='Category',y='Rating',data = df)
plt.xticks(rotation=90)
plt.title('App ratings across different categories')
plt.show()


# In[ ]:


rating = pd.DataFrame(df['Rating'].describe()).T
rating


# **Findings**

# The minimum and maximum rating an app can have are 1 and 5 respectively across all categories.<br>
# The **mean** rating of apps across all categories is **4.2** with a **standard** deviation of **0.51**. Also, the **median** rating of apps across all categories is **4.3**.

# ####  **Category vs. Reviews**

# In[ ]:


sns.set_style('whitegrid')
plt.figure(figsize=(15,8))
sns.scatterplot(y='Category',x='Reviews',data = df,hue='Category',legend=False)
plt.xticks(rotation=90)
plt.title('Number of reviews on the basis of Category')
plt.show()


# In[ ]:


#Number of apps having 0 reviews
len(df[df['Reviews'] == 0])
review_0_category = pd.DataFrame(df[df['Reviews'] == 0]['Category'].describe())
#App having maximum reviews.
max_review_app = df[df['Reviews'] == max(df['Reviews'])]


# **Findings**

# **1.** Most of the apps across different categories have less than 10,000,000 reviews.<br>
# **2.** 596 apps have **0** reviews with most of the apps belonging to **BUSINESS** category.<br>
# **3.** The app having the maximum number of reviews(78158306) is **Facebook** which belong to the **SOCIAL** category.
# 

# The EDA is currently not complete and I will add more findings in the future.<br>
# **Suggestions are welcome**

# In[ ]:




