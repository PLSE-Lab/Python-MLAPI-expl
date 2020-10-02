#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


pd.set_option('display.width', 300)
pd.set_option('display.max_columns', 300)
pd.set_option('display.max_colwidth', 500)
pd.set_option('display.precision',3)

def visualization_settings():
    plt.clf()
    sns.set_style({"xtick.major.size":30,"ytick.major.size":30})
    plt.figure(figsize=(14,6))
    sns.set(font_scale=1.4)


# In[ ]:


df = pd.read_csv('../input/googleplaystore.csv')


# In[ ]:


df.head()


# In[ ]:


df.info()


# In[ ]:


df.shape


# Our dataset contains 10841 samples with 13 columns. Let's review our data column by column. Let's start with App column

# In[ ]:


df['App'].nunique()


# As we can see most of the Apps are unique. 9660 out of 10481 Apps are unique.

# In[ ]:


df[df['App']=='Facebook']


# In[ ]:


df[df['App']=='Instagram']


# For example, Facebook and Instagram have multiple rows with almost same information so we will remove multiple rows of the same Apps in our dataset and will kep only unique ones. 

# In[ ]:


df.drop_duplicates(subset='App', keep='first', inplace=True)


# In[ ]:


df.shape


# In[ ]:


df['App'].head()


# As we see, there are some signs such as **("&", "-", ",")** etc. We don't need those and can remove them 

# ## Removing Redundant Signs from App Names

# In[ ]:


import re
df['App'] = df['App'].map(lambda x: re.sub(r"[^\w\s]", "", x.title())) # Capitalize every word


# We have capitalize every word and removed all punctuations/ signs  rather than words and numbers 

# In[ ]:


df['App'].head()


# I believe this is better. Now we can check how many words there are in App Names

# In[ ]:


df['App_length'] = df['App'].map(lambda x: len(x.split()))
cols = df.columns.tolist()


# In[ ]:


cols.insert(1,'App_length') # We insert new column (App_length) after App column
cols.pop() # We removed the very last column name from the list
df = df[cols] # We have changed the column order in the dataframe


# We have reordered the column orders. The App and App Names are together now.

# In[ ]:


df.head()


# In[ ]:


df.loc[df[['App_length']].idxmax()]


# It is a little bit weird  but the longest App name has 25-word in it. What a cool idea to name the App with such a long name:)

# In[ ]:


df['App_length'].value_counts()


# In[ ]:


#sns.set(style="darkgrid")
visualization_settings()
plt.style.use("ggplot")
ax = sns.countplot(df['App_length'], alpha=0.9)
sns.despine()

# Label customizing
plt.ylabel('Count', fontsize=16)
plt.xlabel('App Name Length', fontsize=16)
plt.title("App Name Length Frequency", fontsize=16);


# In[ ]:


len(df[df['App_length']==3])


# Most common App name length is the ones with `3-word`. There  are 2537 3-word App names. It is almost 1 out of 4 in our database. Broadly speaking we can say that most of the App names are `less than 5 words`. Let's check now the most common 20 words in App names.

# In[ ]:


from collections import Counter
words = Counter("".join(df['App']).split()).items()
sorted_App_names = sorted(words, key=lambda x: x[1], reverse=True) # sorting on App name frequency


# In[ ]:


sorted_App_names[:20]


# The most common 20 words in App Names are shown above.  But just a minute most of the top 20 words in App Names are pretty common words such as conjuctions or prepositoins in English. Let's clean them and check the names again.

# In[ ]:


from nltk.corpus import stopwords
stopword_list= stopwords.words('english')
sorted_App_names = [x for x in sorted_App_names if x[0].lower() not in stopword_list] # stop words are exclueded from App names


# In[ ]:


sorted_App_names[:20]


# It makes much more sense now. As we can see the most common word in App Names is "**Free**".

# In[ ]:


# Let's review Category Column


# In[ ]:


df['Category'].value_counts()


# * The very last category name is 1.9,  it cannot be a category name.  Something must wrong with it so we will dropp it.

# In[ ]:


df = df[df['Category']!='1.9']


# In[ ]:


df['Category'] = df['Category'].map(lambda x: x.title())


# In[ ]:


category_count  = df['Category'].value_counts();
plt.figure(figsize=(20,8))
sns.barplot(category_count.index, category_count.values, alpha=0.8)
plt.ylabel('Count', fontsize=16)
plt.title("App Categories", fontsize=16);
plt.xticks(rotation=90);


# As seen from the graphic Family, Game and Tools categories are the top 3 ones, while Comics, Parenting and Beauty categories are the least used ones. 

# **Let's check out Rating column**

# In[ ]:


df['Rating'].describe()


# In[ ]:


rating_count = df['Rating'].value_counts()
plt.figure(figsize=(20,8))
sns.barplot(rating_count.index, rating_count.values, palette="Blues_d");
plt.ylabel('Count', fontsize=16)
plt.xlabel('Ratings', fontsize=16)
plt.title("Rating Frequency", fontsize=16);


# The rating mean is 4.2. We conclude that many people rate the apps more than 4 stars. 

# In[ ]:


df.groupby('Category')['Rating'].mean().sort_values(ascending=False)


# Highest average mean rating score belongs to "Events" category which is around 4.44, while "Dating" category is with the least mean rating which is around 3.97. Although Events category does have only 64 Apps, it has the highest mean rating score. We can conclude that its user profile tend to give higher rating star. 

# In[ ]:


df.head()


# **Let's review Reviews columns**

# In[ ]:


type(df['Reviews'][0])


# To make math opers on Reviews columns let's convert string type into integer

# In[ ]:


df['Reviews'] = df['Reviews'].map(lambda x: int(x))


# In[ ]:


df['Reviews'].describe()


# Some Apps has been rated almost 8 millions times. On the other hand some Apps has no review at all.

# In[ ]:


df[df['Reviews']>20000000]['App']


# The above applications have been reviewed more than 20 million times. 

# In[ ]:


df.loc[df[['Reviews']].idxmax()]


# **Facebook** is the most rated application in our dataset with more than **78 million** reviews.****

# In[ ]:


plt.figure(figsize=(20,8))
df.groupby('Category')['Reviews'].sum().sort_values(ascending=False).head(10).plot(kind='bar');
plt.ylabel('Count', fontsize=16)
plt.xlabel('Ratings', fontsize=16)
plt.title("Total Reviews Number for Top 10 Category", fontsize=16)
plt.xticks(rotation=0);


# The most rated category is '**Game**' which has more than** 600 million reviews** in total. Game, Communication, Social and Tools have more than **200 million reviews**.

# **Let's review Size column**

# In[ ]:


df['Size'].value_counts().head(10)


# The App size varies but the top ten can be seen above. We can argue that most common Apps' size varies between **11MB and 26 MB**, if we ignore the 'Varies with device'. 

# **Let's check Type now**

# In[ ]:


df['Type'].value_counts()


# There are only **756** apps are paid Apps  and the rest are free.

# In[ ]:


df['Price'] = df['Price'].map(lambda x: re.sub(r"[^\w\.]", "", x)) #remove $ from the price


# In[ ]:


type(df['Price'][0])


# We need to convert string into float

# In[ ]:


df['Price'] = df["Price"].map(lambda x: float(x))


# In[ ]:


df['Price'].mean()


# The average charge for paid Apps is **$1.01. ** 

# In[ ]:


df.loc[df[['Price']].idxmax()]


# The most expensive App is IM Rich Trump which is **$400** and it has been installed more than **10 thousand times.**.People have paid **400 thousads dollars**[](http://) so far for it.

# In[ ]:


df_paid = df[df['Price']>50]


# In[ ]:


df_paid.groupby('App')[['Price']].mean().sort_values(by='Price', ascending=False)


# The Apps more than** $50**  can be seen above. 

# In[ ]:


df_paid[['App', 'Genres', 'Price', 'Installs']].sort_values(by='Price', ascending=False)


# As we can see many people are willing to pay **some good money** for some Apps in certain categories such as **Finance, LifeStyle and Entertainment.**

# We wanted to get basic insights from the App Data we have. I hope you enjoyed my basic findings. Thank you.

# In[ ]:





# In[ ]:




