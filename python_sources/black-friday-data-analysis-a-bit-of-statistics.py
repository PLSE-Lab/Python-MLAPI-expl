#!/usr/bin/env python
# coding: utf-8

# # **Introduction**
# 
# * In this kernel I will make an analysis of Black Friday data. 
# * Some tools of  seaborn and plotly library will be used for visualization.
# * Also I will show you a few perspectives in terms of statistics by explainig some terms "Skewness" and "Kurtosis" of distributions. 
# * Let's start with importing libraries and dataframe.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# plotly
import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
# import warnings
import warnings
# filter warnings
warnings.filterwarnings('ignore')
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
# Any results you write to the current directory are saved as output.
import os
print(os.listdir("../input"))


# In[ ]:


df = pd.read_csv('../input/BlackFriday.csv')
df.info()


# *It seems there are lots of NaN values, let's see how many of them really there are.*

# In[ ]:


df.isnull().sum().sort_values(ascending=False)


# In[ ]:


df.sample(10)


# *Now let's check the number of unique values like User, Product etc. in the dataset*

# In[ ]:


df.nunique()


# *It's time to check some statistical values of data, in terms of **Purchase**. With this information we'll obtain, maybe we can classify the people as** "Poor"** or **"Stinger"**  whatever you call, or vice versa. 

# In[ ]:


df.Purchase.describe()


# *Let's put on some make up to following information with a boxplot within visible std & mean.*
# 
# **Generally, prefer using a violinplot within a box to see either distribution and the statistics.**

# In[ ]:


box1 = go.Box(y=df.Purchase,
              name='Purchase',
              marker = dict(color = 'rgba(15, 100, 150)'),
              boxmean='sd')

databox1 = [box1]
iplot(databox1)


# *I choose to classify the consumers with respect to the mean value of all purchases, if someone is under the average then he/she is said to be "poor".*
# 
# Let's do it by adding a new column as "Wealthiness" to the data.

# In[ ]:


df['Wealthiness'] = np.where(df.Purchase < df.Purchase.mean(), "Poor", "Rich")
df.Wealthiness.value_counts()


# Advertising has as chasing  cars and clothes, working jobs we hate so we can buy the shit we don't need. - *Tyler Durden*
# 
# *Don't mind the score above, there can't be such numerous rich people. Let's visualize it via barplot.*

# In[ ]:


bar1 = go.Bar(x=df.Wealthiness.value_counts().index,
              y=df.Wealthiness.value_counts().values,
              marker = dict(color = 'rgba(150, 180, 32)',
                            line=dict(color='rgb(104,32,0)',width=1.5))
              )
databar = [bar1]
iplot(databar)


# *I think every knows the answer of following question but.. Which gender is addicted to shopping more than heroin? Let's see.*

# In[ ]:


df.groupby('Gender').agg(dict(Purchase=['min', 'mean', 'max']))


# Maybe I act too many sexist above, let's see this also in boxplot. Since it occupy a huge place, I'll no longer use plotly in plots in which seaborn can handle it.

# In[ ]:


plt.figure(figsize=(7,5))
sns.barplot(x=df.Gender.value_counts().index,y=df.Gender.value_counts().values )
plt.show()


# *I wonder which age range made shopping more. And also distribution of those ranges.*

# In[ ]:


df.Age.value_counts()


# In[ ]:


plt.figure(figsize=(7,5))
sns.barplot(x=df.Age.value_counts().index, y=df.Age.value_counts().values)
plt.show()


# *To see the distributions of age ranges, first we need to encode data since it's comprised of strings.*

# In[ ]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df_new = le.fit_transform(df.Age) 

plt.figure(figsize=(7,5))
sns.distplot(df_new,hist=False)
plt.show()


# *As seen above, the distribution is terrible. Since it's a categorical (discrete) data we can't prove it numerically with "Skewness" and "Kurtosis" values. Instead of those, we can check the Purchase values. By the way;*
# 
# **Skewness **: Measure of symmetry, or more precisely, the lack of symmetry. (**Is data skewed to larger or smaller values?**) "0" means data is totally symmetric.
# 
# **Kurtosis ** : Measure of whether the data are heavy-tailed or light-tailed relative to a normal distribution. (**Are there outliers?**) "0" means data is distributed normally if Fisher's definition is used. If Pearson's definition used, then result "3" gives a normal distribution.

# In[ ]:


plt.figure(figsize=(7,5))
sns.distplot(df.Purchase)
print('Skewness:', df.Purchase.skew())
print('Kurtosis:', df.Purchase.kurt())


# * When skewness is zero, that means data is completely symmetric.
# * Low kurtosis value means distribution have light tails, or lack of outliers.
# 
# Unfortunately, we can't afford those 2 conditions. Data is not symmetric and we have outliers which means some people sold their kidneys to buy too many iphones at discount lol. We did the same above with boxplot, but let's see it again here.

# In[ ]:


plt.figure(figsize=(6,6))
sns.boxplot(x=df.Purchase)
plt.show()


# *Is there any correlation among features? I don't think so, but we have to take a glance at it.*

# In[ ]:


df2 = df.drop(['User_ID'],axis=1)

plt.figure(figsize=(8,8))
sns.heatmap(df2.corr(), annot=True, linewidths=0.5, linecolor='black', cmap='Blues')
plt.xticks(rotation=90)
plt.show()


# *I'm sure everyone realized there are a lot of different consumer, it was close to 6k we've check it above.*
# 
# 
# **So how about finding first 10 users spending the most in Black Friday?**

# * Firstly we will create an empty list "moneybox" to store purchases to append purchases of users into it, then for all consumer in "User_ID" we will find their total purchases in black friday.

# In[ ]:


moneybox = []
consumer_list = list(df.User_ID.unique())

for i in consumer_list:
    money = df[df.User_ID == i].Purchase.sum()
    moneybox.append(money)


# In[ ]:


users_purchases = pd.DataFrame(dict(Users=consumer_list, Purchases=moneybox))
indices = (users_purchases.Purchases.sort_values(ascending=False)).index.values[:10]
users_purchases = users_purchases.reindex(indices).iloc[:10]
users_purchases


# * I tried to sort indices to replace with descending sorted ones, but seaborn library didn't let me plot w.r.t. sorted indices. So I created a dataframe in ascending sort called "result".

# In[ ]:


plt.figure(figsize=(8,6))
users_purchases.groupby(["Users"])['Purchases'].sum().sort_values(ascending=False).plot('bar', position=0.2)
plt.xticks(rotation=45)
plt.ylabel('Purchases')
plt.show()


# In[ ]:


result = users_purchases.groupby(["Users"])['Purchases'].aggregate(np.median).reset_index().sort_values('Purchases')

plt.figure(figsize=(8,6))
sns.barplot(x=users_purchases.Users, y=users_purchases.Purchases, order=result['Users'])
plt.xticks(rotation=45)
plt.show()


# *Genders of the people buying different products in "Product Category 1".*

# In[ ]:


plt.figure(figsize=(12,7))
sns.countplot(x=df.Product_Category_1, hue=df.Gender)
plt.show()


# *Marital Status w.r..t to gender of age ranges*

# In[ ]:


df['Marit_of_Genders'] = df.apply(lambda x:'%s.%s' % (x['Gender'],x['Marital_Status']),axis=1)
plt.figure(figsize =(12,7))
sns.countplot(x=df.Age, hue=df.Marit_of_Genders)
plt.show()


# *Marital status w.r.t. genders of people shopping from Product Category 1*

# In[ ]:


plt.figure(figsize =(12,7))
sns.countplot(x=df.Product_Category_1, hue=df.Marit_of_Genders)
plt.show()


# In[ ]:


plt.figure(figsize=(7,5))
df.groupby('Gender')['Age'].value_counts().sort_values().plot('bar')
plt.xticks(rotation=60)
plt.show()


# * Top selling products

# In[ ]:


plt.figure(figsize=(7,5))
df.groupby('Product_ID')['Purchase'].count().nlargest(10).sort_values().plot('barh')
plt.show()


# * I'll be thankful if you Upvotein case you like it, thanks in advance*
# 
# # **END**
