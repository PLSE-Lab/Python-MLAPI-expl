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
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('fivethirtyeight')

import warnings
warnings.filterwarnings("ignore")


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# ****Yelp:****
# 
# Yelp is an American multinational corporation headquartered in San Francisco, California. It develops, hosts and markets Yelp.com and the Yelp mobile app, which publish crowd-sourced reviews about local businesses, as well as the online reservation service Yelp Reservations. The company also trains small businesses in how to respond to reviews, hosts social events for reviewers, and provides data about businesses, including health inspection scores.
# 
# 
# *
# source: Wikipedia*
# 
# 
# **1.1  Businesses on Yelp **

# In[ ]:


yelp_business = pd.read_csv('../input/yelp_business.csv')


# In[ ]:


yelp_business.head()


# Now let's check for the missing values
# 
# There are many missing values in neighborhood and postal code fields.

# **1.2 Missing values in our dataset:**

# In[ ]:


yelp_business.isnull().sum()


# The following heatmap can be used to visualize the missing values effectively

# In[ ]:


plt.figure(figsize=(12,10))
f = sns.heatmap(yelp_business.isnull(),yticklabels=False, cbar=False, cmap = 'viridis')


# let's check how many of the businesses are open using a count plot on seaborn

# **1.3 Distribution of variables**

# In[ ]:


plt.figure(figsize=(6,6))
sns.countplot(x='is_open',data=yelp_business);


# In[ ]:


#let's look at the number of unique values in the stars variable.

yelp_business['stars'].nunique()


# let's use a plot to visualize the frequency of the ratings

# In[ ]:


sns.countplot(x='stars',data=yelp_business);


# We see that most of the businesses got a rating of either 3 or above

# Let's visualize the distribution of number of reviews that a business has, we have applied log since the distribution is extremely skewed

# In[ ]:


sns.distplot(yelp_business['review_count'].apply(np.log1p));


# In[ ]:


yelp_business_attributes = pd.read_csv('../input/yelp_business_attributes.csv')


# Our intuition is that the number of businesses vary a lot from one state to another.
# 
# We can use a tree map to visualize some of them since  a tree map might not be able to fit all the states

# In[ ]:



by_state = yelp_business.groupby('state')


# In[ ]:


import squarify    # pip install squarify (algorithm for treemap)
plt.figure(figsize=(12,12))

a = by_state['business_id'].count()

a.sort_values(ascending=False,inplace=True)

squarify.plot(sizes= a[0:15].values, label= a[0:15].index, alpha=0.9)

plt.axis('off')
plt.tight_layout()


# From the tree chat, we see that Arizona and Nevada has the most number of businesses

# **1.4 Leading Business Categories:**
# 
# Let's look at the leading business categories in out dataset. 
# A business is linked to multiple categories in our dataset, so we have to do a bit of preprocessing.
# 

# In[ ]:




business_cats=';'.join(yelp_business['categories'])
cats=pd.DataFrame(business_cats.split(';'),columns=['category'])
cats_ser = cats.category.value_counts()


cats_df = pd.DataFrame(cats_ser)
cats_df.reset_index(inplace=True)

plt.figure(figsize=(12,10))
f = sns.barplot( y= 'index',x = 'category' , data = cats_df.iloc[0:20])
f.set_ylabel('Category')
f.set_xlabel('Number of businesses');


# **1.5 Top Business Names:**
# 
# The top business names in our dataset can be visualized with a word cloud.

# In[ ]:


from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

plt.figure(figsize=(12,10))

wordcloud = WordCloud(background_color='white',
                          width=1200,
                      stopwords = STOPWORDS,
                          height=1000
                         ).generate(str(yelp_business['name']))


plt.imshow(wordcloud)
plt.axis('off');


# In[ ]:


from sklearn.cross_validation import train_test_split
yelp_business.head()


# In[ ]:


X = pd.get_dummies(yelp_business['city'])
yelp_business = pd.concat([yelp_business,X], axis=1)


# In[ ]:


del X;


# In[ ]:


drop_cols = ['business_id',
 'name',
 'neighborhood',
 'address',
 'city',
 'state',
 'postal_code',
  'is_open',           
 'categories']


# In[ ]:


cols = [ i for i in yelp_business.columns if i not in drop_cols]


# In[ ]:


cols1 = ['latitude',
 'longitude',
 'stars',
 'review_count']


# In[ ]:





# In[ ]:


X = yelp_business[cols1]
y = yelp_business['is_open']


# In[ ]:


from imblearn.over_sampling import SMOTE


# In[ ]:


train_X, test_X, train_y, test_y = train_test_split(X,y,test_size = 0.3, random_state = 42)


# In[ ]:


train_X.fillna(0,inplace=True)
test_X.fillna(0,inplace=True)


# In[ ]:


sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_sample(train_X, train_y)


# In[ ]:


X_res = pd.DataFrame(X_res)
y_res = pd.DataFrame(y_res)
test_X = pd.DataFrame(test_X)
test_y = pd.DataFrame(test_y)


# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix


# In[ ]:


L = [0.0001,0.001,0.01,0.1,1,10]


# In[ ]:


accuracy = []
for i in L:
    LR = LogisticRegression(C=i)
    LR.fit(X_res,y_res)
    pred_y = LR.predict(test_X)
    
    accuracy.append(accuracy_score(test_y,pred_y))


# In[ ]:


accuracy


# In[ ]:


y_res[0].value_counts()


# In[ ]:


LR = LogisticRegression(C=0.001)
LR.fit(X_res,y_res)
pred_y = LR.predict(test_X)


# In[ ]:


confusion_matrix(test_y,pred_y)


# In[ ]:


from sklearn.metrics import accuracy_score

accuracy_score(test_y,pred_y)


# In[ ]:


review = pd.read_csv('../input/yelp_review.csv')
checkin = pd.read_csv('../input/yelp_checkin.csv')


# In[ ]:


review.head()


# In[ ]:


review_busines = review.groupby(by=['review_id'])


# In[ ]:


review_businesid = pd.DataFrame()
review_businesid['25-percentile'] = np.percentile(review_busines['stars'],25)


# In[ ]:


review_businesid['50-percentile'] = np.percentile(review_busines['stars'],50)
review_businesid['75-percentile'] = np.percentile(review_busines['stars'],75)
review_businesid['Mean'] = review_busines['stars'].mean()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


checkin.head()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# **2. Business Attributes**

# In[ ]:


yelp_business_att = pd.read_csv('../input/yelp_business_attributes.csv')


# In[ ]:


plt.figure(figsize=(12,10))
f = sns.heatmap(yelp_business_att.isnull(),cbar=False,yticklabels=False,cmap='viridis')


# From the heatmap, we can infer that there are no missing values in our dataset.

# In[ ]:


sns.factorplot(yelp_business_attributes['AcceptsInsurance'])


# In[ ]:


yelp_business_hours = pd.read_csv('../input/yelp_business_hours.csv')


# In[ ]:


yelp_business_hours.head()


# In[ ]:


yelp_tip = pd.read_csv('../input/yelp_tip.csv')


# In[ ]:


yelp_tip.head()


# In[ ]:


from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

plt.figure(figsize=(12,10))

wordcloud = WordCloud(background_color='white',
                          width=1200,
                      stopwords = STOPWORDS,
                          height=1000
                         ).generate(str(yelp_tip['text']))


plt.imshow(wordcloud)
plt.axis('off')
plt.show()


# In[ ]:





# In[ ]:




