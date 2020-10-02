#!/usr/bin/env python
# coding: utf-8

# # Amazon Alexa Reveiw
# ## Data analysis & sentiment analysis using Google Natural Language API

# ## 1) Loading review data

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


review_data = pd.read_csv('../input/amazon-alexa-reviews/amazon_alexa.tsv', delimiter='\t', encoding='utf-8')


# In[ ]:


review_data.head()


# In[ ]:


review_data.isnull().sum()


# ## 2) Data visualization

# ### 2.1) Overall rating about Amazon Alexa

# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


num_rating5 = review_data['rating'][review_data['rating']==5].count()
num_rating4 = review_data['rating'][review_data['rating']==4].count()
num_rating3 = review_data['rating'][review_data['rating']==3].count()
num_rating2 = review_data['rating'][review_data['rating']==2].count()
num_rating1 = review_data['rating'][review_data['rating']==1].count()


# In[ ]:


x = ['5','4','3','2','1']


# In[ ]:


ratio_rating = [num_rating5 / len(review_data['rating']), num_rating4 / len(review_data['rating']), num_rating3 / len(review_data['rating']), 
                num_rating2 / len(review_data['rating']), num_rating1 / len(review_data['rating'])]


# In[ ]:


sns.barplot(x, ratio_rating)
plt.title("Consumer's rating")
plt.xlabel("Rating")
plt.ylabel("Ratio of rating")
plt.show()


# ### 2.2) Feedback, Rating for Alexa variation

# In[ ]:


var_rate = pd.pivot_table(review_data, index = ['variation'])
var_rate.head()


# #### Bar chart

# In[ ]:


plt.figure(figsize = (30,10))
sns.barplot(x='variation',y='rating', data=review_data)
plt.show()


# #### Violin Plot

# In[ ]:


plt.figure(figsize = (30,10))
sns.violinplot(x='variation',y='rating', data=review_data)
plt.show()


# #### Highest feedback, rated Alexa variation

# In[ ]:


var_rate['feedback'].idxmax(), var_rate['rating'].idxmax()


# #### Lowest feedback, rated Alexa variation

# In[ ]:


var_rate['feedback'].idxmin(), var_rate['rating'].idxmin()


# ## 3) Sentiment Analysis using Google Natural Language API
# ### Reference: https://cloud.google.com/natural-language/docs/reference/libraries#installing_the_client_library
# ### Because of packages setting of Kaggle, I just put my code on kernel without running

# In[ ]:


# from google.cloud import language
# from google.cloud.language import enums
# from google.cloud.language import types


# In[ ]:


# path = ''  # FULL path to your service account key
# client = language.LanguageServiceClient.from_service_account_json(path)


# In[ ]:


# senti_score = list()
# senti_mag = list()


# In[ ]:


# for i in range(len(review_data['verified_reviews'])):
#     text = review_data['verified_reviews'][i]
#     document = types.Document(
#         content = text,
#         type    = enums.Document.Type.PLAIN_TEXT)
#     # Detects the sentiment of the text
#     sentiment = client.analyze_sentiment(document=document).document_sentiment
#     senti_score.append(sentiment.score)
#     senti_mag.append(sentiment.magnitude)
#     print('{} is completed'.format(i))


# In[ ]:


# review_data['sentiment_score'] = senti_score
# review_data['sentiment_magnitude'] = senti_mag
# review_data.head()


# In[ ]:


# review_data.to_csv('Amazon_review.csv')


# ## 4) Data analysis based on sentiment analysis

# ### Loading review data contained semtiment analysis result

# In[ ]:


review = pd.read_csv('../input/amazon-alexa-review-with-sentiment-analysis/Amazon_review.csv')


# #### In data set, there are sentiment score and sentiment magnitude. If sentiment score is positive, then it is positive review. And if sentiment score is negative, it is negative review.

# In[ ]:


review.head()


# ### bar chart of sentiment score

# In[ ]:


plt.figure(figsize = (30,10))
sns.barplot(x='variation',y='sentiment_score', data=review)
plt.show()


# ### Normalization

# In[ ]:


review_norm = pd.pivot_table(review, index=['variation'])


# In[ ]:


rating_max = review_norm['rating'].max()
review_norm['rating'] = review_norm['rating'] / rating_max * 100
score_max = review_norm['sentiment_score'].max()
review_norm['sentiment_score'] = review_norm['sentiment_score'] / score_max * 100
magnitude_max = review_norm['sentiment_magnitude'].max()
review_norm['sentiment_magnitude'] = review_norm['sentiment_magnitude'] / magnitude_max * 100
feedback_max = review_norm['feedback'].max()
review_norm['feedback'] = review_norm['feedback'] / feedback_max * 100


# In[ ]:


review_norm.head()


# In[ ]:


review_norm_sort = review_norm.sort_values(by='rating', ascending=False)


# #### Heatmap

# In[ ]:


target_col = ['feedback','rating','sentiment_score','sentiment_magnitude']

plt.figure()
sns.heatmap(review_norm_sort[target_col], annot=True, fmt='f', linewidths=.5)
plt.show()


# ### 4.1) Average ratings and product types with negative sentiment analysis scores

# In[ ]:


minus_review = review[review['sentiment_score'] < 0]
minus_review.head()


# In[ ]:


pd.pivot_table(minus_review, index=['variation'], values=['sentiment_score'], aggfunc=[np.mean, len], margins=False)


# In[ ]:


minus_review.groupby(['variation'])['sentiment_score'].count().sort_values(ascending=False)


# ### 4.2) Average ratings and product types with positive sentiment analysis scores

# In[ ]:


plus_review = review[review['sentiment_score'] > 0]
plus_review.head()


# In[ ]:


pd.pivot_table(plus_review, index=['variation'], values=['sentiment_score'], aggfunc=[np.mean, len], margins=False)


# In[ ]:


plus_review.groupby(['variation'])['sentiment_score'].count().sort_values(ascending=False)


# ### 4.3) Correlation between sentiment analysis, rating, and feedback

# In[ ]:


from scipy.stats import pearsonr


# In[ ]:


corr = pd.DataFrame()


# In[ ]:


corr['rating'] = review['rating']
corr['feedback'] = review['feedback']
corr['sentiment_score'] = review['sentiment_score']
corr['sentiment_magnitude'] = review['sentiment_magnitude']
corr.head()


# In[ ]:


corr.corr()


# #### correlation heat map

# In[ ]:


plt.figure() 
sns.heatmap(corr.corr(), cmap='BuGn')


# #### p-value of correlation

# In[ ]:


cols = corr.columns
mat = corr.values
arr = np.zeros((len(cols),len(cols)), dtype=object)


# In[ ]:


for xi, x in enumerate(mat.T):
    for yi, y in enumerate(mat.T[xi:]):
        arr[xi, yi+xi] = pearsonr(x,y)[1]
        arr[yi+xi, xi] = arr[xi, yi+xi]


# In[ ]:


p_value = pd.DataFrame(arr, index=cols, columns=cols)


# In[ ]:


p_value


# ## 5) Forecasting rating

# ### Feature = 'feedback', 'sentiment_score', 'sentiment_magnitude'

# In[ ]:


feature = review
del_col = ['date','variation','verified_reviews']
feature = feature.drop(del_col, axis=1)
feature.head()


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


y = feature['rating']
x_feature = ['feedback','sentiment_score','sentiment_magnitude']
x = feature[x_feature]


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state = 42)


# ### 5.1) Decision Tree Classifier

# In[ ]:


from sklearn.tree import DecisionTreeClassifier


# In[ ]:


DT_clf = DecisionTreeClassifier()
DT_clf.fit(X_train, y_train)


# In[ ]:


DT_ypred = DT_clf.predict(X_test)


# In[ ]:


DT_accuracy = DT_clf.score(X_test, y_test)
DT_accuracy


# ### 5.2) Random Forest Classifier

# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


RF_clf = RandomForestClassifier(n_estimators = 100)
RF_clf.fit(X_train, y_train)


# In[ ]:


RF_ypred = RF_clf.predict(X_test)


# In[ ]:


RF_accuracy = RF_clf.score(X_test, y_test)
RF_accuracy


# In[ ]:




