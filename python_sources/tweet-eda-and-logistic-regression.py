#!/usr/bin/env python
# coding: utf-8

# ## Tweet EDA and Logistic Regression
# 
# This is a starter notebook to show how to quickly get up-and-running with sentiment analysis. I will take a quick look at the data and use bag-of-words vectors to train a logistic regression model.

# First, we read in the .csv files.

# In[ ]:


import pandas as pd

train = pd.read_csv('../input/tweet-sentiment-analysis/train.csv')
test = pd.read_csv('../input/tweet-sentiment-analysis/test.csv')


# Now lets take a look at the training data.

# In[ ]:


train.head()


# In[ ]:


print(f"There are {len(train)} tweets in the training data.")
positive = sum(train.target)/len(train)
print(f"{positive:.2f}% of the tweets are positive.")
avg_len = sum([len(text) for text in train.text])/len(train)
print(f"The average length of the tweets is {avg_len:.1f} characters.")


# The `CountVectorizer` from scikit-learn makes it easy to create Bag-of-Words vectors.

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer

count_vect = CountVectorizer()
count_vect.fit(train['text'])

bow_train = count_vect.transform(train['text'])
bow_test = count_vect.transform(test['text'])


# In[ ]:


print(f"There are {len(count_vect.vocabulary_)} words in the vocabulary.")


# That seems like a lot of words. Lets run `CountVectorizer` again but only use the top 15,000 most common words.

# In[ ]:


count_vect = CountVectorizer(max_features=15000)
count_vect.fit(train['text'])

bow_train = count_vect.transform(train['text'])
bow_test = count_vect.transform(test['text'])


# Great! Now we can fit a logistic regression model on those vectors. Scikit-learn also provides a handy `LogisticRegression` function to make modeling easy.

# In[ ]:


from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000).fit(bow_train, train.target)


# The model is trained, time to do some predictions. The `predict_proba` method with give us the probability that a tweet is positive.

# In[ ]:


predictions = model.predict_proba(bow_test)
predictions = [probs[1] for probs in predictions]


# To generate a submission, replace the `Predicted` column in the sample submission with the above predictions.

# In[ ]:


submission = pd.read_csv('../input/tweet-sentiment-analysis/sample_submission.csv')
submission['Predicted'] = predictions


# Finally, I save the submission file to the `/kaggle/working` folder. After I save and run this notebook, I will submit these results to the competition and see how well my model worked.

# In[ ]:


submission.to_csv('/kaggle/working/submission.csv', index=False)

