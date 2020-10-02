#!/usr/bin/env python
# coding: utf-8

# ## NEWS ARTICLE CLASSIFICATION
# 
# ### 1. Load data
# * Records in the file are comma delimited;
# * Column titles are included in the text file;
# * Load the data into a Pandas data frame;
# * View unique values for the `category` column for later transformation to discrete numerical values.

# In[ ]:


import pandas as pd

news_df = pd.read_csv("../input/uci-news-aggregator.csv", sep = ",")
# news_df.CATEGORY.unique()


# ### 2. Preprocess data
# * Transform categories into discrete numerical values;
# * Transform all words to lowercase;
# * Remove all punctuations.

# In[ ]:


import string

news_df['CATEGORY'] = news_df.CATEGORY.map({ 'b': 1, 't': 2, 'e': 3, 'm': 4 })
news_df['TITLE'] = news_df.TITLE.map(
    lambda x: x.lower().translate(str.maketrans('','', string.punctuation))
)

news_df.head()


# ### 3. Split into train and test data sets

# In[ ]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    news_df['TITLE'], 
    news_df['CATEGORY'], 
    random_state = 1
)

print("Training dataset: ", X_train.shape[0])
print("Test dataset: ", X_test.shape[0])


# ### 4. Extract features
# * Apply bag of words processing to the dataset

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer

count_vector = CountVectorizer(stop_words = 'english')
training_data = count_vector.fit_transform(X_train)
testing_data = count_vector.transform(X_test)


# ### 5. Train Multinomial Naive Bayes classifier

# In[ ]:


from sklearn.naive_bayes import MultinomialNB

naive_bayes = MultinomialNB()
naive_bayes.fit(training_data, y_train)


# ### 6. Generate predictions

# In[ ]:


predictions = naive_bayes.predict(testing_data)
predictions


# ### 7. Evaluate model performance
# * This is a multi-class classification. So, for these evaulation scores, explicitly specify `average` = `weighted`

# In[ ]:


from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

print("Accuracy score: ", accuracy_score(y_test, predictions))
print("Recall score: ", recall_score(y_test, predictions, average = 'weighted'))
print("Precision score: ", precision_score(y_test, predictions, average = 'weighted'))
print("F1 score: ", f1_score(y_test, predictions, average = 'weighted'))

