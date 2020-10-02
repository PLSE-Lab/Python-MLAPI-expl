#!/usr/bin/env python
# coding: utf-8

# ## Using nltk to classifiy adjectives in amazon food reviews as positive or negative

# ### Structure
# 
# * Import libraries
# * Load and prepare the Dataset
# * Data exploration
# * Filter adjectives with NLTK
# * Transform Data for Model (Count Vectorizer)
# * Function to classify adjectives as positive or negative
# * Top 50 most used adjectives
# * Conclusion

# ### Libraries

# In[ ]:


import numpy as np 
import pandas as pd 
import nltk
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix


# ### Load and prepare the Dataset
# 
# - loading first 50.000 rows
# - only loading columns (Score, Text) that are necessary for my purpose
# - defining reviews with scores >= 4 as positive reviews and reviews with scores < 4 as negative reviews

# In[ ]:


df = pd.read_csv("../input/Reviews.csv", nrows = 50000, usecols = ["Score", "Text"])
df["Score"] = np.where(df["Score"] >= 4, "positive", "negative")


# ### Data exploration
# 
# - displaying general informations about the data
# - displaying first 5 rows of the data
# - Visualizing the distribution of positive and negative ratings

# In[ ]:


df.info()


# In[ ]:


df.head()


# In[ ]:


sns.countplot(data = df, x= df["Score"]).set_title("Score distribution", fontweight = "bold")
plt.show()


# ### Filter adjectives with NLTK
# 
# 1. Creating a list for each review and split all sentences in a review
# 2. Split all words from each sentence and add tags all words 
# 
# *Review : Sentence 1: [(word 1, tag 1), ... , (word n, tag n)]; Sentence n: [(word 1, tag 1), ... , (word n, tag n)]*
# 3. Add all adjectives to a list (without tag, only word)
# 
# *Review: ['Adjective 1', Adjective 2, ... , Adjective n]*
# 
# 4. Transforming the list of adjectives of each review to one string each review, which is needed for the model later on.
# 
# *Review: [Adjective1 Adjective2 Adjective3 Adjective 4]*

# In[ ]:


texts = df["Text"]

import nltk

texts_transformed = []
for review in texts:
    sentences = nltk.sent_tokenize(review)
    adjectives = []
    
    
    for sentence in sentences:
        words = nltk.word_tokenize(sentence) 
        words_tagged = nltk.pos_tag(words)
        
        adj_add = [adjectives.append(word_tagged[0]) for word_tagged in words_tagged if word_tagged[1] == "JJ"]
                
    texts_transformed.append(" ".join(adjectives)) 


# ### Transform Data for Model (Count Vectorizer)

# In[ ]:


X = texts_transformed
y = df["Score"]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)

cv = CountVectorizer(max_features = 50)
cv.fit(X_train)

X_train = cv.transform(X_train)
X_test = cv.transform(X_test)


# In[ ]:


arr = X_train.toarray()

print(arr.shape)


# ### multinomial Naive Bayes
# 
# * fit model with train data
# * calculate r2-score with test data

# In[ ]:


model = MultinomialNB()
model.fit(X_train, y_train)

print(model.score(X_test, y_test))


# ### Confusion-Matrix
# 
# Due to the fact that the already classified data we trained our model with is not very balanced (way more positives than negatives), we don't know if the  R2 score is reliable. The Model could be just labeling almost everything as positive, even if it should be negative. With the Confusion matrix, we can evaluate the accuracy of the classification more clearer and see exactly where our multinomial naive Bayes has its errors.
# 
# Format of Confusion-Matrix:
# 
# | True negative | False positiv
# | --- | --- |
# | False negativ | True positive

# In[ ]:


y_test_pred = model.predict(X_test)
print(confusion_matrix(y_test, y_test_pred))


# ### Function to classify adjectives as positive or negative

# In[ ]:


def classifier(adjective):
    return model.predict(cv.transform([adjective]))
print(classifier('great'))
print(classifier('bad'))


# ### Top 50 most used adjectives

# In[ ]:


adj = list(zip(model.coef_[0], cv.get_feature_names()))
adj = sorted(adj, reverse = True)
for a in adj:
    print(a)


# ### Conclusion
# 
# Adjectives with higher coefficients (good, great, delicious, free) are correlated to positive reviews and adjectives with lower coefficients (bad, expensive) reduce the likelihood of an adjective having a positive meaning/being in a positive review and thus contribute to being classified as negative by the multinomial Naive Bayes. 

# In[ ]:




