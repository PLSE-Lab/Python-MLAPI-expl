#!/usr/bin/env python
# coding: utf-8

# Hi guys,
# 
# This will be a very short example of how we can utilize TFIDF in combination with Chi2 test to find predictive features (and by that I mean filthy words). If you dare, read on...
# # Data Import
# We'll start by importing the data:

# In[ ]:


import pandas as pd

train = pd.read_csv('../input/train.csv', header = 0)
train.head()


# We'll just check if there are any empty fields:

# In[ ]:


train.info()


# Let's see if we can get some insights into the data by checking some standard metrics on the target fields:

# In[ ]:


train.describe()


# Looks like the mean value for the 'toxic' column is the highest. This means that more comments are labeled as 'toxic' than as 'severe toxic' or any other category. With the limited resources that the kernels provide, it would be best to focus only on predicting for that column.
# 
# To do that, we'll further split our training set into 'train' and 'test' set. This will help us at least partially evaluate our hypothesis.

# In[ ]:


from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import words
from sklearn.model_selection import train_test_split

X, y = train[['comment_text']], train[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.10, random_state = 42)


# # The Vectorizer
# We'll then instantiate a count vectorizer and create a matrix of all the tokens contained in each comment. The matrix will exclude all English stop words and vectorize only valid English words. This will have some consequences:
# 
# * Our algorithm will be optimized for English (other languages will be ignored)
# * Our algorithm will not take into account purposefully misspelled obscenities

# In[ ]:


vectorizer = CountVectorizer(stop_words = 'english',                             lowercase = True,                             max_df = 0.95,                             min_df = 0.05,                             vocabulary = set(words.words()))

vectorized_text = vectorizer.fit_transform(X_train.comment_text)


# We'll now use our vectorized matrix and run TFIDF on it:

# In[ ]:


transformer = TfidfTransformer(smooth_idf = False)
tfidf = transformer.fit_transform(vectorized_text)


# Here comes the interesting part, we'll use the weighted matrix terms to select the 200 best predictors of toxic comments. We can expect that those would be quite obscene terms.

# In[ ]:


from sklearn.feature_selection import SelectKBest, chi2

ch2 = SelectKBest(chi2, k = 200)
best_features = ch2.fit_transform(tfidf, y_train.toxic)


# Fair warning, the next code snippet wil display the distilled essence of online hatred. Scroll further only if you can stomach it... [Otherwise, jump directly to the next section.](#The-Analyzer)

# In[ ]:


filth = [feature for feature, mask in         zip(vectorizer.get_feature_names(), ch2.get_support())         if mask == True]

print(filth)


# # The Analyzer
# We'll now build a new count vectorizer. We'll call it analyzer (analogous to 2 polarizing glasses) and it will vectorize again our input by only counting the predictive obscenities from above. This will give us a new matrix of n features, where n is the number of predictive words.
# 

# In[ ]:


analyzer = CountVectorizer(lowercase = True,                             vocabulary = filth)


# Now, let's define a function that vectorizes comment texts and weighs the vectors using the already trained TFIDF transformer:

# In[ ]:


def get_features(frame):
    result = pd.DataFrame(                transformer.fit_transform(                analyzer.fit_transform(                frame.comment_text)                                         ).todense(),                                            index = frame.index)
    return result


# We'll also define a dictionary which will contain our input train and test data:

# In[ ]:


feature_frames = {}

for frame in ('train', 'test'):
    feature_frames[frame] = get_features(eval('X_%s' % frame))

feature_frames['train'].info()


# # Training
# We can now train our algorithm of choice using the feature frames:

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier

knc = KNeighborsClassifier(n_neighbors = 10)
knc.fit(feature_frames['train'], y_train.toxic)


# # Log Loss and Conclusion
# Finally, we assess our log loss:

# In[ ]:


from sklearn.metrics import log_loss

result = pd.DataFrame(knc.predict_proba(feature_frames['test']), index = feature_frames['test'].index)

result['actual'] = y_test.toxic
result['text'] = X_test.comment_text

print(log_loss(y_test.toxic, result[1]))


# And here are some examples of predictions and their corresponding comments (again, viewer discretion is advised):

# In[ ]:


pd.set_option('max_colwidth', 100)
result[[1, 'actual', 'text']][(result.actual == 1) & (result[1] > 0.5)][:10]


# Afterword:
# 
# * In a live system such a model should use additional matching criteria for pursposefully misspelled obscenities (e.g. 'id10t' instead of 'idiot')
# * The model could be improved by using ngrams 
# * The model could be improved by using an ensemble of models
