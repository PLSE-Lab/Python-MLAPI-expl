#!/usr/bin/env python
# coding: utf-8

# # HapPy Halloween kernel

# ![](https://media.giphy.com/media/pLWfbn1WVKzMQ/giphy.gif)

# This kernel is made by a beginner for beginners.
# This is my first kernel, so I'm looking for advice !

# ## Imports

# In[ ]:


import os
import pandas as pd
import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt
import seaborn as sns
from string import punctuation
from nltk import pos_tag
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk import FreqDist
from wordcloud import WordCloud, STOPWORDS

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix


# In[ ]:


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")


# ## Problem Understanding

# It is always important to know what we are going for.
# The description gives some details:
# > The competition dataset contains text from works of fiction written by spooky authors of the public domain: Edgar Allan Poe, HP Lovecraft and Mary Shelley
# 
# The objective is to accurately **identify the author of the sentences in the test set**.

# ## Data Understanding

# In[ ]:


train.head()
print("--- Shape ---")
print(train.shape)
print("--- Missing values ---")
train.isnull().sum() * 100 / len(train)


# * The dataset goes straight to the point with only 3 columns : ids, texts, and the targets which are authors
# * There is no missing value

# In[ ]:


sns.countplot(train.author)


# In[ ]:


# Takes a column and concatenate strings
def build_corpus(data):
    data = str(data)
    corpus = ""
    for sent in data:
        corpus += str(sent)
    return corpus


# In[ ]:


# Gather text of authors in different dataframes
eap = train[train.author == "EAP"]
hpl = train[train.author == "HPL"]
mws = train[train.author == "MWS"]


# In[ ]:


plt.figure(figsize=(15,10))
plt.subplot(331)
eap_wc = WordCloud(background_color="white", max_words=100, stopwords=STOPWORDS)
eap_wc.generate(build_corpus(eap.text))
plt.title("Edgar Allan Poe", fontsize=20)
plt.imshow(eap_wc, interpolation='bilinear')
plt.axis("off")

plt.subplot(332)
hpl_wc = WordCloud(background_color="white", max_words=100, stopwords=STOPWORDS)
hpl_wc.generate(build_corpus(hpl.text))
plt.title("HP Lovecraft", fontsize=20)
plt.imshow(hpl_wc, interpolation='bilinear')
plt.axis("off")

plt.subplot(333)
mws_wc = WordCloud(background_color="white", max_words=100, stopwords=STOPWORDS)
mws_wc.generate(build_corpus(mws.text))
plt.title("Marry Shelley", fontsize=20)
plt.imshow(mws_wc, interpolation='bilinear')
plt.axis("off")


# The authors seems to use a different lexical field, that gives us idea about using a Tfidf over texts.

# ## Data Preparation

# Scikit-Learn doesn't always handle strings, so let's encode authors.

# In[ ]:


le = LabelEncoder()
author_encoded = le.fit_transform(train.author)


# ## Modelling

# Now, this is what you all have been waiting for (or not ! :) )
# Let's split a train and a test dataset to evaluate our algorithm.
# I decided to use acurracy as the metric as it is easier to interpret.
# Finally, I define the cross validation method that will be used.

# In[ ]:


seed = 12
X_train, X_test, y_train, y_test = train_test_split(train.text, author_encoded, 
    test_size=0.3, random_state=seed)
metric = 'accuracy'
kfold = KFold(n_splits=10, random_state=seed)


# ### Useful classes from [PyData Seattle 2017](https://channel9.msdn.com/Events/PyData/Seattle2017/BRK03)

# In[ ]:


# Return called columns of a DataFrame
class ColumnExtractor(TransformerMixin):
    def __init__(self, cols):
        self.cols = cols
    def transform(self, X):
        Xcols = X[self.cols]
        return Xcols
    def fit(self, X, y=None):
        return self

# Enables to train an estimator within the pipeline
class ModelTransformer(TransformerMixin):
    def __init__(self, model):
        self.model = model
    def fit(self, *args, **kwargs):
        self.model.fit(*args, **kwargs)
        return self
    def transform(self, X, **transform_params):
        return pd.DataFrame(self.model.predict(X))


# ### Feature engineering

# In[ ]:


# Calculate the length of each text
class LengthTransformer(TransformerMixin):
    def transform(self, X, **transform_params):
        return pd.DataFrame(X.apply(lambda x: len(str(x)))) 
    def fit(self, X, y=None, **fit_params):
        return self
    
# Count the number of words in each text
class WordCountTransformer(TransformerMixin):
    def transform(self, X, **transform_params):
        return pd.DataFrame(X.apply(lambda x: len(str(x).split()))) 
    def fit(self, X, y=None, **fit_params):
        return self
    
# Count the number of unique words in each text
class UniqueWordCountTransformer(TransformerMixin):
    def transform(self, X, **transform_params):
        return pd.DataFrame(X.apply(lambda x: len(set(str(x).split())))) 
    def fit(self, X, y=None, **fit_params):
        return self
    
# Calculate the average length of words in each text
class MeanLengthTransformer(TransformerMixin):
    def transform(self, X, **transform_params):
        return pd.DataFrame(X.apply(lambda x: np.mean([len(w) for w in str(x).split()]))) 
    def fit(self, X, y=None, **fit_params):
        return self

# Count the number of punctuation in each sentence
class PunctuationCountTransformer(TransformerMixin):
    def transform(self, X, **transform_params):
        return pd.DataFrame(X.apply(lambda x: len([p for p in str(x) if p in punctuation]))) 
    def fit(self, X, y=None, **fit_params):
        return self

# Count the number of unique words in each text
class StopWordsCountTransformer(TransformerMixin):
    def transform(self, X, **transform_params):
        return pd.DataFrame(X.apply(lambda x: len([sw for sw in str(x).lower().split() if sw in set(stopwords.words("english"))]))) 
    def fit(self, X, y=None, **fit_params):
        return self


# The idea is to feed the classifier with the count of commas, the length of sentences, counts of words but also Tf-Idf.

# In[ ]:


pipeline = Pipeline([
    ('features', FeatureUnion([
        ('text_length', LengthTransformer()),
        ('word_count', WordCountTransformer()),
        ('mean_length', MeanLengthTransformer()),
        ('punctuation_count', PunctuationCountTransformer()),
        ('stop_words_count', StopWordsCountTransformer()),
        ('count_vect', CountVectorizer(lowercase=False)),
        ('tf_idf', TfidfVectorizer())
    ])),
  ('classifier', XGBClassifier(objective='multi:softprob', random_state = 12, eval_metric='mlogloss'))
])


# I really like pipelines, it enables to get rid off useless code lines between steps, and it is really clear to interpret.

# ## Evaluation

# In[ ]:


clf_pipe = pipeline.fit(X_train, y_train)
score_pipe = cross_val_score(clf_pipe, X_train, y_train, cv=kfold, scoring=metric)
print("Mean score = %.3f, Std deviation = %.3f"%(np.mean(score_pipe),np.std(score_pipe)))
score_pipe_test = clf_pipe.score(X_test,y_test)
print("Mean score = %.3f, Std deviation = %.3f"%(np.mean(score_pipe_test),np.std(score_pipe_test)))


# In[ ]:


conf_mat = confusion_matrix(y_test, clf_pipe.predict(X_test))
sns.heatmap(conf_mat, annot=True)
plt.xticks(range(3), ('EAP', 'HPL', 'MWS'), horizontalalignment='left')
plt.yticks(range(3), ('EAP', 'HPL', 'MWS'), rotation=0)


# We can see that it is quite well classified, but we can see that EAP is chosen too often.
# ![](https://media.giphy.com/media/azZYZhswISHw4/giphy.gif)

# ## Submission

# In[ ]:


target_names = ['EAP', 'HPL', 'MWS']
y_pred = pd.DataFrame(clf_pipe.predict(test.text), columns=target_names)
submission = pd.concat([test["id"],y_pred], 1)
submission.to_csv("./submission.csv", index=False)


# ## Comments

# * I scored 0.47318 as multiclass loss. Despise the fact it is well classified, probabilities need to be refined.
# * We could try to study in what extent the use of nouns or adverbs could differenciate the authors
# * Open to ideas
# * Open to advice
# * Hope you enjoyed :)

# ![](https://media.giphy.com/media/3o7btQsLqXMJAPu6Na/giphy.gif)
