# %% [markdown]
#
# 
# NLP - or *Natural Language Processing* - is shorthand for a wide array of techniques designed to help machines learn from text. Natural Language Processing powers everything from chatbots to search engines, and is used in diverse tasks like sentiment analysis and machine translation.
# 
# In this tutorial we'll look at this competition's dataset, use a simple technique to process it, build a machine learning model, and submit predictions for a score!

# %% [code]
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import feature_extraction, linear_model, model_selection, preprocessing
import seaborn as sborn
import matplotlib.pyplot as plt

import nltk
from nltk.corpus import stopwords
import re
import string


# %% [code]
train_df = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")
test_df = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")

# %% [markdown]
# ### A quick look at our data
# 
# Let's look at our data... first, an example of what is NOT a disaster tweet.

# %% [code]
train_df[train_df["target"] == 0]["text"]
sborn.countplot(y=train_df.target);

# %% [markdown]
# And one that is:

# %% [code]
#train_df[train_df["target"] == 1]["text"].values[1]
plt.figure(figsize=(9,6))
sborn.countplot(y=train_df.keyword, order=train_df.keyword.value_counts().iloc[:15].index)
plt.title('Top 15 Keywords')
plt.show()

# %% [markdown]
# ### Building vectors
# 
# The theory behind the model we'll build in this notebook is pretty simple: the words contained in each tweet are a good indicator of whether they're about a real disaster or not (this is not entirely correct, but it's a great place to start).
# 
# We'll use scikit-learn's `CountVectorizer` to count the words in each tweet and turn them into data our machine learning model can process.
# 
# Note: a `vector` is, in this context, a set of numbers that a machine learning model can work with. We'll look at one in just a second.

# %% [code]
stopwords = stopwords.words('english')

count_vectorizer = feature_extraction.text.CountVectorizer(stop_words = stopwords)

## let's get counts for the first 5 tweets in the data
example_train_vectors = count_vectorizer.fit_transform(train_df["text"][0:5])

# %% [code]
## we use .todense() here because these vectors are "sparse" (only non-zero elements are kept to save space)
print(example_train_vectors[0].todense().shape)
print(example_train_vectors[0].todense())

# %% [markdown]
# The above tells us that:
# 1. There are 54 unique words (or "tokens") in the first five tweets.
# 2. The first tweet contains only some of those unique tokens - all of the non-zero counts above are the tokens that DO exist in the first tweet.
# 
# Now let's create vectors for all of our tweets.

# %% [code]
train_vectors = count_vectorizer.fit_transform(train_df["text"])

## note that we're NOT using .fit_transform() here. Using just .transform() makes sure
# that the tokens in the train vectors are the only ones mapped to the test vectors - 
# i.e. that the train and test vectors use the same set of tokens.
test_vectors = count_vectorizer.transform(test_df["text"])
train_vectors.shape

# %% [code]
def cleanup_preprocessor(text):
    '''
    Make text lowercase, remove text in square brackets,remove links,remove special characters
    and remove words containing numbers.
    '''
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W"," ",text) # remove special chars
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    
    return text

# %% [markdown]
# ### Our model
# 
# As we mentioned above, we think the words contained in each tweet are a good indicator of whether they're about a real disaster or not. The presence of particular word (or set of words) in a tweet might link directly to whether or not that tweet is real.
# 
# What we're assuming here is a _linear_ connection. So let's build a linear model and see!

# %% [code]

count_vectorizer =  feature_extraction.text.CountVectorizer(token_pattern=r'\w{1,}',
                   ngram_range=(1, 2), stop_words = stopwords,preprocessor=cleanup_preprocessor)

count_vectorizer.fit(train_df['text'])

train_vectors = count_vectorizer.transform(train_df['text'])
test_vectors = count_vectorizer.transform(test_df['text'])
'''
tfidf_vectorizer =  feature_extraction.text.TfidfVectorizer( min_df=3,  max_features=None,analyzer='word',token_pattern=r'\w{1,}',
            ngram_range=(1, 3), use_idf=1,smooth_idf=1,sublinear_tf=1,
            stop_words = stopwords)

train_tfidf = tfidf.fit_transform(train_df['text'])
test_tfidf = tfidf.transform(test_df["text"])
'''
## Our vectors are really big, so we want to push our model's weights
## toward 0 without completely discounting different words - ridge regression 
## is a good way to do this.
clf = linear_model.RidgeClassifier()
#clf = linear_model.LogisticRegression(C=1.0)

# %% [markdown]
# Let's test our model and see how well it does on the training data. For this we'll use `cross-validation` - where we train on a portion of the known data, then validate it with the rest. If we do this several times (with different portions) we can get a good idea for how a particular model or method performs.
# 
# The metric for this competition is F1, so let's use that here.

# %% [code]
scores = model_selection.cross_val_score(clf, train_vectors, train_df["target"], cv=5, scoring="f1")
#scores = model_selection.cross_val_score(clf, ttrain_tfidf, train_df["target"], cv=5)
scores

# %% [markdown]
# The above scores aren't terrible! It looks like our assumption will score roughly 0.65 on the leaderboard. There are lots of ways to potentially improve on this (TFIDF, LSA, LSTM / RNNs, the list is long!) - give any of them a shot!
# 
# In the meantime, let's do predictions on our training set and build a submission for the competition.

# %% [code]
clf.fit(train_vectors, train_df["target"])

# %% [code]
sample_submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")

# %% [code]
sample_submission["target"] = clf.predict(test_vectors)

# %% [code]
sample_submission.head(10)

# %% [code]
sample_submission.to_csv("submission.csv", index=False)

# %% [markdown]
# Now, in the viewer, you can submit the above file to the competition! Good luck!