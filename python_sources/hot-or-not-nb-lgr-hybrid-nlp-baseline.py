#!/usr/bin/env python
# coding: utf-8

# # Hot or Not? NB+LGR Hybrid NLP Baseline
# 
# When embarking on a new modeling project it is always wise to start out with a nice, reliable baseline. A good baseline model should be quick to fit, have just a few hyperparameters, and be fairly stable. The idea  is to set a lower-bound on your performance metrics before you start in on the heavy-compute time models. That way if the accuracy of your RNN is dramatically worse than your baseline, you'll know that you need to investigate what's going on before you invest a lot of time on parameter tuning.
# 
# For NLP problems [Jeremy Howard](https://www.kaggle.com/jhoward/nb-svm-strong-linear-baseline), former President and Chief Scientist at Kaggle, suggests using a technique proposed by [Wang & Manning (2012)](https://nlp.stanford.edu/pubs/sidaw12_simple_sentiment.pdf) combining Naive-Bayes with a linear classifier. The method is pretty neat and while it typically doesn't stand up to the latest-and-greatest deep learners, it has at times shown a significant improvement over Naive-Bayes or Logistic Regression alone.
# 
# Forming generative and discriminative hybrid models is an interesting approach. Generative classifiers assume some functional form for $P(X|Y)$ and $P(Y)$ then use Bayes rule to calculate $P(Y|X)$ and assign labels. Discriminative classifiers assume a functional form for $P(Y|X)$ and then estimate parameters directly from the data. Discriminant models are generally expected to yield better results, but are prone to overfitting, Generative models can outperform discriminant ones if the assumed generative function is correct. Feeding the results of a generative model into a discriminative one can smooth errors, providing a form of regularization by scaling discriminant inputs by a prior distribution. 
# 
# In this kernel we'll take a closer look at how this technique works on a toy example and then apply it to a real Kaggle dataset to see it in action.

# In[ ]:


import numpy as np
import pandas as pd
import re
import random

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import KFold
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_curve
from scipy.sparse import csr_matrix

from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib
import matplotlib.colors as mc


# ## How it works with a toy example
# 
# Lets take a look at how using Naive-Bayes to pre-transform the data works using a toy example before applying this technique to a real data set. Here are eight food reviews and associated labels. Looks like our reviewers were focused on eating pizza (yum!) and olives (ew!).

# In[ ]:


# fictitious food reviews
docs = ['The pizza is great!',
       'This is amazingly great pizza.',
       'This is amazingly good pizza.',
       'This pizza is terribly good.',
       'These are terrible olives.',
       'These olives are terrible.',
       'These are amazingly gross olives.',
       'Olives are amazingly terrible!']
y = [1,1,1,1,0,0,0,0]


# After applying some basic text preprocessing (stemming, stop words, lowercase, drop punctuation, drop numeric), the corpus is converted into a bag of words document term matrix. What we can see is that  'pizza', 'good' and 'great' are only found in positive reviews, while 'olives' (which are revolting) and 'gross' are only found in negative reviews. 'Terrible' is found in both good and bad reviews, but most often in bad. 'Amazingly' is evenly distributed amongst good and bad reviews. Intuitively 'olive' and 'pizza' are the strongest signals for classification, while 'amazingly' doesn't really impart much useful information at all.

# In[ ]:


# basic text processing
# omit stop words and apply stemming
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english') + ['this', 'that', 'these'])
def process_text(text):
    # Lower case
    text = [ t.lower() for t in text.split(' ') ]
    # Drop punctuation and numerics
    text = [ re.sub(r'[^A-Za-z]', '', t) for t in text ]
    # Drop stopwords
    text = [ t for t in text if t not in stop_words ]
    # Apply porter stemming
    text = ' '.join([ stemmer.stem(t) for t in text ])
    return text
    
docs_processed = [ process_text(t) for t in docs ]

ctv = CountVectorizer()
doc_term_matrix = ctv.fit_transform(docs_processed)

x = pd.DataFrame(doc_term_matrix.todense(),
                 columns = ctv.get_feature_names())


# In[ ]:


# Some adjustements for the sake of visualization
df = x.copy()
df = df.apply(lambda x: x/3)
df['target'] = [1 if x == 0 else 0.66 for x in y]
df.index = docs

cols = ["#ffffff", "#000000", "#e74c3c", "#2ecc71"]
cm = sns.color_palette(cols)

fig, axarr = plt.subplots(1, 1, figsize=(10, 4))
sns.heatmap(df[['target'] + list(df.columns[:-1])],
            cmap=cm,
            linecolor='white',
            linewidths=2,
            cbar=False)
_ = plt.title('Bag of Words')


# This NBTransformer class takes a document term matrix (X) and labels (y), and scales X by the log of the ratio of the probability of the word given class 1 versus the probability of the word given class 0. For each word (feature), $p$ is count of the word in class 1, $\lVert p \lVert_1$ is the total of all words in class 1, and $q$ is the count of the word in class 0. The transformer class includes a smoothing parameter $\alpha$, for details see the [MultinomialNB User Guide](https://scikit-learn.org/stable/modules/naive_bayes.html#multinomial-naive-bayes).
# $$ r = log \left( \frac{\frac{p}{\lVert p \lVert_1 }}{\frac{q}{\lVert q \lVert_1}}\right) = log \left(\frac{P(X|C=1)}{P(X|C=0)}\right) $$

# In[ ]:


class NBTransformer(BaseEstimator, TransformerMixin):
    """
    Implementation of Wang, S. and Manning C.D. Baselines and Bigrams: 
    Simple, Good Sentiment and Topic Classification (2012)
    https://nlp.stanford.edu/pubs/sidaw12_simple_sentiment.pdf
    
    Adapted from: 
        Jeremy Howard Machine Learning for Coders Lesson 11
        Kaggle @jhoward https://www.kaggle.com/jhoward/nb-svm-strong-linear-baseline
        Kaggle @Ren https://www.kaggle.com/ryanzhang/tfidf-naivebayes-logreg-baseline
    """    
    def __init__(self, alpha=1):
        self.r = None
        self.alpha = alpha

    def fit(self, X, y):
        # Convert X matrix to 1's and 0's
        X = csr_matrix(X)
        y = csr_matrix(y).T
        y_not = csr_matrix(np.ones((y).shape[0])).T - y
        # compute smoothed log prob ratio
        # use sparse pointwise multiplication, more memory efficient
        # than slicing
        p = self.alpha + X.multiply(y).sum(0)
        q = self.alpha + X.multiply(y_not).sum(0)
        # computed using L1 norm per Wang and Manning
        # (difference is accounted for by logistic regression bias term in otherwise)
        # adjusted smoothing to be equivalent to sklearn MultinomialNB using vocab size
        # (minor deviation from Wang and Manning)
        self.r = csr_matrix(np.log(
            (p / (self.alpha * X.shape[1] + X.multiply(y).sum())) /
            (q / (self.alpha * X.shape[1] + X.multiply(y_not).sum()))
        ))
        return self

    def transform(self, X, y=None):
        return X.multiply(self.r)
    
    def fit_transform(self, X, y):
        self.fit(X, y)
        return X.multiply(self.r)


# We end up with the scaled document term matrix below. Here we can see that 'amazingli' is has disappeared, the counts were reduced to nearly zero (not quite zero because of the slight word imbalance between classes). 'Pizza' and 'olive' are now the (absolute) largest values, as expected. And 'terrible' suggests a negative review, but is a weaker signal than the word 'olive'. 
# 
# This fits exactly with the intuition discussed above. We have effectively created a prior that we can feed directly into the linear classifier of our choosing, such as logistic regression or SVC. The final results will be in line with the Naive-Bayes classifier _unless_ the linear classifer has a strong reason to override Naive-Bayes because due to error. Voila! Now lets try this out on some real data!

# In[ ]:


doc_term_matrix_nb = NBTransformer().fit_transform(doc_term_matrix,y)
df = pd.DataFrame(doc_term_matrix_nb.todense(),
                 columns = ctv.get_feature_names())
df['target'] = [df.min().min() if x == 0 else df.max().max() for x in y]
df.index = docs

cols = sns.diverging_palette(145, 280, s=85, l=25, n=10000)
cm = sns.color_palette(cols)

fig, axarr = plt.subplots(1, 1, figsize=(10, 4))
sns.heatmap(df[['target'] + list(df.columns[:-1])],
            cmap=cm,
            linecolor='white',
            linewidths=2)
_ = plt.title('Naive Bayes Scaled Bag of Words')


# ## Applied to Real Data
# 
# Next we'll apply the hybrid model to some data scraped by [@milesh1](https://www.kaggle.com/milesh1) from the [Reddit Rateme](https://www.reddit.com/r/Rateme/). On this subreddit users can submit pictures other users rate their attractiveness out of 10 ([they're real good humans, Brent](https://twitter.com/dog_rates)). 
# For the purposes of this experiment we'll try to predict scores of 7 or up, because 7 sounds like a very respectable hotness score.

# In[ ]:


df = pd.read_csv('../input/comments.csv', header=None, names=['comment', 'rating'])

# At least a 7 sounds like a good place to be.
hotness_threshold = 7
df['hot'] = df.rating.apply(lambda x: 1 if x >= hotness_threshold else 0)

# find and remove comments that contain only a score (e.g. "8/10" or "8/10.")
df['score_only'] = df.comment.apply(lambda x: re.match(r"\d*[\.]?\d\/\d{1,2}[\.]?$",x) is not None)
df = df.loc[np.logical_not(df['score_only']),:].reset_index()
df = df.drop('score_only', axis=1)

print('{}% rated attractive (threshold score of {})'.format(int(100*df.hot.mean()), hotness_threshold))

_ = sns.distplot(df.rating,
             bins=range(0,11),
             kde=False,
             hist_kws=dict(edgecolor="w", linewidth=2))
plt.title('Distribution of Ratings')
_ = plt.axvline(7, 0,100000, color='r')


# After applying some very basic preprocessing, a document term matrix including bigrams was created using CountVectorizer. We then fit Naive-Bayes and Logistic Regression for comparison purposes with 4-fold cross-validation. Finally, we fit the hybrid Naive-Bayes using a pipeline.

# In[ ]:


# Again applying just some basic preprocessing
def preprocessing(doc):
    # if the entirety of the comment is a numeric rating (e.g. 3/10),
    # replace with identifier token for use as feature
    if re.match(r"\d*[\.]?\d\/\d{1,2}[\.]?$", doc) is not None:
        return 'NO_COMMENT'
    # space non-alpha characters
    doc = re.sub('([^a-zA-Z]+)', r' ', doc)
    # lowercase + drop punc / numbers / special characters
    tokens = [token.lower() for token in doc.split(' ') if token.isalpha()]
    # some comments are only a rating score, lets identify those
    return ' '.join(tokens)

# Create bag of words including bigrams
cv = CountVectorizer(preprocessor=preprocessing,
                     stop_words='english',
                     analyzer='word',
                     ngram_range=(1,2),
                     max_df=0.9999,
                     min_df=0.0001)

x = cv.fit_transform(df.comment)
y = df.hot


# In[ ]:


get_ipython().run_cell_magic('time', '', 'nb = MultinomialNB()\nkf = KFold(n_splits = 4)\n\nscores_nb = []\nfor it, iv in kf.split(x):\n    nb.fit(x[it,:], y[it])\n    probs = nb.predict_proba(x[iv,:])\n    scores_nb.append(average_precision_score(y[iv], probs[:,1]))\n    \nfpr_nb, tpr_nb, thresholds = roc_curve(y[iv], probs[:,1])\npre_nb, rec_nb, thresholds = precision_recall_curve(y[iv], probs[:,1])')


# In[ ]:


get_ipython().run_cell_magic('time', '', "lgr = LogisticRegression(solver='liblinear', \n                         class_weight='balanced',\n                         C=0.5)\n\nscores_lgr = []\nfor it, iv in kf.split(x):\n    lgr.fit(x[it,:], y[it])\n    probs = lgr.predict_proba(x[iv,:])\n    scores_lgr.append(average_precision_score(y[iv], probs[:,1]))\n    \nfpr_lgr, tpr_lgr, thresholds = roc_curve(y[iv], probs[:,1])\npre_lgr, rec_lgr, thresholds = precision_recall_curve(y[iv], probs[:,1])")


# In[ ]:


get_ipython().run_cell_magic('time', '', "# Hybrid model\npipeline = Pipeline([('NB', NBTransformer()),\n                     ('Logistic Regression', LogisticRegression(solver='liblinear', \n                                                                class_weight='balanced',\n                                                                C=0.5))])\n\nscores_nblgr = []\nfor it, iv in kf.split(x):\n    pipeline.fit(x[it,:], y[it])\n    probs = pipeline.predict_proba(x[iv,:])\n    scores_nblgr.append(average_precision_score(y[iv], probs[:,1]))\n    \nfpr_nblgr, tpr_nblgr, thresholds = roc_curve(y[iv], probs[:,1])\npre_nblgr, rec_nblgr, thresholds = precision_recall_curve(y[iv], probs[:,1])")


# ## Conclusion
# Alas, no big winner here, the hybrid model only _very slightly_ outperformed Logistic Regression as measured by average precision. We can also see that the ROC and Precision-Recall curves are almost exactly overlapping.

# In[ ]:


print("Naive-Bayes: \t\t\t\t{}".format(round(np.mean(scores_nb),4)))
print("Logistic Regression: \t\t\t{}".format(round(np.mean(scores_lgr),4)))
print("Naive-Bayes + Logistic Regression: \t{}".format(round(np.mean(scores_nblgr),4)))


# In[ ]:


fig, axarr = plt.subplots(1, 2, figsize=(16, 4))
plt.subplot(121)
plt.plot(fpr_nb, tpr_nb, label='NB')
plt.plot(fpr_lgr, tpr_lgr, label='LGR')
plt.plot(fpr_nblgr, tpr_nblgr, label='NB+LGR')
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.title('ROC Curve')

plt.subplot(122)
plt.plot(rec_nb, pre_nb, label='NB')
plt.plot(rec_lgr, pre_lgr, label='LGR')
plt.plot(rec_nblgr, pre_nblgr, label='NB+LGR')
plt.xlabel('recall')
plt.ylabel('precision')
plt.title('Precision-Recall Curve')
_ = plt.legend(loc='best')


# That said, the hybrid model took negligibly longer to fit and technically _did_ perform better. It certainly meets the essential criteria for a baseline model, it is a fast, easy and reliable addition to the tool kit. Definitely worth trying out next time you start a new NLP competition.
# 
# Do you have an example dataset with more dramatic results? I'd love to know about it!

# In[ ]:




