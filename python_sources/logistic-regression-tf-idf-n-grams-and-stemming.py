#!/usr/bin/env python
# coding: utf-8

# # Logistic Regression with TF-IDF, n-grams and stemming

# In this kernel I will compare different sets of TF-IDF features to find best one to classify each toxicity type. I will use unigrams, unigrams together with bigrams, and apply stemming and lemmatization techniques for each of the these TF-IDF matrices.
# 
# In our case best ROC-AUC scores are achieved with logistic regression (it was compared to linear SVM and naive bayes classifiers). So we will use it for cross-validation and training in this kernel.

# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, TransformerMixin 
from nltk.stem.snowball import SnowballStemmer
from nltk import WordNetLemmatizer, pos_tag, word_tokenize
from nltk.corpus import wordnet


# In[3]:


train_data = pd.read_csv('../input/train.csv')
train_data.head()


# In[4]:


train_data.info()


# In[5]:


label_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']


# In[6]:


def combined_cv_scores(X, ys, params):
    """ CV scores for different set of labels (ys) """
    scores = {}
    for col in ys.columns:
        clf = LogisticRegression(C=params[col])
        s = cross_val_score(clf, X, ys[col], scoring='roc_auc')
        scores[col] = np.mean(s)
        print('{}: {}, mean {}'.format(col, s, np.mean(s)))
    return scores


# ## Unigrams

# With TfidfVectorizer we can extract "bag of words" from the text and apply TF-IDF (term frequency - inverse document frequency) weights. Experimentaly I've figured out that applying sublinear tf scaling (1 + log(tf)) improves overall results. First let's extract unigrams from the text data.

# In[7]:


vectorizer_unigram = TfidfVectorizer(sublinear_tf=True)
X_unigram = vectorizer_unigram.fit_transform(train_data['comment_text'])
X_unigram.shape


# Using grid search we can tune parameters of logistic regression (l2 penalty works best). Tuning is out of the scope of this kernel as it takes quite a lot of time. I'm just setting optimal C parameter for each of six labels and fixing corresponding cross validation scores:

# In[8]:


# 'C' parameter of logistic regression, obtained by GridSearchCV 
params_unigram = {
    'toxic': 4,
    'severe_toxic': 2,
    'obscene': 3,
    'threat': 4,
    'insult': 3,
    'identity_hate': 3,
}


# In[9]:


get_ipython().run_cell_magic('time', '', 'scores_unigram = combined_cv_scores(X_unigram, train_data[label_names], params_unigram)')


# ## Bigrams

# Now let's extract unigrams as well as bigrams from the text (and of course apply tf-idf vectorizing).

# In[10]:


vectorizer_bigram = TfidfVectorizer(ngram_range=(1,2), sublinear_tf=True)
X_bigram = vectorizer_bigram.fit_transform(train_data['comment_text'])
X_bigram.shape


# Again let's set C parameters obtained from the grid search and get corresponding CV scores:

# In[11]:


# 'C' parameter of logistic regression, obtained by GridSearchCV 
params_bigram = {
    'toxic': 50,
    'severe_toxic': 4,
    'obscene': 30,
    'threat': 40,
    'insult': 12,
    'identity_hate': 12,
}


# In[12]:


get_ipython().run_cell_magic('time', '', 'scores_bigram = combined_cv_scores(X_bigram, train_data[label_names], params_bigram)')


# ## Stemming, unigrams

# Now we'll use stemming technique - cutting the words to their root form. Let's use Snowball english stemmer algorithm from the nltk package. Ignooring english stopwords do not improve the scoring, so we'll not enable this option. First let's apply stemming to unigrams.

# In[13]:


stemmer = SnowballStemmer('english', ignore_stopwords=False)

class StemmedTfidfVectorizer(TfidfVectorizer):
    
    def __init__(self, stemmer, *args, **kwargs):
        super(StemmedTfidfVectorizer, self).__init__(*args, **kwargs)
        self.stemmer = stemmer
        
    def build_analyzer(self):
        analyzer = super(StemmedTfidfVectorizer, self).build_analyzer()
        return lambda doc: (self.stemmer.stem(word) for word in analyzer(doc.replace('\n', ' ')))


# In[14]:


vectorizer_stem_u = StemmedTfidfVectorizer(stemmer=stemmer, sublinear_tf=True)
X_train_stem_u = vectorizer_stem_u.fit_transform(train_data['comment_text'])
X_train_stem_u.shape


# In[15]:


# 'C' parameter of logistic regression, obtained by GridSearchCV 
params_stem_u = {
    'toxic': 3,
    'severe_toxic': 1,
    'obscene': 3,
    'threat': 4,
    'insult': 2,
    'identity_hate': 2,
}


# In[16]:


get_ipython().run_cell_magic('time', '', 'scores_stem_u = combined_cv_scores(X_train_stem_u, train_data[label_names], params_stem_u)')


# ## Stemming, bigrams

# Next apply stemming to unigrams+bigrams features.

# In[17]:


vectorizer_stem_b = StemmedTfidfVectorizer(stemmer=stemmer, ngram_range=(1,2), sublinear_tf=True)
X_train_stem_b = vectorizer_stem_b.fit_transform(train_data['comment_text'])
X_train_stem_b.shape


# In[18]:


# 'C' parameter of logistic regression, obtained by GridSearchCV 
params_stem_b = {
    'toxic': 20,
    'severe_toxic': 3,
    'obscene': 20,
    'threat': 40,
    'insult': 8,
    'identity_hate': 12,
}


# In[19]:


get_ipython().run_cell_magic('time', '', 'scores_stem_b = combined_cv_scores(X_train_stem_b, train_data[label_names], params_stem_b)')


# ## Lemmatization

# Now let's apply lemmatization - getting grammatically correct normal form of the word with the use of morphology. We will use WordNetLemmatizer from nltk package and part-of-speech word tagging.

# In[20]:


def lemmatize(text):
    """ Tokenize text and lemmatize word tokens """
    def get_pos(tag):
        if tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('R'):
            return wordnet.ADV
        return wordnet.NOUN
    
    wnl = WordNetLemmatizer()
    return [wnl.lemmatize(token, get_pos(tag)) for token, tag in pos_tag(word_tokenize(text))]


# In[21]:


vectorizer_lemma_u = TfidfVectorizer(tokenizer=lemmatize, sublinear_tf=True)
X_train_lemma_u = vectorizer_lemma_u.fit_transform(train_data['comment_text'])
X_train_lemma_u.shape


# In[22]:


# 'C' parameter of logistic regression, obtained by GridSearchCV 
params_lemma_u = {
    'toxic': 4,
    'severe_toxic': 2,
    'obscene': 4,
    'threat': 4,
    'insult': 3,
    'identity_hate': 3,
}


# In[23]:


get_ipython().run_cell_magic('time', '', 'scores_stem_b = combined_cv_scores(X_train_lemma_u, train_data[label_names], params_lemma_u)')


# The score is not improving even compared with pure unigrams.

# ## Predict

# As we can see stemming gives best results. Severe_toxic and identity_hate type classification gives best score with unigram features, the rest of the types do best with bigrams. So now we can apply corresponding vectorizers to test data and make predictions.

# In[24]:


models = {
    'toxic': {'classifier': LogisticRegression(C=20), 'features': 'stem_b'},
    'severe_toxic': {'classifier': LogisticRegression(C=3), 'features': 'stem_b'},
    'obscene': {'classifier': LogisticRegression(C=3), 'features': 'stem_u'},
    'threat': {'classifier': LogisticRegression(C=40), 'features': 'stem_b'},
    'insult': {'classifier': LogisticRegression(C=8), 'features': 'stem_b'},
    'identity_hate': {'classifier': LogisticRegression(C=2), 'features': 'stem_u'},
}


# In[25]:


def fit_predict_results(models, train_features, test_features, train_data, test_data):
    result = pd.DataFrame(columns=(['id']+list(models.keys())))
    result.id = test_data['id']
    for label, model in models.items():
        clf = model['classifier']
        X = train_features[model['features']]
        clf.fit(X, train_data[label])
        X = test_features[model['features']]
        predicts = clf.predict_proba(X)
        result[label] = predicts[:,1]
    return result


# In[26]:


# read test data
test_data = pd.read_csv('../input/test.csv')
test_data.head()


# In[27]:


# Test unigrams with stemming
X_test_stem_u = vectorizer_stem_u.transform(test_data['comment_text'])
X_test_stem_u.shape


# In[28]:


# Test bigrams with stemming
X_test_stem_b = vectorizer_stem_b.transform(test_data['comment_text'])
X_test_stem_b.shape


# In[29]:


get_ipython().run_cell_magic('time', '', "train_features = {\n    'stem_u': X_train_stem_u,\n    'stem_b': X_train_stem_b\n}\ntest_features = {\n    'stem_u': X_test_stem_u,\n    'stem_b': X_test_stem_b\n}\n\nresult = fit_predict_results(models, train_features, test_features, train_data, test_data)")


# Just for check - comments classified as threat.

# In[30]:


result[result.threat>0.5][:5]


# In[31]:


test_data['comment_text'][1053]


# In[33]:


result.to_csv('submission.csv', index=False)

