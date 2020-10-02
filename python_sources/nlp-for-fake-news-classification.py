#!/usr/bin/env python
# coding: utf-8

# # NLP for Fake News Classification
# 
# A dataset from [Real or Fake](https://www.kaggle.com/rchitic17/real-or-fake) is provided containing different news articles.
# 
# We want to build a  model that can classify if a given article is considered fake or not. We will use a subset of the data for training and the remaining for testing our model.
# 
# ## Outline
# 
# We separate the project in 3 steps:
# 
# **Data Loading and Processing:** Load the data and analyze it to obtain an accurate picture of it, its features, its values (and whether they are incomplete or wrong), its data types among others. We also do the required processing.
# 
# **Feature Engineering / Modeling:** Once we have the data, we create some features and then the modeling stage begins, we set multiple baselines making use of different models, we will hopefully produce a model that fits our expectations of performance. Once we have that model, a process of tuning it to the training data would be performed.
# 
# **Results and Conclusions:** Finally, with our tuned model, we  predict against the test set, and finally, outline our conclusions.

# In[ ]:


import re
import eli5
import spacy
import nltk as nl
import pandas as pd
from sklearn.base import clone
import matplotlib.pyplot as plt
from scipy.sparse import hstack
from nltk.corpus import stopwords
from ml_helper.helper import Helper
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from nltk.stem.snowball import SnowballStemmer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from scikitplot.metrics import plot_confusion_matrix
from sklearn.preprocessing import FunctionTransformer
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import accuracy_score as metric_scorer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
nl.download('stopwords')
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Setting Key Values
# 
# The following values are used throught the code, this cell gives a central source where they can be managed.

# In[ ]:


KEYS = {
    "SEED": 1,
    "DATA_PATH": "../input/fake_or_real_news.csv",
    "TARGET": "label",
    "METRIC": "accuracy",
    "TIMESERIES": False,
    "SPLITS": 3,
    "ESTIMATORS": 150,
    "ITERATIONS": 500,
}

hp = Helper(KEYS)


# ## Data Loading and Processing
# 
# Here we load the necessary data, splitting 80% for training and the remaining 20% for testing. We also review its types and print its first rows.

# In[ ]:


df = pd.read_csv(KEYS["DATA_PATH"], header=0, names=["id", "title", "text", "label"])
train, test = train_test_split(df, test_size=0.20, random_state=KEYS["SEED"])


# ### Data Types

# In[ ]:


train.dtypes


# ### First Rows of the Data

# In[ ]:


train.head()


# ### Missing Data
# 
# We check if there is any missing data.

# In[ ]:


hp.missing_data(df)


# ### Sample News

# In[ ]:


train.iloc[10,2]


# ### Check target variable balance
# We review the distribution of values in the target variable.

# In[ ]:


hp.target_distribution(train)


# The distribution is perfect, therefore no resampling is needed.

# In[ ]:


train["concat"] = train["title"] + train["text"]
test["concat"] = test["title"] + test["text"]
train.head()


# ## Tokenizing data
# We convert the different articles to a matrix of token counts

# In[ ]:


count_vect = CountVectorizer()
base_train = count_vect.fit_transform(train["concat"])
print(base_train.shape)


# ## Normalized TF/TF-IDF Representation
# 
# Now we transform the matrix of token counts to a normalized tf or tf-idf representation, where tf represents term frequency and tf-idf represents the frequency times the inverse document frequency, that way the importance/scale of certain repeated tokens throughout the text is reduced.

# In[ ]:


tfidf_transformer = TfidfTransformer()
base_train = tfidf_transformer.fit_transform(base_train)
print(base_train.shape)


# ### 2 Component PCA on TF-IDF Representation
# Now we will reduce dimensionality on the TF-IDF Representation to 2 components to see how they differ.

# In[ ]:


pcaed = TruncatedSVD(n_components=2).fit_transform(base_train)
pcaed = pd.concat([pd.DataFrame(pcaed).reset_index(drop=True), train["label"].reset_index(drop=True)], axis=1)
pcaed.head()


# In[ ]:


pctrue=pcaed[pcaed["label"]=="REAL"]
pcfake=pcaed[pcaed["label"]=="FAKE"]
plt.figure(figsize = (12,8))
plt.xlabel('Principal Component 1', fontsize = 15)
plt.ylabel('Principal Component 2', fontsize = 15)
plt.title('2 Component PCA on TF-IDF Representation', fontsize = 20)
plt.scatter(pctrue[0], pctrue[1], color="blue")
plt.scatter(pcfake[0], pcfake[1], color="red")
plt.legend(["REAL", "FAKE"])


# We can see that they are very similar and that there is no clear distinction between the two.

# ## Baseline
# 
# In order to test the performance of our feature engineering steps, we will create several initial baseline models, that way we will see how our efforts increase the models predictive power.
# 
# ### Train Function
# Here we define the train function which will be used with the different models, it performs a cross validation score on 80% of the training data and a final validation on the remaining 20%.

# In[ ]:


basepipe = Pipeline([
    ('vect', TfidfVectorizer(stop_words="english", ngram_range=(1,2), sublinear_tf=True))
])
    

models = [
    {"name": "naive", "model": MultinomialNB()},
    {"name": "logistic_regression", "model": LogisticRegression(solver="lbfgs", max_iter=KEYS["ITERATIONS"], random_state=KEYS["SEED"])},
    {"name": "svm", "model": SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=KEYS["SEED"])},
    {"name": "pac", "model":  PassiveAggressiveClassifier(max_iter=1000, random_state=KEYS["SEED"], tol=1e-3)},
]

all_scores = hp.pipeline(train[["concat", "label"]], models, basepipe, note="Base models")


# We can see that the Passive Aggressive classifier was the best performing one during our baseline.

# ## Feature Engineering

# ### Stemming
# Here we stem the news text with two different stemmers Porter and Snowball. Overall, we perform the following transformations:
# 
# - Stemming of the words
# 
# - Removing punctuation, stop words and other characters
# 
# - Convert to Lower case and split string into words (tokenization)

# In[ ]:


def stemmer(df, stem = "snow"):
    if stem == "port":
        stemmer = PorterStemmer()
    else:
        stemmer = SnowballStemmer(language='english')
    df = df.apply(lambda x: " ".join([stemmer.stem(i) for i in re.sub("[^a-zA-Z]", " ", x).split()]).lower())
    return df


# ### Original

# In[ ]:


train.iloc[3:4]["concat"]


# ### Stemmed

# In[ ]:


stemmer(train["concat"].iloc[3:4])


# Stemming is closely related to lemmatization. The difference is that a stemmer operates on a single word without knowledge of the context, and therefore cannot discriminate between words which have different meanings depending on part of speech. However, stemmers are typically easier to implement and run faster, and the reduced accuracy may not matter for some applications.

# ## Lemmatizing and POS Tagging with Spacy

# Using Spacy we will do the following transformations:
# 
# - Lemmatization of the words
# 
# - POS Tagging of the words
# 
# - Removing punctuation, stop words and other characters
# 
# - Convert to Lower case and split string into words (tokenization)
# 
# - We remove words with a frequency of less than 0.01% or with more than 0.99%
# 
# Here is a sample of the lemmatization and tagging:

# In[ ]:


nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "textcat"])
doc = nlp(train['concat'][3])
print(doc.text[:50])
print('----------------------------------------------------')
for token in doc[:5]:
    print(f'Token: {token.text}, Lemma: {token.lemma_}, POS: {token.pos_}')


# ### Defining our own Tokenizer
# 
# We define our own tokenizer, which lemmatizes, lowercases and adds POS tagging. We also try removing stop words, punctuations and digits with a second tokenizer.

# In[ ]:


def tokenizer(text):
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "textcat"])
    return [token.lemma_.lower().strip() + token.pos_ for token in nlp(text)]


# In[ ]:


def strict_tokenizer(text):
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "textcat"])
    return [token.lemma_.lower().strip() + token.pos_ for token in nlp(text)
        if 
            not token.is_stop and not nlp.vocab[token.lemma_].is_stop
            and not token.is_punct
            and not token.is_digit
    ]


# ## Pipelines
# Now we define 2 different pipelines that will be tested, one with Stemming and another one with Lemmatization.

# ### Pipeline 1: Snowball Stemmer

# In[ ]:


snow_pipe = Pipeline([
    ('snow_stem', FunctionTransformer(stemmer, validate=False)),
    ('vect', TfidfVectorizer(stop_words="english", ngram_range=(1,2), sublinear_tf=True)),
])


# ### Pipeline 2: Porter Stemmer

# In[ ]:


port_pipe = Pipeline([
    ('port_stem', FunctionTransformer(stemmer, kw_args={"stem": "port"}, validate=False)),
    ('vect', TfidfVectorizer(stop_words="english", ngram_range=(1,2), sublinear_tf=True)),
])


# ### Pipeline 3: Lemmatizer with POS Tagging

# In[ ]:


lemm_pipe = Pipeline([
    ('lemma_vect', TfidfVectorizer(analyzer = 'word', max_df=0.99, min_df=0.01, ngram_range=(1,2), tokenizer=tokenizer))
])


# ### Pipeline 4: Strict Lemmatizer with POS Tagging

# In[ ]:


strict_lemm_pipe = Pipeline([
    ('strict_lemma_vect', TfidfVectorizer(analyzer = 'word', max_df=0.99, min_df=0.01, ngram_range=(1,2), tokenizer=strict_tokenizer))
])


# And these are the models we will test with the different pipelines, based on the results obtained from our baselines

# In[ ]:


models = [
    {"name": "logistic_regression", "model": LogisticRegression(solver="lbfgs", max_iter=KEYS["ITERATIONS"], random_state=KEYS["SEED"])},
    {"name": "pac", "model":  PassiveAggressiveClassifier(max_iter=KEYS["ITERATIONS"], random_state=KEYS["SEED"], tol=1e-3)},
]


# Testing all pipelines consecutively.

# In[ ]:


all_scores = hp.pipeline(train[["concat", "label"]], models, snow_pipe, all_scores=all_scores, quiet = True)
all_scores = hp.pipeline(train[["concat", "label"]], models, port_pipe, all_scores=all_scores, quiet = True)
all_scores = hp.pipeline(train[["concat", "label"]], models, lemm_pipe, all_scores=all_scores, quiet = True)
all_scores = hp.pipeline(train[["concat", "label"]], models, strict_lemm_pipe, all_scores=all_scores)


# ### Pipeline Performance by Model

# In[ ]:


hp.plot_models(all_scores)


# ### Top Pipelines per Model
# 
# Here we show the top pipelines per model.

# In[ ]:


hp.show_scores(all_scores, top=True)


# ## Hyperparameter Tuning
# 
# Now that we have the best performing model which is a Passive Agressive Classifier with lemmatization and POS tagging of the concatenated text and title, we will do a 5 fold cross validated randomized grid search over it to get the best parameters for the model.

# In[ ]:


grid = {
    "pac__C": [1.0, 10.0],
    "pac__tol": [1e-2, 1e-3],
    "pac__max_iter": [500, 1000],
}

final_scores, pipe = hp.cross_val(train[["concat", "label"]], model=clone(hp.top_pipeline(all_scores)), grid=grid)
final_scores


# In[ ]:


print(pipe.best_params_)
final_pipe = pipe.best_estimator_


# ## Explaining the model
# Here we see the associated weights to the different tokens and wether they are positive or negative with regards to classifying a news as REAL.
# 
# It seems like proper punctuation ('s in the end of words, use of commas, etc) really help the model to see if it is a real or fake news, also having many exclamation signs does the inverse.

# In[ ]:


eli5.show_weights(final_pipe, top=30, target_names=train.label)


# ### Explained prediction on one news article
# Usign the same library we will attempt to understand how a specific news item from the test set is predicted.

# In[ ]:


test.concat.iloc[2]


# In[ ]:


eli5.show_prediction(final_pipe.named_steps['pac'], test.concat.iloc[2], vec = final_pipe.named_steps['lemma_vect'], top=30, target_names=train.label)


# From the weights of the different positive and negative tokens the news has we can see how it would predict the article.

# ## Predictions and Results on Test Set
# 
# Now with our final pipeline we perform preditions on the entire test set.
# 
# To review the performance of the model, accuracy is not enough, therefore we plot a confusion matrix and print a classification report.
# 
# ### Accuracy

# In[ ]:


predictions = final_pipe.predict(test["concat"])
metric_scorer(test["label"], predictions)


# ### Classification Report

# In[ ]:


print(classification_report(test["label"], predictions))


# ### Confusion Matrix

# In[ ]:


plot_confusion_matrix(test["label"], predictions)


# # Conclusions
# 
# The classification report obtained from our final model on a 20% holdout of the data shows its accuracy, precision (how often the predictions are correct) and the recall (how many of the total observations in the set are correctly classified), also its f1-score (harmonic average of both). The weighted average for all of them stands at 96% which means that it can classify which articles are fake with great efficacy.
# 
# This information is extremely useful to multiple actors, including social networks and end consumers since it can help them differentiate between real and fake stories, which often lead to skewed views of current world events.
