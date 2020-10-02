#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install jcopml')


# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import json
import pprint

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # Import Data

# In[ ]:


df = pd.read_csv("../input/ndsc-beginner/train.csv")
df.tail(20)


# In[ ]:


df.info()


# # Add Information to df

# In[ ]:


with open("../input/ndsc-beginner/categories.json", 'rb') as handle:
    category_details = json.load(handle)


# In[ ]:


pprint.pprint(category_details)


# #### create dictionary of category type & product type

# In[ ]:


category_mapper = {}
product_type_mapper = {}

for cat in category_details.keys():
    for key, value in category_details[cat].items():
#         print(key)
#         print(value)
        category_mapper[value] = key
        product_type_mapper[value] = cat


# In[ ]:


category_mapper, product_type_mapper


# #### join the dictionary into dataframe

# In[ ]:


df['category_name'] = df['Category'].map(category_mapper)
df['product_type'] = df['Category'].map(product_type_mapper)


# In[ ]:


df.head()


# In[ ]:


df.groupby(['Category', 'category_name', 'product_type']).sum()


# In[ ]:


# df.category_name.value_counts()


# # 1) CATEGORY CLASSIFICATION BASED ON TEXT (TITLE FEATURE)

# In[ ]:


import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from jcopml.pipeline import num_pipe, cat_pipe
from jcopml.plot import plot_missing_value
from jcopml.feature_importance import mean_score_decrease


# In[ ]:


from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from string import punctuation

# sw is list of stopwords (ID + EN) and punctuation
sw = stopwords.words("indonesian") + stopwords.words("english") + list(punctuation)


# ## Import Data

# In[ ]:


df.to_csv("df_complete_with_cat_prod.csv")
df.head()


# In[ ]:


# check dataset of category
# print('length of category: ', len(df.Category.unique()))
# print(df.Category.value_counts(normalize=True))


# ## Encoding with FastText

# In[ ]:


import torch
import os

import pandas as pd
from tqdm.auto import tqdm

from nltk.tokenize import word_tokenize
from gensim.models import FastText

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device


# #### Prepare Corpus

# In[ ]:


sentences = [word_tokenize(text.lower()) for text in tqdm(df.title)]


# In[ ]:


sentences[:5]


# #### Train FastText Model

# In[ ]:


model = FastText(sentences, size=128, window=5, min_count=3, workers=4, iter=100, sg=0, hs=0)


# In[ ]:


from jcopml.utils import save_model
save_model(model, "product_detection_title.fasttext")


# In[ ]:


# save
os.makedirs("model/fasttext/", exist_ok=True)
model.save("model/fasttext/product_detection_title.fasttext")


# In[ ]:


# load
model = FastText.load("../input/ndsc-2019-product-detection-title-fasttext/product_detection_title.fasttext")


# #### Continue training

# In[ ]:


# text = [
#     ['missha', 'line', 'fighting', 'foundation', 'moisture', '2', 'refills'],
#     ['cushion', 'moisture','chafing', 'relief', 'gel', '2', 'refills']
# ]


# In[ ]:


# model.train(text, total_examples=len(text), epochs=2)
# model.save("model/fasttext/product_detection_title.fasttext")


# #### Model information

# In[ ]:


w2v = model.wv


# In[ ]:


# w2v.index2word
# w2v.vectors
w2v.vector_size


# #### Sanity check

# In[ ]:


# similar word
w2v.similar_by_word('foundation', topn=5)


# #### Higher order visualization

# In[ ]:


from umap import UMAP
import numpy as np
import pandas as pd
import plotly.express as px


# In[ ]:


X = UMAP().fit_transform(w2v.vectors)
df = pd.DataFrame(X, columns=["umap1", "umap2"])
df["title"] = w2v.index2word


# In[ ]:


fig = px.scatter(df, x="umap1", y="umap2", text="text")
fig.update_traces(textposition='top center')
fig.update_layout(
    height=800,
    title_text='Reduced FastText Visualization'
)
fig.show()


# #### ENCODING TEXT

# In[ ]:


w2v = FastText.load("../input/ndsc-2019-product-detection-title-fasttext/product_detection_title.fasttext").wv


# In[ ]:


def simple_encode_sentence(sentence, w2v, stopwords=None):
    if stopwords is None:
        vecs = [w2v[word] for word in word_tokenize(sentence)]
    else:
        vecs = [w2v[word] for word in word_tokenize(sentence) if word not in stopwords]
        
    sentence_vec = np.mean(vecs, axis=0) #setiap kalimat dicari rata2 dari vektor per kata
    return sentence_vec

def better_encode_sentence(sentence, w2v, stopwords=None):
    if stopwords is None:
        vecs = [w2v[word] for word in word_tokenize(sentence)]
    else:
        vecs = [w2v[word] for word in word_tokenize(sentence) if word not in stopwords]
    
    vecs = [vec / np.linalg.norm(vec) for vec in vecs if np.linalg.norm(vec) > 0]
    sentence_vec = np.mean(vecs, axis=0) #setiap kalimat dicari rata2 dari vektor per kata
    return sentence_vec


# In[ ]:


vecs = [better_encode_sentence(sentence, w2v, stopwords=sw) for sentence in tqdm(df.title)]
vecs = np.array(vecs)
vecs


# In[ ]:


df_vecs = pd.DataFrame(vecs)
df_vecs.to_csv('vectorized_sentences.csv')


# ## Dataset Splitting

# In[ ]:


X = vecs
y = df.Category

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_train.shape, X_test.shape, y_train.shape, y_test.shape


# In[ ]:


X_train, y_train


# ## Training

# In[ ]:


from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import RandomizedSearchCV
from jcopml.tuning import random_search_params as rsp

from sklearn.feature_extraction.text import CountVectorizer


# rsp.logreg_params
# ??model.fit()
# Pipeline.fit does not accept the average parameter. You can pass parameters to specific steps of your pipeline using the stepname__parameter format, e.g. `Pipeline.fit(X, y, logisticregression__sample_weight=sample_weight)`.

# In[ ]:


pipeline = Pipeline([
    ('algo', LogisticRegression(solver='lbfgs', n_jobs=-1, random_state=42, class_weight="balanced"))
])

model_logreg = RandomizedSearchCV(pipeline, rsp.logreg_params, cv=4, scoring='f1_micro', n_iter=50, n_jobs=-1, verbose=1, random_state=42)
model_logreg.fit(X_train, y_train)


print(model_logreg.best_params_)
print(model_logreg.score(X_train, y_train), model_logreg.best_score_, model_logreg.score(X_test, y_test))


# In[ ]:


pipeline = Pipeline([
#     ('prep', CountVectorizer(tokenizer=word_tokenize, stop_words=sw_indo)), # this step has been done in Encoding process with w2v, train set has been encoded
    ('algo', SGDClassifier(random_state=42, tol=None))
])

parameter = {
    'algo__loss': ['hinge', 'log'],
    'algo__penalty': ['l2', 'l1'],
    'algo__alpha': [0.0001, 0.0002, 0.0003], 
    'algo__max_iter': [5, 6, 7, 8, 9, 10],
    'algo__tol': [0.0001, 0.0002, 0.0003]
}

model_sgd = RandomizedSearchCV(pipeline, parameter, cv=2, scoring='f1_micro', n_iter=20, n_jobs=-1, verbose=1, random_state=42)
model_sgd.fit(X_train, y_train)

print(model_sgd.best_params_)
print(model_sgd.score(X_train, y_train), model_sgd.best_score_, model_sgd.score(X_test, y_test))

# Fitting 2 folds for each of 20 candidates, totalling 40 fits
# [Parallel(n_jobs=-1)]: Using backend LokyBackend with 4 concurrent workers.
# [Parallel(n_jobs=-1)]: Done  40 out of  40 | elapsed: 24.8min finished
# /opt/conda/lib/python3.7/site-packages/sklearn/linear_model/_stochastic_gradient.py:573: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
#   ConvergenceWarning)
# {'algo__tol': 0.0001, 'algo__penalty': 'l2', 'algo__max_iter': 10, 'algo__loss': 'hinge', 'algo__alpha': 0.0001}
# 0.6196323965107295 0.6242902574949558 0.6201405608934693


# pipeline = Pipeline([
# #     ('prep', CountVectorizer(tokenizer=word_tokenize, stop_words=sw_indo)), # this step has been done in Encoding process with w2v, train set has been encoded
#     ('algo', SGDClassifier(random_state=42, tol=None))
# ])
# ----------------------------------------------------------------------------------------------------------
# Training based on the best parameter.

# pipeline = Pipeline([
# #     ('prep', CountVectorizer(tokenizer=word_tokenize, stop_words=sw_indo)), # this step has been done in Encoding process with w2v, train set has been encoded
#     ('algo', SGDClassifier(tol=0.0001, penalty='l2', max_iter=10, loss='hinge', alpha=0.0001, random_state=42))
# ])

# model_sgd = pipeline
# model_sgd.fit(X_train, y_train)

# print(model_sgd.best_params_)
# print(model_sgd.score(X_train, y_train), model_sgd.best_score_, model_sgd.score(X_test, y_test))


# ## Save Model

# In[ ]:


# import os
# from jcopml.utils import save_model


# In[ ]:


os.makedirs("model/linear_svm/", exist_ok=True)
model_sgd.save("model/linear_svm/linear_svm.pkl")


# In[ ]:


# model_logreg.save("model/fasttext/product_detection_title_model_logreg.pkl")
model_sgd.save("model/fasttext/product_detection_title_model_sgd.pkl")


# ## Submit Prediction

# In[ ]:


df_submit = pd.read_csv("../input/ndsc-beginner/test.csv")
df_submit.head()


# In[ ]:


df.head()


# In[ ]:


submit_vecs = [better_encode_sentence(sentence, w2v, stopwords=sw) for sentence in tqdm(df_submit.title)]
submit_vecs = np.array(submit_vecs)
submit_vecs


# In[ ]:


target = model_sgd.predict(submit_vecs)
target


# In[ ]:


df_submit_final = pd.DataFrame({
    "itemid": df_submit.itemid,
    "Category": target
})

# set itemid as index
df_submit_final.set_index('itemid', inplace=True)


# In[ ]:


df_submit_final


# In[ ]:


df_submit_final.to_csv('product_detection.csv')


# # 2) CATEGORY CLASSIFICATION BASED ON IMAGE

# In[ ]:




