#!/usr/bin/env python
# coding: utf-8

# # [Real or Not? NLP with Disaster Tweets](https://www.kaggle.com/c/nlp-getting-started)

# ### I plan to research the possibilities and compare different models with different parameters of Simple Transformers models to solve the issue "Real or Not? NLP with Disaster Tweets"

# I plan to study of **each of Simple Transformers models** with different parameters without K-fold since cross-validation complicates the analysis of the model:
# 
# * visualize **embeddings** (I use the function from my notebook in this competition with about 500 forks: [NLP - EDA, Bag of Words, TF IDF, GloVe, BERT](https://www.kaggle.com/vbmokin/nlp-eda-bag-of-words-tf-idf-glove-bert));
# * study **outliers** (forecast errors) (I use the functions of my notebook with more than 700 forks: [TSE2020 - RoBERTa (CNN) - Outlier Analysis, 3chr](https://www.kaggle.com/vbmokin/tse2020-roberta-cnn-outlier-analysis-3chr));
# * build a **confusion matrix** (from the same notebook [NLP - EDA, Bag of Words, TF IDF, GloVe, BERT](https://www.kaggle.com/vbmokin/nlp-eda-bag-of-words-tf-idf-glove-bert));
# 
# I commit the result of each prediction and save (parameters and LB) it in a section with successful commits.
# 
# This notebook use my public dataset with cleaning data for this competition [NLP with Disaster Tweets - cleaning data](https://www.kaggle.com/vbmokin/nlp-with-disaster-tweets-cleaning-data):
# * `train_data_cleaning.csv`
# * `test_data_cleaning.csv`
# 
# to speed up and increase the accuracy of calculations.
# 
# See all models in list of **transformers** library: https://huggingface.co/transformers/pretrained_models.html

# # Acknowledgements
# 
# This kernel uses such good notebooks and resources: 
# * libraries [transformers](https://huggingface.co/transformers) and [simpletransformers](https://github.com/ThilinaRajapakse/simpletransformers)
# * dataset [NLP with Disaster Tweets - cleaning data](https://www.kaggle.com/vbmokin/nlp-with-disaster-tweets-cleaning-data)
# * notebook [SimpleTransformers + Hyperparam Tuning + k-fold CV](https://www.kaggle.com/szelee/simpletransformers-hyperparam-tuning-k-fold-cv)
# * notebook [NLP with DT cleaning: Simple Transformers predict](https://www.kaggle.com/vbmokin/nlp-with-dt-cleaning-simple-transformers-predict)
# * notebook [NLP with Disaster Tweets - EDA and Cleaning data](https://www.kaggle.com/vbmokin/nlp-with-disaster-tweets-eda-and-cleaning-data)
# * notebook [NLP - EDA, Bag of Words, TF IDF, GloVe, BERT](https://www.kaggle.com/vbmokin/nlp-eda-bag-of-words-tf-idf-glove-bert)
# * notebook [TSE2020 - RoBERTa (CNN) - Outlier Analysis, 3chr](https://www.kaggle.com/vbmokin/tse2020-roberta-cnn-outlier-analysis-3chr)

# <a class="anchor" id="0.1"></a>
# ## Table of Contents
# 
# 1. [All commits](#1)
#     - [Commit now](#1.1)
#     - [Successful and most interesting commits without KFolds](#1.2)
#         - [DistilBERT](#1.2.1)
#             - [distilbert-base-uncased](#1.2.1.1)
#             - [distilbert-base-cased](#1.2.1.2)
#         - [RoBERTa](#1.2.2)
#             - [distilroberta-base](#1.2.2.1)
#             - [roberta-base](#1.2.2.2)
#         - [ALBERT](#1.2.3)
#             - [albert-base-v1](#1.2.3.1)
#             - [albert-xlarge-v2](#1.2.3.2)
#         - [BERT](#1.2.4)
#             - [bert-base-uncased](#1.2.4.1)
#             - [bert-base-cased](#1.2.4.2)
#             - [bert-base-multilingual-cased](#1.2.4.3)            
#     - [Successful commits with KFolds](#1.3)
#         - [DistilBERT](#1.3.1)
#             - [distilbert-base-uncased](#1.3.1.1)            
# 1. [Import libraries](#2)
# 1. [Download data](#3)
# 1. [EDA](#4)
# 1. [Model training and prediction](#5)
#     - [Without KFold](#5.1)
#     - [With KFold](#5.2)
# 1. [Submission](#6)
# 1. [Visualization of model outputs for all training data](#7)
# 1. [Outlier Analysis](#8)
#     - [Word Cloud visualization](#8.1)
#     - [Punctuation marks repetition analysis](#8.2)
# 1. [Showing Confusion Matrices](#9)
# 1. [Resume](#10)

# ## 1. All commits<a class="anchor" id="1"></a>
# 
# [Back to Table of Contents](#0.1)

# * DATA1 - dataset 1 (commits 1-23, 43,...) - original dataset of the competition 
# * DATA2 - dataset 2 (commits 24-42, 44,...) - cleaned dataset from [NLP with Disaster Tweets - cleaning data](https://www.kaggle.com/vbmokin/nlp-with-disaster-tweets-cleaning-data)

# ## 1.1. Commit now <a class="anchor" id="1.1"></a>
# 
# [Back to Table of Contents](#0.1)

# In[ ]:


model_type = 'distilbert'
model_name = 'distilbert-base-uncased'
with_kfold = False
weight = [0.43, 0.57]
dataset = 'DATA2'  # or 'DATA1'
n_splits = 1   # if with_kfold then must be n_splits > 1
seed = 100
model_args =  {'fp16': False,
               'train_batch_size': 4,
               'gradient_accumulation_steps': 2,
               'do_lower_case': True,
               'learning_rate': 1e-05,
               'overwrite_output_dir': True,
               'manual_seed': seed,
               'num_train_epochs': 2}


# ## 1.2. Successful and most interesting commits without KFolds <a class="anchor" id="1.2"></a>
# 
# [Back to Table of Contents](#0.1)

# ## 1.2.1. DistilBERT <a class="anchor" id="1.2.1"></a>
# 
# [Back to Table of Contents](#0.1)

# ## 1.2.1.1. distilbert-base-uncased <a class="anchor" id="1.2.1.1"></a>
# 12-layer, 768-hidden, 12-heads, 110M parameters. Trained on lower-cased English text.
# 
# [Back to Table of Contents](#0.1)

# * DATA2 - Commit 67 (LB = 0.84278): lr = 1e-05, num_epochs = 2, seed = 100, acc = 0.887, num_outliers = 864(26.5%), weight = [0.44, 0.56]
# * DATA2 - Commit 62 (LB = 0.84186): lr = 1e-05, num_epochs = 2, seed = 100, acc = 0.886, num_outliers = 865(26.5%), weight = [0.45, 0.55]
# * DATA2 - Commit 56 (LB = 0.84155): lr = 1e-05, num_epochs = 2, seed = 100, acc = 0.887, num_outliers = 857(26.3%), weight = [0.4, 0.6]
# * DATA1 - Commit 6 (LB = 0.84125): lr = 1e-05, num_epochs = 2, acc = 0.888, num_outliers = 852(26.1%)
# * DATA2 - Commit 55 (LB = 0.84125): lr = 9e-06, num_epochs = 2, seed = 100, acc = 0.884, num_outliers = 884(27.1%), weight = [0.4, 0.6]
# * DATA2 - Commit 32 (LB = 0.84125): lr = 9e-06, num_epochs = 2, seed = 100, acc = 0.883, num_outliers = 893(27.4%)
# * DATA2 - Commit 49 (LB = 0.84033): lr = 1e-05, num_epochs = 2, seed = 100, acc = 0.886, num_outliers = 871(26.7%)
# * DATA2 - Commit 25 (LB = 0.84002): lr = 1e-05, seed = 100, acc = 0.885, num_outliers = 873(26.8%)
# * DATA2 - Commit 54 (LB = 0.83971): lr = 9e-06, num_epochs = 2, seed = 100, acc = 0.882, num_outliers = 895(27.4%), weigh = [0.6, 0.4]
# * DATA2 - Commit 57 (LB = 0.83971): lr = 1e-05, num_epochs = 2, seed = 100, acc = 0.887, num_outliers = 862(26.4%), weight = [0.3, 0.7]
# * DATA2 - Commit 58 (LB = 0.83879): lr = 1.5e-05, num_epochs = 2, seed = 100, acc = 0.9, num_outliers = 763(23.4%), weight = [0.4, 0.6]
# * DATA2 - Commit 45 (LB = 0.83849): lr = 1e-05, num_epochs = 2, seed = 42, acc = 0.887, num_outliers = 857(26.3%)
# * DATA1 - Commit 9 (LB = 0.83726): lr = 2e-05, acc = 0.905, num_outliers = 720(22.1%)
# * DATA2 - Commit 29 (LB = 0.83604): lr = 3e-05, acc = 0.866, num_outliers = 1018(31.2%)
# * DATA1 - Commit 5 (LB = 0.83389): lr = 1e-05, seed = 100, acc = 0.915, num_outliers = 649(19.9%)
# * DATA2 - Commit 30 (LB = 0.83174): lr = 4e-05, num_epochs = 2, seed = 100, acc = 0.917, num_outliers = 632(19.4%)

# ## 1.2.1.2. distilbert-base-cased <a class="anchor" id="1.2.1.2"></a>
# 6-layer, 768-hidden, 12-heads, 65M parameters. The DistilBERT model distilled from the BERT model bert-base-cased checkpoint
# 
# [Back to Table of Contents](#0.1)

# * DATA2 - Commit 28 (LB = 0.82194): lr = 1e-05, num_epochs = 2, seed = 100, acc = 0.883, num_outliers = 887(27.2%)

# ## 1.2.2. RoBERTa <a class="anchor" id="1.2.2"></a>
# 
# [Back to Table of Contents](#0.1)

# ### 1.2.2.1. distilroberta-base <a class="anchor" id="1.2.2.1"></a>
# 6-layer, 768-hidden, 12-heads, 82M parameters. The DistilRoBERTa model distilled from the RoBERTa model roberta-base checkpoint.
# 
# [Back to Table of Contents](#0.1)

# * DATA1 - Commit 22 (LB = 0.83818): lr = 1e-05, num_epochs = 2, seed = 100, acc = 0.878, num_outliers = 930(28.5%)
# * DATA1 - Commit 21 (LB = 0.83358): lr = 2e-05, num_epochs = 2, seed = 1, acc = 0.892, num_outliers = 819(25.1%)

# ### 1.2.2.2. roberta-base <a class="anchor" id="1.2.2.2"></a>
# 12-layer, 768-hidden, 12-heads, 125M parameters. RoBERTa using the BERT-base architecture
# 
# [Back to Table of Contents](#0.1)

# * DATA2 - Commit 51 (LB = 0.83512): lr = 1e-05, num_epochs = 2, seed = 100, acc = 0.885, num_outliers = 873(26.8%)
# * DATA1 - Commit 50 (LB = 0.83021): lr = 1e-05, num_epochs = 2, seed = 100, acc = 0.886, num_outliers = 868(26.6%)
# * DATA1 - Commit 52 (LB = 0.82899): lr = 9e-06, num_epochs = 2, seed = 100, acc = 0.887, num_outliers = 862(26.4%)

# ## 1.2.3. ALBERT <a class="anchor" id="1.2.3"></a>
# 
# [Back to Table of Contents](#0.1)

# ### 1.2.3.1. albert-base-v1 <a class="anchor" id="1.2.3.1"></a>
# 12 repeating layers, 128 embedding, 768-hidden, 12-heads, 11M parameters. ALBERT base model
# 
# [Back to Table of Contents](#0.1)

# * DATA2 - Commit 27 (LB = 0.83328): lr = 1e-05, num_epochs = 2, seed = 100, acc = 0.868, num_outliers = 1002(30.7%)

# ### 1.2.3.2. albert-xlarge-v2 <a class="anchor" id="1.2.3.2"></a>
# 24 repeating layers, 128 embedding, 2048-hidden, 16-heads, 58M parameters. ALBERT xlarge model with no dropout, additional training data and longer training
# 
# [Back to Table of Contents](#0.1)

# * DATA2 - Commit 26: lr = 1e-05, num_epochs = 2, seed = 100, acc = 0.572, num_outliers = 3259(99.9%)
# 
# I don't send this submission file. It's not a good solution.

# ## 1.2.4. BERT <a class="anchor" id="1.2.4"></a>
# 
# [Back to Table of Contents](#0.1)

# "**bert-large-uncased**" get error (see unsuccessful commits 3, 4, 42): "OSError: [Errno 28] No space left on device"

# ### 1.2.4.1. bert-base-uncased <a class="anchor" id="1.2.4.1"></a>
# 12-layer, 768-hidden, 12-heads, 110M parameters. Trained on lower-cased English text.
# 
# [Back to Table of Contents](#0.1)

# * DATA2 - Commit 36 (LB = 0.84063): lr = 9e-06, num_epochs = 2, seed = 100, acc = 0.898, num_outliers = 780(23.9%)
# * DATA2 - Commit 34 (LB = 0.84033): lr = 1e-05, num_epochs = 2, seed = 100, acc = 0.9, num_outliers = 763(23.4%)
# * DATA2 - Commit 37 (LB = 0.83818): lr = 2e-05, num_epochs = 2, seed = 100, acc = 0.921, num_outliers = 601(18.4%)
# * DATA2 - Commit 38 (LB = 0.83634): lr = 8e-06, num_epochs = 2, seed = 100, acc = 0.893, num_outliers = 812(24.9%)
# * DATA1 - Commit 44 (LB = 0.83512): lr = 9e-06, num_epochs = 2, seed = 100, acc = 0.898, num_outliers = 775(23.8%)
# * DATA1 - Commit 43 (LB = 0.83144): lr = 9e-06, num_epochs = 1, seed = 100, acc = 0.869, num_outliers = 999(30.6%)

# ### 1.2.4.2. bert-base-cased <a class="anchor" id="1.2.4.2"></a>
# 12-layer, 768-hidden, 12-heads, 110M parameters. Trained on cased English text.
# 
# [Back to Table of Contents](#0.1)

# * DATA2 - Commit 39 (LB = 0.83573): lr = 9e-06, num_epochs = 2, seed = 100, acc = 0.896, num_outliers = 794(24.3%)

# ### 1.2.4.3. bert-base-multilingual-cased <a class="anchor" id="1.2.4.3"></a>
# 12-layer, 768-hidden, 12-heads, 110M parameters. Trained on cased text in the top 104 languages with the largest Wikipedias.
# 
# [Back to Table of Contents](#0.1)

# * DATA1 - Commit 60 (LB = 0.82807): lr = 1e-05, num_epochs = 2, seed = 100, acc = 0.885, num_outliers = 878(26.9%), weight = [0.5, 0.5]

# ## 1.3. Successful commits with KFolds <a class="anchor" id="1.3"></a>
# 
# [Back to Table of Contents](#0.1)

# ## 1.3.1. DistilBERT <a class="anchor" id="1.3.1"></a>
# 
# [Back to Table of Contents](#0.1)

# ## 1.3.1.1. distilbert-base-uncased <a class="anchor" id="1.3.1.1"></a>
# 12-layer, 768-hidden, 12-heads, 110M parameters. Trained on lower-cased English text.
# 
# [Back to Table of Contents](#0.1)

# * DATA2 - Commit 33 (LB = 0.83910): lr = 5e-05, n_splits = 5, num_epochs = 3, seed = 1, acc = 0.828, num_outliers = 1313(40.2%)
# * DATA2 - Commit 59 (LB = 0.83879): lr = 1e-05, n_splits = 10, num_epochs = 2, seed = 100, acc = 0.84, num_outliers = 1217(37.3%), weight = [0.4, 0.6]
# * DATA1 - Commit 13 (LB = 0.83450): lr = 9e-06, n_splits = 5, num_epochs = 2, seed = 100, acc = 0.837, num_outliers = 1240(38.0%)

# ## 2. Import libraries<a class="anchor" id="2"></a>
# 
# [Back to Table of Contents](#0.1)

# In[ ]:


get_ipython().system('pip install --upgrade transformers')
get_ipython().system('pip install simpletransformers')


# In[ ]:


import os, re, string
import random

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from wordcloud import WordCloud, STOPWORDS
from collections import Counter

import numpy as np
import pandas as pd
import sklearn
from sklearn.decomposition import PCA, TruncatedSVD

import torch

from simpletransformers.classification import ClassificationModel
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, KFold

import warnings
warnings.simplefilter('ignore')

pd.set_option('max_rows', 100)
pd.set_option('max_colwidth', 2000)


# In[ ]:


random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True


# ## 3. Download data<a class="anchor" id="3"></a>
# 
# [Back to Table of Contents](#0.1)

# ### See my posts about this issue "[Cleaning dataset for this competition](https://www.kaggle.com/c/nlp-getting-started/discussion/166426)"

# Thanks to [NLP with Disaster Tweets - EDA and Cleaning data](https://www.kaggle.com/vbmokin/nlp-with-disaster-tweets-eda-and-cleaning-data)

# In[ ]:


if dataset == 'DATA1':
    # Original dataset of the competition
    train_data = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')[['text', 'target']]
    test_data = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')[['text']]
    
elif (dataset == 'DATA2') or (dataset == 'DATA2b'):
    # Cleaned dataset from https://www.kaggle.com/vbmokin/nlp-with-disaster-tweets-cleaning-data
    train_data = pd.read_csv('../input/nlp-with-disaster-tweets-cleaning-data/train_data_cleaning.csv')[['text', 'target']]
    test_data = pd.read_csv('../input/nlp-with-disaster-tweets-cleaning-data/test_data_cleaning.csv')[['text']]

# Original dataset of the competition
sample_submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")


# In[ ]:


train_data['target'].hist()


# In[ ]:


print("Weights which I offer for 0 and 1:", weight)


# In[ ]:


train_data


# In[ ]:


train_data.info()


# In[ ]:


test_data['text']


# In[ ]:


test_data.info()


# ## 4. EDA<a class="anchor" id="4"></a>
# 
# [Back to Table of Contents](#0.1)

# ### See my posts about this issue "[Punctuation marks repetition in incorrectly classified text](https://www.kaggle.com/c/nlp-getting-started/discussion/166248)"

# In[ ]:


def subtext_repeation_in_df(df, col, subtext, num):
    # Calc statistics as table for character repetition (1...num times) from subtext list in the df[col]
    
    text = "".join(df[col])
    result = pd.DataFrame(columns = ['subtext', 'count'])
    i = 0
    if (len(df) > 0) and (len(subtext) > 0):
        for c in subtext:
            for j in range(num):
                cs = c*(j+1)
                result.loc[i,'count'] = text.count(cs)
                if c == ' ':
                    cs = cs.replace(' ','<space>')
                result.loc[i,'subtext'] = cs                
                i += 1
    print('Number of all data is', len(df))
    result = result[result['count'] > 0].reset_index(drop=True)
    display(result.sort_values(by='subtext'))
    
    print('Text examples')
    problem_examples = pd.DataFrame(columns = ['problem_examples'])
    problem_examples['problem_examples'] = ''
    for i in range(len(result)):
        problem_examples.loc[i,'problem_examples'] = df[df[col].str.find(result.loc[i,'subtext'])>-1].reset_index(drop=True).loc[0, col]
    problem_examples = problem_examples.drop_duplicates()
    display(problem_examples)


# In[ ]:


# Analysis of punctuation marks repetition in training data
print('Statistics for punctuation marks repetition in training data')
subtext_repeation_in_df(train_data, 'text', list(string.punctuation), 10)


# In[ ]:


# Analysis of punctuation marks repetition in test data
print('Statistics for punctuation marks repetition in test data')
subtext_repeation_in_df(test_data, 'text', list(string.punctuation), 10)


# ## 5. Model training and prediction<a class="anchor" id="5"></a>
# 
# [Back to Table of Contents](#0.1)

# ## 5.1. Without KFold<a class="anchor" id="5.1"></a>
# 
# [Back to Table of Contents](#0.1)

# In[ ]:


# Model training without KFold
if not with_kfold:
    model = ClassificationModel(model_type, model_name, args=model_args, weight=weight) 
    model.train_model(train_data)
    result, model_outputs, wrong_predictions = model.eval_model(train_data, acc=sklearn.metrics.accuracy_score)
    y_preds, _, = model.predict(test_data['text'])
    pred_train, _ = model.predict(train_data['text'])


# In[ ]:


if not with_kfold:
    acc = result['acc']
    print('acc =',acc)


# ## 5.2. With KFold<a class="anchor" id="5.2"></a>
# 
# [Back to Table of Contents](#0.1)

# In[ ]:


# Model training with KFold
if with_kfold:
    kf = KFold(n_splits=n_splits, random_state=seed, shuffle=True)

    results = []
    wrong_predictions = []
    y_preds = np.zeros(test_data.shape[0])
    pred_train = np.zeros(train_data.shape[0])
    
    first_fold = True
    for train_index, val_index in kf.split(train_data):
        train_df = train_data.iloc[train_index]
        val_df = train_data.iloc[val_index]

        # Model training
        model = ClassificationModel(model_type, model_name, args=model_args)
        model.train_model(train_df)

        # Validation data prediction
        result, model_outputs_fold, wrong_predictions_fold = model.eval_model(val_df, acc=sklearn.metrics.accuracy_score)
        pred_train[val_index], _ = model.predict(val_df['text'].reset_index(drop=True))
        
        # Save fold results
        if first_fold:
            model_outputs = model_outputs_fold
            first_fold = False
        else: model_outputs = np.vstack((model_outputs,model_outputs_fold))
        
        wrong_predictions += wrong_predictions_fold
        results.append(result['acc'])

        # Test data prediction
        y_pred, _ = model.predict(test_data['text'])
        y_preds += y_pred / n_splits


# In[ ]:


# Thanks to https://www.kaggle.com/szelee/simpletransformers-hyperparam-tuning-k-fold-cv
# CV accuracy result output
if with_kfold:
    for i, result in enumerate(results, 1):
        print(f"Fold-{i}: {result}")
    
    acc = np.mean(results)

    print(f"{n_splits}-fold CV accuracy result: Mean: {acc} Standard deviation:{np.std(results)}")


# ## 6. Submission<a class="anchor" id="6"></a>
# 
# [Back to Table of Contents](#0.1)

# In[ ]:


y_preds[:] = y_preds[:]>=0.5
y_preds = y_preds.astype(int)
np.mean(y_preds)


# In[ ]:


# Data prediction and submission
sample_submission["target"] = y_preds
sample_submission.to_csv("submission.csv", index=False)
y_preds[:20]


# ## 7. Visualization of model outputs for all training data <a class="anchor" id="7"></a>
# 
# [Back to Table of Contents](#0.1)

# In[ ]:


# Visualization of model outputs for each rows of training data
def plot_data_lavel(data, labels):
    colors = ['orange','blue']
    plt.scatter(data[:,0], data[:,1], s=8, alpha=.8, c=labels, cmap=matplotlib.colors.ListedColormap(colors))
    orange_patch = mpatches.Patch(color='orange', label='Not')
    blue_patch = mpatches.Patch(color='blue', label='Real')
    plt.legend(handles=[orange_patch, blue_patch], prop={'size': 30})

fig = plt.figure(figsize=(16, 16))          
plot_data_lavel(model_outputs, train_data['target'].values)
plt.show()


# ## 8. Outlier Analysis<a class="anchor" id="8"></a>
# 
# [Back to Table of Contents](#0.1)

# This chapter uses notebook [TSE2020 - RoBERTa (CNN) - Outlier Analysis, 3chr](https://www.kaggle.com/vbmokin/tse2020-roberta-cnn-outlier-analysis-3chr)

# ## 8.1. Word Cloud visualization <a class="anchor" id="8.1"></a>
# 
# [Back to Table of Contents](#0.1)

# In[ ]:


stop_words = list(STOPWORDS) + list('0123456789') + ['rt', 'amp', 'us', 'will', 'via', 'dont', 'cant', 'u', 'work', 'im',
                               'got', 'back', 'first', 'one', 'two', 'know', 'going', 'time', 'go', 'may', 'youtube', 'say', 'day', 'love', 
                               'still', 'see', 'watch', 'make', 'think', 'even', 'right', 'left', 'take', 'want', 'http', 'https', 'co']


# In[ ]:


def plot_word_cloud(x, col, num_common_words, stop_words):
    # Building the WordCloud for the num_common_words most common data in x[col] without words from list stop_words
    
    corpus = " ".join(x[col].str.lower())
    corpus = corpus.translate(str.maketrans('', '', string.punctuation))
    corpus_without_stopwords = [word for word in corpus.split() if word not in stop_words]
    common_words = Counter(corpus_without_stopwords).most_common(num_common_words)
    
    plt.figure(figsize=(12,8))
    word_cloud = WordCloud(stopwords = stop_words,
                           background_color='black',
                           max_font_size = 80
                           ).generate(" ".join(corpus_without_stopwords))
    plt.imshow(word_cloud)
    plt.axis('off')
    plt.show()
    return common_words


# In[ ]:


# Training data visualization as WordCloud
print('Word Cloud for training data without stopwords and apostrophes')
plot_word_cloud(train_data, 'text', 50, stop_words)


# In[ ]:


# Test data visualization as WordCloud
print('Word Cloud for test data without stopwords and apostrophes')
plot_word_cloud(test_data, 'text', 50, stop_words)


# In[ ]:


# Form DataFrame with outliers
outliers = pd.DataFrame(columns = ['text', 'label'])
for i in range(len(wrong_predictions)):
    outliers.loc[i, 'text'] = wrong_predictions[i].text_a
    outliers.loc[i, 'label'] = wrong_predictions[i].label


# In[ ]:


outliers


# In[ ]:


# Outliers visualization as WordCloud
print('Word Cloud for outliers without stopwords and apostrophes in the training data predictions')
outliers_top50 = plot_word_cloud(outliers, 'text', 50, stop_words)


# In[ ]:


outliers_top50


# ## 8.2. Analysis of punctuation marks repetition in text <a class="anchor" id="8.2"></a>
# 
# [Back to Table of Contents](#0.1)

# In[ ]:


# Analysis of punctuation marks repetition in outliers
print('Statistics for punctuation marks repetition in outliers')
subtext_repeation_in_df(outliers, 'text', list(string.punctuation), 10)


# ## 9. Showing Confusion Matrices<a class="anchor" id="9"></a>
# 
# [Back to Table of Contents](#0.1)

# In[ ]:


# Thanks to https://www.kaggle.com/marcovasquez/basic-nlp-with-tensorflow-and-wordcloud and https://www.kaggle.com/vbmokin/nlp-eda-bag-of-words-tf-idf-glove-bert
# Showing Confusion Matrix
def plot_cm(y_true, y_pred, title, figsize=(5,5)):
    cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true))
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    cm = pd.DataFrame(cm, index=np.unique(y_true), columns=np.unique(y_true))
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    fig, ax = plt.subplots(figsize=figsize)
    plt.title(title)
    sns.heatmap(cm, cmap= "YlGnBu", annot=annot, fmt='', ax=ax)


# In[ ]:


# Showing Confusion Matrix for ST Bert model
plot_cm(pred_train, train_data['target'].values, 'Confusion matrix for ST Bert model', figsize=(7,7))


# ## 10. Resume<a class="anchor" id="10"></a>
# 
# [Back to Table of Contents](#0.1)

# In[ ]:


num_outliers_per_cent = round(len(outliers)*100/len(test_data), 1)


# In[ ]:


acc_round = round(acc,3)
print('acc =', acc, '=', acc_round)


# In[ ]:


if n_splits == 1:
    n_splits_res = ""
else: n_splits_res = f"n_splits = {n_splits}, "


# In[ ]:


print(f"Model - {model_type}, {model_name}")
print(f"* {dataset} - Commit __ (LB = 0._____): lr = {model_args['learning_rate']}, {n_splits_res}num_epochs = {model_args['num_train_epochs']}, seed = {seed}, acc = {acc_round}, num_outliers = {len(outliers)}({num_outliers_per_cent}%), weight = {weight}")


# I hope you find this kernel useful and enjoyable.

# Your comments and feedback are most welcome.

# [Go to Top](#0)
