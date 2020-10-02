#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import tensorflow_hub as hub
from tqdm import tqdm
from tqdm import tqdm_notebook
from sklearn.metrics import auc
from sklearn.metrics import classification_report
import seaborn as sns
import matplotlib.pyplot as plt

from transformers import AutoTokenizer,BertTokenizer,TFBertModel,TFOpenAIGPTModel,OpenAIGPTTokenizer,DistilBertTokenizer, TFDistilBertModel,XLMTokenizer, TFXLMModel
from transformers import TFAutoModel, AutoTokenizer
from kaggle_datasets import KaggleDatasets
from sklearn.metrics import roc_curve,confusion_matrix,auc
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, processors
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import matplotlib as mpl


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.initializers import Constant
import re

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# # **Main Purpose of the NoteBook**
# <p1> The purpose of this notebook is really simple, I was curious as how different Transformers Architecture will perform over the given data. This notebook is really simple to understand, there is no fancy EDA or any data exploration. **The sole purpose of this notebook is to just compare different Transformers Architecture. I will be judging the Model performance over the accuracy, AUC,recall and precision**<p1>
# 
# <p2> NoteBook also can be used as a introduction to as how tensorflow keras can be integrated with hugging face. 
# I have tried to make this notebook quite user friendly and easy to understand so that every one can understand the basic of hugging face </p2>
# 
# ![](https://huggingface.co/front/thumbnails/models.png)
# 
# <p3>The following Transformers Architecture have been tested in the notebook</p3>
# 
# ### 1. BERT
# ### 2. OpenAIGPT
# ### 3. Transformer XL
# ### 4. XLM
# ### 5. XLM-Roberta-Large
# 
# 
# ![](https://media-exp1.licdn.com/dms/image/C5112AQHdCkQOA8sjlQ/article-cover_image-shrink_600_2000/0?e=1596067200&v=beta&t=aXNCLbGrTOsjxeWZkzx5ksTNThTUFVaEBPl1_vsf6K8)
#     
# <p4> However I also later on added some classical LSTM models with and without attention mechanims in order to  see its performance on the dataset. The same metric system will be used to judge the performance of LSTMs,The following two type of LSTM models have been used </p4>
#     
#     
#   
# ### 6.LSTM with Glovec Embedding
# ### 7.LSTM with Attention Mechanism
#  
# <p5>I also have given useful links for each of the model used so that you can further expand your knowledge</p5>
#     
#     
# <font color='red'>This is my first notebook and I have tried to put into a lot of effort. Please upvote, it will really encourage me to make better and more helpful notebooks in future</font>   
# 
#     
# <font color='read'> The notebook takes much time to run therefore I cannot run the entire notebook in one go. Therefore, I have ran notebook sequentailly and attached the output of each model performance in the markdown. If you want to check any model just run the relevant section while also executing the functions which are common to all models. Again, the purpose of notebook is to show how without any substaintial hyperparameter tuning different transformers and attention mechanism LSTMs perform on the task</font>

# ## Credits:
# <p1> I have used some help from other notebooks, therefore I would like to give the credit to desire individuals:
#     <ul>
#         <li> 1. [xhlulu](https://www.kaggle.com/xhlulu/jigsaw-tpu-distilbert-with-huggingface-and-keras) </li>
#         <li> 2. [tanulsingh077](https://www.kaggle.com/tanulsingh077/deep-learning-for-nlp-zero-to-transformers-bert/notebook) </li>
#     </ul>
# 

# ## TPU Distribution Strategy for Efficent Model Training

# In[ ]:


# Detect hardware, return appropriate distribution strategy
try:
    # TPU detection. No parameters necessary if TPU_NAME environment variable is
    # set: this is always the case on Kaggle.
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Running on TPU ', tpu.master())
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    # Default distribution strategy in Tensorflow. Works on CPU and single GPU.
    strategy = tf.distribute.get_strategy()


# In[ ]:


train1 = pd.read_csv("/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv")
train2 = pd.read_csv("/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-unintended-bias-train.csv")
train2.toxic = train2.toxic.round().astype(int)

valid_raw = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/validation.csv')
test_raw = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/test.csv')
sub = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/sample_submission.csv')


# In[ ]:


# Combine train1 with a subset of train2
train_raw = pd.concat([
    train1[['comment_text', 'toxic']],
    train2[['comment_text', 'toxic']].query('toxic==1'),
    train2[['comment_text', 'toxic']].query('toxic==0').sample(n=100000, random_state=0)
])


# In[ ]:


valid_raw['toxic'].value_counts().plot(kind='bar')


# In[ ]:


train_raw['toxic'].value_counts().plot(kind='bar')


# In[ ]:


neg, pos = np.bincount(train_raw['toxic'])
total = neg + pos
print('Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(
    total, pos, 100 * pos / total))


# # 1. BERT

# ![BERT](https://www.researchgate.net/profile/Wei_Shi102/publication/336995759/figure/fig1/AS:824051559825408@1573480610178/The-architecture-from-BERT-Devlin-et-al-2019-for-fine-tuning-of-implicit-discourse.ppm)

# ## Usefull Links About BERT
# <p1> The following links are quite useful if you to further your knowledge about BERT
# <ui>
#     <li>[illustrated-bert](http://jalammar.github.io/illustrated-bert/) </li>
#     <li>[a-visual-guide-to-using-bert-for-the-first-time](http://jalammar.github.io/a-visual-guide-to-using-bert-for-the-first-time/) </li>
#     <li>[Chris Mccromick AI](https://www.youtube.com/watch?v=_eSGWNqKeeY&t=1s) </li> 
#     <li>[Chris Mccromick AI-2](https://www.youtube.com/watch?v=l8ZYCvgGu0o) </li>
#     <li>[CS224n Video](https://www.youtube.com/watch?v=S-CspeZ8FHc)
#     <li>[Orginal Paper](https://arxiv.org/abs/1810.04805)</li>
# </ui>
# 

# In[ ]:


# First load the real tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


# # Hyperparameters-Transformers

# In[ ]:


BATCH_SIZE = 16 * strategy.num_replicas_in_sync
EPOCHS=2
LEARNING_RATE=1e-5
early_stopping=early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', 
    verbose=1,
    patience=10,
    mode='max',
    restore_best_weights=True)
AUTO = tf.data.experimental.AUTOTUNE
GCS_DS_PATH = KaggleDatasets().get_gcs_path()
max_seq_length = 192


# In[ ]:



def single_encoding_function(text,tokenizer,name='BERT'):
    input_ids=[]
    if name=='BERT':
        tokenizer.pad_token ='[PAD]'
    elif name=='OPENAIGPT2':
        tokenizer.pad_token='<unk>'
    elif name=='Transformer XL':
        print(tokenizer.eos_token)
        tokenizer.pad_token= tokenizer.eos_token
    elif name=='DistilBert':
        tokenizer.pad_token='[PAD]'
    
    for sentence in tqdm(text):
        encoded=tokenizer.encode(sentence,max_length=max_seq_length,pad_to_max_length=True)
        input_ids.append(encoded)
    return input_ids


# In[ ]:


X_train=np.array(single_encoding_function(train_raw['comment_text'].values.tolist(),tokenizer,name="BERT"))
y_train=np.array(train_raw['toxic'])
X_valid=np.array(single_encoding_function(valid_raw['comment_text'].values.tolist(),tokenizer,name="BERT"))
y_valid=np.array(valid_raw['toxic'])
X_test=np.array(single_encoding_function(test_raw['content'].values.tolist(),tokenizer,name="BERT"))


# In[ ]:


steps_per_epoch = X_train.shape[0] // BATCH_SIZE


# ## Making Tensor Data Pipeline

# In[ ]:


def make_data():
    train = (
        tf.data.Dataset
        .from_tensor_slices((X_train, y_train))
        .repeat()
        .shuffle(2048)
        .batch(BATCH_SIZE)
        .prefetch(AUTO))

    valid = (
        tf.data.Dataset
        .from_tensor_slices((X_valid, y_valid))
        .batch(BATCH_SIZE)
        .cache()
        .prefetch(AUTO)
    )

    test = (
        tf.data.Dataset
        .from_tensor_slices(X_test)
        .batch(BATCH_SIZE)
    )
    return train,valid,test


# In[ ]:


train,valid,test=make_data()


# ## Making TF Model for Different Transformers

# In[ ]:


def build_model(transformer_layer,max_len=max_seq_length):
    input_word_ids = tf.keras.layers.Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    sequence_output = transformer_layer(input_word_ids)[0]
    
    cls_token = sequence_output[:, 0, :]
    out = tf.keras.layers.Dense(1, activation='sigmoid')(cls_token)
    
    model = tf.keras.Model(inputs=input_word_ids, outputs=out)
    
    
    return model


# ## Functions to plot Metrics and loss to Compare Different Transformers
# 
# ![](https://miro.medium.com/fit/c/1838/551/1*aPYAckB1ZtfcqoY8wZ915w.jpeg)

# ### Note:
# <p1> These are four function whic will help me to compare the performance of different models. The first two functions being used will plot loss and other choosen metrics over the number of epochs</p1>
# 
# <p2>The last two function will plot confusion matrix and roc for the models respectively</p2>

# ### Useful links to understand Choosen Metrics:
# <p1> These are the link which can help you to understand the metrics which I have used to compare the performance of different models</p1>
# <ui>
#     <li>[Confusion Matrix](https://www.coursera.org/lecture/python-machine-learning/confusion-matrices-basic-evaluation-metrics-90kLk)</li>
#     <li>[ROC,Precision,Recall,AUC](https://www.coursera.org/lecture/python-machine-learning/precision-recall-and-roc-curves-8v6DL)</li>
# </ui>
#     

# In[ ]:


mpl.rcParams['figure.figsize'] = (12, 10)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

def plot_loss(history):
# Use a log scale to show the wide range of values.
    plt.semilogy(history.epoch,  history.history['loss'],
               color='red', label='Train Loss')
    plt.semilogy(history.epoch,  history.history['val_loss'],
          color='green', label='Val Loss',
          linestyle="--")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
  
    plt.legend()
    
    
def plot_metrics(history):
    metrics =  ['loss', 'auc', 'precision', 'recall']
    for n, metric in enumerate(metrics):
        name = metric.replace("_"," ").capitalize()
        plt.subplot(2,2,n+1)
        plt.plot(history.epoch,  history.history[metric], color=colors[0], label='Train')
        plt.plot(history.epoch, history.history['val_'+metric],
                 color=colors[0], linestyle="--", label='Val')
        plt.xlabel('Epoch')
        plt.ylabel(name)
        if metric == 'loss':
            plt.ylim([0, plt.ylim()[1]])
        elif metric == 'auc':
            plt.ylim([0.8,1])
        else:
            plt.ylim([0,1])

        plt.legend()

def plot_cm(y_true, y_pred, title):
    ''''
    input y_true-Ground Truth Labels
          y_pred-Predicted Value of Model
          title-What Title to give to the confusion matrix
    
    Draws a Confusion Matrix for better understanding of how the model is working
    
    return None
    
    '''
    
    figsize=(10,10)
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

def roc_curve_plot(fpr,tpr,roc_auc):
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' %roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

        


# ## Model Compilation Under TPUs

# In[ ]:


get_ipython().run_cell_magic('time', '', "def compile_model(name):\n    with strategy.scope():\n        METRICS = [\n          tf.keras.metrics.TruePositives(name='tp'),\n          tf.keras.metrics.FalsePositives(name='fp'),\n          tf.keras.metrics.TrueNegatives(name='tn'),\n          tf.keras.metrics.FalseNegatives(name='fn'), \n          tf.keras.metrics.BinaryAccuracy(name='accuracy'),\n          tf.keras.metrics.Precision(name='precision'),\n          tf.keras.metrics.Recall(name='recall'),\n          tf.keras.metrics.AUC(name='auc')]\n        if name=='bert-base-uncased':\n            transformer_layer = (\n                TFBertModel.from_pretrained(name)\n            )\n        elif name=='openai-gpt':\n            transformer_layer = (\n                TFOpenAIGPTModel.from_pretrained(name)\n            )\n        elif name=='distilbert-base-cased':\n            transformer_layer = (\n                TFDistilBertModel.from_pretrained(name)\n            )\n        elif name=='xlm-mlm-en-2048':\n            transformer_layer = (\n                TFBertModel.from_pretrained(name)\n            )\n        elif name=='jplu/tf-xlm-roberta-large':\n            transformer_layer = (\n                TFAutoModel.from_pretrained(name)\n            )\n        model = build_model(transformer_layer, max_len=max_seq_length)\n        model.compile(optimizer=tf.keras.optimizers.Adam(\n        learning_rate=LEARNING_RATE), loss='binary_crossentropy', metrics=METRICS)\n    return model")


# In[ ]:


steps_per_epoch=X_train.shape[0]//BATCH_SIZE

model=compile_model('bert-base-uncased')
print(model.summary())
history=model.fit(
    train,steps_per_epoch=steps_per_epoch,
    epochs=EPOCHS,callbacks=[early_stopping], validation_data=valid
)


# ## BERT Performance-Loss

# In[ ]:


plot_loss(history)


# <img src='https://www.kaggleusercontent.com/kf/36129650/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..8pFTexbxmoqrx0OFHZ_YFw.qJ_XoPOQxUJaJIyQCzZNDB5Rs-ATsNEA5ZhM9mB5xxiYm-VfPhTiXbJNb8ocdZP9zzurfVgGID4gkAVkdl09D3jx1naobYVbt5me1OmtbWTDUdVKo-hmeMtM3Qf7vKbDl92GRz67Z97b6ghs7pG_9YxQWh5se171PcAuDFsBs4fzwZ_jZ1sw8g4LYOftybCvwmAh3yfLlY2KkV6RkDRgix3PPS6Qxpc2iy91GRDskkwKX2GDcGSycMAccPOuelzVBs-1gQEnuctTXiBLBiLbM1eIcxYbhKtmZn3mqjU4nPv-2qFTh4UKjIOnMSVg_dwtlxZ4I858gd1a0wI4We43kGi0lORZ0CwJDJgemHAcnkcjPYLllJyqwj2pL5zrEwexAf-3oR0DoDZ952DD1xlVR_K5vtt_P2aFybGiTfzSjAfnWpVb-2-QUIF6Esbz25MYXuZo2Trk7HqDt3ApBPMqzdvBjRu5MTcF-cFtkBKIWcBWqYZcTB5IDcmy9PWVDSyKcLksJNHiPPB3Ei3LuE0Z8GfKUGPyBXRe1PYAHZg27QvnmvxoOIi-epUuUoCN2GRm6p6Ekyg-RRb5RXCmJU41UF2h7JxBKPoGmlVvlYMjnrn4vZatLbWVbVS7GBpkWHYb6IIIkqaYFDXICaFgVAIu2C6_sPf2-LQUXNDkOYPYYCKN_FHlw1paRnPDMNRs0J4T.5IAqYeDNXIJujd6C8JCl2Q/__results___files/__results___32_0.png'>

# ## Bert Performance-Metrics

# In[ ]:


plot_metrics(history)


# <img src='https://www.kaggleusercontent.com/kf/36129650/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..8pFTexbxmoqrx0OFHZ_YFw.qJ_XoPOQxUJaJIyQCzZNDB5Rs-ATsNEA5ZhM9mB5xxiYm-VfPhTiXbJNb8ocdZP9zzurfVgGID4gkAVkdl09D3jx1naobYVbt5me1OmtbWTDUdVKo-hmeMtM3Qf7vKbDl92GRz67Z97b6ghs7pG_9YxQWh5se171PcAuDFsBs4fzwZ_jZ1sw8g4LYOftybCvwmAh3yfLlY2KkV6RkDRgix3PPS6Qxpc2iy91GRDskkwKX2GDcGSycMAccPOuelzVBs-1gQEnuctTXiBLBiLbM1eIcxYbhKtmZn3mqjU4nPv-2qFTh4UKjIOnMSVg_dwtlxZ4I858gd1a0wI4We43kGi0lORZ0CwJDJgemHAcnkcjPYLllJyqwj2pL5zrEwexAf-3oR0DoDZ952DD1xlVR_K5vtt_P2aFybGiTfzSjAfnWpVb-2-QUIF6Esbz25MYXuZo2Trk7HqDt3ApBPMqzdvBjRu5MTcF-cFtkBKIWcBWqYZcTB5IDcmy9PWVDSyKcLksJNHiPPB3Ei3LuE0Z8GfKUGPyBXRe1PYAHZg27QvnmvxoOIi-epUuUoCN2GRm6p6Ekyg-RRb5RXCmJU41UF2h7JxBKPoGmlVvlYMjnrn4vZatLbWVbVS7GBpkWHYb6IIIkqaYFDXICaFgVAIu2C6_sPf2-LQUXNDkOYPYYCKN_FHlw1paRnPDMNRs0J4T.5IAqYeDNXIJujd6C8JCl2Q/__results___files/__results___34_0.png'>

# ## BERT Performance-Confusion Matrix

# In[ ]:


y_predict=model.predict(valid, verbose=1)
y_predict[ y_predict> 0.5] = 1
y_predict[y_predict <= 0.5] = 0
plot_cm(y_valid, y_predict, 'BERT-Confusion Matrix')


# <img src='https://www.kaggleusercontent.com/kf/36129650/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..K5ST4JjFGBMeTtwX1jC9XQ.0KxxSju7AC0c3tRP2w-rTijGPiYJGNZnjThGQx-BG3govElMkJpIJw6gcpAozeKQSH_Q3_Cjh6a38SnweVyY3X6eW1YMl8UvsJNUsQnuKGTTig69_hUGyoPmepANHm4vZnwUN3vljN9RmyRVaXuvHTkgGCs6KASszIFNgE1v309GpSOtQy7Um8_75fzG-szb7mNmMBlkZ_vBs4-4f-Wmf_8JwkQit6kgDM9qXzE2pEq0btxta2CuVyCYUkJ-qV6aCQprqSy68UTOjD6tdwlASMi9hOLyeogQrSoHLzPWgf_Xg3Pm6dwwjrFwDw2MpFoT47szD5MaXTLoT6cxUAdIRZVNMHed7HsC9r-PDwIdDiZKy8Sss5DUrj3jhE02_HCx9cJLYsaLIvPwJVrfKb7aYvaULHhRw4rLaHtxo4r0B7bKLTGkkgkWUmXVv7yDViYdk48ClVic05ZpfjUmpDo9TcD1s7mUCOcoLdqJh7U8UXbBqGBMHpVD95MOFpwAyTC9mnvDuiIiWVxRr8buuC_QnjRITs4CPYhrFG_-5zXC5B4KDUG23AcbCwK7IkSW4PKnufdJHlgKq3g9nNJUcvgE_mB10rGX2YGzJYpIFIegR6HjzR9UuWLIvm7cxOxWTjUSd-8lzFPNFfWyGWgXsyvdtEVa9mkAKJ63A12oFLIou9Gj1L3wMFXBTFsY3n8EB29h.vwq3hDsu0b8kXlx_-yqfkQ/__results___files/__results___36_1.png'>

# ## BERT Performance-ROC Curve

# In[ ]:


y_predict_prob=model.predict(valid, verbose=1)
fpr, tpr, _ = roc_curve(y_valid,y_predict_prob)
roc_auc = auc(fpr, tpr)
roc_curve_plot(fpr,tpr,roc_auc)


# <img src='https://www.kaggleusercontent.com/kf/36129650/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..K5ST4JjFGBMeTtwX1jC9XQ.0KxxSju7AC0c3tRP2w-rTijGPiYJGNZnjThGQx-BG3govElMkJpIJw6gcpAozeKQSH_Q3_Cjh6a38SnweVyY3X6eW1YMl8UvsJNUsQnuKGTTig69_hUGyoPmepANHm4vZnwUN3vljN9RmyRVaXuvHTkgGCs6KASszIFNgE1v309GpSOtQy7Um8_75fzG-szb7mNmMBlkZ_vBs4-4f-Wmf_8JwkQit6kgDM9qXzE2pEq0btxta2CuVyCYUkJ-qV6aCQprqSy68UTOjD6tdwlASMi9hOLyeogQrSoHLzPWgf_Xg3Pm6dwwjrFwDw2MpFoT47szD5MaXTLoT6cxUAdIRZVNMHed7HsC9r-PDwIdDiZKy8Sss5DUrj3jhE02_HCx9cJLYsaLIvPwJVrfKb7aYvaULHhRw4rLaHtxo4r0B7bKLTGkkgkWUmXVv7yDViYdk48ClVic05ZpfjUmpDo9TcD1s7mUCOcoLdqJh7U8UXbBqGBMHpVD95MOFpwAyTC9mnvDuiIiWVxRr8buuC_QnjRITs4CPYhrFG_-5zXC5B4KDUG23AcbCwK7IkSW4PKnufdJHlgKq3g9nNJUcvgE_mB10rGX2YGzJYpIFIegR6HjzR9UuWLIvm7cxOxWTjUSd-8lzFPNFfWyGWgXsyvdtEVa9mkAKJ63A12oFLIou9Gj1L3wMFXBTFsY3n8EB29h.vwq3hDsu0b8kXlx_-yqfkQ/__results___files/__results___38_1.png'>

# # 2. OpenAIGPT

# ![](http://www.topbots.com/wp-content/uploads/2019/04/cover_GPT_web-1280x640.jpg)

# ## Usefull Links About OPENAI-GPT
# <p1> The following links are quite useful if you to further your knowledge about OPENAI-GPT(Please note most of these resources are related to open-ai GPT-2 but are sufficent to understand to orginal open-ai-gpt as well</p1>
# <ui>
#     <li>[illustrated-gpt2](http://jalammar.github.io/illustrated-gpt2/) </li>
#     <li>[openai-gpt-language-modeling](https://towardsdatascience.com/openai-gpt-language-modeling-on-gutenberg-with-tensorflow-keras-876f9f324b6c) </li>
#     <li>[better-language-models](https://openai.com/blog/better-language-models/) </li> 
#     <li>[Orginal Paper](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)</li>
# </ui>

# In[ ]:


# # First load the real tokenizer
tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')


# In[ ]:


X_train=np.array(single_encoding_function(train_raw['comment_text'],tokenizer,'OPENAIGPT2'))
y_train=np.array(train_raw['toxic'])
X_valid=np.array(single_encoding_function(valid_raw['comment_text'],tokenizer,'OPENAIGPT2'))
y_valid=np.array(valid_raw['toxic'])
X_test=np.array(single_encoding_function(test_raw['content'],tokenizer,'OPENAIGPT2'))


# In[ ]:


steps_per_epoch = X_train.shape[0] // BATCH_SIZE


# In[ ]:


train,valid,test=make_data()


# In[ ]:



model=compile_model('openai-gpt')
print(model.summary())

history=model.fit(
    train,steps_per_epoch=steps_per_epoch,
    epochs=EPOCHS,callbacks=[early_stopping], validation_data=valid
)


# ## OpenAIGPT Performance-Loss

# In[ ]:


plot_loss(history)


# <img src='https://www.kaggleusercontent.com/kf/36129650/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..K5ST4JjFGBMeTtwX1jC9XQ.0KxxSju7AC0c3tRP2w-rTijGPiYJGNZnjThGQx-BG3govElMkJpIJw6gcpAozeKQSH_Q3_Cjh6a38SnweVyY3X6eW1YMl8UvsJNUsQnuKGTTig69_hUGyoPmepANHm4vZnwUN3vljN9RmyRVaXuvHTkgGCs6KASszIFNgE1v309GpSOtQy7Um8_75fzG-szb7mNmMBlkZ_vBs4-4f-Wmf_8JwkQit6kgDM9qXzE2pEq0btxta2CuVyCYUkJ-qV6aCQprqSy68UTOjD6tdwlASMi9hOLyeogQrSoHLzPWgf_Xg3Pm6dwwjrFwDw2MpFoT47szD5MaXTLoT6cxUAdIRZVNMHed7HsC9r-PDwIdDiZKy8Sss5DUrj3jhE02_HCx9cJLYsaLIvPwJVrfKb7aYvaULHhRw4rLaHtxo4r0B7bKLTGkkgkWUmXVv7yDViYdk48ClVic05ZpfjUmpDo9TcD1s7mUCOcoLdqJh7U8UXbBqGBMHpVD95MOFpwAyTC9mnvDuiIiWVxRr8buuC_QnjRITs4CPYhrFG_-5zXC5B4KDUG23AcbCwK7IkSW4PKnufdJHlgKq3g9nNJUcvgE_mB10rGX2YGzJYpIFIegR6HjzR9UuWLIvm7cxOxWTjUSd-8lzFPNFfWyGWgXsyvdtEVa9mkAKJ63A12oFLIou9Gj1L3wMFXBTFsY3n8EB29h.vwq3hDsu0b8kXlx_-yqfkQ/__results___files/__results___48_0.png'>

# ## OpenAIGPT Performance-Metrics

# In[ ]:


plot_metrics(history)


# <img src='https://www.kaggleusercontent.com/kf/36129650/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..K5ST4JjFGBMeTtwX1jC9XQ.0KxxSju7AC0c3tRP2w-rTijGPiYJGNZnjThGQx-BG3govElMkJpIJw6gcpAozeKQSH_Q3_Cjh6a38SnweVyY3X6eW1YMl8UvsJNUsQnuKGTTig69_hUGyoPmepANHm4vZnwUN3vljN9RmyRVaXuvHTkgGCs6KASszIFNgE1v309GpSOtQy7Um8_75fzG-szb7mNmMBlkZ_vBs4-4f-Wmf_8JwkQit6kgDM9qXzE2pEq0btxta2CuVyCYUkJ-qV6aCQprqSy68UTOjD6tdwlASMi9hOLyeogQrSoHLzPWgf_Xg3Pm6dwwjrFwDw2MpFoT47szD5MaXTLoT6cxUAdIRZVNMHed7HsC9r-PDwIdDiZKy8Sss5DUrj3jhE02_HCx9cJLYsaLIvPwJVrfKb7aYvaULHhRw4rLaHtxo4r0B7bKLTGkkgkWUmXVv7yDViYdk48ClVic05ZpfjUmpDo9TcD1s7mUCOcoLdqJh7U8UXbBqGBMHpVD95MOFpwAyTC9mnvDuiIiWVxRr8buuC_QnjRITs4CPYhrFG_-5zXC5B4KDUG23AcbCwK7IkSW4PKnufdJHlgKq3g9nNJUcvgE_mB10rGX2YGzJYpIFIegR6HjzR9UuWLIvm7cxOxWTjUSd-8lzFPNFfWyGWgXsyvdtEVa9mkAKJ63A12oFLIou9Gj1L3wMFXBTFsY3n8EB29h.vwq3hDsu0b8kXlx_-yqfkQ/__results___files/__results___50_0.png'>

# ## OpenAIGPT-Confusion Matrix

# In[ ]:


y_predict=model.predict(valid, verbose=1)
y_predict[ y_predict> 0.5] = 1
y_predict[y_predict <= 0.5] = 0
plot_cm(y_valid, y_predict, 'OpenAIGPT-Confusion Matrix')


# <img src='https://www.kaggleusercontent.com/kf/36129650/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..K5ST4JjFGBMeTtwX1jC9XQ.0KxxSju7AC0c3tRP2w-rTijGPiYJGNZnjThGQx-BG3govElMkJpIJw6gcpAozeKQSH_Q3_Cjh6a38SnweVyY3X6eW1YMl8UvsJNUsQnuKGTTig69_hUGyoPmepANHm4vZnwUN3vljN9RmyRVaXuvHTkgGCs6KASszIFNgE1v309GpSOtQy7Um8_75fzG-szb7mNmMBlkZ_vBs4-4f-Wmf_8JwkQit6kgDM9qXzE2pEq0btxta2CuVyCYUkJ-qV6aCQprqSy68UTOjD6tdwlASMi9hOLyeogQrSoHLzPWgf_Xg3Pm6dwwjrFwDw2MpFoT47szD5MaXTLoT6cxUAdIRZVNMHed7HsC9r-PDwIdDiZKy8Sss5DUrj3jhE02_HCx9cJLYsaLIvPwJVrfKb7aYvaULHhRw4rLaHtxo4r0B7bKLTGkkgkWUmXVv7yDViYdk48ClVic05ZpfjUmpDo9TcD1s7mUCOcoLdqJh7U8UXbBqGBMHpVD95MOFpwAyTC9mnvDuiIiWVxRr8buuC_QnjRITs4CPYhrFG_-5zXC5B4KDUG23AcbCwK7IkSW4PKnufdJHlgKq3g9nNJUcvgE_mB10rGX2YGzJYpIFIegR6HjzR9UuWLIvm7cxOxWTjUSd-8lzFPNFfWyGWgXsyvdtEVa9mkAKJ63A12oFLIou9Gj1L3wMFXBTFsY3n8EB29h.vwq3hDsu0b8kXlx_-yqfkQ/__results___files/__results___52_1.png'>

# ## OpenAIGPT-ROC

# In[ ]:


y_predict_prob=model.predict(valid, verbose=1)
fpr, tpr, _ = roc_curve(y_valid,y_predict_prob)
roc_auc = auc(fpr, tpr)
roc_curve_plot(fpr,tpr,roc_auc)


# # 3-DistilBert 

# ![](https://jalammar.github.io/images/distilBERT/bert-distilbert-tutorial-sentence-embedding.png)

# ## Usefull Links About DistilBERT
# <p1> The following links are quite useful if you to further your knowledge about Transformer-XL</p1>
# <ui>
#     <li>[distilbert](https://medium.com/huggingface/distilbert-8cf3380435b5) </li>
#     <li>[a-visual-guide-to-using-dsitillbert-for-the-first-time](http://jalammar.github.io/a-visual-guide-to-using-bert-for-the-first-time/) </li> 
#     <li>[Orginal Paper](https://arxiv.org/pdf/1910.01108.pdf)</li>
# </ui>

# In[ ]:


# # First load the real tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased')


# In[ ]:


X_train=np.array(single_encoding_function(train_raw['comment_text'],tokenizer,'DistilBert'))
y_train=np.array(train_raw['toxic'])
X_valid=np.array(single_encoding_function(valid_raw['comment_text'],tokenizer,'DistilBert'))
y_valid=np.array(valid_raw['toxic'])
X_test=np.array(single_encoding_function(test_raw['content'],tokenizer,'DistilBert'))


# In[ ]:


train,valid,test=make_data()


# In[ ]:


steps_per_epoch = X_train.shape[0] // BATCH_SIZE


# In[ ]:



model=compile_model('distilbert-base-cased')
print(model.summary())

history=model.fit(
    train,steps_per_epoch=steps_per_epoch,
    epochs=EPOCHS,callbacks=[early_stopping], validation_data=valid
)


# ## DistilBert Performance-Loss

# In[ ]:


plot_loss(history)


# ## DistilBert  XL Performance-Metrics

# In[ ]:


plot_metrics(history)


# ## DistilBert  Performance-Confusion Matrix

# In[ ]:


y_predict=model.predict(valid, verbose=1)
y_predict[ y_predict> 0.5] = 1
y_predict[y_predict <= 0.5] = 0
plot_cm(y_valid, y_predict, 'Transformer XL Performance-Confusion Matrix')


# ## DistilBert  Performace-ROC

# In[ ]:


y_predict_prob=model.predict(valid, verbose=1)
fpr, tpr, _ = roc_curve(y_valid,y_predict_prob)
roc_auc = auc(fpr, tpr)
roc_curve_plot(fpr,tpr,roc_auc)


# # 4-XLM

# ![](https://i.pinimg.com/564x/88/66/9b/88669b6455bde00c747fd9535e7ee98e.jpg)

# ## Usefull Links About XLM
# <p1> The following links are quite useful if you to further your knowledge about XLM</p1>
# <ui>
#     <li>[xlm-enhancing-bert-for-cross-lingual-language-model](https://towardsdatascience.com/xlm-enhancing-bert-for-cross-lingual-language-model-5aeed9e6f14b#:~:text=Background,(or%20sub%2Dwords).)</li>
#     <li>[GitHub XLM](https://github.com/facebookresearch/XLM) </li>
#     <li>[a-deep-dive-into-multilingual-nlp-models](https://peltarion.com/blog/data-science/a-deep-dive-into-multilingual-nlp-models)</li>
#     <li>[Orginal Paper](a-deep-dive-into-multilingual-nlp-models)</li>
# </ui>

# In[ ]:


# # First load the real tokenizer
tokenizer = XLMTokenizer.from_pretrained('xlm-mlm-en-2048')


# In[ ]:


X_train=np.array(single_encoding_function(train_raw['comment_text'],tokenizer,'XLM'))
y_train=np.array(train_raw['toxic'])
X_valid=np.array(single_encoding_function(valid_raw['comment_text'],tokenizer,'XLM'))
y_valid=np.array(valid_raw['toxic'])
X_test=np.array(single_encoding_function(test_raw['content'],tokenizer,'XLM'))


# In[ ]:


train,valid,test=make_data()


# In[ ]:



model=compile_model('xlm-mlm-en-2048')
print(model.summary())

history=model.fit(
    train,steps_per_epoch=steps_per_epoch,
    epochs=EPOCHS,callbacks=[early_stopping], validation_data=valid
)


# ## XLM Performance-Loss

# In[ ]:


plot_loss(history)


# ## XLM Performance-Metrics

# In[ ]:


plot_metrics(history)


# ## XLM-Confusion Matrix

# In[ ]:


y_predict=model.predict(valid, verbose=1)
y_predict[ y_predict> 0.5] = 1
y_predict[y_predict <= 0.5] = 0
plot_cm(y_valid, y_predict, 'XLM-Confusion Matrix')


# ## XLM-ROC Curve

# In[ ]:


y_predict_prob=model.predict(valid, verbose=1)
fpr, tpr, _ = roc_curve(y_valid,y_predict_prob)
roc_auc = auc(fpr, tpr)
roc_curve_plot(fpr,tpr,roc_auc)


# # 5-XLM-Roberta-large -the Choosen one :D

# <img src='https://www.etcentric.org/wp-content/uploads/2018/07/FAIR_Facebook_AI_Research.jpg'>

# ## Usefull Links About XLM-Roberta-Large
# <p1> The following links are quite useful if you to further your knowledge about XLM-Roberta-Large</p1>
# <ui>
#     <li>[xlm-roberta-the-multilingual-alternative](https://towardsdatascience.com/xlm-roberta-the-multilingual-alternative-for-non-english-nlp-cf0b889ccbbf) </li>
#     <li>[Github-XLMR](https://github.com/pytorch/fairseq/tree/master/examples/xlmr) </li>
#     <li>[bert-roberta-distilbert-xlnet-one-us](https://www.kdnuggets.com/2019/09/bert-roberta-distilbert-xlnet-one-use.html)</li>
#     <li>[Explaining Roberta Technique](https://arxiv.org/pdf/1907.11692.pdf)</li>
#     <li>[FaceBook AI post](https://ai.facebook.com/blog/-xlm-r-state-of-the-art-cross-lingual-understanding-through-self-supervision/)
#     <li>[Orginal Paper](https://arxiv.org/pdf/1911.02116.pdf)</li>
# </ui>

# ## Youtube Video-Explaining XLM-Roberta-Large!!!-Really Useful

# In[ ]:


from IPython.display import YouTubeVideo

YouTubeVideo("Ot6A3UFY72c", width=800, height=300)



# In[ ]:


def regular_encode(texts, tokenizer, maxlen=512):
    enc_di = tokenizer.batch_encode_plus(
        texts, 
        return_attention_masks=False, 
        return_token_type_ids=False,
        pad_to_max_length=True,
        max_length=maxlen
    )
    
    return np.array(enc_di['input_ids'])


# In[ ]:


tokenizer = AutoTokenizer.from_pretrained('jplu/tf-xlm-roberta-large')


# In[ ]:


get_ipython().run_cell_magic('time', '', '\nX_train = regular_encode(train_raw.comment_text.values, tokenizer, maxlen=max_seq_length)\nX_valid = regular_encode(valid_raw.comment_text.values, tokenizer, maxlen=max_seq_length)\nX_test = regular_encode(test_raw.content.values, tokenizer, maxlen=max_seq_length)\n\ny_train = train_raw.toxic.values\ny_valid = valid_raw.toxic.values')


# In[ ]:


steps_per_epoch = X_train.shape[0] // BATCH_SIZE


# In[ ]:


train,valid,test=make_data()


# In[ ]:



final_model=compile_model('jplu/tf-xlm-roberta-large')
print(final_model.summary())

history=final_model.fit(
    train,steps_per_epoch=steps_per_epoch,
    epochs=EPOCHS,callbacks=[early_stopping], validation_data=valid
)


# ## XLM-Roberta-large-Loss

# In[ ]:


plot_loss(history)


# ## XLM-Roberta-large-Metrics

# In[ ]:


plot_metrics(history)


# ## XLM-Roberta-large-Confusion Matrix

# In[ ]:


y_predict=final_model.predict(valid, verbose=1)
y_predict[ y_predict> 0.5] = 1
y_predict[y_predict <= 0.5] = 0
plot_cm(y_valid, y_predict, 'XLM-Roberta-Confusion Matrix')


# ## XLM-Roberta-large-ROC Curve

# In[ ]:


y_predict_prob=final_model.predict(valid, verbose=1)
fpr, tpr, _ = roc_curve(y_valid,y_predict_prob)
roc_auc = auc(fpr, tpr)
roc_curve_plot(fpr,tpr,roc_auc)


# In[ ]:


steps_per_epoch = X_valid.shape[0] // BATCH_SIZE
train_history_2 = final_model.fit(
    valid.repeat(),
    steps_per_epoch=steps_per_epoch,
    epochs=EPOCHS
)


# # 6-**Glovec Embedding LSTM**

# ![](https://user-images.githubusercontent.com/10358317/43802664-22c08c8a-9a9f-11e8-83e1-fea4bf334f6e.png)

# ## Usefull Links About LSTM
# <p1> The following links are quite useful if you to further your knowledge about LSTMs</p1>
# <ui>
#     <li>[Understanding-LSTMs](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)</li>
#     <li>[illustrated-guide-to-lstms-and-gru-s-a-step-by-step-explanation](https://towardsdatascience.com/illustrated-guide-to-lstms-and-gru-s-a-step-by-step-explanation-44e9eb85bf21)</li>
#     <li>[Andrew Ng Explaining LSTM](https://www.coursera.org/lecture/nlp-sequence-models/long-short-term-memory-lstm-KXoay)</li>
#     <li>[Orginal Paper](https://www.bioinf.jku.at/publications/older/2604.pdf)</li>
# </ui>
# </p1>
# 
# <p2> The following links are quite useful if you want to understand about Glovec </p2>
# <ui>
#     <li>[CS224n Notes](http://web.stanford.edu/class/cs224n/readings/cs224n-2019-notes02-wordvecs2.pdf)</li>
#     <li>[Anrew Ng Explaining Glovec](https://www.coursera.org/lecture/nlp-sequence-models/glove-word-vectors-IxDTG)</li>
#     <li>[CS22n Video](https://www.youtube.com/watch?v=ASn7ExxLZws&t=1s)</li>
#     <li>[Orginal Paper](http://nlp.stanford.edu/pubs/glove.pdf)</li>
# </ui>
# </p2>

# ## Hyperparameter for LSTMs 

# In[ ]:


max_seq_length = 512
embedding_dim=300
BATCH_SIZE = 32
EPOCHS=1
LEARNING_RATE=1e-5
early_stopping=early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', 
    verbose=1,
    patience=2,
    mode='max',
    restore_best_weights=True)
AUTO = tf.data.experimental.AUTOTUNE


# ## DataPipeline for LSTMs

# In[ ]:



tokenizer = Tokenizer(split=' ', oov_token='<unw>', filters=' ')
tokenizer.fit_on_texts(train_raw['comment_text'].values)

# this takes our sentences and replaces each word with an integer
X_train = tokenizer.texts_to_sequences(train_raw['comment_text'].values)
X_train=np.array(pad_sequences(X_train, max_seq_length))
y_train=np.array(train_raw['toxic'])

X_valid = tokenizer.texts_to_sequences(valid_raw['comment_text'].values)
X_valid = np.array(pad_sequences(X_valid, max_seq_length))
y_valid=np.array(valid_raw['toxic'])

X_test = tokenizer.texts_to_sequences(test_raw['content'].values)
X_test = np.array(pad_sequences(X_test, max_seq_length))


# In[ ]:


print('The X_train,X_valid,X_test shape respectively is {}-{}-{}'.format(X_train.shape,X_valid.shape,X_test.shape))


# In[ ]:


train,valid,test=make_data()


# ## Glovec Embedding

# In[ ]:


get_ipython().system('wget http://nlp.stanford.edu/data/glove.6B.zip')
get_ipython().system('unzip glove*.zip')


# In[ ]:


embeddings_index = {}

f = open(os.path.join(os.getcwd(), 'glove.6B.{}d.txt'.format(str(embedding_dim))))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))


# In[ ]:


# first create a matrix of zeros, this is our embedding matrix
word_index = tokenizer.word_index
num_words=len(word_index)+1
embedding_matrix = np.zeros((num_words, embedding_dim))

# for each word in out tokenizer lets try to find that work in our w2v model
for word, i in tqdm(word_index.items()):
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # we found the word - add that words vector to the matrix
        embedding_matrix[i] = embedding_vector
    else:
        # doesn't exist, assign a random vector
        embedding_matrix[i] = np.random.randn(embedding_dim)


# In[ ]:


steps_per_epoch=X_train.shape[0]//BATCH_SIZE
with strategy.scope():
    model = Sequential()

    model.add(Embedding(num_words,
                        embedding_dim,
                        embeddings_initializer=Constant(embedding_matrix),
                        input_length=max_seq_length,
                        trainable=True))
    model.add(SpatialDropout1D(0.2))
    model.add(Bidirectional(LSTM(64, return_sequences=True)))
    model.add(Bidirectional(LSTM(32)))
    model.add(Dropout(0.25))
    model.add(Dense(units=1, activation='sigmoid'))

    METRICS = [
              tf.keras.metrics.TruePositives(name='tp'),
              tf.keras.metrics.FalsePositives(name='fp'),
              tf.keras.metrics.TrueNegatives(name='tn'),
              tf.keras.metrics.FalseNegatives(name='fn'), 
              tf.keras.metrics.BinaryAccuracy(name='accuracy'),
              tf.keras.metrics.Precision(name='precision'),
              tf.keras.metrics.Recall(name='recall'),
              tf.keras.metrics.AUC(name='auc')]
    model.compile(loss ='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE),metrics =METRICS)

history=model.fit(
train,steps_per_epoch=steps_per_epoch,epochs=EPOCHS,callbacks=[early_stopping], validation_data=valid)


# ## LSTM with Glovec Embedding Loss

# In[ ]:


plot_loss(history)


# ## LSTM with Glovec Embedding-Metrics

# In[ ]:


plot_metrics(history)


# ## LSTM with Glovec Embedding-Confusion Matrix

# In[ ]:


y_predict=model.predict(valid, verbose=1)
y_predict[y_predict> 0.5] = 1
y_predict[y_predict <= 0.5] = 0
plot_cm(y_valid, y_predict, 'LSTM with Glovec-Confusion Matrix')


# ## LSTM with Glovec Embedding-ROC

# In[ ]:


y_predict_prob=model.predict(valid, verbose=1)
fpr, tpr, _ = roc_curve(y_valid,y_predict_prob)
roc_auc = auc(fpr, tpr)
roc_curve_plot(fpr,tpr,roc_auc)


# # 7-**LSTM with attention Mechanism**

# ![](https://www.mdpi.com/sensors/sensors-19-00861/article_deploy/html/images/sensors-19-00861-g004.png)

# ## Usefull Links About LSTM with Attention
# <p1> The following links are quite useful if you to further your knowledge about LSTMs</p1>
# <ui>
#     <li>[comprehensive-guide-attention-mechanism-deep-learning](https://www.analyticsvidhya.com/blog/2019/11/comprehensive-guide-attention-mechanism-deep-learning/)</li>
#     <li>[attention-long-short-term-memory-recurrent-neural-networks](https://machinelearningmastery.com/attention-long-short-term-memory-recurrent-neural-networks/)</li>
#     <li>[Andrew Ng Explaining LSTM with Attention](https://www.youtube.com/watch?v=SysgYptB198)</li>
#     <li>[Orginal Paper](https://www.aclweb.org/anthology/D16-1058.pdf)</li>
# </ui>

# In[ ]:


with strategy.scope():
    # Define an input sequence and process it.
    inputs = Input(shape=(max_seq_length,))
    embedding=Embedding(num_words,
                        embedding_dim,
                        embeddings_initializer=Constant(embedding_matrix),
                        input_length=max_seq_length,
                        trainable=True)(inputs)
    

    # Apply dropout to prevent overfitting
    embedded_inputs = Dropout(0.2)(embedding)
    
    # Apply Bidirectional LSTM over embedded inputs
    lstm_outs =Bidirectional(
        LSTM(300, return_sequences=True)
    )(embedded_inputs)
    
    # Apply dropout to LSTM outputs to prevent overfitting
    lstm_outs = Dropout(0.2)(lstm_outs)
    
    # Attention Mechanism - Generate attention vectors
    attention_vector = TimeDistributed(Dense(1))(lstm_outs)
    attention_vector = Reshape((X_train.shape[1],))(attention_vector)
    attention_vector = Activation('softmax', name='attention_vec')(attention_vector)
    attention_output = Dot(axes=1)([lstm_outs, attention_vector])
    
    # Last layer: fully connected with softmax activation
    fc = Dense(300, activation='relu')(attention_output)
    output = Dense(1, activation='sigmoid')(fc)
    
    model=tf.keras.Model(inputs,output)
    
    
    METRICS = [
              tf.keras.metrics.TruePositives(name='tp'),
              tf.keras.metrics.FalsePositives(name='fp'),
              tf.keras.metrics.TrueNegatives(name='tn'),
              tf.keras.metrics.FalseNegatives(name='fn'), 
              tf.keras.metrics.BinaryAccuracy(name='accuracy'),
              tf.keras.metrics.Precision(name='precision'),
              tf.keras.metrics.Recall(name='recall'),
              tf.keras.metrics.AUC(name='auc')]
    model.compile(loss ='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE),metrics =METRICS)
    print(model.summary())

history=model.fit(
train,steps_per_epoch=steps_per_epoch,epochs=EPOCHS,callbacks=[early_stopping], validation_data=valid)


# ## LSTM with Attention Mechanism -Loss

# In[ ]:


plot_loss(history)


# ## LSTM with Attention Mechanism-Metrics

# In[ ]:


plot_metrics(history)


# ## LSTM with Attention Mechanism-Confusion Matrix

# In[ ]:


y_predict=model.predict(valid, verbose=1)
y_predict[ y_predict> 0.5] = 1
y_predict[y_predict <= 0.5] = 0
plot_cm(y_valid, y_predict, 'LSTM with Attention Mechanism-Confusion Matrix')


# ## LSTM with Attention Mechanism-ROC

# In[ ]:


fy_predict_prob=model.predict(valid, verbose=1)
fpr, tpr, _ = roc_curve(y_valid,y_predict_prob)
roc_auc = auc(fpr, tpr)
roc_curve_plot(fpr,tpr,roc_auc)


# # Final Submission

# In[ ]:


sub['toxic'] = final_model.predict(test, verbose=1)
sub.to_csv('submission.csv', index=False)


# In[ ]:




