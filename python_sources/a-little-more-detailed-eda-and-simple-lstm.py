#!/usr/bin/env python
# coding: utf-8

# *Disclaimer: The Visualizations and dataframes in this kernel may be considered profane, vulgar, or offensive.*

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns
import gc

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


# change default plot config
plt.rc('figure', figsize=(14.4, 8.1), dpi=72)
plt.rc('font', size=13)


# In[ ]:


train_df = pd.read_csv("../input/train.csv", parse_dates=['created_date'])


# In[ ]:


train_df.info()
display(train_df.head())
display(train_df.describe())


# In[ ]:


# Subgroups
toxicity_subtypes = [
    'severe_toxicity', 'obscene', 'identity_attack',
    'insult', 'threat', 'sexual_explicit'
]

identities = [
    'asian', 'atheist', 'bisexual',
    'black', 'buddhist', 'christian', 'female', 'heterosexual', 'hindu',
    'homosexual_gay_or_lesbian', 'intellectual_or_learning_disability',
    'jewish', 'latino', 'male', 'muslim', 'other_disability',
    'other_gender', 'other_race_or_ethnicity', 'other_religion',
    'other_sexual_orientation', 'physical_disability',
    'psychiatric_or_mental_illness', 'transgender', 'white'
]

metadata = [
    'created_date', 'publication_id', 'parent_id', 'article_id',
    'rating', 'funny', 'wow', 'sad', 'likes', 'disagree'
]

annotation = ['identity_annotator_count', 'toxicity_annotator_count']


# In[ ]:


# Feture Engineering for visualization
train_df['rating'] = train_df['rating'].map({"approved": 1, "rejected": 0})
train_df["is_toxic"] = np.where(train_df["target"].values >= 0.5, 1, 0).astype("int32")


# ## Distributions

# In[ ]:


for col in train_df.columns:
    if col in ["rating", "is_toxic"]:
        sns.countplot(train_df[col])
    elif train_df[col].dtype.name in ["float64", "int64"] and col not in ["id", "comment_text", "article_id", "parent_id", "publication_id"]:
        sns.distplot(train_df.loc[train_df[col].notna() & train_df["is_toxic"].eq(0), col], label="is_toxic=0")
        sns.distplot(train_df.loc[train_df[col].notna() & train_df["is_toxic"].eq(1), col], label="is_toxic=1")
        plt.legend()
    else:
        continue
    plt.title(f"Distribution of `{col}` in train")
    plt.show()


# ## Explore frequent words

# In[ ]:


import unicodedata
import sys
from nltk.corpus import stopwords
from wordcloud import WordCloud

puncts_trans = {i: ' ' for i in range(sys.maxunicode)
                if unicodedata.category(chr(i)).startswith('P')}

del puncts_trans[ord("'")]

puncts = [chr(i) for i in puncts_trans.keys()]
# print("Puncts:", puncts)

stop_words = stopwords.words('english')
other_frequent_words = ["people", "don", "doesn", "didn", "can",
                        "could", "like", "would", "one", "get",
                        't', 's', 'i', 'you']


def freqs_plot_incond(df, cond=''):
    new_comment_text = df["comment_text"].str.translate(puncts_trans).str.lower()
    tokenized = ' '.join(new_comment_text.values.tolist()).split()
    tokenized = [word for word in tokenized if word not in stop_words + other_frequent_words]
    s = pd.Series(tokenized)
    del tokenized
    gc.collect()
    freq = s.value_counts().to_dict()
    del s
    gc.collect()
    if len(freq) == 0:
        print(f"The {cond} has no words")
        return
    wc = WordCloud(width=800, height=450)
    pic_mat = wc.fit_words(freq).to_array()
    plt.imshow(pic_mat)
    plt.title(f"Frequent words in condition of {cond}")
    plt.show()
    del pic_mat, wc
    gc.collect()


# ### Frequent words in identities mentioned

# In[ ]:


for col in identities:
    freqs_plot_incond(train_df.loc[train_df[col].gt(0.2)], f"mentioned identity `{col}` > 0.2")


# ### Frequent words in toxicity subtypes

# In[ ]:


for col in toxicity_subtypes:
    freqs_plot_incond(train_df.loc[train_df[col].gt(0.2)], f"subtype `{col}` > 0.2")


# In[ ]:


del train_df
gc.collect()


# ## Prepare datasets for training

# In[ ]:


train_df = pd.read_csv("../input/train.csv", usecols=identities+toxicity_subtypes+["id", "target", "comment_text"])
test_df = pd.read_csv("../input/test.csv")
train_df.info()
test_df.info()


# In[ ]:


train_df["is_toxic"] = np.where(train_df["target"].values >= 0.5, 1, 0).astype("int32")


# In[ ]:


get_ipython().run_cell_magic('time', '', 'from tensorflow.keras.preprocessing.text import Tokenizer\nfrom tensorflow.keras.preprocessing.sequence import pad_sequences\n\nSIZEOF_VOCAB = 49999  # size of vocabulary\nINPUT_LENGTH = 192  # max length for each sequence\n\n# fit on text in all the datasets\n# text_to_fit = pd.concat([train_df["comment_text"], test_df["comment_text"]])\n\n# fit on text in condition of:\ntext_to_fit = train_df.loc[((train_df["target"]>0.3)\n                             &(train_df[toxicity_subtypes].gt(0.2).any(axis=1)))\n                           |((train_df["target"]>0.3)\n                             &(train_df[identities].gt(0.2).any(axis=1))), "comment_text"]\n\ntokenizer = Tokenizer(num_words=SIZEOF_VOCAB,\n                      filters=\'\'.join(puncts) + \'\\n\\t\\r\',\n                     )\ntokenizer.fit_on_texts(text_to_fit)\nprint(len(tokenizer.word_index))\n\nword_counts = pd.Series(dict(tokenizer.word_counts.items())).sort_values(ascending=False)\nfig, ax = plt.subplots(figsize=(14.4, 10.8))\nsns.barplot(x=word_counts[:50], y=word_counts[:50].index, ax=ax)\nax.set_title("Top frequent words in tokenizer")\nplt.show()\n\ntrain_text_seq = tokenizer.texts_to_sequences(train_df["comment_text"])\ntest_text_seq = tokenizer.texts_to_sequences(test_df["comment_text"])\n\n# Find out the lengths of words in each sequence\nfor seq, title in zip([train_text_seq, test_text_seq], ["Train", "Test"]):\n    s = pd.Series([len(x) for x in seq])\n    sns.boxplot(s)\n    plt.title(f"Distribution of number of words in each comment in {title}")\n    plt.show()\n\ntrain_features = pad_sequences(train_text_seq, maxlen=INPUT_LENGTH).astype("int32")\ntest_features = pad_sequences(test_text_seq, maxlen=INPUT_LENGTH).astype("int32")\n\ntrn_istoxic = train_df["is_toxic"].values\ntrn_aux_target = train_df["target"].values\ntrn_subtypes = train_df[toxicity_subtypes].values\n\ngc.collect()')


# ## Build and train models

# In[ ]:


from tensorflow.keras import Model
import tensorflow.keras.layers as L
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.regularizers import l1_l2

from sklearn.metrics import roc_auc_score, confusion_matrix

EMBEDDING_SIZE = 512
BATCH_SIZE = 512
RECURRENT_UNITS = 128
LR = 0.005
reg_l1 = 0.3
reg_l2 = 0.5


def model_fn():
    
    inp = L.Input(shape=(INPUT_LENGTH,))
    emb = L.Embedding(input_dim=SIZEOF_VOCAB+1, output_dim=EMBEDDING_SIZE,
                      input_length=INPUT_LENGTH,
                      trainable=True)(inp)
    
    drop_0 = L.SpatialDropout1D(rate=0.15)(emb)
    bi_lstm_0 = L.Bidirectional(L.CuDNNLSTM(RECURRENT_UNITS,
                                            recurrent_regularizer=l1_l2(l1=reg_l1, l2=reg_l2),
                                            return_sequences=True))(drop_0)
    bi_lstm_1 = L.Bidirectional(L.CuDNNLSTM(RECURRENT_UNITS,
                                            recurrent_regularizer=l1_l2(l1=reg_l1, l2=reg_l2),
                                            return_sequences=False))(bi_lstm_0)

    out_main = L.Dense(1, activation='sigmoid', name="main_proba")(bi_lstm_1)
    out_aux = L.Dense(1, activation='sigmoid', name="aux_proba")(bi_lstm_1)
    out_subtypes_probas = L.Dense(6, activation='sigmoid', name="subtypes_proba")(bi_lstm_1)
    
    model = Model(inputs=inp, outputs=[out_main, out_aux, out_subtypes_probas])
    model.compile(Adam(lr=LR), loss=binary_crossentropy, metrics=['acc'])

    return model


# In[ ]:


get_ipython().run_cell_magic('time', '', '\nmodel = model_fn()\nmodel.summary()\n\nhist = model.fit(\n    train_features,\n    [trn_istoxic, trn_aux_target, trn_subtypes],\n    batch_size=BATCH_SIZE,\n    epochs=3,\n    callbacks=[\n     LearningRateScheduler(lambda epoch: max([LR-0.3*LR*epoch, 0.001]), verbose=1),\n    ],\n    validation_split=0.1, shuffle=True,\n)\n\ntrn_pred, trn_aux, _ = model.predict(train_features)\n\ntest_pred, test_aux, sub_pred = model.predict(test_features)')


# ## Validation on trainset

# In[ ]:


gc.collect()

# AUC
print("Train's istoxic,main AUC: {}".format(roc_auc_score(trn_istoxic, trn_pred)))
print("Train's istoxic,aux AUC: {}".format(roc_auc_score(trn_istoxic, trn_aux)))

# Plot Confusion Matrices
main_cm = confusion_matrix(trn_istoxic, np.where(trn_pred>=0.5, 1, 0))
sns.heatmap(main_cm, annot=True)
plt.xlabel("Actual classes")
plt.ylabel("Predicted classes")
plt.title("Confusion Matrix in Train")
plt.show()


# ## Submission

# In[ ]:


submission = pd.DataFrame({
    "id": test_df["id"],
    "prediction": test_pred.flatten(),
})

submission.to_csv("submission.csv", index=False)


# ## Exploration on predicted testset

# In[ ]:


subtypes = pd.DataFrame(data=sub_pred, columns=toxicity_subtypes)
submission[toxicity_subtypes] = subtypes[toxicity_subtypes]
submission["target"] = test_aux.flatten()
submission["comment_text"] = test_df["comment_text"]


# In[ ]:


# Find out the distribution of subtypes
for col in toxicity_subtypes+["target"]:
    sns.distplot(submission[col])
    plt.title(f"Distribution of `{col}` in test")
    plt.show()


# In[ ]:


for col in toxicity_subtypes+["target"]:
    freqs_plot_incond(submission.loc[submission[col].gt(0.2)], f"`{col}` > 0.2")


# ## Next step:
# 
# * Explore on extra datasets containing preset embedding weights.
