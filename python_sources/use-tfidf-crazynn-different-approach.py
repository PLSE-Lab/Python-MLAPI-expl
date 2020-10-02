#!/usr/bin/env python
# coding: utf-8

# ### Congratulations to Everyone!
# 
# This kernel does not give a comparable score to the BERT models; but I tried to make a different use of USE model by providing question title as "context" to the **"question-response"** architecture of **USE** while exctracting embeddings.
# 
# Also I played a little with NN architecture instead of using a plain FC network. 
# 
# USE Embedding extraction and NN were my own castles in this sandbox and had fun creating this. Even though it is not a LB jumper, it jumped above other USE and TfIdf models. Love getting creative in this platform and wanted to share.
# 
# The kernel scored better on public LB (0.363) when compared to direct use(not using the qa model but just treating t,q,a of our data as separate sentences.) of USE for extracting embeddings and plain FC-NN.
# 
# ****
# After obtaining embeddings for title-question-answer trio: 
# - I Obtained TfIdf vectors and reduced them with TSVD
# - Calculated different similarity measures between the trio.
# 
# After merging these features, they were fed to an NN where: 
# - Two NNs take care of USE and TfIdf vectors.
# - Title, question, answers are fed to 3 seperate dense layers.
# - Their outputs are concatenated as **"title-question"** and **"question-answer-similaritymeasures"** and send to two seperate dense hidden layers.
# - Question labels are the outputs of the first hidden layer.
# - Answer labels are the outputs of the second hidden layer.
# - Question-Answer labels are concatenated
# - Outputs of USE-NN and TfIdf NN are averaged in a layer.
# 
# 
# 
# For the rest I thank to the kernels below.
# https://www.kaggle.com/abhishek/distilbert-use-features-oof
# https://www.kaggle.com/abazdyrev/use-features-oof

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


from collections import OrderedDict,defaultdict
import scipy.spatial.distance as sci_dist
from scipy.stats import spearmanr,rankdata

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow.keras.backend as K
import tensorflow.keras as keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Dropout, Lambda, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback

from sklearn.model_selection import KFold
from sklearn.linear_model import MultiTaskElasticNet

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


# In[ ]:


sub = pd.read_csv('../input/google-quest-challenge/sample_submission.csv')
train = pd.read_csv('../input/google-quest-challenge/train.csv')
test = pd.read_csv('../input/google-quest-challenge/test.csv')
full = pd.concat([train,test],axis=0)
train_size = len(train)


# In[ ]:


module_url = "../input/universalsentenceencodermodels/universal-sentence-encoder-models/use-qa"
large_module_url = "../input/universalsentenceencodermodels/universal-sentence-encoder-models/use-large"
module = hub.load(module_url)
large_module = hub.load(large_module_url)


# In[ ]:


labels = ['question_asker_intent_understanding',
       'question_body_critical', 'question_conversational',
       'question_expect_short_answer', 'question_fact_seeking',
       'question_has_commonly_accepted_answer',
       'question_interestingness_others', 'question_interestingness_self',
       'question_multi_intent', 'question_not_really_a_question',
       'question_opinion_seeking', 'question_type_choice',
       'question_type_compare', 'question_type_consequence',
       'question_type_definition', 'question_type_entity',
       'question_type_instructions', 'question_type_procedure',
       'question_type_reason_explanation', 'question_type_spelling',
       'question_well_written', 'answer_helpful',
       'answer_level_of_information', 'answer_plausible', 'answer_relevance',
       'answer_satisfaction', 'answer_type_instructions',
       'answer_type_procedure', 'answer_type_reason_explanation',
       'answer_well_written']

text_features = ['question_title','question_body','answer']


# In[ ]:


question_labels = ['question_asker_intent_understanding',
       'question_body_critical', 'question_conversational',
       'question_expect_short_answer', 'question_fact_seeking',
       'question_has_commonly_accepted_answer',
       'question_interestingness_others', 'question_interestingness_self',
       'question_multi_intent', 'question_not_really_a_question',
       'question_opinion_seeking', 'question_type_choice',
       'question_type_compare', 'question_type_consequence',
       'question_type_definition', 'question_type_entity',
       'question_type_instructions', 'question_type_procedure',
       'question_type_reason_explanation', 'question_type_spelling',
       'question_well_written']

answer_labels = ['answer_helpful',
       'answer_level_of_information', 'answer_plausible', 'answer_relevance',
       'answer_satisfaction', 'answer_type_instructions',
       'answer_type_procedure', 'answer_type_reason_explanation',
       'answer_well_written']


# In[ ]:


texts = full[text_features]


# - Use tensorflow dataset elements for feature extraction part of the pipeline

# In[ ]:


contexts = tf.data.Dataset.from_tensor_slices(texts.question_title.values.tolist())
questions = tf.data.Dataset.from_tensor_slices(texts.question_body.values.tolist())
answers = tf.data.Dataset.from_tensor_slices(texts.answer.values.tolist())
tf_text_data = tf.data.Dataset.zip((contexts,questions,answers))
text_batches = tf_text_data.batch(32)


# - Extract embeddings batch by batch for resource mgmt.

# In[ ]:


embeddings = defaultdict(list)
for i,batch in enumerate(text_batches):
    print(f'Extracting embeddings from batch {i+1}...')
    
    context = tf.constant(batch[0])
    questions = tf.constant(batch[1])
    answers = tf.constant(batch[2])
    
    embeddings['title_embs'].append(large_module(context).numpy())
    embeddings['question_embs'].append(module.signatures['question_encoder'](questions)['outputs'].numpy())
    embeddings['answer_embs'].append(module.signatures['response_encoder'](input=answers,context=context)['outputs'].numpy())
    
                        


# In[ ]:


question_embeddings = np.vstack(embeddings['question_embs'])
answer_embeddings = np.vstack(embeddings['answer_embs'])
title_embeddings = np.vstack(embeddings['title_embs'])


# ### TF-IDF

# In[ ]:


tfidf_title = TfidfVectorizer(ngram_range=(1,3))
tfidf_question = TfidfVectorizer(ngram_range=(1,3))
tfidf_answer = TfidfVectorizer(ngram_range=(1,3))


# In[ ]:


tsvd_title = TruncatedSVD(n_components=100)
tsvd_question = TruncatedSVD(n_components=100)
tsvd_answer = TruncatedSVD(n_components=100)


# In[ ]:


train_texts, test_texts = texts[:train_size], texts[train_size:]


# In[ ]:


title_vec_train = tfidf_title.fit_transform(train_texts.question_title)
title_vec_test = tfidf_title.transform(test_texts.question_title)

question_vec_train = tfidf_question.fit_transform(train_texts.question_body)
question_vec_test = tfidf_question.transform(test_texts.question_body)

answer_vec_train = tfidf_answer.fit_transform(train_texts.answer)
answer_vec_test = tfidf_answer.transform(test_texts.answer)


# In[ ]:


title_train = tsvd_title.fit_transform(title_vec_train)
title_test = tsvd_title.transform(title_vec_test)

question_train = tsvd_question.fit_transform(question_vec_train)
question_test = tsvd_question.transform(question_vec_test)                         

answer_train = tsvd_answer.fit_transform(answer_vec_train)
answer_test = tsvd_answer.transform(answer_vec_test)            


# In[ ]:


train_x = np.hstack([title_train,question_train,answer_train])
test_x = np.hstack([title_test,question_test,answer_test])


# ### Distance and Similarity Features

# In[ ]:


qa_train_similarity_matrix = np.inner(question_embeddings[:train_size],answer_embeddings[:train_size])
qa_test_similarity_matrix = np.inner(question_embeddings[train_size:],answer_embeddings[train_size:])

qt_train_similarity_matrix = np.inner(question_embeddings[:train_size],title_embeddings[:train_size])
qt_test_similarity_matrix = np.inner(question_embeddings[train_size:],title_embeddings[train_size:])

ta_train_similarity_matrix = np.inner(title_embeddings[:train_size],answer_embeddings[:train_size])
ta_test_similarity_matrix = np.inner(title_embeddings[train_size:],answer_embeddings[train_size:])


# In[ ]:


qa_train_sim_score = np.diag(qa_train_similarity_matrix)
qa_test_sim_score = np.diag(qa_test_similarity_matrix)

qt_train_sim_score = np.diag(qt_train_similarity_matrix)
qt_test_sim_score = np.diag(qt_test_similarity_matrix)

ta_train_sim_score = np.diag(ta_train_similarity_matrix)
ta_test_sim_score = np.diag(ta_test_similarity_matrix)


# In[ ]:


l2_dist = lambda x, y: np.linalg.norm(x-y,axis=1)
cos_dist = lambda x, y: 1 - np.sum(x*y,axis=1)


# In[ ]:


#Question Answer Pairs
qa_l2_dist_train = l2_dist(question_embeddings[:train_size],answer_embeddings[:train_size])
qa_l2_dist_test = l2_dist(question_embeddings[train_size:],answer_embeddings[train_size:])

qa_cos_dist_train = cos_dist(question_embeddings[:train_size],answer_embeddings[:train_size])
qa_cos_dist_test = cos_dist(question_embeddings[train_size:],answer_embeddings[train_size:])

#Question Title Pairs
qt_l2_dist_train = l2_dist(question_embeddings[:train_size],title_embeddings[:train_size])
qt_l2_dist_test = l2_dist(question_embeddings[train_size:],title_embeddings[train_size:])

qt_cos_dist_train = cos_dist(question_embeddings[:train_size],title_embeddings[:train_size])
qt_cos_dist_test = cos_dist(question_embeddings[train_size:],title_embeddings[train_size:])

#Title Answer Pairs
ta_l2_dist_train = l2_dist(title_embeddings[:train_size],answer_embeddings[:train_size])
ta_l2_dist_test = l2_dist(title_embeddings[train_size:],answer_embeddings[train_size:])

ta_cos_dist_train = cos_dist(title_embeddings[:train_size],answer_embeddings[:train_size])
ta_cos_dist_test = cos_dist(title_embeddings[train_size:],answer_embeddings[train_size:])


# ### Merge Features

# In[ ]:


X_train = np.hstack([title_embeddings[:train_size],question_embeddings[:train_size],answer_embeddings[:train_size],
                         qa_train_sim_score.reshape(-1,1),qt_train_sim_score.reshape(-1,1),ta_train_sim_score.reshape(-1,1),
                         qa_l2_dist_train.reshape(-1,1),qa_cos_dist_train.reshape(-1,1),
                         qt_l2_dist_train.reshape(-1,1),qt_cos_dist_train.reshape(-1,1),
                         ta_l2_dist_train.reshape(-1,1),ta_cos_dist_train.reshape(-1,1),
                         train_x
                    ])

X_test = np.hstack([title_embeddings[train_size:],question_embeddings[train_size:],answer_embeddings[train_size:],
                         qa_test_sim_score.reshape(-1,1),qt_test_sim_score.reshape(-1,1),ta_test_sim_score.reshape(-1,1),
                         qa_l2_dist_test.reshape(-1,1),qa_cos_dist_test.reshape(-1,1),
                         qt_l2_dist_test.reshape(-1,1),qt_cos_dist_test.reshape(-1,1),
                         ta_l2_dist_test.reshape(-1,1),ta_cos_dist_test.reshape(-1,1),
                         test_x
                   ])


# In[ ]:


y_train = train[labels].values


# In[ ]:


X_train.shape,y_train.shape,X_test.shape


# ### TF Model

# In[ ]:


class SpearmanRhoCallback(Callback):
    def __init__(self, training_data, validation_data, patience, model_name):
        self.x = training_data[0]
        self.y = training_data[1]
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]
        
        self.patience = patience
        self.value = -1
        self.bad_epochs = 0
        self.model_name = model_name

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        y_pred_val = self.model.predict(self.x_val)
        rho_val = np.mean([spearmanr(self.y_val[:, ind], y_pred_val[:, ind] + np.random.normal(0, 1e-7, y_pred_val.shape[0])).correlation for ind in range(y_pred_val.shape[1])])
        if rho_val >= self.value:
            self.value = rho_val
        else:
            self.bad_epochs += 1
        if self.bad_epochs >= self.patience:
            print("Epoch %05d: early stopping Threshold" % epoch)
            self.model.stop_training = True
            #self.model.save_weights(self.model_name)
        print('\rval_spearman-rho: %s' % (str(round(rho_val, 4))), end=100*' '+'\n')
        return rho_val

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return


# In[ ]:


def crazy_model():    
    inp = Input(shape=(X_train.shape[1],))

    title_input = inp[:,:512]
    question_input = inp[:,512:1024]
    answer_input = inp[:,1024:1536]
    
    similarity_input = inp[:,1536:1545]
    
    tfidf_title_input = inp[:,1545:1645]
    tfidf_question_input = inp[:,1645:1745]
    tfidf_answer_input = inp[:,1745:]
    
    #USE Model
    title_hidden = Dense(256, activation='relu')(title_input)
    title_hidden = Dropout(0.2)(title_hidden)
    question_hidden = Dense(256, activation='relu')(question_input)
    question_hidden = Dropout(0.2)(question_hidden)
    answer_hidden = Dense(256, activation='relu')(answer_input)
    answer_hidden = Dropout(0.2)(answer_hidden)
    
    title_out = Dense(128, activation='relu')(title_hidden)
    question_out = Dense(128, activation='relu')(question_hidden)
    answer_out = Dense(128, activation='relu')(answer_hidden)
    
    qa_pair_vector = Concatenate(axis=1)([question_out,answer_out,similarity_input])
    qa_hidden = Dense(128,activation='relu')(qa_pair_vector)
    qa_out = Dense(len(answer_labels),activation='sigmoid')(Concatenate(axis=1)([qa_hidden,answer_hidden]))

    qt_pair_vector = Concatenate(axis=1)([title_out,question_out])
    qt_hidden = Dense(128,activation='relu')(qt_pair_vector)
    qt_out = Dense(len(question_labels),activation='sigmoid')(Concatenate(axis=1)([qt_hidden,question_hidden]))
    
    use_output = Concatenate(axis=1)([qt_out,qa_out])
    
    #TF-IDF Model
    tfidf_title_hidden = Dense(256, activation='relu')(tfidf_title_input)
    tfidf_title_hidden = Dropout(0.2)(tfidf_title_hidden)
    tfidf_question_hidden = Dense(256, activation='relu')(tfidf_question_input)
    tfidf_question_hidden = Dropout(0.2)(tfidf_question_hidden)
    tfidf_answer_hidden = Dense(256, activation='relu')(tfidf_answer_input)
    tfidf_title_hidden = Dropout(0.2)(tfidf_title_hidden)
    
    tfidf_title_out = Dense(128, activation='relu')(tfidf_title_hidden)
    tfidf_question_out = Dense(128, activation='relu')(tfidf_question_hidden)
    tfidf_answer_out = Dense(128, activation='relu')(tfidf_answer_hidden)
    
    tfidf_qa_pair_vector = Concatenate(axis=1)([tfidf_question_out,tfidf_answer_out,similarity_input])
    tfidf_qa_hidden = Dense(128,activation='relu')(tfidf_qa_pair_vector)
    tfidf_qa_out = Dense(len(answer_labels),activation='sigmoid')(Concatenate(axis=1)([tfidf_qa_hidden,tfidf_answer_hidden]))

    tfidf_qt_pair_vector = Concatenate(axis=1)([tfidf_title_out,tfidf_question_out])
    tfidf_qt_hidden = Dense(128,activation='relu')(tfidf_qt_pair_vector)
    tfidf_qt_out = Dense(len(question_labels),activation='sigmoid')(Concatenate(axis=1)([tfidf_qt_hidden,tfidf_question_hidden]))
    
    tfidf_output = Concatenate(axis=1)([tfidf_qt_out,tfidf_qa_out])
    
    
    output = keras.layers.Average()([use_output,tfidf_output])
    
    model = Model(inputs=inp, outputs=output)
    
    model.compile(
        optimizer=Adam(lr=1e-4),
        loss=['binary_crossentropy']
    )
    print(similarity_input.shape)
    model.summary()
    return model
    


# In[ ]:


all_predictions = []

kf = KFold(n_splits=5, random_state=42, shuffle=True)
for ind, (tr, val) in enumerate(kf.split(X_train)):
    X_tr = X_train[tr]
    y_tr = y_train[tr]
    X_vl = X_train[val]
    y_vl = y_train[val]
    
    model = crazy_model()
    model.fit(
        X_tr, y_tr, epochs=100, batch_size=32, validation_data=(X_vl, y_vl), verbose=True, 
        callbacks=[SpearmanRhoCallback(training_data=(X_tr, y_tr), validation_data=(X_vl, y_vl),
                                       patience=5, model_name=f'best_model_batch{ind}.h5')]
    )
    all_predictions.append(model.predict(X_test))
    
model = crazy_model()
model.fit(X_train, y_train, epochs=33, batch_size=32, verbose=False)
all_predictions.append(model.predict(X_test))
    
kf = KFold(n_splits=5, random_state=2019, shuffle=True)
for ind, (tr, val) in enumerate(kf.split(X_train)):
    X_tr = X_train[tr]
    y_tr = y_train[tr]
    X_vl = X_train[val]
    y_vl = y_train[val]
    
    model = MultiTaskElasticNet(alpha=0.001, random_state=42, l1_ratio=0.5)
    model.fit(X_tr, y_tr)
    all_predictions.append(model.predict(X_test))
    
model = MultiTaskElasticNet(alpha=0.001, random_state=42, l1_ratio=0.5)
model.fit(X_train, y_train)
all_predictions.append(model.predict(X_test))


# In[ ]:


test_preds = np.array([np.array([rankdata(c) for c in p.T]).T for p in all_predictions]).mean(axis=0)
max_val = test_preds.max() + 1
test_preds = test_preds/max_val + 1e-12


# In[ ]:


sub[labels] = test_preds


# In[ ]:


sub.to_csv('submission.csv',index=False)


# In[ ]:


sub.head()


# In[ ]:




