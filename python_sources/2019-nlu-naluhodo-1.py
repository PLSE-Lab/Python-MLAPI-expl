#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
get_ipython().run_line_magic('matplotlib', 'inline')


# ---
# 
# # Loading models
# 
# 
# We build our models using sklearn and Keras.

# In[ ]:


cd ../input/nlunaluhodo


# In[ ]:


from load_embeddings import DataEmbeddings, BertEmbeddings, SoftAuxEmbeddings, HardAuxEmbeddings
from BiLSTM import BiLstm
from LSTM import Lstm
from Bert_LSTM import Bert_Lstm
from BOW_LSTM import Bow_Lstm
from BOW_Logistic import Bow_Logistic
from Soft_Aux import Soft_Aux
from Soft_Main import Soft_Main
from Hard_Main import Hard_Main
from evaluation import pred_classes_dt, pred_classes_f1, plot_roc_curve, plot_confusion_matrix


# # Loading Data
# 
# The data comes from [Berkeley Enron Email Analysis Project](http://bailando.sims.berkeley.edu/enron_email.html). It is a multilabel classification problems about Emails. Here we use the first main classification of the emails and try to build a model to classify emails into 6 classes:
# 
# 1.1 Company Business, Strategy, etc.<br/>
# 1.2 Purely Personal<br/>
# 1.3 Personal but in professional context (e.g., it was good working with you)<br/>
# 1.4 Logistic Arrangements (meeting scheduling, technical support, etc)<br/>
# 1.5 Employment arrangements (job seeking, hiring, recommendations, etc)<br/>
# 1.6 Document editing/checking (collaboration)
# 
# data_index is a boostrap index to upsample our data, it is used as increasing weight in multitask learning.

# In[ ]:


cd ../../working


# In[ ]:


data_index = pickle.load(open('../input/upsamplekfoldindex/data_index.p', 'rb'))
beeap_1 = pd.read_csv('../input/beeapfinal/beeap_1.csv')
beeap_1.drop('Unnamed: 0', axis=1, inplace=True)


# ---
# 
# # Models
# 
# ## BOW Logistic; BOW LSTM; GloVe LSTM; GloVe BiLSTM

# In[ ]:


dataset = DataEmbeddings(label_column = [str(i+1) for i in range(6)])
X_train, y_train, X_valid, y_valid, X_test, y_test, embedding_layer = dataset.load_data_embeddings(beeap_1, '../input/glove-6b/glove.6B.300d.txt')

bowlr = Bow_Logistic()
bowlr.fit(X_train, y_train, X_valid, y_valid)
bowlr_pred = bowlr.predict(X_test)
    
bowlstm = Bow_Lstm()
bowlstm.fit(X_train, y_train, X_valid, y_valid)
bowlstm_pred = bowlstm.predict(X_test, batch_size = bowlstm.arguments['batch_size'], verbose = 1)
    
lstm = Lstm()
lstm.fit(X_train, y_train, X_valid, y_valid, embedding_layer)
lstm_pred = lstm.predict(X_test, batch_size = lstm.arguments['batch_size'], verbose = 1)
    
bilstm = BiLstm()
bilstm.fit(X_train, y_train, X_valid, y_valid, embedding_layer)
bilstm_pred = bilstm.predict(X_test, batch_size = bilstm.arguments['batch_size'], verbose = 1)


# ## GloVe Embedding Customization BiLSTM
# 
# We trained our own word embeddings using the algorithm provided with GloVe. We hope that this embedding can learn some information for words that appears only in this corpus like[](http://) names and email addresses.

# In[ ]:


dataset = DataEmbeddings(label_column = [str(i+1) for i in range(6)])
X_train_ec, y_train_ec, X_valid_ec, y_valid_ec, X_test_ec, y_test_ec, embedding_layer_ec = dataset.load_data_embeddings(beeap_1, '../input/glove-ec/glove_ec/GloVe_ec.300B.txt')
bilstm_ec = BiLstm()
bilstm_ec.fit(X_train_ec, y_train_ec, X_valid_ec, y_valid_ec, embedding_layer_ec)
bilstm_ec_pred = bilstm_ec.predict(X_test_ec, batch_size = bilstm_ec.arguments['batch_size'], verbose = 1)


# ## BiLSTM with Embeddings from Bert
# 
# We extract the weight of the embedding layer from Bert and see if it performs well with a BiLSTM model.**

# In[ ]:


bert_dataset = BertEmbeddings()
X_train_bert, y_train_bert, X_valid_bert, y_valid_bert, X_test_bert, y_test_bert = bert_dataset.load_bert_embeddings('../input/bertembeddingpickle/beeap_1_bert_ec_768d/beeap_1_bert_ec_768D.p')

bertlstm = Bert_Lstm()
bertlstm.arguments['learning_rate'] = 3e-4
bertlstm.arguments['units'] = 64
bertlstm.arguments['drop_out_rate'] = 0.2
bertlstm.fit(X_train_bert, y_train_bert, X_valid_bert, y_valid_bert)
bertlstm_pred = bertlstm.predict(X_test_bert, batch_size = bertlstm.arguments['batch_size'], verbose = 1)


# ---
# 
# # Multitask Learning
# 
# ## Loading Auxiliary Data
# 
# Emails in the Enron corpus has manually labeled folders by users, we want to predict the right folder with our model to see if it can provide us additional information or understanding about the dataset.

# In[ ]:


from sklearn.preprocessing import LabelEncoder

emails_w_labels = []
for i in open('../input/multitask-labels/multitask_emails/emails_w_labels'):
    emails_w_labels += [i.split(', ')]
emails_w_labels = pd.DataFrame(emails_w_labels, columns=['Y', 'contents'])

labelEncoder=LabelEncoder()
emails_w_labels['labels'] = labelEncoder.fit_transform(emails_w_labels['Y'])

emails_w_labels = emails_w_labels[emails_w_labels['labels'].isin(  (emails_w_labels['labels'].value_counts()>30)                                                 [:(emails_w_labels['labels'].value_counts()>30) .value_counts()[1]].index  )]


# In[ ]:


from sklearn.preprocessing import LabelBinarizer

emails_w_labels[emails_w_labels['labels'].isin(  (emails_w_labels['labels'].value_counts()>30)                                                 [:(emails_w_labels['labels'].value_counts()>30) .value_counts()[1]].index  )]

labelBinarizer = LabelBinarizer()
label_column = emails_w_labels['labels'].unique()
df = pd.DataFrame(labelBinarizer.fit_transform(emails_w_labels['labels']), columns = label_column)
df['contents'] = emails_w_labels['contents'].values


# ---
# 
# ## Soft Multitask Model
# 
# We build two identical models for both the main task(the 6 class classification that we focused on) and the auxiliary task and make restrictions on some layers to force them to be similar.
# 
# For now, we only achieve the "strongest" restriction, that the main task has exactly the same layers as the auxiliary and is only able to fit using the last output layer.

# In[ ]:


soft_aux_dataset = SoftAuxEmbeddings(content_column='contents', label_column=emails_w_labels['labels'].unique())
X_train_aux, y_train_aux, X_valid_aux, y_valid_aux, X_test_aux, y_test_aux, embedding_layer_aux = soft_aux_dataset.load_soft_aux_embeddings(df, '../input/glove-6b/glove.6B.300d.txt')


# In[ ]:


bilstm_aux = Soft_Aux()
bilstm_aux.fit(X_train_aux, y_train_aux, X_valid_aux, y_valid_aux, embedding_layer_aux)


# In[ ]:


dataset = DataEmbeddings(data_index, label_column=[str(i+1) for i in range(6)])
X_train_up, y_train_up, X_valid_up, y_valid_up, X_test_up, y_test_up, embedding_layer_up = dataset.load_data_embeddings(beeap_1, '../input/glove-6b/glove.6B.300d.txt')


# In[ ]:


soft_multitask_model = Soft_Main()
soft_multitask_model.fit(X_train_up, y_train_up, X_valid_up, y_valid_up, embedding_layer_up, Soft_Aux=bilstm_aux)
soft_multitask_pred = soft_multitask_model.predict(X_test_up, batch_size = soft_multitask_model.arguments['batch_size'], verbose = 1)


# ---
# 
# ## Hard Multitask Model
# 
# We combine the dataset and let our model to learn from both of them simultaneously. Since emails from one dataset don't have labels from the other, we need a mask on loss function for evaluation.

# In[ ]:


hard_aux_dataset = HardAuxEmbeddings(data_index, label_column = [str(i+1) for i in range(6)], label_column_aux = emails_w_labels['labels'].unique())
X_train_hard, y_train_hard, X_valid_hard, y_valid_hard, X_test_hard, y_test_hard, embedding_layer_hard = hard_aux_dataset.load_hard_aux_embeddings(beeap_1, df, '../input/glove-6b/glove.6B.300d.txt')


# In[ ]:


hard_multitask_model = Hard_Main()
hard_multitask_model.fit(X_train_hard, y_train_hard, X_valid_hard, y_valid_hard, embedding_layer_hard)
hard_multitask_pred = hard_multitask_model.predict(X_test_hard, batch_size = hard_multitask_model.arguments['batch_size'], verbose = 1)[:X_test_hard.shape[0]-15133, :6]


# ---
# 
# # Results

# Load in the prediction results for fine tuned Bert model.

# In[ ]:


import pickle

bert_result = pickle.load(open('../input/bert-result/bert_result (2).p', 'rb'))


# In[ ]:


from sklearn.metrics import f1_score

macro_F1 = []
micro_F1 = []

bowlr_y_pred_f1 = pred_classes_f1(bowlr.predict(X_train), y_train, bowlr_pred)
macro_F1 += [f1_score(y_test, bowlr_y_pred_f1, average='macro')]
micro_F1 += [f1_score(y_test, bowlr_y_pred_f1, average='micro')]
print('Bow Logistic macro F1 score: {0:0.4f}'.format(macro_F1[0]))
print('Bow Logistic micro F1 score: {0:0.4f}\n'.format(micro_F1[0]))

bowlstm_y_pred_f1 = pred_classes_f1(bowlstm.predict(X_train, batch_size = bowlstm.arguments['batch_size'], verbose = 1), y_train, bowlstm_pred)
macro_F1 += [f1_score(y_test, bowlstm_y_pred_f1, average='macro')]
micro_F1 += [f1_score(y_test, bowlstm_y_pred_f1, average='micro')]
print('Bow LSTM macro F1 score: {0:0.4f}'.format(macro_F1[1]))
print('Bow LSTM micro F1 score: {0:0.4f}\n'.format(micro_F1[1]))

lstm_y_pred_f1 = pred_classes_f1(lstm.predict(X_train, batch_size = lstm.arguments['batch_size'], verbose = 1), y_train, lstm_pred)
macro_F1 += [f1_score(y_test, lstm_y_pred_f1, average='macro')]
micro_F1 += [f1_score(y_test, lstm_y_pred_f1, average='micro')]
print('LSTM macro F1 score: {0:0.4f}'.format(macro_F1[2]))
print('LSTM micro F1 score: {0:0.4f}\n'.format(micro_F1[2]))

bilstm_y_pred_f1 = pred_classes_f1(bilstm.predict(X_train, batch_size = bilstm.arguments['batch_size'], verbose = 1), y_train, bilstm_pred)
macro_F1 += [f1_score(y_test, bilstm_y_pred_f1, average='macro')]
micro_F1 += [f1_score(y_test, bilstm_y_pred_f1, average='micro')]
print('BiLSTM macro F1 score: {0:0.4f}'.format(macro_F1[3]))
print('BiLSTM micro F1 score: {0:0.4f}'.format(micro_F1[3]))


# In[ ]:


bilstm_ec_y_pred_f1 = pred_classes_f1(bilstm_ec.predict(X_train, batch_size = 1024, verbose = 1), y_train, bilstm_ec_pred)
macro_F1 += [f1_score(y_test, bilstm_ec_y_pred_f1, average='macro')]
micro_F1 += [f1_score(y_test, bilstm_ec_y_pred_f1, average='micro')]
print('BiLSTM E.C. macro F1 score: {0:0.4f}'.format(macro_F1[4]))
print('BiLSTM E.C. micro F1 score: {0:0.4f}'.format(micro_F1[4]))


# In[ ]:


bertlstm_y_pred_f1 = pred_classes_f1(bertlstm.predict(X_train_bert, batch_size = 1024, verbose = 1), y_train_bert, bertlstm_pred)
macro_F1 += [f1_score(y_test_bert, bertlstm_y_pred_f1, average='macro')]
micro_F1 += [f1_score(y_test_bert, bertlstm_y_pred_f1, average='micro')]
print('Bert BiLSTM macro F1 score: {0:0.4f}'.format(macro_F1[5]))
print('Bert BiLSTM micro F1 score: {0:0.4f}'.format(micro_F1[5]))


# In[ ]:


soft_multitask_y_pred_f1 = pred_classes_f1(soft_multitask_model.predict(X_train_up, batch_size = soft_multitask_model.arguments['batch_size'], verbose = 1), y_train_up, soft_multitask_pred)
macro_F1 += [f1_score(y_test_up, soft_multitask_y_pred_f1, average='macro')]
micro_F1 += [f1_score(y_test_up, soft_multitask_y_pred_f1, average='micro')]
print('Multitask macro F1 score: {0:0.4f}'.format(macro_F1[6]))
print('Multitask micro F1 score: {0:0.4f}'.format(micro_F1[6]))


# In[ ]:


hard_multitask_y_pred_f1 = pred_classes_f1(hard_multitask_model.predict(X_train_hard, batch_size = hard_multitask_model.arguments['batch_size'], verbose = 1)[:y_train_hard.shape[0]-52963, :6], y_train_hard[:y_train_hard.shape[0]-52963, :6], hard_multitask_pred)
macro_F1 += [f1_score(y_test_hard[:y_test_hard.shape[0]-15133, :6], hard_multitask_y_pred_f1, average='macro')]
micro_F1 += [f1_score(y_test_hard[:y_test_hard.shape[0]-15133, :6], hard_multitask_y_pred_f1, average='micro')]
print('Multitask macro F1 score: {0:0.4f}'.format(macro_F1[7]))
print('Multitask micro F1 score: {0:0.4f}'.format(micro_F1[7]))


# In[ ]:


bert_y_pred_f1 = pred_classes_f1(bert_result['y_train_pred'], bert_result['y_train'], bert_result['y_test_pred'])
macro_F1 += [f1_score(bert_result['y_test'], bert_y_pred_f1, average='macro')]
micro_F1 += [f1_score(bert_result['y_test'], bert_y_pred_f1, average='micro')]
print('Bert macro F1 score: {0:0.4f}'.format(macro_F1[8]))
print('Bert micro F1 score: {0:0.4f}'.format(micro_F1[8]))


# In[ ]:


plt.figure(figsize=(18, 6))
plt.plot(range(9), macro_F1, linewidth=2, marker='D', label='macro F1 score')
plt.plot(range(9), micro_F1, linewidth=2, marker='D', label='micro F1 score')
plt.grid(axis='y')
plt.legend(loc=4, fontsize=11)
plt.xticks(range(9), ['Logistic\n Regression', 'Bag of Word\n LSTM', 'GloVe\n LSTM', 'GloVe\n BiLSTM', 'GloVe E.C.\n BiLSTM', 'Soft_Multitask\n BiLSTM', 'Hard_Multitask\n BiLSTM', 'Bert embedding\n BiLSTM', 'Bert'], fontsize=13)
plt.yticks(fontsize=13)
plt.title('F1 Scores', fontsize=16)
plt.show()


# We can see that the Bert model has the highest score, and our customized embedding also has a nice performance.
# 
# Still need to find out why Bert Embedding BiSLTM model has a bad F1 score with a good AUC score.

# In[ ]:


plot_roc_curve(y_test, bowlr_pred, title='ROC Curves for Bow Logistic', micro=True, per_class=True)


# In[ ]:


plot_roc_curve(y_test, bowlstm_pred, title='ROC Curves for Bow LSTM', micro=True, per_class=True)


# In[ ]:


plot_roc_curve(y_test, lstm_pred, title='ROC Curves for LSTM', micro=True, per_class=True)


# In[ ]:


plot_roc_curve(y_test, bilstm_pred, title='ROC Curves for BiLSTM', micro=True, per_class=True)


# In[ ]:


plot_roc_curve(y_test, bilstm_ec_pred, title='ROC Curves for BiLSTM E.C.', micro=True, per_class=True)


# In[ ]:


plot_roc_curve(y_test_up, soft_multitask_pred, title='ROC Curves for Soft Multitask', micro=True, per_class=True)


# In[ ]:


plot_roc_curve(y_test_hard[:y_test_hard.shape[0]-15133, :6], hard_multitask_pred, title='ROC Curves for Hard Multitask', micro=True, per_class=True)


# In[ ]:


plot_roc_curve(y_test_bert, bertlstm_pred, title='ROC Curves for Bert BiLSTM', micro=True, per_class=True)


# In[ ]:


plot_roc_curve(bert_result['y_test'], bert_result['y_test_pred'], title='ROC Curves for Bert', micro=True, per_class=True)


# In[ ]:


from sklearn.metrics import roc_auc_score

macro_auc = []
micro_auc = []

macro_auc += [roc_auc_score(y_test, bowlr_pred)]
macro_auc += [roc_auc_score(y_test, bowlstm_pred)]
macro_auc += [roc_auc_score(y_test, lstm_pred)]
macro_auc += [roc_auc_score(y_test, bilstm_pred)]
macro_auc += [roc_auc_score(y_test, bilstm_ec_pred)]
macro_auc += [roc_auc_score(y_test_up, soft_multitask_pred)]
macro_auc += [roc_auc_score(y_test_hard[:y_test_hard.shape[0]-15133, :6], hard_multitask_pred)]
macro_auc += [roc_auc_score(y_test_bert, bertlstm_pred)]
macro_auc += [roc_auc_score(bert_result['y_test'], bert_result['y_test_pred'])]

micro_auc += [roc_auc_score(y_test, bowlr_pred, average='micro')]
micro_auc += [roc_auc_score(y_test, bowlstm_pred, average='micro')]
micro_auc += [roc_auc_score(y_test, lstm_pred, average='micro')]
micro_auc += [roc_auc_score(y_test, bilstm_pred, average='micro')]
micro_auc += [roc_auc_score(y_test, bilstm_ec_pred, average='micro')]
micro_auc += [roc_auc_score(y_test_up, soft_multitask_pred, average='micro')]
micro_auc += [roc_auc_score(y_test_hard[:y_test_hard.shape[0]-15133, :6], hard_multitask_pred, average='micro')]
micro_auc += [roc_auc_score(y_test_bert, bertlstm_pred, average='micro')]
micro_auc += [roc_auc_score(bert_result['y_test'], bert_result['y_test_pred'], average='micro')]


# In[ ]:


from sklearn.metrics import roc_curve, auc
from itertools import cycle
from scipy import interp

def roc_data(y_test, y_pred):
    fpr = {}
    tpr = {}
    roc_auc = {}
    for i in range(6):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    fpr['micro'], tpr['micro'], _ = roc_curve(y_test.ravel(), y_pred.ravel())
    roc_auc['micro'] = auc(fpr['micro'], tpr['micro'])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(6)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(6):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= 6

    fpr['macro'] = all_fpr
    tpr['macro'] = mean_tpr
    roc_auc['macro'] = auc(fpr['macro'], tpr['macro'])
    return [fpr['macro'], tpr['macro'], roc_auc['macro'], fpr['micro'], tpr['micro'], roc_auc['micro']]


# In[ ]:


bowlr_roc = roc_data(y_test, bowlr_pred)
bowlstm_roc = roc_data(y_test, bowlstm_pred)
lstm_roc = roc_data(y_test, lstm_pred)
bilstm_roc = roc_data(y_test, bilstm_pred)
bilstm_ec_roc = roc_data(y_test, bilstm_ec_pred)
soft_multitask_roc = roc_data(y_test_up, soft_multitask_pred)
hard_multitask_roc = roc_data(y_test_hard[:y_test_hard.shape[0]-15133, :6], hard_multitask_pred)
bertlstm_roc = roc_data(y_test_bert, bertlstm_pred)
bert_roc = roc_data(bert_result['y_test'], bert_result['y_test_pred'])


# In[ ]:


plt.figure(figsize=(10, 10))
plt.plot(bowlr_roc[0], bowlr_roc[1], label='macro ROC Curve Logistic (AUC = {0:0.4f})'.format(bowlr_roc[2]), alpha=0.8, linewidth=2, c='C8')
plt.plot(bowlstm_roc[0], bowlstm_roc[1], label='macro ROC Curve Bow LSTM (AUC = {0:0.4f})'.format(bowlstm_roc[2]), alpha=0.8, linewidth=2, c='C7')
plt.plot(lstm_roc[0], lstm_roc[1], label='macro ROC Curve GloVe LSTM (AUC = {0:0.4f})'.format(lstm_roc[2]), alpha=0.8, linewidth=2, c='C6')
plt.plot(bilstm_roc[0], bilstm_roc[1], label='macro ROC Curve GloVe BiLSTM (AUC = {0:0.4f})'.format(bilstm_roc[2]), alpha=0.8, linewidth=2, c='C5')
plt.plot(bilstm_ec_roc[0], bilstm_ec_roc[1], label='macro ROC Curve GloVe E.C. BiLSTM (AUC = {0:0.4f})'.format(bilstm_ec_roc[2]), alpha=0.8, linewidth=2, c='C4')
plt.plot(soft_multitask_roc[0], soft_multitask_roc[1], label='macro ROC Curve Soft_Multitask (AUC = {0:0.4f})'.format(soft_multitask_roc[2]), alpha=0.8, linewidth=2, c='C3')
plt.plot(hard_multitask_roc[0], hard_multitask_roc[1], label='macro ROC Curve Hard_Multitask (AUC = {0:0.4f})'.format(hard_multitask_roc[2]), alpha=0.8, linewidth=2, c='C2')
plt.plot(bertlstm_roc[0], bertlstm_roc[1], label='macro ROC Curve Bert LSTM (AUC = {0:0.4f})'.format(bertlstm_roc[2]), alpha=0.8, linewidth=2, c='C1')
plt.plot(bert_roc[0], bert_roc[1], label='macro ROC Curve Bert (AUC = {0:0.4f})'.format(bert_roc[2]), alpha=0.8, linewidth=2, c='C0')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xticks(fontsize=13)
plt.xticks(fontsize=13)
plt.xlabel('False Positive Rate', fontsize=16)
plt.ylabel('True Positive Rate', fontsize=16)
plt.legend(loc=4)
plt.title('ROC Curves', fontsize=16)
plt.show()


# In[ ]:


plt.figure(figsize=(10, 10))
plt.plot(bowlr_roc[3], bowlr_roc[4], label='micro ROC Curve Logistic (AUC = {0:0.4f})'.format(bowlr_roc[5]), alpha=0.8, linewidth=2, c='C8')
plt.plot(bowlstm_roc[3], bowlstm_roc[4], label='micro ROC Curve Bow LSTM (AUC = {0:0.4f})'.format(bowlstm_roc[5]), alpha=0.8, linewidth=2, c='C7')
plt.plot(lstm_roc[3], lstm_roc[4], label='micro ROC Curve GloVe LSTM (AUC = {0:0.4f})'.format(lstm_roc[5]), alpha=0.8, linewidth=2, c='C6')
plt.plot(bilstm_roc[3], bilstm_roc[4], label='micro ROC Curve GloVe BiLSTM (AUC = {0:0.4f})'.format(bilstm_roc[5]), alpha=0.8, linewidth=2, c='C5')
plt.plot(bilstm_ec_roc[3], bilstm_ec_roc[4], label='micro ROC Curve GloVe E.C. BiLSTM (AUC = {0:0.4f})'.format(bilstm_ec_roc[5]), alpha=0.8, linewidth=2, c='C4')
plt.plot(soft_multitask_roc[3], soft_multitask_roc[4], label='micro ROC Curve Soft_Multitask (AUC = {0:0.4f})'.format(soft_multitask_roc[5]), alpha=0.8, linewidth=2, c='C3')
plt.plot(hard_multitask_roc[3], hard_multitask_roc[4], label='micro ROC Curve Hard_Multitask (AUC = {0:0.4f})'.format(hard_multitask_roc[5]), alpha=0.8, linewidth=2, c='C2')
plt.plot(bertlstm_roc[3], bertlstm_roc[4], label='micro ROC Curve Bert BiLSTM (AUC = {0:0.4f})'.format(bertlstm_roc[5]), alpha=0.8, linewidth=2, c='C1')
plt.plot(bert_roc[3], bert_roc[4], label='micro ROC Curve Bert (AUC = {0:0.4f})'.format(bert_roc[5]), alpha=0.8, linewidth=2, c='C0')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xticks(fontsize=13)
plt.xticks(fontsize=13)
plt.xlabel('False Positive Rate', fontsize=16)
plt.ylabel('True Positive Rate', fontsize=16)
plt.legend(loc=4)
plt.title('ROC Curves', fontsize=16)
plt.show()


# Here the best model is BiLSTM with Bert Embedding, and our customized embedding also performs very nice.
# 
# The Bert model does not have an outstanding AUC score. Maybe our dataset is too small, or maybe we need to understand more about Bert to take full use of it.
