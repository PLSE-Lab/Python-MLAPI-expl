#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
import os
sys.path.append("..") 

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


# In[ ]:


import warnings
import pandas as pd
import numpy as np
from glob import glob


# In[ ]:


from frankner.preprocessing import ContextNER
from frankner.model import BiLSTM
from frankner.metrics import F1Metrics
from frankner.metrics import all_metrics
from frankner.metrics import all_metrics_fold
from frankner.utils import build_matrix_embeddings


# In[ ]:


from tcc_utils import save_best_model
from tcc_utils import save_stats_to_disk
from tcc_utils import pip_predict_metric


# In[ ]:


from seqeval.metrics import f1_score
from seqeval.metrics import recall_score
from seqeval.metrics import precision_score


# In[ ]:


warnings.filterwarnings('ignore')


# In[ ]:


train = pd.read_csv('../data/tcc/TRAIN_AUGMENTATION.csv')
test = pd.read_csv('../data/tcc/TEST.csv')


# In[ ]:


#building vocab
all_words_vocab = train.Word.to_list() + test.Word.to_list()


# In[ ]:


ner_train = ContextNER(train, all_words_vocab)
ner_test = ContextNER(test, all_words_vocab, max_len=ner_train.max_len)


# <h1 align="center">
#     <img alt="Frank NER" src="https://raw.githubusercontent.com/mpgxc/experimental.Frankner/master/.github/logo.png" width="350px" />
# </h1>

# In[ ]:


twitter_wb = list(map(lambda x: x if x.find('twitter') != -1 else '', glob('../Embeddings/GloVe/*')))


# In[ ]:


twitter_wb


# ### Loading the selected Word Embeddings

# In[ ]:


get_ipython().run_cell_magic('time', '', "\nglove_tweet = \\\nbuild_matrix_embeddings(path='../Embeddings/GloVe/glove.twitter.27B.200d.txt',\n                        num_tokens=ner_train.num_words, \n                        embedding_dim=200, \n                        word_index=ner_train.word2idx)\n\n# glove = \\\nbuild_matrix_embeddings(path='../Embeddings/GloVe/glove.840B.300d.txt',\n                        num_tokens=ner_train.num_words, \n                        embedding_dim=300, \n                        word_index=ner_train.word2idx)\n\nfasttext = \\\nbuild_matrix_embeddings(path='../Embeddings/FastText/crawl-300d-2M.vec',\n                        num_tokens=ner_train.num_words, \n                        embedding_dim=300, \n                        word_index=ner_train.word2idx)")


# ### Concatenate/ Combinate the all Word Embeddings

# In[ ]:


wb_concatenate = np.concatenate([glove, fasttext, glove_tweet], axis=1)


# In[ ]:


wb_concatenate.shape


# In[ ]:


BATCH_SIZE = 16
EPOCHS = 30
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2
N_MODELS = 5


# In[ ]:


get_ipython().run_cell_magic('time', '', "\nAll_Historys = {}\n\nfor index_model in range(N_MODELS):\n    \n    file_name_model = '{}_model_crf_only_twitter_word_embeddings'.format(index_model + 1)\n    \n    print('\\nModel [{}] Training...'.format(1 + index_model))\n    \n    model = \\\n    BiLSTM(isa_crf=True,\n           words_weights= wb_concatenate,\n           pre_trained=True,\n           max_len=ner_train.max_len,\n           num_words=ner_train.num_words,\n           num_tags=ner_train.num_tags,\n           learning_rate=LEARNING_RATE,\n           dropout=0.5)\n    \n    All_Historys[file_name_model] = \\\n    model.fit(ner_train.X_array, \n              ner_train.y_array, \n              batch_size=BATCH_SIZE, \n              epochs=EPOCHS, \n              validation_split=VALIDATION_SPLIT,\n              callbacks=[F1Metrics(ner_train.idx2tag), \n                         save_best_model('models/', file_name_model)])\n    \n    print('-' * 50)\n    \n    pip_predict_metric(ner_train, \n                       ner_test, \n                       model,\n                       path_folder='models/',\n                       name_model=file_name_model)")


# ### Instantiating the model

# In[ ]:


save_stats_to_disk(All_Historys, path_folder='plots/', file_name='8 - Modelo Final + Tweet Embeddings')


# ### Evaluating metrics average and standard deviation

# In[ ]:


get_ipython().run_cell_magic('time', '', "\nall_f1_score = []\nall_recall_score = []\nall_precision_score = []\n\nfor index_model in range(N_MODELS):\n\n    file_name_model = '{}_model_crf_only_twitter_word_embeddings'.format(index_model + 1)\n    \n    model = \\\n    BiLSTM(isa_crf=True,\n           words_weights=wb_concatenate,\n           pre_trained=True,\n           max_len=ner_train.max_len,\n           num_words=ner_train.num_words,\n           num_tags=ner_train.num_tags,\n           learning_rate=LEARNING_RATE,\n           dropout=0.5)\n\n    model.load_weights('models/' + file_name_model + '.h5')\n\n    y_pred, y_true = \\\n    np.argmax(model.predict(ner_test.X_array), axis=-1), \\\n    np.argmax(ner_test.y_array, -1)\n    \n    pred_tag, true_tag = \\\n    ner_train.parser2categorical(y_pred, y_true) \n    \n    print('-' * 50)\n    print(file_name_model + '[{}] - Metrics...'.format(index_model + 1))\n    \n    all_precision_score.append(precision_score(pred_tag, true_tag))\n    all_recall_score.append(recall_score(pred_tag, true_tag))\n    all_f1_score.append(f1_score(pred_tag, true_tag))\n    \n    all_metrics_fold(pred_tag, true_tag)")


# In[ ]:


print("Average Precision: \t%.2f%%  |  std: (+/- %.2f%%)"%      (np.mean(all_precision_score) * 100,       np.std(all_precision_score) * 100))

print("Average Recall: \t%.2f%%  |  std: (+/- %.2f%%)"%      (np.mean(all_recall_score) * 100,       np.std(all_recall_score) * 100))

print("Average F1: \t\t%.2f%%  |  std: (+/- %.2f%%)" %      (np.mean(all_f1_score) * 100,       np.std(all_f1_score) * 100))


# In[ ]:




