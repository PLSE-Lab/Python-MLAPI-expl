#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
print(os.listdir("../input/nvidiaapex/repository/NVIDIA-apex-39e153a"))
print(os.listdir("../input/bert-pretrained-models"))
print(os.listdir("../input/jigsaw-unintended-bias-in-toxicity-classification"))
print(os.listdir("../input/ppbert/pytorch-pretrained-bert/pytorch-pretrained-BERT"))

# Any results you write to the current directory are saved as output.


# In[ ]:


# Installing Nvidia Apex
get_ipython().system(' pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ../input/nvidiaapex/repository/NVIDIA-apex-39e153a')


# In[ ]:


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import random
import pkg_resources
import seaborn as sns
import time
import scipy.stats as stats
import gc
import re
import operator 
import sys
from sklearn import metrics
from sklearn import model_selection
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from nltk.stem import PorterStemmer
from sklearn.metrics import roc_auc_score
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')
from tqdm import tqdm, tqdm_notebook
import os
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
import warnings
warnings.filterwarnings(action='once')
import pickle
from apex import amp
import shutil


# In[ ]:


# config
device=torch.device('cuda')
MAX_SEQUENCE_LENGTH = 222
SEED = 1234
EPOCHS = 1
Data_dir="../input/jigsaw-unintended-bias-in-toxicity-classification"
Input_dir = "../input"
BERT_MODEL_PATH = '../input/bert-pretrained-models/uncased_l-12_h-768_a-12/uncased_L-12_H-768_A-12'
num_to_load=250000                         #Train size to match time limit
IDENTITY_COLUMNS = [
    'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',
    'muslim', 'black', 'white', 'psychiatric_or_mental_illness'
]
AUX_COLUMNS = ['target', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']
TEXT_COLUMN = 'comment_text'
TARGET_COLUMN = 'target'
OUTPUT_MODEL_FILE = "bert_pytorch.bin"
LEARNING_RATE = 2e-5
BATCH_SIZE = 32
ACCUMULATION_STEPS = 2


# In[ ]:


# Add the Bart Pytorch repo to the PATH
# using files from: https://github.com/huggingface/pytorch-pretrained-BERT
package_dir_a = "../input/ppbert/pytorch-pretrained-bert/pytorch-pretrained-BERT"
sys.path.insert(0, package_dir_a)

from pytorch_pretrained_bert import convert_tf_checkpoint_to_pytorch
from pytorch_pretrained_bert import BertTokenizer, BertForSequenceClassification,BertAdam


# In[ ]:


# Translate model from tensorflow to pytorch

convert_tf_checkpoint_to_pytorch.convert_tf_checkpoint_to_pytorch(
    BERT_MODEL_PATH + '/bert_model.ckpt',
    BERT_MODEL_PATH + '/bert_config.json',
    'pytorch_model.bin'
)


# In[ ]:


shutil.copyfile(BERT_MODEL_PATH + '/bert_config.json', 'bert_config.json')
print(os.listdir("."))


# In[ ]:


# This is the Bert configuration file
from pytorch_pretrained_bert import BertConfig

bert_config = BertConfig('bert_config.json')


# In[ ]:


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def convert_lines(example, max_seq_length,tokenizer):
    """Converting the lines to BERT format.
    Thanks to https://www.kaggle.com/httpwwwfszyc/bert-in-keras-taming"""
    max_seq_length -=2
    all_tokens = []
    longer = 0
    for text in tqdm_notebook(example):
        tokens_a = tokenizer.tokenize(text)
        if len(tokens_a)>max_seq_length:
            tokens_a = tokens_a[:max_seq_length]
            longer += 1
        one_token = tokenizer.convert_tokens_to_ids(["[CLS]"]+tokens_a+["[SEP]"])+[0] * (max_seq_length - len(tokens_a))
        all_tokens.append(one_token)
    print('#sequences truncated to maxlen {}: {}'.format(max_seq_length, longer))
    return np.array(all_tokens)


# In[ ]:



seed_everything(SEED)
train_df = pd.read_csv(os.path.join(Data_dir,"train.csv")).sample(num_to_load)
test_df = pd.read_csv(os.path.join(Data_dir,"test.csv"))
train_df.shape, test_df.shape


# In[ ]:


get_ipython().run_cell_magic('time', '', '# Make sure all comment_text values are strings\ntrain_df[TEXT_COLUMN] = train_df[TEXT_COLUMN].astype(str)\ntrain_df[TEXT_COLUMN] = train_df[TEXT_COLUMN].fillna("DUMMY_VALUE")\ntest_df[TEXT_COLUMN] = test_df[TEXT_COLUMN].astype(str)\n#train_df=train_df.fillna(0)\n#train_df = train_df.drop([\'comment_text\'],axis=1)\n# convert target to 0,1\ntrain_df[TARGET_COLUMN]=(train_df[TARGET_COLUMN]>=0.5).astype(float)\n\nfor col in IDENTITY_COLUMNS + [TARGET_COLUMN]:\n    train_df[col] = (train_df[col]>=0.5).astype(float)\n    \ntokenizer = BertTokenizer.from_pretrained(BERT_MODEL_PATH, cache_dir=None,do_lower_case=True)\nx_train = convert_lines(train_df[TEXT_COLUMN],MAX_SEQUENCE_LENGTH,tokenizer)\ny_train = train_df[TARGET_COLUMN].values\ny_aux_train = train_df[AUX_COLUMNS].values\nx_test = convert_lines(test_df[TEXT_COLUMN],MAX_SEQUENCE_LENGTH,tokenizer)')


# In[ ]:


x_train.shape, x_test.shape, y_train.shape, y_aux_train.shape


# In[ ]:


get_ipython().run_cell_magic('time', '', '# [y_train, y_aux_train]\n# len(AUX_COLUMNS + [TARGET_COLUMN])\n# y_train_torch = torch.tensor(np.hstack([y_train[:, np.newaxis], y_aux_train]), dtype=torch.float32).cuda()\nds = torch.utils.data.TensorDataset(torch.tensor(x_train,dtype=torch.long), torch.tensor(y_train,dtype=torch.float).unsqueeze(1))\nmodel = BertForSequenceClassification.from_pretrained(".",cache_dir=None,num_labels=1)\nmodel.zero_grad()\nmodel = model.to(device)\nparam_optimizer = list(model.named_parameters())\nno_decay = [\'bias\', \'LayerNorm.bias\', \'LayerNorm.weight\']\noptimizer_grouped_parameters = [\n    {\'params\': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], \'weight_decay\': 0.01},\n    {\'params\': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], \'weight_decay\': 0.0}\n    ]\nnum_train_optimization_steps = int(EPOCHS*len(ds) / BATCH_SIZE / ACCUMULATION_STEPS)\n\noptimizer = BertAdam(optimizer_grouped_parameters,\n                     lr=LEARNING_RATE,\n                     warmup=0.05,\n                     t_total=num_train_optimization_steps)\n\nmodel, optimizer = amp.initialize(model, optimizer, opt_level="O1",verbosity=0)\nmodel=model.train()\n\n\ntq = tqdm_notebook(range(EPOCHS))\nfor epoch in tq:\n    loader = torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)\n    avg_loss = 0.\n    avg_accuracy = 0.\n    lossf=None\n    tk0 = tqdm_notebook(enumerate(loader),total=len(loader),leave=False)\n    optimizer.zero_grad()   # Bug fix - thanks to @chinhuic\n    for i,(x_batch, y_batch) in tk0:\n#        optimizer.zero_grad()\n        y_pred = model(x_batch.to(device), attention_mask=(x_batch>0).to(device), labels=None)\n        loss =  F.binary_cross_entropy_with_logits(y_pred,y_batch.to(device))\n        with amp.scale_loss(loss, optimizer) as scaled_loss:\n            scaled_loss.backward()\n        if (i+1) % ACCUMULATION_STEPS == 0:             # Wait for several backward steps\n            optimizer.step()                            # Now we can do an optimizer step\n            optimizer.zero_grad()\n        if lossf:\n            lossf = 0.98*lossf+0.02*loss.item()\n        else:\n            lossf = loss.item()\n        tk0.set_postfix(loss = lossf)\n        avg_loss += loss.item() / len(loader)\n        avg_accuracy += torch.mean(((torch.sigmoid(y_pred[:,0])>0.5) == (y_batch[:,0]>0.5).to(device)).to(torch.float) ).item()/len(loader)\n    tq.set_postfix(avg_loss=avg_loss,avg_accuracy=avg_accuracy)\n\ntorch.save(model.state_dict(), OUTPUT_MODEL_FILE)\nprint(os.listdir("."))')


# In[ ]:


# Make inference on test data
# The following 2 lines are not needed but show how to download the model for prediction
model = BertForSequenceClassification(bert_config,num_labels=1)
model.load_state_dict(torch.load(OUTPUT_MODEL_FILE))
model.to(device)
for param in model.parameters():
    param.requires_grad=False
model.eval()
predictions = np.zeros((len(x_test)))
ds = torch.utils.data.TensorDataset(torch.tensor(x_test,dtype=torch.long))
loader = torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)

tk0 = tqdm_notebook(loader)
for i,(x_batch,)  in enumerate(tk0):
    pred = model(x_batch.to(device), attention_mask=(x_batch>0).to(device), labels=None)
    predictions[i*BATCH_SIZE:(i+1)*BATCH_SIZE] = pred[:,0].detach().cpu().squeeze().numpy()

predictions.shape


# In[ ]:



# prepare submission
submission = pd.DataFrame.from_dict({
    'id': test_df.id,
    'prediction': predictions
})
submission.head()


# In[ ]:


submission.shape


# In[ ]:


submission.to_csv('submission.csv', index=False)
print(os.listdir("."))

