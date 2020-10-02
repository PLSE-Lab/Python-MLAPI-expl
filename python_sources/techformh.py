#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np
import pandas as pd
from sklearn.metrics import *
from sklearn.model_selection import *
from scipy.special import softmax
import re
from sklearn.model_selection import StratifiedKFold
from simpletransformers.classification.classification_model import ClassificationModel
from sklearn.metrics import mean_squared_error as mse
from tqdm import tqdm
import warnings
from torch import nn
import matplotlib.pyplot as plt
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
from collections import defaultdict 

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


# !pip install torch-lr-finder


# In[ ]:


# !pip install simpletransformers


# In[ ]:


df_train = pd.read_csv("/kaggle/input/mentalhealth/Train.csv")
df_test = pd.read_csv("/kaggle/input/mentalhealth/Test.csv")
df_sub = pd.read_csv("/kaggle/input/mentalhealth/sub.csv")


# In[ ]:


df_train.drop('ID',axis=1,inplace=True)
df_test.drop('ID',axis=1,inplace=True)


# In[ ]:


df_temp = pd.concat([df_train,df_test])
df_temp['len'] = df_temp['text'].apply(lambda x: len(x.split()))
df_temp['len'].describe()


# In[ ]:


df_train['text']


# In[ ]:


# df_train['Model_Info']= df_train['Model_Info'].apply(lambda x: x.lower())
# df_test['Model_Info'] = df_test['Model_Info'].apply(lambda x: x.lower())
# df_train['Additional_Description'] = df_train['Additional_Description'].apply(lambda x:x.lower())
# df_test['Additional_Description'] = df_test['Additional_Description'].apply(lambda x:x.lower())
# df_train['text']= df_train['text'].apply(lambda x: re.sub(r"[^A-Za-z]", " ", x))
# df_test['text'] = df_test['text'].apply(lambda x: re.sub(r"[^A-Za-z]", " ", x))
# df_train['length'] = df_train['text'].apply(lambda x: len(x.split()))


# In[ ]:


# df_test['len'] = df_test['text'].apply(lambda x: len(x.split()))


# In[ ]:


# df_train['text'].apply(lambda x:len(x.split())).describe()


# 

# In[ ]:


label_num = {
    'Depression':0,
    'Drugs':1,
    'Suicide':2,
    'Alcohol':3
}


# In[ ]:


df_train['label'] = df_train['label'].map(label_num)


# In[ ]:


def get_model(model_type, model_name, n_epochs = 4, train_batch_size = 16,  seq_len = 40, lr = 6e-5):
    model = ClassificationModel(model_type, model_name,num_labels=4,use_cuda=True,args={'train_batch_size':train_batch_size,
                                                                         'reprocess_input_data': True,
                                                                         'overwrite_output_dir': True,
                                                                         'fp16': False,
                                                                         'do_lower_case':True,
                                                                         'num_train_epochs': n_epochs,
                                                                         'max_seq_length': seq_len,
                                                                         'regression': False,
                                                                         'manual_seed': 2,
                                                                         "learning_rate":lr,
                                                                         "save_eval_checkpoints": False,
                                                                         "save_model_every_epoch": False,
                                                                                        "silent":True,})
    return model


# In[ ]:


# df_train['label'] = df_train['label'].map(label_num)


# In[ ]:


tmp = pd.DataFrame()
tmp['text'] = df_train['text']
tmp['labels'] = df_train['label']
tmp_test = pd.DataFrame()
tmp_test['text'] = df_test['text']
tmp_test['labels'] = 0


# In[ ]:


# tmp_test = df_test.copy()
# tmp_test['labels'] = 0


# In[ ]:


pd.DataFrame((len(tmp_test),4))


# In[ ]:


tmp_trn, tmp_val = train_test_split(tmp, test_size=0.20, random_state=22,stratify=tmp['labels'])


# In[ ]:


train1 = tmp_trn


# In[ ]:


err=[]
err_holdout = []
oof = np.zeros((len(tmp_test), 4))
splits=10
fold=StratifiedKFold(n_splits=splits, shuffle=True, random_state=22)
i=1
for train_index, test_index in fold.split(train1,train1['labels']):
    train1_trn, train1_val = train1.iloc[train_index], train1.iloc[test_index]
    model = get_model('roberta','roberta-base',n_epochs=3)
    model.train_model(train1_trn)
    raw_outputs_val = model.eval_model(train1_val)[1]
    raw_outputs_val = softmax(raw_outputs_val,axis=1)
    print(f"Log_Loss: {log_loss(train1_val['labels'], raw_outputs_val)}")
    err.append(log_loss(train1_val['labels'], raw_outputs_val))
    raw_outputs_hold = model.eval_model(tmp_val)[1]
    raw_outputs_hold = softmax(raw_outputs_hold,axis=1)
    print(f"Log_Loss: {log_loss(tmp_val['labels'], raw_outputs_hold)}")
    err_holdout.append(log_loss(tmp_val['labels'], raw_outputs_hold))
    raw_outputs_test = model.eval_model(tmp_test)[1]
    raw_outputs_test = softmax(raw_outputs_test,axis=1)
    oof+=raw_outputs_test
print("Mean LogLoss: ",np.mean(err))
print("Mean LogLoss Holdout: ",np.mean(err_holdout))
oof = oof/splits
# final=pd.DataFrame()
# final['ID']=test['ID']
# final['target']=np.mean(y_pred_tot, 0)
# print(final.shape)
# final.to_csv('20fold_rbl_2_3e5_32_128.csv',index=False)


# In[ ]:


def get_learning(x):
    return [x*i for i in range(1,11,1)]


# In[ ]:


def get_final_rates(start,end,rates=None):
    if rates == None:
        rates = get_learning(start)
    else:
        rate = get_learning(start)
        for i in rate:
            rates.append(i)
    start = rates[-1]    
    if start != end:
        get_final_rates(start,end,rates)
    
    return rates;


# In[ ]:


def def_value(): 
    return "None"


# In[ ]:


d = defaultdict(def_value)
d['loss'] = [1,2,3,4]


# In[ ]:


d['loss'].append(1)


# In[ ]:


def get_lr(start,end):
    rates = get_final_rates(start,end,rates=None)
    loss_vals = []
    lr_vals = []
    for lr in rates:
        model = get_model('roberta','roberta-base',n_epochs = 1, train_batch_size = 16,  seq_len = 40, lr = lr)
        model.train_model(tmp_trn,eval_df=tmp_val)
        result,model_out,wrong_preds = model.eval_model(tmp_val)
        out = softmax(model_out,axis=1)
        loss = log_loss(y_pred=out,y_true=tmp_val['labels'])
        loss_vals.append(loss)
        lr_vals.append(lr)
    return loss_vals,lr_vals


# In[ ]:


loss_vals,lr_vals = get_lr(start=1e-7,end=1e-2)


# In[ ]:


loss_vals


# In[ ]:


for i in range(len(loss_vals)):
    if loss_vals[i] == min(loss_vals):
        print(lr_vals[i-3])
        


# In[ ]:


plt.plot(np.log(lr_vals),loss_vals)


# In[ ]:


model = get_model('roberta','roberta-base',n_epochs = 1, train_batch_size = 16,  seq_len = 40, lr = 1e-4)
model.train_model(tmp_trn,eval_df=tmp_val)
result,model_out,wrong_preds = model.eval_model(tmp_val)
out = softmax(model_out,axis=1)
print(log_loss(y_pred=out,y_true=tmp_val['labels']))


# In[ ]:


model


# In[ ]:


# from torch_lr_finder import LRFinder
# from torch import optim
# criterion = nn.LogSoftmax()
# optimizer = optim.Adam(model.parameters(), lr=0.1, weight_decay=1e-2)
# lr_finder = LRFinder(model,criterion=criterion,device="cuda")
# lr_finder.range_test(trainloader, val_loader=val_loader, end_lr=1, num_iter=100, step_mode="linear")
# lr_finder.plot(log_lr=False)
# lr_finder.reset()


# In[ ]:





# 

# In[ ]:


result


# In[ ]:


model_out


# In[ ]:


# from math import sqrt
# print(sqrt(mean_squared_error(y_pred=np.abs(model_out),y_true=np.abs(tmp_val['labels']))))


# In[ ]:





# 

# In[ ]:





# In[ ]:


# preds_val = np.clip(result[1], 0, 1)


# In[ ]:


preds_val


# In[ ]:


print(log_loss(y_pred=preds_val,y_true=tmp_val['labels']))


# In[ ]:


tmp_test


# In[ ]:


test_preds = model.eval_model(tmp_test)


# In[ ]:


test_preds


# In[ ]:


test_preds_fin = softmax(test_preds[1],axis=1)


# In[ ]:


test_preds_fin


# In[ ]:


tmp_test


# In[ ]:


oof[308]


# In[ ]:


df_sub['Depression'] = oof[:,0]
df_sub['Drugs'] = oof[:,1]
df_sub['Suicide'] = oof[:,2]
df_sub['Alcohol'] = oof[:,3]


# In[ ]:





# In[ ]:


df_sub['Depression'] = df_sub['Depression'].apply(lambda x: 1 if x > 0.60 else x)
df_sub['Drugs'] = df_sub['Drugs'].apply(lambda x: 1 if x > 0.60 else x)
df_sub['Suicide'] = df_sub['Suicide'].apply(lambda x: 1 if x > 0.60 else x)
df_sub['Alcohol'] = df_sub['Alcohol'].apply(lambda x: 1 if x > 0.60 else x)


# In[ ]:


# df_sub['Price'].std()


# In[ ]:


df_sub.to_csv('zindi10SKF.csv',index=False)


# In[ ]:


Batch 16
Learning_rate 2e-5
seq_len(20)

Batch 8
Learning_rate 1e-5
seq_len(22)
There is relationship with seq_len and batch_size

