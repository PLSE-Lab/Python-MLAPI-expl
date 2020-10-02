#!/usr/bin/env python
# coding: utf-8

# ## Kaggle competition  Santander customer transaction using Neural Network in PyTorch
# 
# It's a fairly simple neural network with enhancements targeted for later. 
# Steps:
# 1. Load data
# 2. Add features
# 3. Model training using 5-fold StratifiedKfol

# In[47]:


import torch
import numpy as np
from torch import nn, optim
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import StratifiedKFold
import torch.nn.functional as F
import torchvision

from sklearn.metrics import accuracy_score, confusion_matrix,f1_score, precision_recall_curve, average_precision_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('ggplot')
import os
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)


# ### Check for GPU

# In[2]:


cuda = torch.cuda.is_available()
if cuda:
    device = "cuda"
    print("cuda available")
torch.cuda.get_device_name(0)    


# ### Load data and scale between (-1,1) using standard scaler

# In[9]:


os.listdir('../input')


# In[41]:


get_ipython().run_cell_magic('time', '', "df_train_raw = pd.read_csv('../input/train.csv')\ndf_test_raw = pd.read_csv('../input/test.csv')")


# In[67]:


get_ipython().run_cell_magic('time', '', 'df_train = df_train_raw.copy()\ndf_test = df_test_raw.copy()')


# In[68]:


train_cols = [col for col in df_train.columns if col not in ['ID_code', 'target']]
y_train = df_train['target']


# In[69]:


df_train.shape


# In[70]:


ss = StandardScaler()
rs = RobustScaler() 
df_train[train_cols] = ss.fit_transform(df_train[train_cols])
df_test[train_cols] = ss.fit_transform(df_test[train_cols])


# In[71]:


interactions= {'var_81':['var_53','var_139','var_12','var_76'],
               'var_12':['var_139','var_26','var_22', 'var_53','var_110','var_13'],
               'var_139':['var_146','var_26','var_53', 'var_6', 'var_118'],
               'var_53':['var_110','var_6'],
              'var_26':['var_110','var_109','var_12'],
              'var_118':['var_156'],
              'var_9':['var_89'],
              'var_22':['var_28','var_99','var_26'],
              'var_166':['var_110'],
              'var_146':['var_40','var_0'],
              'var_80':['var_12']}


# In[72]:


get_ipython().run_cell_magic('time', '', "for col in train_cols:\n        df_train[col+'_2'] = df_train[col] * df_train[col]\n        df_train[col+'_3'] = df_train[col] * df_train[col]* df_train[col]\n#         df_train[col+'_4'] = df_train[col] * df_train[col]* df_train[col]* df_train[col]\n        df_test[col+'_2'] = df_test[col] * df_test[col]\n        df_test[col+'_3'] = df_test[col] * df_test[col]* df_test[col]")


# In[73]:


get_ipython().run_cell_magic('time', '', "for df in [df_train, df_test]:\n    df['sum'] = df[train_cols].sum(axis=1)  \n    df['min'] = df[train_cols].min(axis=1)\n    df['max'] = df[train_cols].max(axis=1)\n    df['mean'] = df[train_cols].mean(axis=1)\n    df['std'] = df[train_cols].std(axis=1)\n    df['skew'] = df[train_cols].skew(axis=1)\n    df['kurt'] = df[train_cols].kurtosis(axis=1)\n    df['med'] = df[train_cols].median(axis=1)")


# In[74]:


get_ipython().run_cell_magic('time', '', "for key in interactions:\n    for value in interactions[key]:\n        df_train[key+'_'+value+'_mul'] = df_train[key]*df_train[value]\n        df_train[key+'_'+value+'_div'] = df_train[key]/df_train[value]\n        df_train[key+'_'+value+'_sum'] = df_train[key] + df_train[value]\n        df_train[key+'_'+value+'_sub'] = df_train[key] - df_train[value]\n        \n        df_test[key+'_'+value+'_mul'] = df_test[key]*df_test[value]\n        df_test[key+'_'+value+'_div'] = df_test[key]/df_test[value]\n        df_test[key+'_'+value+'_sum'] = df_test[key] + df_test[value]\n        df_test[key+'_'+value+'_sub'] = df_test[key] - df_test[value]")


# In[75]:


df_train_raw.columns


# In[80]:


df_train['num_zero_rows'] = (df_train_raw[train_cols] == 0).astype(int).sum(axis=1)
df_test['num_zero_rows'] = (df_test_raw[train_cols] == 0).astype(int).sum(axis=1)


# In[81]:


df_train.head()


# In[82]:


all_train_columns = [col for col in df_train.columns if col not in ['ID_code', 'target']]


# ### Define a neural network

# In[85]:


class classifier(nn.Module):
    
    def __init__(self,input_dim, hidden_dim, dropout = 0.4):
        super().__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(p = dropout)
    
    
    def forward(self,x):
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        
        return x   


# In[87]:


model = classifier(len(all_train_columns),200)
print(model)
del model


# ### Defining the Stratified K fold

# In[90]:


folds = StratifiedKFold(n_splits = 5, shuffle = True)


# In[91]:


#Definining the sigmoid function to calculate final results
def sigmoid(x):
    return 1/(1 + np.exp(-x))


# In[92]:


get_ipython().run_cell_magic('time', '', 'oof = np.zeros(len(df_train))\npredictions = np.zeros(len(df_test))\n\n# Defining the parameters\nbatch_size = 1000\nn_epochs = 10\n\n#Loader for test dataset\ntest_x = torch.from_numpy(df_test[all_train_columns].values).float()#.cuda()\ntest_dataset = torch.utils.data.TensorDataset(test_x)\ntestloader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, shuffle = False)\n\n#Starting the tranining on each fold\nfor n_fold, (train_idx, val_idx) in enumerate(folds.split(df_train, y_train)):\n    running_train_loss, running_val_loss = [],[]\n    val_loss_min = np.Inf\n#     print(i, train_idx.shape, val_idx.shape)\n\n    print("Fold number: ", n_fold+1)\n    \n    #Defining the train loader iterator\n    train_x_fold = torch.from_numpy(df_train.iloc[train_idx][all_train_columns].values).float().cuda()\n    train_y_fold = torch.from_numpy(y_train[train_idx].values).float().cuda()\n    train_fold_dataset = torch.utils.data.TensorDataset(train_x_fold,train_y_fold)\n    trainloader = torch.utils.data.DataLoader(train_fold_dataset, batch_size = batch_size, shuffle = True)\n    \n    #Defining the validation dataset loader iterator\n    val_x_fold = torch.from_numpy(df_train.iloc[val_idx][all_train_columns].values).float().cuda()\n    val_y_fold = torch.from_numpy(y_train[val_idx].values).float().cuda()\n    val_fold_dataset = torch.utils.data.TensorDataset(val_x_fold,val_y_fold )\n    valloader = torch.utils.data.DataLoader(val_fold_dataset, batch_size = batch_size, shuffle = False)\n    \n    #Initiating model, optimizer and loss function\n    model = classifier(len(all_train_columns),200)\n    model.cuda()\n    optimizer = optim.Adam(model.parameters(), lr = 0.004)\n    # criterion = nn.CrossEntropyLoss() # also number of outputs should be 2\n    criterion = nn.BCEWithLogitsLoss()\n    \n    #Starting the neural network training\n    for epoch in range(n_epochs):\n        train_loss = 0\n        \n        for train_x_batch, train_y_batch in trainloader:\n            model.train()\n            optimizer.zero_grad()\n            output = model(train_x_batch)\n            loss = criterion(output, train_y_batch.view(-1,1))\n            loss.backward()\n            optimizer.step()\n            train_loss += loss.item()/len(trainloader)\n        \n        #Evaluating on validation dataset\n        with torch.no_grad():\n            val_loss = 0\n            model.eval()\n            val_preds = []\n            val_true = []\n            \n            #Stratified Kfold is splitting the dataset a bit weird.. \n            #so have to introduce list methods to store validation and test predictions\n            for i, (val_x_batch, val_y_batch) in enumerate(valloader):\n                val_output = model(val_x_batch)\n                val_loss += (criterion(val_output, val_y_batch.view(-1,1)).item())/len(valloader)\n                batch_output = sigmoid(val_output.cpu().numpy().squeeze())\n                try:\n                    batch_output = list(batch_output)\n                except TypeError:\n                    batch_output =[batch_output]\n                val_preds.extend(batch_output)\n                \n#                 batch_true = val_y_batch.cpu().numpy().squeeze()\n#                 try:\n#                     batch_true = list(batch_true)\n#                 except TypeError:\n#                     batch_true =[batch_true]\n#                 val_true.extend(batch_true)\n                \n        running_train_loss.append(train_loss)\n        running_val_loss.append(val_loss)\n        \n        \n        print("Epoch: {}   Training loss: {:.6f}   Validation Loss: {:.6f}    Val_auc:{:.5f}".format(epoch+1,\n                                                                              train_loss,\n                                                                               val_loss,\n                                                                               roc_auc_score(y_train[val_idx].values,\n                                                                                             val_preds))\n         )\n        \n        #Saving the model only if validation loss is going down in the epoch\n        if val_loss <= val_loss_min:\n            print("Validation loss decresed from {:.6f} ----> {:.6f} Saving Model".format(val_loss_min,val_loss))\n            torch.save(model.state_dict(), "san_cust_tran_torch.pt")\n            val_loss_min = val_loss\n            \n        \n    oof[val_idx] = val_preds    \n    print("Fold {} metrics:   Avg Training loss: {:.4f}   Avg Validation Loss: {:.4f}   Val_auc:{:.5f}".format(n_fold+1,\n                                                                              np.mean(running_train_loss),\n                                                                               np.mean(running_val_loss),\n                                                                               roc_auc_score(y_train[val_idx].values,\n                                                                                             oof[val_idx])))\n    \n    #Predicting on test set with the best model in the fold\n    y_test_pred_fold = []\n    print("Saving test results for best model")\n    for (test_x_batch,) in testloader:\n        model.load_state_dict(torch.load("san_cust_tran_torch.pt"))\n        model.cpu()\n        test_output = model(test_x_batch)\n        test_batch_output = sigmoid(test_output.detach().numpy().squeeze())\n        try:\n            test_batch_output = list(test_batch_output)\n        except TypeError:\n            test_batch_output =[test_batch_output]\n        y_test_pred_fold.extend(test_batch_output)\n    predictions += np.array(y_test_pred_fold)/folds.n_splits \n    \n    print("end of fold: ",n_fold+1,"\\n")\n        \n        ')


# ### Checking best iteration

# In[93]:


plt.figure(figsize = (8,4))
plt.title("Train vs val loss on last epoch")
plt.plot(running_train_loss, label = "train")
plt.plot(running_val_loss, label = "val")
plt.legend()
plt.show()


# ### Preparing submission file

# In[94]:


sub = pd.DataFrame({'ID_code': df_test.ID_code.values,
                   'target': predictions})
sub.to_csv('sub_pytorch_simplenn.csv', index = False)


# 1. The first submission gave the LB score of 0.854

# In[ ]:




