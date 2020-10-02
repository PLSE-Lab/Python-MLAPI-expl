#!/usr/bin/env python
# coding: utf-8

# ## Setup Directories
# setup directories to the models and data.

# In[ ]:


# Setup target & Name
MODEL_NAME   = 'model1'
TARGET       = 'target'

# Create Input Paths
WORK_DIR          = "../working/"
Input_dir         = "../input/"
output_model_file = "bert_pytorch.bin"
save_model_file   = "bert_pytorch.bin"

# Generate Directories & Paths
BERT_MODEL_PATH       = Input_dir + 'bert-pretrained-models/uncased_l-12_h-768_a-12/uncased_L-12_H-768_A-12/'
package_dir_a         = Input_dir + "ppbert/pytorch-pretrained-bert/pytorch-pretrained-BERT"
Data_dir              = Input_dir + "jigsaw-unintended-bias-in-toxicity-classification"
BERT_MODEL_PATH_SETUP = Input_dir + "kaggle-model-ii/"


# ## Model Parameters
# Setup the model training parameters

# In[ ]:


# Preprocessing Parameters
num_to_load  = 1100000 # Train size to match time limit
valid_size   = 10000   # Validation Size
SEED         = 5432
EPOCHS       = 1
NUM_OUTPUTS  = 7
acc_steps    = 1

# Model Parameters
MAX_SEQUENCE_LENGTH = 200
lr                  = 1e-5
batch_size          = 64
WARM_UP             = 0.01


# ## Setup Variables & Packages
# setup variables and packages for the model

# In[ ]:


# Setup Bert Path
BERT_LOAD    = BERT_MODEL_PATH_SETUP + output_model_file
SUBGROUP_AUC = 'subgroup_auc'
BPSN_AUC     = 'bpsn_auc'     # stands for background positive, subgroup negative
BNSP_AUC     = 'bnsp_auc'     # stands for background negative, subgroup positive
no_decay      = ['bias', 'LayerNorm.bias', 'LayerNorm.weight'] # FOR MODEL
identity_cols = ['male','female','homosexual_gay_or_lesbian','christian','jewish','muslim','black','white','psychiatric_or_mental_illness']
y_columns     = [TARGET]

# Import Packages
import os, numpy as np,  pandas as pd
print(os.listdir("../input/nvidiaapex/repository/NVIDIA-apex-39e153a"))
get_ipython().system(' pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ../input/nvidiaapex/repository/NVIDIA-apex-39e153a')
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import torch.nn as nn
import seaborn as sns
import torch.utils.data
import torch.nn.functional as F
import scipy.stats as stats
from sklearn import metrics
from sklearn import model_selection
from sklearn.metrics import roc_auc_score
from keras.preprocessing import text, sequence
import datetime, pkg_resources, time, gc, re, operator, sys
import shutil, pickle, warnings
from nltk.stem import PorterStemmer
from tqdm import tqdm, tqdm_notebook
from IPython.core.interactiveshell import InteractiveShell
from apex import amp
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')
InteractiveShell.ast_node_interactivity = "all"
warnings.filterwarnings(action='once')
device=torch.device('cuda')
sys.path.insert(0, package_dir_a)
from pytorch_pretrained_bert import convert_tf_checkpoint_to_pytorch, BertConfig
from pytorch_pretrained_bert import BertTokenizer, BertForSequenceClassification,BertAdam
shutil.copyfile(BERT_MODEL_PATH_SETUP + 'bert_config.json', WORK_DIR + 'bert_config.json') # copy file
bert_config = BertConfig(BERT_MODEL_PATH_SETUP + "bert_config.json")                       # Translate model from tensorflow to pytorch


# In[ ]:


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def custom_loss(data, targets):
    bce_loss_1 = nn.BCEWithLogitsLoss(weight=targets[:,1:2])(data[:,:1],targets[:,:1])
    bce_loss_2 = nn.BCEWithLogitsLoss()(data[:,1:],targets[:,2:])
    the_score = (bce_loss_1 * loss_weight) + bce_loss_2
    return the_score

def convert_lines(example, max_seq_length,tokenizer):
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
    return np.array(all_tokens)

# ==========================
# ==== Result Analysis =====
# ==========================

def calculate_overall_auc(df, model_name):
    true_labels = df[TARGET]>0.5
    predicted_labels = df[model_name]
    return metrics.roc_auc_score(true_labels, predicted_labels)

def power_mean(series, p):
    total = sum(np.power(series, p))
    return np.power(total / len(series), 1 / p)

def get_final_metric(bias_df, overall_auc, POWER=-5, OVERALL_MODEL_WEIGHT=0.25):
    bias_score = np.average([
        power_mean(bias_df[SUBGROUP_AUC], POWER),
        power_mean(bias_df[BPSN_AUC], POWER),
        power_mean(bias_df[BNSP_AUC], POWER)])
    return (OVERALL_MODEL_WEIGHT * overall_auc) + ((1 - OVERALL_MODEL_WEIGHT) * bias_score)


def compute_auc(y_true, y_pred):
    try:
        return metrics.roc_auc_score(y_true, y_pred)
    except ValueError:
        return np.nan

def compute_subgroup_auc(df, subgroup, label, model_name):
    subgroup_examples = df[df[subgroup]>0.5]
    return compute_auc((subgroup_examples[label]>0.5), subgroup_examples[model_name])

def compute_bpsn_auc(df, subgroup, label, model_name):
    """Computes the AUC of the within-subgroup negative examples and the background positive examples."""
    subgroup_negative_examples = df[(df[subgroup]>0.5) & (df[label]<=0.5)]
    non_subgroup_positive_examples = df[(df[subgroup]<=0.5) & (df[label]>0.5)]
    examples = subgroup_negative_examples.append(non_subgroup_positive_examples)
    return compute_auc(examples[label]>0.5, examples[model_name])

def compute_bnsp_auc(df, subgroup, label, model_name):
    """Computes the AUC of the within-subgroup positive examples and the background negative examples."""
    subgroup_positive_examples = df[(df[subgroup]>0.5) & (df[label]>0.5)]
    non_subgroup_negative_examples = df[(df[subgroup]<=0.5) & (df[label]<=0.5)]
    examples = subgroup_positive_examples.append(non_subgroup_negative_examples)
    return compute_auc(examples[label]>0.5, examples[model_name])

def compute_bias_metrics_for_model(dataset, subgroups, model, label_col, include_asegs=False):
    """Computes per-subgroup metrics for all subgroups and one model."""
    records = []
    for subgroup in subgroups:
        record = {
            'subgroup': subgroup,
            'subgroup_size': len(dataset[dataset[subgroup]>0.5])
        }
        record[SUBGROUP_AUC] = compute_subgroup_auc(dataset, subgroup, label_col, model)
        record[BPSN_AUC] = compute_bpsn_auc(dataset, subgroup, label_col, model)
        record[BNSP_AUC] = compute_bnsp_auc(dataset, subgroup, label_col, model)
        records.append(record)
    return pd.DataFrame(records).sort_values('subgroup_auc', ascending=True)


# ## Setup Training Data
# Setup the sequence training data for the model

# In[ ]:


get_ipython().run_cell_magic('time', '', '\n# setup tokenizer\ntokenizer = BertTokenizer.from_pretrained(BERT_MODEL_PATH, cache_dir=None,do_lower_case=True)\n\n# setup records\ntrain_df = pd.read_csv(os.path.join(Data_dir,"train.csv"))\n\n# sample training dataset\ntrain_df = train_df.sample(num_to_load+valid_size,random_state=SEED)\n\n# Make sure all comment_text values are strings\ntrain_df[\'comment_text\'] = train_df[\'comment_text\'].astype(str) \n\n# create sequence data\nseq_data = convert_lines(train_df["comment_text"].fillna("DUMMY_VALUE"),MAX_SEQUENCE_LENGTH,tokenizer)\n\n# drop comment text\ntrain_df = train_df.drop([\'comment_text\'],axis=1)')


# ### Generate Targets
# Generate weighted targets for main prediction and underlying subgroups 

# In[ ]:


train_df['valid_target'] = (train_df[y_columns]>=0.5).astype(float)

# Overall
weights = np.ones((len(seq_data),)) / 4

# Subgroup
weights += (train_df[identity_cols].fillna(0).values>=0.5).sum(axis=1).astype(bool).astype(np.int) / 4

# Background Positive, Subgroup Negative
weights += (((train_df['target'].values>=0.5).astype(bool).astype(np.int) +  (train_df[identity_cols].fillna(0).values<0.5).sum(axis=1).astype(bool).astype(np.int) ) > 1 ).astype(bool).astype(np.int) / 4

# Background Negative, Subgroup Positive
weights += (((train_df['target'].values<0.5).astype(bool).astype(np.int) +  (train_df[identity_cols].fillna(0).values>=0.5).sum(axis=1).astype(bool).astype(np.int) ) > 1 ).astype(bool).astype(np.int) / 4

# loss weights
loss_weight = 1.0 / weights.mean()

# set y train vector
y_train = np.vstack([(train_df['target'].values>=0.5).astype(np.int),weights]).T

# obtain aux 
y_aux_train = train_df[identity_cols].fillna(0).values

# setup y aux train
y_aux_train = train_df[['target', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']]

# y train mix
y_train = np.hstack([y_train, y_aux_train])


# ## Setup Training Dataset
# setup a torch dataset for training.

# In[ ]:


# Setup Training Data
X = seq_data[:num_to_load]                
y = y_train[:num_to_load]

# Setup Validation Data
X_val = seq_data[num_to_load:]                
y_val = train_df['valid_target'].values[num_to_load:]

# setup train and test dataframe? maybe iffy.... bc of head and tail
test_df  = train_df.tail(valid_size).copy()
train_df = train_df.head(num_to_load)

# setup torch dataset
train_dataset = torch.utils.data.TensorDataset(torch.tensor(X,dtype=torch.long), torch.tensor(y,dtype=torch.float))


# In[ ]:


gc.collect()

# Seed Everything
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

# Set model to being deterministic
torch.backends.cudnn.deterministic = False

# setup default model
model = BertForSequenceClassification(bert_config,num_labels = NUM_OUTPUTS)

# load trained model into the default model
model.load_state_dict(torch.load(BERT_LOAD))

# set the device to cuda GPU computing
model.to(device)

# zero the model gradiant
model.zero_grad()

# setup model to cuda cores
model = model.to(device)

# setup parameter optimizer
param_optimizer = list(model.named_parameters())

optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

# create new train table (May need to add .copy())
train = train_dataset

# obtain number of train optimization steps
num_train_optimization_steps = int(EPOCHS*len(train)/batch_size/acc_steps)

# setup the optimizer
optimizer = BertAdam(optimizer_grouped_parameters, lr = lr, warmup = WARM_UP, t_total = num_train_optimization_steps)

# initialize model
model, optimizer = amp.initialize(model, optimizer, opt_level="O1",verbosity=0)

# setup model for training
model=model.train()


# In[ ]:


# garbage collect
gc.collect()

# setup notebook display load bar for loop.
tq = tqdm_notebook(range(EPOCHS))

for epoch in tq:
    
    # perform train collector
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    
    # set performance parameters
    avg_loss     = 0.
    avg_accuracy = 0.
    lossf        = None
    
    # Setup Display
    tk0 = tqdm_notebook(enumerate(train_loader),total=len(train_loader),leave=False)
    
    # zero optimizer gradiant
    optimizer.zero_grad()
    
    # empty cache
    torch.cuda.empty_cache()
    
    for i,(x_batch, y_batch) in tk0:
        
        # predict y results
        y_pred = model(x_batch.to(device), attention_mask=(x_batch>0).to(device), labels=None)
        
        # calculate loss
        y_batch_device = y_batch.to(device);
        loss = custom_loss(y_pred,y_batch_device)
       
        # amp scale loss optimizer
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            
            # scaled loss backward
            scaled_loss.backward()
            
        # Wait for several backward steps    
        if (i+1) % acc_steps == 0:       
            
            # Perform an additional optimiztion step
            optimizer.step()  
            
            # zero gradiant for the optimizer
            optimizer.zero_grad()
            
        # drill into loss function     
        if lossf:
            lossf = 0.98*lossf+0.02*loss.item()
        else:
            lossf = loss.item()
            
        tk0.set_postfix(loss = lossf)
        avg_loss     += loss.item() / len(train_loader)
        avg_accuracy += torch.mean(((torch.sigmoid(y_pred[:,0])>0.5) == (y_batch[:,0]>0.5).to(device)).to(torch.float)).item()/len(train_loader)
    tq.set_postfix(avg_loss=avg_loss,avg_accuracy=avg_accuracy)


# save models
torch.save(model.state_dict(), save_model_file)


# In[ ]:


# setup default model
model = BertForSequenceClassification(bert_config,num_labels = NUM_OUTPUTS)

# load trained model into the default model
model.load_state_dict(torch.load(output_model_file))

# set the device to cuda GPU computing
model.to(device)


# In[ ]:


# check model parameters
for param in model.parameters():
    param.requires_grad=False

# set model to evaluation    
model.eval()

# setup torch dataset
valid = torch.utils.data.TensorDataset(torch.tensor(X_val,dtype=torch.long))

# perform train loader
valid_loader = torch.utils.data.DataLoader(valid, batch_size=batch_size, shuffle=False)
    


# In[ ]:


# setup validation prediction array
valid_preds = np.zeros((len(X_val)))

# setup tqdm load bar
tk0 = tqdm_notebook(valid_loader)

for i,(x_batch)  in enumerate(tk0):
    
    # extract and transform x inputs
    x_batch = x_batch[0].long()
    
    # generate predictions
    pred = model(x_batch.to(device), attention_mask=(x_batch>0).to(device), labels=None)
    
    # validate
    valid_preds[i*batch_size:(i+1)*batch_size] = sigmoid(pred[:,0].detach().cpu().squeeze().numpy())
    


# In[ ]:


test_df[MODEL_NAME] = valid_preds 
bias_metrics_df     = compute_bias_metrics_for_model(test_df, identity_cols, MODEL_NAME, 'target')
write_df            = get_final_metric(bias_metrics_df, calculate_overall_auc(test_df, MODEL_NAME))
print(write_df)
bias_metrics_df


# In[ ]:


test_df.to_csv('validation_results.csv', index=False)
bias_metrics_df.to_csv('bias_metrics_results.csv', index=False)

