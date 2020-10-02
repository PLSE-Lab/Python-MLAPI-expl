#!/usr/bin/env python
# coding: utf-8

# ## Just wanted to build a baseline GPT2 kernel even if the results are not promising
# ### May we will use it in blend of blend of blend at the end !

# In[ ]:


## Because Kaggle has stopped internet to fight with bad bad guys . I am installing them from local downloads .
import os
#os.system('pip install ../input/sacremoses/sacremoses-master/')
#os.system('pip install ../input/transformers-2-3-0/')


# In[ ]:


get_ipython().system('pip install ../input/sacremoses/sacremoses-master/')
get_ipython().system('pip install ../input/transformers-2-3-0/')


# In[ ]:


## Loading Libraries  -No they are not for books
import transformers, sys, os, gc
import numpy as np, pandas as pd, math
import torch, random, os, multiprocessing, glob
import torch.nn.functional as F
import torch, time

from ml_stratifiers import MultilabelStratifiedShuffleSplit, MultilabelStratifiedKFold
from scipy.stats import spearmanr
from torch import nn
from torch.utils import data
from torch.utils.data import DataLoader, Dataset
from transformers import (
    BertTokenizer, BertModel, BertForSequenceClassification, BertConfig,
    WEIGHTS_NAME, CONFIG_NAME, AdamW, get_linear_schedule_with_warmup, 
    get_cosine_schedule_with_warmup,    GPT2Config,GPT2Model,
    GPT2LMHeadModel,
    GPT2Tokenizer,
)
from transformers.modeling_gpt2 import GPT2PreTrainedModel
from tqdm import tqdm
print(transformers.__version__)


# # Data Understanding

# In[ ]:


## Common Variables for Notebook 
ROOT = '../input/google-quest-challenge/' ## This is the root of all evil.


# In[ ]:


pretrained_bert = '../input/gpt2-models/'


# In[ ]:


## Make results reproducible .Else noone will believe you .
import random

def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


# In[ ]:


## This is for fast experiements , but we dont have enough GPU anyway, so thats a bummer!
class PipeLineConfig:
    def __init__(self, lr, warmup,accum_steps, epochs, seed, expname,head_tail,freeze,question_weight,answer_weight,fold,train):
        self.lr = lr
        self.warmup = warmup
        self.accum_steps = accum_steps
        self.epochs = epochs
        self.seed = seed
        self.expname = expname
        self.head_tail = head_tail
        self.freeze = freeze
        self.question_weight = question_weight
        self.answer_weight =answer_weight
        self.fold = fold
        self.train = train


# In[ ]:


config_1 = PipeLineConfig(3e-5,0.05,4,0,42,'uncased_1',True,False,0.7,0.3,8,False) ## These are experiments . You can do as much as you want as long as inference is faster 
config_2 = PipeLineConfig(4e-5,0.03,4,6,2019,'uncased_2',True,False,0.8,0.2,5,False)## Adding various different seeds , folds and learning rate and mixing them and then doing inference .
config_3 = PipeLineConfig(4e-5,0.03,4,4,2019,'small_test_3',True ,False, 0.8,0.2,3,True) ## For Small tests in Kaggle , less number of fold , less number of epochs.
config_4 = PipeLineConfig(4e-5,0.05,1,4,2019,'small_test_4',True ,False, 0.8,0.2,3,True)
config_5 = PipeLineConfig(3e-5,0.05,1,6,42,'gpt2class',True ,False, 0.8,0.2,8,True)
config_6 = PipeLineConfig(3e-5,0.05,1,6,42,'gpt2cnn',True ,False, 0.8,0.2,8,True)
## I am doing first experiement
config = config_5
## Note : If you want to train and just not copy this and submit then change the last parameter above to "True" it will kick off the training process.


# In[ ]:


seed_everything(config.seed)


# In[ ]:


## load the data 
train = pd.read_csv(ROOT+'train.csv')
test = pd.read_csv(ROOT+'test.csv')
sub = pd.read_csv(ROOT+'sample_submission.csv')


# In[ ]:


## Get the shape of the data
train_len, test_len ,sub_len = len(train.index), len(test.index),len(sub.index)
print(f'train size: {train_len}, test size: {test_len} , sample size: {sub_len}')


# In[ ]:


## These are those target data , many of which has created so much controversy . Mr Spearman has also commented NaN for many of them .
target_cols = ['question_asker_intent_understanding', 'question_body_critical', 
               'question_conversational', 'question_expect_short_answer', 
               'question_fact_seeking', 'question_has_commonly_accepted_answer', 
               'question_interestingness_others', 'question_interestingness_self', 
               'question_multi_intent', 'question_not_really_a_question', 
               'question_opinion_seeking', 'question_type_choice',
               'question_type_compare', 'question_type_consequence',
               'question_type_definition', 'question_type_entity', 
               'question_type_instructions', 'question_type_procedure', 
               'question_type_reason_explanation', 'question_type_spelling', 
               'question_well_written', 'answer_helpful',
               'answer_level_of_information', 'answer_plausible', 
               'answer_relevance', 'answer_satisfaction', 
               'answer_type_instructions', 'answer_type_procedure', 
               'answer_type_reason_explanation', 'answer_well_written']


# In[ ]:





# In[ ]:


# From the Ref Kernel's
from math import floor, ceil

def _get_masks(tokens, max_seq_length):
    """Mask for padding"""
    if len(tokens)>max_seq_length:
        raise IndexError("Token length more than max seq length!")
    return [1]*len(tokens) + [0] * (max_seq_length - len(tokens))

def _get_segments(tokens, max_seq_length):
    """Segments: 0 for the first sequence, 1 for the second"""
    
    if len(tokens) > max_seq_length:
        raise IndexError("Token length more than max seq length!")
        
    segments = []
    first_sep = True
    current_segment_id = 0
    
    for token in tokens:
        segments.append(current_segment_id)
        if token == "[SEP]": ## We dont need it for gpt2
            if first_sep:
                first_sep = False 
            else:
                current_segment_id = 1
    return segments + [0] * (max_seq_length - len(tokens))

def _get_ids(tokens, tokenizer, max_seq_length):
    """Token ids from Tokenizer vocab"""
    
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = token_ids + [0] * (max_seq_length-len(token_ids))
    return input_ids

def _trim_input(title, question, answer, max_sequence_length=512, t_max_len=30, q_max_len=241, a_max_len=241):
    
    #350+128+30 = 508 + 4 = 512
    
    t = tokenizer.tokenize(title)
    q = tokenizer.tokenize(question)
    a = tokenizer.tokenize(answer)
    
    t_len = len(t)
    q_len = len(q)
    a_len = len(a)

    if (t_len+q_len+a_len) > max_sequence_length:
        
        if t_max_len > t_len:
            t_new_len = t_len
            a_max_len = a_max_len + floor((t_max_len - t_len)/2)
            q_max_len = q_max_len + ceil((t_max_len - t_len)/2)
        else:
            t_new_len = t_max_len
      
        if a_max_len > a_len:
            a_new_len = a_len 
            q_new_len = q_max_len + (a_max_len - a_len)
        elif q_max_len > q_len:
            a_new_len = a_max_len + (q_max_len - q_len)
            q_new_len = q_len
        else:
            a_new_len = a_max_len
            q_new_len = q_max_len
            
            
        if t_new_len+a_new_len+q_new_len != max_sequence_length:
            raise ValueError("New sequence length should be %d, but is %d"%(max_sequence_length, (t_new_len + a_new_len + q_new_len)))
        q_len_head = round(q_new_len/2)
        q_len_tail = -1* (q_new_len -q_len_head)
        a_len_head = round(a_new_len/2)
        a_len_tail = -1* (a_new_len -a_len_head)        
        t = t[:t_new_len]
       # q = q[:q_new_len]
      
        q = q[:q_len_head]+q[q_len_tail:]
      #  print(len(q1))
        #a = a[:a_new_len]
        a = a[:a_len_head]+a[a_len_tail:]
    
    return t, q, a

def _convert_to_gpt2_inputs(title, question, answer, tokenizer, max_sequence_length):
    """Converts tokenized input to ids, masks and segments for GPT2"""
    
    stoken = title +  question + answer ## No CLS or SEP tokens for GPT2

    input_ids = _get_ids(stoken, tokenizer, max_sequence_length)
    input_masks = _get_masks(stoken, max_sequence_length)
    input_segments = _get_segments(stoken, max_sequence_length)

    return [input_ids, input_masks, input_segments]

def compute_input_arays(df, columns, tokenizer, max_sequence_length):
    
    input_ids, input_masks, input_segments = [], [], []
    for _, instance in df[columns].iterrows():
        t, q, a = instance.question_title, instance.question_body, instance.answer
        t, q, a = _trim_input(t, q, a, max_sequence_length)
        ids, masks, segments = _convert_to_gpt2_inputs(t, q, a, tokenizer, max_sequence_length)
        input_ids.append(ids)
        input_masks.append(masks)
        input_segments.append(segments)
    return [
        torch.from_numpy(np.asarray(input_ids, dtype=np.int32)).long(), 
        torch.from_numpy(np.asarray(input_masks, dtype=np.int32)).long(),
        torch.from_numpy(np.asarray(input_segments, dtype=np.int32)).long(),
    ]

def compute_output_arrays(df, columns):
    return np.asarray(df[columns])


# In[ ]:


class QuestDataset(torch.utils.data.Dataset):
    def __init__(self, inputs, lengths, labels = None):
        
        self.inputs = inputs
        if labels is not None:
            self.labels = labels
        else:
            self.labels = None
        self.lengths = lengths

    def __getitem__(self, idx):
        
        input_ids       = self.inputs[0][idx]
        input_masks     = self.inputs[1][idx]
        input_segments  = self.inputs[2][idx]
        lengths         = self.lengths[idx]
        if self.labels is not None: # targets
            labels = self.labels[idx]
            return input_ids, input_masks, input_segments, labels, lengths
        return input_ids, input_masks, input_segments, lengths

    def __len__(self):
        return len(self.inputs[0])


# In[ ]:


class SpatialDropout(nn.Dropout2d):
    def forward(self, x):
        x = x.unsqueeze(2)    # (N, T, 1, K)
        x = x.permute(0, 3, 2, 1)  # (N, K, 1, T)
        x = super(SpatialDropout, self).forward(x)  # (N, K, 1, T), some features are masked
        x = x.permute(0, 3, 2, 1)  # (N, T, 1, K)
        x = x.squeeze(2)  # (N, T, K)
        return x

##Sakami gave this 
class GPT2ClassificationHeadModel(GPT2Model):

    def __init__(self, config, clf_dropout=0.1, n_class=30):
        super(GPT2ClassificationHeadModel, self).__init__(config)
        self.transformer = GPT2Model(config)
        self.dropout = nn.Dropout(clf_dropout)
        self.linear = nn.Linear(config.n_embd*3 , n_class)

       # nn.init.normal_(self.linear.weight, std=0.02)
       # nn.init.normal_(self.linear.bias, 0)
        
        self.init_weights()

    def forward(self, input_ids, position_ids=None,attention_mask=None, token_type_ids=None, labels=None, past=None):
        hidden_states, presents = self.transformer(input_ids, position_ids,attention_mask, token_type_ids, past)
        avg_pool = torch.mean(hidden_states, 1)
        max_pool, _ = torch.max(hidden_states, 1)
        h_conc = torch.cat((avg_pool, max_pool, hidden_states[:, -1, :]), 1)
        logits = self.linear(self.dropout(h_conc))
        return logits


# In[ ]:


class NeuralNet3(nn.Module):
    def __init__(self,pretrained_bert):
        super(NeuralNet3, self).__init__()
        self.config = GPT2Config.from_pretrained(pretrained_bert, output_hidden_states=False)
        self.gptmodel = GPT2Model(self.config)
        self.gpt = self.gptmodel.from_pretrained( pretrained_bert,config =self.config)
         
        self.encoded_dropout = SpatialDropout( 0.2 )
        self.pooled_dropout = nn.Dropout( 0.2 )            

        
        dense_hidden_units = self.gpt.config.hidden_size * 3
        
        self.linear1 = nn.Linear( dense_hidden_units, dense_hidden_units )
        self.linear2 = nn.Linear( dense_hidden_units, dense_hidden_units )
        
        self.classifier = nn.Sequential(
            nn.Linear( dense_hidden_units, 30 ),
        )
        
    def forward( self, ids, masks, segments ):  
        
        outputs = self.gpt( input_ids=ids, attention_mask=masks, token_type_ids=segments)

        last_hidden_state,present = self.gpt( input_ids=ids, attention_mask=masks, token_type_ids=segments)

        last_hidden_state = self.encoded_dropout(last_hidden_state)

        avg_pool = torch.mean( last_hidden_state, 1 )
        max_pool, _ = torch.max( last_hidden_state, 1 )

        h_conc = torch.cat( ( avg_pool, max_pool, last_hidden_state[:, -1, :] ), 1 )
        h_conc_linear  = F.relu(self.linear1(h_conc))
        hidden = h_conc + h_conc_linear 
            
        h_conc_linear  = F.relu(self.linear2(hidden))
        hidden = hidden + h_conc_linear      
        
        return self.classifier( hidden )


# In[ ]:


##From combat-wombat
class GPT2CNN(GPT2PreTrainedModel):
    def __init__(self, config, num_labels):
        super().__init__(config)
        self.transformer = GPT2Model(config)
        self.cnn1 = nn.Conv1d(768, 256, kernel_size=3, padding=1)
        self.cnn2 = nn.Conv1d(256, num_labels, kernel_size=3, padding=1)

        self.init_weights()

    def forward(
        self,
        input_ids,
        position_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
        past=None,
    ):
        x, _ = self.transformer(input_ids, position_ids,attention_mask, token_type_ids,labels, past)
        x = x.permute(0, 2, 1)
        x = F.relu(self.cnn1(x))
        x = self.cnn2(x)
        output, _ = torch.max(x, 2)
        return output


# In[ ]:


from tqdm.notebook import tqdm


# In[ ]:


def train_model(train_loader, optimizer, criterion, scheduler,config):
    
    model.train()
    avg_loss = 0.
    
    tk0 = tqdm(enumerate(train_loader),total =len(train_loader))
    
    for idx, batch in tk0:

        input_ids, input_masks, input_segments, labels, _ = batch
        input_ids, input_masks, input_segments, labels = input_ids.to(device), input_masks.to(device), input_segments.to(device), labels.to(device)            
        
        output_train = model(ids = input_ids.long(),
                             masks = input_masks,
                             segments = input_segments,
                            )
        logits = output_train #output preds
        loss = criterion(logits,labels)
        #loss = config.question_weight*criterion(logits[:,0:21], labels[:,0:21]) + config.answer_weight*criterion(logits[:,21:30], labels[:,21:30])
        loss.backward()
        
        optimizer.step()
       # scheduler.step()
        optimizer.zero_grad()
        
        avg_loss += loss.item() / len(train_loader)
        del input_ids, input_masks, input_segments, labels

    torch.cuda.empty_cache()
    gc.collect()
    return avg_loss

def val_model(val_loader, val_shape, batch_size=4):

    avg_val_loss = 0.
    model.eval() # eval mode
    
    valid_preds = np.zeros((val_shape, len(target_cols)))
    original = np.zeros((val_shape, len(target_cols)))
    
    tk0 = tqdm(enumerate(val_loader))
    with torch.no_grad():
        
        for idx, batch in (tk0):
            input_ids, input_masks, input_segments, labels, _ = batch
            input_ids, input_masks, input_segments, labels = input_ids.to(device), input_masks.to(device), input_segments.to(device), labels.to(device)            
            
            output_val = model(ids = input_ids.long(),
                             masks = input_masks,
                             segments = input_segments,
                            )
            logits = output_val #output preds
            avg_val_loss += criterion(logits, labels).item() / len(val_loader)
            valid_preds[idx*batch_size : (idx+1)*batch_size] = logits.detach().cpu().squeeze().numpy()
            original[idx*batch_size : (idx+1)*batch_size]    = labels.detach().cpu().squeeze().numpy()
        
        score = 0
        preds = torch.sigmoid(torch.tensor(valid_preds)).numpy()
        
        # np.save("preds.npy", preds)
        # np.save("actuals.npy", original)
        
        rho_val = np.mean([spearmanr(original[:, i], preds[:,i]).correlation for i in range(preds.shape[1])])
        print('\r val_spearman-rho: %s' % (str(round(rho_val, 5))), end = 100*' '+'\n')
        
        for i in range(len(target_cols)):
            print(i, spearmanr(original[:,i], preds[:,i]))
            score += np.nan_to_num(spearmanr(original[:, i], preds[:, i]).correlation)
    return avg_val_loss, score*1.0/len(target_cols)


# In[ ]:


def predict_result(model, test_loader, batch_size=32):
    
    test_preds = np.zeros((len(test), len(target_cols)))
    
    model.eval();
    tk0 = tqdm(enumerate(test_loader))
    for idx, x_batch in tk0:
        with torch.no_grad():
            outputs = model(ids = x_batch[0].to(device), 
                            masks = x_batch[1].to(device),
                            segments = x_batch[2].to(device),
                           )
            predictions = outputs[0]
            test_preds[idx*batch_size : (idx+1)*batch_size] = predictions.detach().cpu().squeeze().numpy()

    output = torch.sigmoid(torch.tensor(test_preds)).numpy()        
    return output


# In[ ]:


get_ipython().system('ls ../input')


# In[ ]:


### Here is the GPT2 part 
tokenizer = GPT2Tokenizer.from_pretrained('../input/gpt2-models/', cache_dir=None)
input_categories = list(train.columns[[1,2,5]]); input_categories

#bert_model_config = '../input/pretrained-bert-models-for-pytorch/bert-base-uncased/bert_config.json'
gpt2_config =  GPT2Config.from_pretrained('../input/gpt2-models/')
#gpt2_config.n_embd = 30

gpt2_model = 'gpt2'
#do_lower_case = 'uncased' in bert_model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
#output_model_file = 'bert_pytorch.bin'

from sklearn.model_selection import GroupKFold

test_inputs = compute_input_arays(test, input_categories, tokenizer, max_sequence_length=512)
lengths_test = np.argmax(test_inputs[0] == 0, axis=1)
lengths_test[lengths_test == 0] = test_inputs[0].shape[1]


# In[ ]:


# 'Cyclical Learning Rates for Training Neural Networks'- Leslie N. Smith, arxiv 2017
#       https://arxiv.org/abs/1506.01186
#       https://github.com/bckenstler/CLR

class CyclicScheduler1():

    def __init__(self, min_lr=0.001, max_lr=0.01, period=10 ):
        super(CyclicScheduler1, self).__init__()

        self.min_lr = min_lr
        self.max_lr = max_lr
        self.period = period

    def __call__(self, time):

        #sawtooth
        #r = (1-(time%self.period)/self.period)

        #cosine
        time= time%self.period
        r = (np.cos(time/self.period *np.pi)+1)/2

        lr = self.min_lr + r*(self.max_lr-self.min_lr)
        return lr

    def __str__(self):
        string = 'CyclicScheduler\n'                 + 'min_lr=%0.3f, max_lr=%0.3f, period=%8.1f'%(self.min_lr, self.max_lr, self.period)
        return string


class CyclicScheduler2():

    def __init__(self, min_lr=0.001, max_lr=0.01, period=10, max_decay=1.0, warm_start=0 ):
        super(CyclicScheduler2, self).__init__()

        self.min_lr = min_lr
        self.max_lr = max_lr
        self.period = period
        self.max_decay = max_decay
        self.warm_start = warm_start
        self.cycle = -1

    def __call__(self, time):
        if time<self.warm_start: return self.max_lr

        #cosine
        self.cycle = (time-self.warm_start)//self.period
        time = (time-self.warm_start)%self.period

        period = self.period
        min_lr = self.min_lr
        max_lr = self.max_lr *(self.max_decay**self.cycle)


        r   = (np.cos(time/period *np.pi)+1)/2
        lr = min_lr + r*(max_lr-min_lr)

        return lr



    def __str__(self):
        string = 'CyclicScheduler\n'                 + 'min_lr=%0.4f, max_lr=%0.4f, period=%8.1f'%(self.min_lr, self.max_lr, self.period)
        return string


#tanh curve
class CyclicScheduler3():

    def __init__(self, min_lr=0.001, max_lr=0.01, period=10, max_decay=1.0, warm_start=0 ):
        super(CyclicScheduler3, self).__init__()

        self.min_lr = min_lr
        self.max_lr = max_lr
        self.period = period
        self.max_decay = max_decay
        self.warm_start = warm_start
        self.cycle = -1

    def __call__(self, time):
        if time<self.warm_start: return self.max_lr

        #cosine
        self.cycle = (time-self.warm_start)//self.period
        time = (time-self.warm_start)%self.period

        period = self.period
        min_lr = self.min_lr
        max_lr = self.max_lr *(self.max_decay**self.cycle)

        r   = (np.tanh(-time/period *16 +8)+1)*0.5
        lr = min_lr + r*(max_lr-min_lr)

        return lr



    def __str__(self):
        string = 'CyclicScheduler\n'                 + 'min_lr=%0.3f, max_lr=%0.3f, period=%8.1f'%(self.min_lr, self.max_lr, self.period)
        return string


class NullScheduler():
    def __init__(self, lr=0.01 ):
        super(NullScheduler, self).__init__()
        self.lr    = lr
        self.cycle = 0

    def __call__(self, time):
        return self.lr

    def __str__(self):
        string = 'NullScheduler\n'                 + 'lr=%0.5f '%(self.lr)
        return string


# net ------------------------------------
# https://github.com/pytorch/examples/blob/master/imagenet/main.py ###############
def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def get_learning_rate(optimizer):
    lr=[]
    for param_group in optimizer.param_groups:
        lr +=[ param_group['lr'] ]

    assert(len(lr)==1) #we support only one param_group
    lr = lr[0]

    return lr


# In[ ]:


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


# In[8]:

NUM_FOLDS = 6  # change this
SEED = config.seed
BATCH_SIZE = 4
epochs = config.epochs   # change this
ACCUM_STEPS = 1

kf = GroupKFold(n_splits = NUM_FOLDS)

test_set = QuestDataset(inputs=test_inputs, lengths=lengths_test, labels=None)
test_loader  = DataLoader(test_set, batch_size=32, shuffle=False)
result = np.zeros((len(test), 30))

y_train = train[target_cols].values # dummy

print(bcolors.FAIL, f"For Every Fold, Train {epochs} Epochs", bcolors.ENDC)

for fold, (train_index, val_index) in enumerate(kf.split(X=train.values, groups=train.question_body)):
    if fold > 2:
        break
    print(bcolors.HEADER, "Current Fold:", fold, bcolors.ENDC)

    train_df, val_df = train.iloc[train_index], train.iloc[val_index]
    print("Train and Valid Shapes are", train_df.shape, val_df.shape)
    
    print(bcolors.HEADER, "Preparing train datasets....", bcolors.ENDC)
    
    inputs_train = compute_input_arays(train_df, input_categories, tokenizer, max_sequence_length=512)
    outputs_train = compute_output_arrays(train_df, columns = target_cols)
    outputs_train = torch.tensor(outputs_train, dtype=torch.float32)
    lengths_train = np.argmax(inputs_train[0] == 0, axis=1)
    lengths_train[lengths_train == 0] = inputs_train[0].shape[1]
    
    print(bcolors.HEADER, "Preparing Valid datasets....", bcolors.ENDC)
    
    inputs_valid = compute_input_arays(val_df, input_categories, tokenizer, max_sequence_length=512)
    outputs_valid = compute_output_arrays(val_df, columns = target_cols)
    outputs_valid = torch.tensor(outputs_valid, dtype=torch.float32)
    lengths_valid = np.argmax(inputs_valid[0] == 0, axis=1)
    lengths_valid[lengths_valid == 0] = inputs_valid[0].shape[1]
    
    print(bcolors.HEADER, "Preparing Dataloaders Datasets....", bcolors.ENDC)

    train_set    = QuestDataset(inputs=inputs_train, lengths=lengths_train, labels=outputs_train)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    
    valid_set    = QuestDataset(inputs=inputs_valid, lengths=lengths_valid, labels=outputs_valid)
    valid_loader = DataLoader(valid_set, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    
    model = NeuralNet3(pretrained_bert)
    model.zero_grad();
    model.to(device);
    torch.cuda.empty_cache()
    model.train();
    
    i = 0
    best_avg_loss   = 100.0
    best_score      = -1.
    best_param_loss = None
    best_param_score = None

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, eps=4e-5)
    criterion = nn.BCEWithLogitsLoss()
    #scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=config.warmup, num_training_steps= epochs*len(train_loader)//ACCUM_STEPS)
    scheduler = CyclicScheduler2( min_lr=0.000002, max_lr=0.000035, period=4, warm_start=0, max_decay=0.9 )

    print("Training....")
    
    for epoch in range(epochs):
        lr = scheduler( epoch )
        adjust_learning_rate(optimizer, lr )
        torch.cuda.empty_cache()
        
        start_time   = time.time()
        avg_loss     = train_model(train_loader, optimizer, criterion, scheduler,config)
        avg_val_loss, score = val_model(valid_loader, val_shape=val_df.shape[0],batch_size =BATCH_SIZE)
        elapsed_time = time.time() - start_time

        print(bcolors.WARNING, 'Epoch {}/{} \t loss={:.4f} \t val_loss={:.4f} \t score={:.6f} \t time={:.2f}s'.format(
            epoch + 1, epochs, avg_loss, avg_val_loss, score, elapsed_time),
        bcolors.ENDC
        )

        if best_avg_loss > avg_val_loss:
            i = 0
            best_avg_loss = avg_val_loss 
            best_param_loss = model.state_dict()

        if best_score < score:
            best_score = score
            best_param_score = model.state_dict()
            torch.save(best_param_score, 'best_param_score_{}_{}.pt'.format(config.expname ,fold+1))
        else:
            i += 1

    model.load_state_dict(best_param_score)
    result += predict_result(model, test_loader)
    torch.save(best_param_score, 'best_param_score_{}_{}.pt'.format(config.expname ,fold+1))

    del train_df, val_df, model, optimizer, criterion, scheduler
    torch.cuda.empty_cache()
    del valid_loader, train_loader, valid_set, train_set
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    gc.collect()

NUM_FOLDS =1
result /= NUM_FOLDS
print(result)


# In[ ]:


submission = pd.read_csv(r'../input/google-quest-challenge/sample_submission.csv')
submission.loc[:, 'question_asker_intent_understanding':] = result

submission.to_csv('submission.csv', index=False)
submission.head()


# In[ ]:


for i in range(30):
    print(submission[target_cols[i]].value_counts())
    


# In[ ]:




