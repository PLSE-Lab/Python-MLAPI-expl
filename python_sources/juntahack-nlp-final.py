#!/usr/bin/env python
# coding: utf-8

# # JanataHack_AV_NLP Hackathon
# 
# **Final Approach**
# 
# Hi All. I am releasing this kernel as many of us would love to see the appraoch of the winning kernels in the different competitions that adds to the great learning one gets while competing in those. If you like this please upvote the kernel. In case you have some suggestion or some feedback you would like to provide, please feel free to comment!! Enjoy :)

# In[ ]:


import numpy as np
import pandas as pd


# In[ ]:


path = '/kaggle/input/janata-data/'


# In[ ]:


path1 = '/kaggle/input/covid-19-tweet-classification/'


# In[ ]:


trainD = pd.read_csv(path+'train.csv')
testD = pd.read_csv(path+'test.csv')
gameoverviews = pd.read_csv(path+'game_overview.csv')
testD.head()


# In[ ]:


trainD = pd.read_csv(path1+'updated_train.csv')
testD = pd.read_csv(path1+'updated_test.csv')


# In[ ]:


gameoverviews.head()


# In[ ]:


trainD.head()


# In[ ]:


bigTrain=trainD.merge(gameoverviews,how='left',on='title')


# In[ ]:


bigTrain.head()


# In[ ]:


bigTest=testD.merge(gameoverviews,how='left',on='title')


# # Straight Into Transfer Learning - Roberta + FastAI

# In[ ]:


from fastai import *
from fastai.text import *
import os


# **Installing HuggingFace Transformers**

# In[ ]:


get_ipython().run_cell_magic('bash', '', 'pip install -q transformers')


# In[ ]:


from pathlib import Path 

import os

import torch
import torch.optim as optim

import random 

from fastai.callbacks import *

# transformers
from transformers import PreTrainedModel, PreTrainedTokenizer, PretrainedConfig

from transformers import BertForSequenceClassification, BertTokenizer, BertConfig
from transformers import RobertaForSequenceClassification, RobertaTokenizer, RobertaConfig
from transformers import XLNetForSequenceClassification, XLNetTokenizer, XLNetConfig
from transformers import XLMForSequenceClassification, XLMTokenizer, XLMConfig
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer, DistilBertConfig
from transformers import AlbertForSequenceClassification, AlbertTokenizer, AlbertConfig


# In[ ]:


import fastai
import transformers
print('fastai version :', fastai.__version__)
print('transformers version :', transformers.__version__)


# # Setting Model Classes to try out

# In[ ]:


MODEL_CLASSES = {
    'bert': (BertForSequenceClassification, BertTokenizer, BertConfig),
    'xlnet': (XLNetForSequenceClassification, XLNetTokenizer, XLNetConfig),
    'xlm': (XLMForSequenceClassification, XLMTokenizer, XLMConfig),
    'roberta': (RobertaForSequenceClassification, RobertaTokenizer, RobertaConfig),
    'distilbert': (DistilBertForSequenceClassification, DistilBertTokenizer, DistilBertConfig),
    'albert':(AlbertForSequenceClassification,AlbertTokenizer, AlbertConfig)
}


# In[ ]:


# Parameters
seed = 10
use_fp16 = True
bs = 4

model_type = 'roberta'
pretrained_model_name = 'roberta-large'


# In[ ]:


model_class, tokenizer_class, config_class = MODEL_CLASSES[model_type]


# **Setting Seed**

# In[ ]:


def seed_all(seed_value):
    random.seed(seed_value) # Python
    np.random.seed(seed_value) # cpu vars
    torch.manual_seed(seed_value) # cpu  vars
    
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # gpu vars
        torch.backends.cudnn.deterministic = True  #needed
        torch.backends.cudnn.benchmark = False


# In[ ]:


seed_all(seed)


# # Using Tokenizer Class

# In[ ]:


class TransformersBaseTokenizer(BaseTokenizer):
    """Wrapper around PreTrainedTokenizer to be compatible with fast.ai"""
    def __init__(self, pretrained_tokenizer: PreTrainedTokenizer, model_type = 'bert', **kwargs):
        self._pretrained_tokenizer = pretrained_tokenizer
        self.max_seq_len = pretrained_tokenizer.max_len
        self.model_type = model_type

    def __call__(self, *args, **kwargs): 
        return self

    def tokenizer(self, t:str) -> List[str]:
        """Limits the maximum sequence length and add the spesial tokens"""
        CLS = self._pretrained_tokenizer.cls_token
        SEP = self._pretrained_tokenizer.sep_token
        if self.model_type in ['roberta']:
            tokens = self._pretrained_tokenizer.tokenize(t, add_prefix_space=True)[:self.max_seq_len - 2]
            tokens = [CLS] + tokens + [SEP]
        else:
            tokens = self._pretrained_tokenizer.tokenize(t)[:self.max_seq_len - 2]
            if self.model_type in ['xlnet']:
                tokens = tokens + [SEP] +  [CLS]
            else:
                tokens = [CLS] + tokens + [SEP]
        return tokens


# In[ ]:


transformer_tokenizer = tokenizer_class.from_pretrained(pretrained_model_name)
transformer_base_tokenizer = TransformersBaseTokenizer(pretrained_tokenizer = transformer_tokenizer, model_type = model_type)
fastai_tokenizer = Tokenizer(tok_func = transformer_base_tokenizer, pre_rules=[], post_rules=[])


# ** Converting Text to Features**

# In[ ]:


class TransformersVocab(Vocab):
    def __init__(self, tokenizer: PreTrainedTokenizer):
        super(TransformersVocab, self).__init__(itos = [])
        self.tokenizer = tokenizer
    
    def numericalize(self, t:Collection[str]) -> List[int]:
        "Convert a list of tokens `t` to their ids."
        return self.tokenizer.convert_tokens_to_ids(t)
        #return self.tokenizer.encode(t)

    def textify(self, nums:Collection[int], sep=' ') -> List[str]:
        "Convert a list of `nums` to their tokens."
        nums = np.array(nums).tolist()
        return sep.join(self.tokenizer.convert_ids_to_tokens(nums)) if sep is not None else self.tokenizer.convert_ids_to_tokens(nums)
    
    def __getstate__(self):
        return {'itos':self.itos, 'tokenizer':self.tokenizer}

    def __setstate__(self, state:dict):
        self.itos = state['itos']
        self.tokenizer = state['tokenizer']
        self.stoi = collections.defaultdict(int,{v:k for k,v in enumerate(self.itos)})


# In[ ]:


transformer_vocab =  TransformersVocab(tokenizer = transformer_tokenizer)
numericalize_processor = NumericalizeProcessor(vocab=transformer_vocab)

tokenize_processor = TokenizeProcessor(tokenizer=fastai_tokenizer, include_bos=False, include_eos=False)

transformer_processor = [tokenize_processor, numericalize_processor]


# ** Padding Sequence ** 

# In[ ]:


pad_first = bool(model_type in ['xlnet'])
pad_idx = transformer_tokenizer.pad_token_id


# ** Creating Data Bunch **

# In[ ]:


data_transclas = (TextList.from_df(bigTrain, cols=['title','user_review','overview','tags','developer','publisher'], processor=transformer_processor)
             .split_by_rand_pct(0.1,seed=seed)
             .label_from_df(cols= 'user_suggestion').databunch(bs=bs, pad_first=pad_first, pad_idx=pad_idx))


# In[ ]:


data_transclas = (TextList.from_df(trainD, cols=['text'], processor=transformer_processor)
             .split_by_rand_pct(0.1,seed=seed)
             .label_from_df(cols= 'target').databunch(bs=bs, pad_first=pad_first, pad_idx=pad_idx))


# # Checking the Tokens

# In[ ]:


print('[CLS] token :', transformer_tokenizer.cls_token)
print('[SEP] token :', transformer_tokenizer.sep_token)
print('[PAD] token :', transformer_tokenizer.pad_token)
data_transclas.show_batch()


# In[ ]:


print('[CLS] id :', transformer_tokenizer.cls_token_id)
print('[SEP] id :', transformer_tokenizer.sep_token_id)
print('[PAD] id :', pad_idx)
test_one_batch = data_transclas.one_batch()[0]
print('Batch shape : ',test_one_batch.shape)
print(test_one_batch)


# # Custom Transformer Model with attention mask (Need to understand More)

# In[ ]:


# defining our model architecture 
class CustomTransformerModel(nn.Module):
    def __init__(self, transformer_model: PreTrainedModel):
        super(CustomTransformerModel,self).__init__()
        self.transformer = transformer_model
        
    def forward(self, input_ids, attention_mask=None):
        
        # attention_mask
        # Mask to avoid performing attention on padding token indices.
        # Mask values selected in ``[0, 1]``:
        # ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.
        attention_mask = (input_ids!=pad_idx).type(input_ids.type()) 
        
        logits = self.transformer(input_ids,
                                  attention_mask = attention_mask)[0]   
        return logits


# # Tweaking the Configs for this use case

# In[ ]:


config = config_class.from_pretrained(pretrained_model_name)
config.num_labels = 2
config.use_bfloat16 = use_fp16
print(config)


# # Defining The Model

# In[ ]:


transformer_model = model_class.from_pretrained(pretrained_model_name, config = config)

custom_transformer_model = CustomTransformerModel(transformer_model = transformer_model)


# # Creating Learner Instance

# In[ ]:


from fastai.callbacks import *
from transformers import AdamW
from functools import partial

CustomAdamW = partial(AdamW, correct_bias=False)

learner = Learner(data_transclas, 
                  custom_transformer_model, 
                  opt_func = CustomAdamW, 
                  metrics=[accuracy, error_rate])

# Show graph of learner stats and metrics after each epoch.
learner.callbacks.append(ShowGraph(learner))

# Put learn in FP16 precision mode. --> Seems to not working
if use_fp16: learner = learner.to_fp16()


# In[ ]:


print(learner.model)


# # Layer Splitting

# In[ ]:


list_layers = [learner.model.transformer.roberta.embeddings,
              learner.model.transformer.roberta.encoder.layer[0],
              learner.model.transformer.roberta.encoder.layer[1],
              learner.model.transformer.roberta.encoder.layer[2],
              learner.model.transformer.roberta.encoder.layer[3],
              learner.model.transformer.roberta.encoder.layer[4],
              learner.model.transformer.roberta.encoder.layer[5],
              learner.model.transformer.roberta.encoder.layer[6],
              learner.model.transformer.roberta.encoder.layer[7],
              learner.model.transformer.roberta.encoder.layer[8],
              learner.model.transformer.roberta.encoder.layer[9],
              learner.model.transformer.roberta.encoder.layer[10],
              learner.model.transformer.roberta.encoder.layer[11],
              learner.model.transformer.roberta.pooler]


# In[ ]:


learner.split(list_layers)
num_groups = len(learner.layer_groups)
print('Learner split in',num_groups,'groups')
print(learner.layer_groups)


# # Training Model Starts - Progressive Unfreezing

# In[ ]:


learner.freeze_to(-1)


# In[ ]:


learner.summary()


# # LR finder

# In[ ]:


#learner.lr_find()


# In[ ]:


#learner.recorder.plot(skip_end=10,suggestion=True)


# # Model Fitting

# In[ ]:


learner.fit_one_cycle(1,max_lr=1e-4,moms=(0.8,0.7))


# In[ ]:


#learner.save('first_cycle')


# In[ ]:


#seed_all(seed)
#learner.load('first_cycle');


# # Repeat the same

# In[ ]:


learner.freeze_to(-2)


# In[ ]:


# learner.lr_find()
# learner.recorder.plot(skip_end=10,suggestion=True)


# In[ ]:


lr = 1e-5


# In[ ]:


learner.fit_one_cycle(1, max_lr=slice(lr*0.95**num_groups, lr), moms=(0.8, 0.9))


# In[ ]:


#learner.save('second_cycle')


# In[ ]:


# seed_all(seed)
# import gc
# del learner
# gc.collect()
# learner = Learner(data_transclas, 
#                   custom_transformer_model, 
#                   opt_func = CustomAdamW, 
#                   metrics=[accuracy, error_rate])

# # Show graph of learner stats and metrics after each epoch.
# learner.callbacks.append(ShowGraph(learner))

# # Put learn in FP16 precision mode. --> Seems to not working
# if use_fp16: learner = learner.to_fp16()
# learner.split(list_layers)
# learner.load('second_cycle');
learner.freeze_to(-3)
learner.fit_one_cycle(1, max_lr=slice(lr*0.95**num_groups, lr), moms=(0.8, 0.9))


# In[ ]:


#learner.save('third_cycle')


# In[ ]:


# seed_all(seed)
# del learner
# gc.collect()
# learner = Learner(data_transclas, 
#                   custom_transformer_model, 
#                   opt_func = CustomAdamW, 
#                   metrics=[accuracy, error_rate])

# # Show graph of learner stats and metrics after each epoch.
# learner.callbacks.append(ShowGraph(learner))

# # Put learn in FP16 precision mode. --> Seems to not working
# if use_fp16: learner = learner.to_fp16()
# learner.split(list_layers)
# learner.load('third_cycle');


# In[ ]:


learner.unfreeze()


# In[ ]:


learner.fit_one_cycle(2, max_lr=slice(lr*0.95**num_groups, lr), moms=(0.8, 0.9))


# # Saving The final Model

# In[ ]:


learner.export(file = 'transformer_robertalarge_full.pkl');


# In[ ]:


path = '/kaggle/working'
export_learner = load_learner(path, file = 'transformer_robertalarge_full.pkl')


# # Getting Final PRedictions

# In[ ]:


from tqdm._tqdm_notebook import tqdm_notebook
tqdm_notebook.pandas()
bigTest['user_suggestion']= bigTest.progress_apply(lambda x: learner.predict(x)[0], axis=1)
bigTest[['review_id','user_suggestion']].to_csv('submission.csv',index=False)


# In[ ]:


from tqdm._tqdm_notebook import tqdm_notebook
tqdm_notebook.pandas()
testD['target']= testD.progress_apply(lambda x: learner.predict(x)[2][1], axis=1)
testD[['ID','target']].to_csv('submission3.csv',index=False)


# In[ ]:


from IPython.display import HTML

def create_download_link(title = "Download CSV file", filename = "data.csv"):  
    html = '<a href={filename}>{title}</a>'
    html = html.format(title=title,filename=filename)
    return HTML(html)

# create a link to download the dataframe which was saved with .to_csv method
create_download_link(filename='submission1.csv')


# In[ ]:




