#!/usr/bin/env python
# coding: utf-8

# I've been taking the [fastai MOOC](https://course.fast.ai/) and wanted to get my hands dirty implementing its Universal Language Model Fine-tuning for Text Classification ("ULMFiT")[1]. However,  it seems impossible to use ULMFiT without breaking the competition rules (no internet/gpu). So instead of joining the competition, I thought I'd just use this as a learning experience.
# 
# Note: I stole most of the code from the original fastai jupyter notebook [2]. If you are new to deep learning, I highly recommend checking out their [website](https://course.fast.ai/).

# In[ ]:


get_ipython().system('pip uninstall -y torch')
get_ipython().system('conda install -y pytorch=0.3.1.0 cuda80 -c soumith')
import torch
get_ipython().system('pip install fastai==0.7.0')
get_ipython().system('pip uninstall -y torchtext')
get_ipython().system('pip install torchtext==0.2.3')


# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import gc
import datetime
from sklearn.model_selection import train_test_split
from sklearn import model_selection

#https://www.kaggle.com/bguberfain/a-simple-model-using-the-market-and-news-data
from itertools import chain

from fastai.text import *
import html

from kaggle.competitions import twosigmanews
env = twosigmanews.make_env()


# ### Data Preparation
# We will need to take the 2 dataframes and clean them up.

# In[ ]:


(market_train_df, news_train_df) = env.get_training_data()


# Defining tags for the language model. This allows the model to learn when a sentence starts and also data fields, if any.

# In[ ]:


BOS = 'xbos'  # beginning-of-sentence tag
FLD = 'xfld'  # data field tag

PATH=Path('/kaggle/working')


# ULMFiT involves 2 main types of models:
# 1. Language Model (LM)
# 1. Classifier Model (CLAS)
# 
# We first train a language model on a large generic language corpus like wikipedia, then use **discriminative differential fine-tuning** and **slanted triangular learning rates** on the news headlines dataset to fine tune the model. After that, we fit a 'linear classifier head' on top of the language model to do our classification.

# In[ ]:


CLAS_PATH=Path('classifier/')
CLAS_PATH.mkdir(exist_ok=True)

LM_PATH=Path('lang_model/')
LM_PATH.mkdir(exist_ok=True)


# Due to kernel limitations and lack of patience, I've shrunk the dataset to a really tiny subsample. ULMFiT is supposed to work even for really small datasets (100 labelled examples) - this is one of the breakthroughs of the paper [2]. I guess this can be another test to see if it works for news headlines as well.

# In[ ]:


toy = True
if toy:
    market_train_df = market_train_df.tail(4_000)
    news_train_df = news_train_df.tail(12_000)
else:
    market_train_df = market_train_df.tail(3_000_000)
    news_train_df = news_train_df.tail(6_000_000)


# Aligning the time in both datasets for joining later.

# In[ ]:


# Split date into before and after 22h (the time used in train data)
# E.g: 2007-03-07 23:26:39+00:00 -> 2007-03-08 00:00:00+00:00 (next day)
#      2009-02-25 21:00:50+00:00 -> 2009-02-25 00:00:00+00:00 (current day)
news_train_df['time'] = (news_train_df['time'] - np.timedelta64(22,'h')).dt.ceil('1D')

# Round time of market_train_df to 0h of curret day
market_train_df['time'] = market_train_df['time'].dt.floor('1D')


# In[ ]:


#function to combine dataframes. Stole and adapted from https://www.kaggle.com/bguberfain/a-simple-model-using-the-market-and-news-data
def join_market_news(market_train_df, news_train_df):
    # Fix asset codes (str -> list)
    news_train_df['assetCodes'] = news_train_df['assetCodes'].str.findall(f"'([\w\./]+)'")    
    
    #rename headline column to text for later
    news_train_df = news_train_df.rename(columns={'headline':'text'})
    
    # Expand assetCodes
    assetCodes_expanded = list(chain(*news_train_df['assetCodes']))
    assetCodes_index = news_train_df.index.repeat( news_train_df['assetCodes'].apply(len) )
    
    assert len(assetCodes_index) == len(assetCodes_expanded)
    df_assetCodes = pd.DataFrame({'level_0': assetCodes_index, 'assetCode': assetCodes_expanded})

    # Create expanded news (will repeat every assetCodes' row)
    news_cols = ['time', 'assetCodes', 'text']
    news_train_df_aggregated = pd.merge(df_assetCodes, news_train_df[news_cols], left_on='level_0', right_index=True, suffixes=(['','_old']))

    # Free memory
    del news_train_df, df_assetCodes

    # Flat columns
    #news_train_df_aggregated.columns = ['_'.join(col).strip() for col in news_train_df_aggregated.columns.values]

    # Join with train
    market_train_df = pd.merge(market_train_df, news_train_df_aggregated, on=['time', 'assetCode'], how='inner')
    #market_train_df = market_train_df.join(news_train_df_aggregated, on=['time', 'assetCode'])

    # Free memory
    del news_train_df_aggregated
    
    return market_train_df


# In[ ]:


df = join_market_news(market_train_df, news_train_df)
df.reset_index(inplace=True)


# In[ ]:


#collect free memory if any
del news_train_df
del market_train_df
gc.collect()


# In[ ]:


#reduce file size from https://www.kaggle.com/c/two-sigma-financial-news/discussion/68265 and https://www.kaggle.com/gemartin/load-data-reduce-memory-usage
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype.name
        
        if col_type not in ['object', 'category', 'datetime64[ns, UTC]']:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        #else:
            #df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df

df = reduce_mem_usage(df)


# Creating binary labels, neg and pos. I kept it simple, with either 1 for positive returns and 0 for none/negative returns.

# In[ ]:


CLASSES = ['neg', 'pos']


# In[ ]:


df['labels'] = df.returnsOpenNextMktres10.apply(lambda x: 1 if x>0 else 0)


# In[ ]:


news = df[['labels', 'text']]


# In[ ]:


del df
gc.collect()


# In[ ]:


#Creating train/val set
df_trn, df_val = train_test_split(news, test_size=0.33, random_state=42)


# In[ ]:


df_trn.head(10)


# In[ ]:


df_trn.to_csv(CLAS_PATH/'train.csv',header=False, index=False)
df_val.to_csv(CLAS_PATH/'test.csv',header=False, index=False)
(CLAS_PATH/'classes.txt').open('w', encoding='utf-8').writelines(f'{o}\n' for o in CLASSES)


# In[ ]:


trn_texts,val_texts = sklearn.model_selection.train_test_split(
    np.concatenate([df_trn.iloc[:,1],df_val.iloc[:,1]]), test_size=0.1)


# In[ ]:


df_trn = pd.DataFrame({'text':trn_texts, 'labels':[0]*len(trn_texts)})
df_val = pd.DataFrame({'text':val_texts, 'labels':[0]*len(val_texts)})

df_trn.to_csv(LM_PATH/'train.csv', header=False, index=False)
df_val.to_csv(LM_PATH/'test.csv', header=False, index=False)


# Helper functions clean up, grab the text and labels while adding the tags defined earlier and tokenises the text using spacy. Thanks to fastai, there is multi-processing which helps to speed things up.
# 
# For a quick explanation of tokenisation and other NLP terms, [spacy's docs site](https://spacy.io/usage/spacy-101#annotations-token) has great code snippets that you can run to see what each NLP term really means. 

# In[ ]:


re1 = re.compile(r'  +')

def fixup(x):
    x = x.replace('#39;', "'").replace('amp;', '&').replace('#146;', "'").replace(
        'nbsp;', ' ').replace('#36;', '$').replace('\\n', "\n").replace('quot;', "'").replace(
        '<br />', "\n").replace('\\"', '"').replace('<unk>','u_n').replace(' @.@ ','.').replace(
        ' @-@ ','-').replace('\\', ' \\ ')
    return re1.sub(' ', html.unescape(x))


# In[ ]:


def get_texts(df, n_lbls=1):
    labels = df.iloc[:,range(n_lbls)].values.astype(np.int64)
    texts = f'\n{BOS} {FLD} 1 ' + df[n_lbls].astype(str)
    for i in range(n_lbls+1, len(df.columns)): texts += f' {FLD} {i-n_lbls} ' + df[i].astype(str)
    texts = list(texts.apply(fixup).values)

    tok = Tokenizer().proc_all_mp(partition_by_cores(texts))
    return tok, list(labels)


# In[ ]:


def get_all(df, n_lbls):
    tok, labels = [], []
    for i, r in enumerate(df):
        print(i)
        tok_, labels_ = get_texts(r, n_lbls)
        tok += tok_;
        labels += labels_
    return tok, labels


# Chunking helps to reduce the memory load as pandas will not read the entire file at once, but in chunks. Probably not very important our use case as I'm using a very small sample.

# In[ ]:


chunksize=24000


# In[ ]:


df_trn = pd.read_csv(LM_PATH/'train.csv', header=None, chunksize=chunksize)
df_val = pd.read_csv(LM_PATH/'test.csv', header=None, chunksize=chunksize)


# In[ ]:


#tokenising
tok_trn, trn_labels = get_all(df_trn, 0)
tok_val, val_labels = get_all(df_val, 0)


# In[ ]:


(LM_PATH/'tmp').mkdir(exist_ok=True)


# In[ ]:


np.save(LM_PATH/'tmp'/'tok_trn.npy', tok_trn)
np.save(LM_PATH/'tmp'/'tok_val.npy', tok_val)


# In[ ]:


tok_trn = np.load(LM_PATH/'tmp'/'tok_trn.npy')
tok_val = np.load(LM_PATH/'tmp'/'tok_val.npy')


# Showing top 25 most common words. Most of the tokens are stop words. Only interesting ones to me are research and $. In case you're wondering, t_up means the word after the tag is in caps.

# In[ ]:


freq = Counter(p for o in tok_trn for p in o)
freq.most_common(25)


# Limiting the maximum vocabulary we will allow in our model to 60,000 (reason explained in the [lesson notebook](https://github.com/fastai/fastai/blob/master/courses/dl2/imdb.ipynb)) and the minimum frequency of a word to be considered in the vocabulary. 

# In[ ]:


max_vocab = 60000
min_freq = 2


# itos stands for integer to string. It's essentially indexing each word in our vocabulary like a dictionary. Adding 2 more tokens to the dictionary, padding and unknown, i.e. new words

# In[ ]:


itos = [o for o,c in freq.most_common(max_vocab) if c>min_freq]
itos.insert(0, '_pad_')
itos.insert(0, '_unk_')


# Going the way round, mapping string to integer. As you can see, we don't have that many tokens.

# In[ ]:


stoi = collections.defaultdict(lambda:0, {v:k for k,v in enumerate(itos)})
len(itos)


# In[ ]:


trn_lm = np.array([[stoi[o] for o in p] for p in tok_trn])
val_lm = np.array([[stoi[o] for o in p] for p in tok_val])


# In[ ]:


np.save(LM_PATH/'tmp'/'trn_ids.npy', trn_lm)
np.save(LM_PATH/'tmp'/'val_ids.npy', val_lm)
pickle.dump(itos, open(LM_PATH/'tmp'/'itos.pkl', 'wb'))


# In[ ]:


trn_lm = np.load(LM_PATH/'tmp'/'trn_ids.npy')
val_lm = np.load(LM_PATH/'tmp'/'val_ids.npy')
itos = pickle.load(open(LM_PATH/'tmp'/'itos.pkl', 'rb'))


# In[ ]:


vs=len(itos)
vs,len(trn_lm)


# ### Creating our universal language model using wikitext103
# 
# Now that we have gotten our data in the right shape and form, we are going to create our universal language model. fastai has made available the weights that are used in wikitext103 model, so if we download it and initialise the model with the weights, we don't need to train the model again.

# In[ ]:


get_ipython().system(' wget -nH -r -np -P {PATH} http://files.fast.ai/models/wt103/bwd_wt103.h5')
get_ipython().system(' wget -nH -r -np -P {PATH} http://files.fast.ai/models/wt103/bwd_wt103_enc.h5')
get_ipython().system(' wget -nH -r -np -P {PATH} http://files.fast.ai/models/wt103/fwd_wt103.h5')
get_ipython().system(' wget -nH -r -np -P {PATH} http://files.fast.ai/models/wt103/fwd_wt103_enc.h5')
get_ipython().system(' wget -nH -r -np -P {PATH} http://files.fast.ai/models/wt103/itos_wt103.pkl')


# To match the weights of wiki103, our language model needs a embedding size of 400, 1150 hidden layers and 3 layers.

# In[ ]:


em_sz,nh,nl = 400,1150,3


# In[ ]:


PRE_PATH = PATH/'models'/'wt103'
PRE_LM_PATH = PRE_PATH/'fwd_wt103.h5'


# Loading weights in pytorch

# In[ ]:


wgts = torch.load(PRE_LM_PATH, map_location=lambda storage, loc: storage)


# Average weight is assigned to tokens that are unknown to the imdb model. The model can then be fine tuned to its correct weight when training.

# In[ ]:


enc_wgts = to_np(wgts['0.encoder.weight'])
row_m = enc_wgts.mean(0)


# Assigning -1 to each token that is not in wikitext corpus using defaultdict

# In[ ]:


itos2 = pickle.load((PRE_PATH/'itos_wt103.pkl').open('rb'))
stoi2 = collections.defaultdict(lambda:-1, {v:k for k,v in enumerate(itos2)})


# Assigning mean weights to unknown tokens (-1)

# In[ ]:


new_w = np.zeros((vs, em_sz), dtype=np.float32)
for i,w in enumerate(itos):
    r = stoi2[w]
    new_w[i] = enc_wgts[r] if r>=0 else row_m


# Assigning weights to encoders/decoders in a tensor

# In[ ]:


wgts['0.encoder.weight'] = T(new_w)
wgts['0.encoder_with_dropout.embed.weight'] = T(np.copy(new_w))
wgts['1.decoder.weight'] = T(np.copy(new_w))


# **Definitions: **
# * wd = weight decay (similar idea to L2 regularisation)
# * bptt = backpropagation through time, i.e. how long a sequence of words we will feed the GPU at one time.
# * bs = batch size (Because of the small sample used, we lower the batch size.)
# * optn_fn = optimisation function, set as Adam. For RNN models, we need to lower the momentum (beta1) from it's default 0.9.

# In[ ]:


wd=1e-7
bptt=70
bs=10
opt_fn = partial(optim.Adam, betas=(0.8, 0.99))


# We use LanguageModelLoader to load our training and validation text concatenated together by bs and bptt and load it into our LanguageModelData object.

# In[ ]:


trn_dl = LanguageModelLoader(np.concatenate(trn_lm), bs, bptt)
val_dl = LanguageModelLoader(np.concatenate(val_lm), bs, bptt)
md = LanguageModelData(PATH, 1, vs, trn_dl, val_dl, bs=bs, bptt=bptt)


# Dropouts are placed in different layers - this is a featuer of the AWD-LSTM model. Jeremy recommends keeping the dropout ratios as follows, and tweaking the factor instead. The default is 0.7, but for smaller datasets, it's better to have a higher dropout to avoid overfitting. I choose 0.9 to test things out.

# In[ ]:


drops = np.array([0.25, 0.1, 0.2, 0.02, 0.15])*0.9


# The metric we keep track of is accuracy. In classification models, often times we use cross entropy loss as our metric. However, cross entropy loss encourages the model to be very confident in its classification. In our case, we really just care whether the model is correct or not, so accuracy is a better metric.

# In[ ]:


learner= md.get_model(opt_fn, em_sz, nh, nl, 
    dropouti=drops[0], dropout=drops[1], wdrop=drops[2], dropoute=drops[4], dropouth=drops[4])

learner.metrics = [accuracy]
learner.freeze_to(-1)


# In[ ]:


learner.model.load_state_dict(wgts)


# In[ ]:


lr=1e-3
lrs = lr


# Running the learner for one epoch to train the embeddings in the last layer. Earlier on, we initialised unknown tokens with mean weights, so this will improve the embeddings for those new tokens.

# **Slanted triangular learning rate**
# 
# use_clr implements slanted triangular learning rate, which essentially increases your learning rate quickly and slowly decays it over a longer period. Visually, it looks like a slanted triangle, hence the name. The first parameter is the ratio between the highest and the lowest learning rate; the second parameter is the ratio between the first peak and the last peak. 

# In[ ]:


learner.fit(lrs/2, 1, wds=wd, use_clr=(32,2), cycle_len=1)


# In[ ]:


learner.save('lm_last_ft')
learner.load('lm_last_ft')
learner.unfreeze()


# In[ ]:


# search for a learning rate, then run for 15 epoches
learner.lr_find(start_lr=lrs/10, end_lr=lrs*10, linear=True)
learner.sched.plot()


# In[ ]:


learner.fit(lrs, 1, wds=wd, use_clr=(20,10), cycle_len=15)


# We save the encoder, not the entire model as that is what we will use as the 'backbone' to attach our classifier head

# In[ ]:


#Saving the model
#learner.save('lm1')

#Saving the RNN encoder (rnn_enc)
learner.save_encoder('lm1_enc')
learner.sched.plot_loss()


# #### Creating Classifier tokens
# Loading classifier data (with labels) and tokenising them. Steps are identical to when we were creating our bespoke language model.

# In[ ]:


df_trn = pd.read_csv(CLAS_PATH/'train.csv', header=None, chunksize=chunksize)
df_val = pd.read_csv(CLAS_PATH/'test.csv', header=None, chunksize=chunksize)


# In[ ]:


#get tokens
tok_trn, trn_labels = get_all(df_trn, 1)
tok_val, val_labels = get_all(df_val, 1)


# In[ ]:


(CLAS_PATH/'tmp').mkdir(exist_ok=True)

np.save(CLAS_PATH/'tmp'/'tok_trn.npy', tok_trn)
np.save(CLAS_PATH/'tmp'/'tok_val.npy', tok_val)

np.save(CLAS_PATH/'tmp'/'trn_labels.npy', trn_labels)
np.save(CLAS_PATH/'tmp'/'val_labels.npy', val_labels)


# In[ ]:


tok_trn = np.load(CLAS_PATH/'tmp'/'tok_trn.npy')
tok_val = np.load(CLAS_PATH/'tmp'/'tok_val.npy')


# Loading back the same vocabulary so the index is the same.

# In[ ]:


itos = pickle.load((LM_PATH/'tmp'/'itos.pkl').open('rb'))
stoi = collections.defaultdict(lambda:0, {v:k for k,v in enumerate(itos)})
len(itos)


# In[ ]:


trn_clas = np.array([[stoi[o] for o in p] for p in tok_trn])
val_clas = np.array([[stoi[o] for o in p] for p in tok_val])


# In[ ]:


np.save(CLAS_PATH/'tmp'/'trn_ids.npy', trn_clas)
np.save(CLAS_PATH/'tmp'/'val_ids.npy', val_clas)


# #### Creating Classifier Model
# The steps taken are quite similar to creating a language model

# In[ ]:


trn_clas = np.load(CLAS_PATH/'tmp'/'trn_ids.npy')
val_clas = np.load(CLAS_PATH/'tmp'/'val_ids.npy')


# In[ ]:


trn_labels = np.squeeze(np.load(CLAS_PATH/'tmp'/'trn_labels.npy'))
val_labels = np.squeeze(np.load(CLAS_PATH/'tmp'/'val_labels.npy'))


# In[ ]:


bptt,em_sz,nh,nl = 70,400,1150,3
vs = len(itos)
opt_fn = partial(optim.Adam, betas=(0.8, 0.99))
bs = 48


# In[ ]:


min_lbl = trn_labels.min()
trn_labels -= min_lbl
val_labels -= min_lbl
c=int(trn_labels.max())+1


# Classifier model reads one headline at a time. Headlines are of varying lengths, so padding is required - this is done automagically in fastai. SortishSampler helps to create batches that are of similar length so padding is reduced.

# In[ ]:


trn_ds = TextDataset(trn_clas, trn_labels)
val_ds = TextDataset(val_clas, val_labels)
trn_samp = SortishSampler(trn_clas, key=lambda x: len(trn_clas[x]), bs=bs//2)
val_samp = SortSampler(val_clas, key=lambda x: len(val_clas[x]))
trn_dl = DataLoader(trn_ds, bs//2, transpose=True, num_workers=1, pad_idx=1, sampler=trn_samp)
val_dl = DataLoader(val_ds, bs, transpose=True, num_workers=1, pad_idx=1, sampler=val_samp)
md = ModelData(PATH, trn_dl, val_dl)


# In[ ]:


# part 1
dps = np.array([0.4, 0.5, 0.05, 0.3, 0.1])


# In[ ]:


dps = np.array([0.4,0.5,0.05,0.3,0.4])*0.5


# #### Concat pooling
# Because signals may be present in different parts of the headline, we may lose information if we only consider the last hidden state. Hence, we combine the last hidden state, the max pooled and average representation of the hidden states.

# In[ ]:


m = get_rnn_classifer(bptt, 20*70, c, vs, emb_sz=em_sz, n_hid=nh, n_layers=nl, pad_token=1,
          layers=[em_sz*3, 50, c], drops=[dps[4], 0.1],
          dropouti=dps[0], wdrop=dps[1], dropoute=dps[2], dropouth=dps[3])


# In[ ]:


opt_fn = partial(optim.Adam, betas=(0.7, 0.99))


# In[ ]:


learn = RNN_Learner(md, TextModel(to_gpu(m)), opt_fn=opt_fn)
learn.reg_fn = partial(seq2seq_reg, alpha=2, beta=1)
learn.clip=.25
learn.metrics = [accuracy]


# #### Discriminative (differential) learning rates
# For different layers, we apply different learning rates. For lower layers, we use lower learning rates - this idea is analogous to deep learning in computer vision, where lower layers contain more generic information hence not needing much fine tuning.
# 
# The general formula to decrease the learning rate is by dividing the learning rate of the previous layer by 2.6.

# In[ ]:


lr=3e-3
lrm = 2.6
lrs = np.array([lr/(lrm**4), lr/(lrm**3), lr/(lrm**2), lr/lrm, lr])


# In[ ]:


lrs=np.array([1e-4,1e-4,1e-4,1e-3,1e-2])


# In[ ]:


wd = 1e-7
wd = 0
learn.load_encoder('lm1_enc')


# #### Gradual Unfreezing
# To prevent catastrophic forgetting, i.e. overfitting the classifier model, we unfreeze the last layer first, fine-tune that layer only, then proceed to fine-tune the one below it, until all layers are fine-tuned.

# In[ ]:


learn.freeze_to(-1)


# In[ ]:


learn.lr_find(lrs/1000)
learn.sched.plot()


# In[ ]:


learn.fit(lrs, 1, wds=wd, cycle_len=1, use_clr=(8,3))


# In[ ]:


learn.save('clas_0')


# In[ ]:


learn.load('clas_0')


# In[ ]:


learn.freeze_to(-2)


# In[ ]:


learn.fit(lrs, 1, wds=wd, cycle_len=1, use_clr=(8,3))


# In[ ]:


learn.save('clas_1')


# In[ ]:


learn.load('clas_1')


# In[ ]:


learn.unfreeze()


# In[ ]:


learn.fit(lrs, 1, wds=wd, cycle_len=14, use_clr=(32,10))


# We achieved pretty decent accuracy without doing anything different from the original notebook and using a really small subset of the data.

# In[ ]:


learn.sched.plot_loss()


# In[ ]:


learn.save('clas_2')


# ### References
# [1] Universal Language Model Fine-tuning for Text Classification, https://arxiv.org/abs/1801.06146
# 
# [2] Fastai Lesson 10, https://github.com/fastai/fastai/blob/master/courses/dl2/imdb.ipynb
# 
# 
