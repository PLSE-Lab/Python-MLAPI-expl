#!/usr/bin/env python
# coding: utf-8

# # TensorFlow BERTweet

# This notebook uses BERTweet from:
# 
# https://github.com/VinAIResearch/BERTweet
# 
# https://arxiv.org/abs/2005.10200
# 
# (see discussion here: https://www.kaggle.com/c/tweet-sentiment-extraction/discussion/152861)

# This is mostly adjusting and combining code from the following:
# 
# https://www.kaggle.com/cdeotte/tensorflow-roberta-0-705 (for majority of script)
# 
# https://www.kaggle.com/al0kharba/tensorflow-roberta-0-712 (for inference and CNN head, switched from dropout to batch normalization)
# 
# https://www.kaggle.com/christofhenkel/setup-tokenizer (for tokenizer)
# 
# https://www.kaggle.com/nandhuelan/bertweet-first-look (for offsets and decoding)

# Changes from V2: Changed CNN head and added LR schedule
# 
# Changes from V3: Adjusted epochs and batch size to get fold 4 training to learn (last iteration was stuck and produced a 0.65 jaccard)
# 
# Changes from V6: Adjusted post-processing for unk tokens. Added jaccard scores for sentiments. Set neutral equal to text instead of relying on model (jaccard is higher in CV when we do this).

# # Load  data and libraries

# See tips on how to install offline: https://www.kaggle.com/c/severstal-steel-defect-detection/discussion/113195

# In[ ]:


internet_on = False


# In[ ]:


if internet_on==True:
    get_ipython().system('pip install fairseq fastBPE ')


# In[ ]:


if internet_on==True:
    get_ipython().system('ls /root/.cache/pip/wheels/df/60/ff/1764bce64cccd9d2c06ba19e5f6f4108ad29e2d48e1068c684')


# In[ ]:


if internet_on==True:
    get_ipython().system('ls /root/.cache/pip/wheels/fb/85/9b/286072121774d5b8b0253ab66271b558069189cbe795bc6084')


# In[ ]:


if internet_on==True:
    get_ipython().system('mv /root/.cache/pip/wheels/df/60/ff/1764bce64cccd9d2c06ba19e5f6f4108ad29e2d48e1068c684/* /kaggle/working')
    get_ipython().system('mv /root/.cache/pip/wheels/fb/85/9b/286072121774d5b8b0253ab66271b558069189cbe795bc6084/* /kaggle/working')


# Note: I could not figure out how to get sacrebleu, but found it loaded to Kaggle already

# In[ ]:


if internet_on==False:
    get_ipython().system('pip install ../input/fairseq-and-fastbpe/sacrebleu-1.4.9-py3-none-any.whl ')


# In[ ]:


# These files were saved in version 1 of this notebook when internet_on was True
if internet_on==False:
    get_ipython().system('pip install ../input/v1-fairseq-fastbpe/fastBPE-0.1.0-cp36-cp36m-linux_x86_64.whl')
    get_ipython().system('pip install ../input/v1-fairseq-fastbpe/fairseq-0.9.0-cp36-cp36m-linux_x86_64.whl')


# In[ ]:


import pandas as pd, numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from sklearn.model_selection import StratifiedKFold
from transformers import *
import tokenizers
print('TF version',tf.__version__)


# In[ ]:


#From: https://www.kaggle.com/christofhenkel/setup-tokenizer
from types import SimpleNamespace
from fairseq.data.encoders.fastbpe import fastBPE
from fairseq.data import Dictionary

class BERTweetTokenizer():
    
    def __init__(self,pretrained_path = 'pretrained_models/BERTweet_base_transformers/'):
        
        self.bpe = fastBPE(SimpleNamespace(bpe_codes= pretrained_path + "bpe.codes"))
        self.vocab = Dictionary()
        self.vocab.add_from_file(pretrained_path + "dict.txt")
        self.cls_token_id = 0
        self.pad_token_id = 1
        self.sep_token_id = 2
        self.pad_token = '<pad>'
        self.cls_token = '<s>'
        self.sep_token = '</s>'
        
    def bpe_encode(self,text):
        return self.bpe.encode(text)
    
    def encode(self,text,add_special_tokens=False):
        subwords = self.bpe.encode(text)
        input_ids = self.vocab.encode_line(subwords, append_eos=False, add_if_not_exist=False).long().tolist()
        return input_ids
    
    def tokenize(self,text):
        return self.bpe_encode(text).split()
    
    def convert_tokens_to_ids(self,tokens):
        input_ids = self.vocab.encode_line(' '.join(tokens), append_eos=False, add_if_not_exist=False).long().tolist()
        return input_ids
    
    #from: https://www.kaggle.com/nandhuelan/bertweet-first-look
    def decode_id(self,id):
        return self.vocab.string(id, bpe_symbol = '@@')
    
    def decode_id_nospace(self,id):
        return self.vocab.string(id, bpe_symbol = '@@ ')


# In[ ]:


tokenizer = BERTweetTokenizer('/kaggle/input/bertweet-base-transformers/')


# In[ ]:


tokenizer.encode('positive')


# In[ ]:


tokenizer.encode('negative')


# In[ ]:


tokenizer.encode('neutral')


# In[ ]:


tokenizer.decode_id([14058])


# In[ ]:


def read_train():
    train=pd.read_csv('../input/tweet-sentiment-extraction/train.csv')
    train['text']=train['text'].astype(str)
    train['selected_text']=train['selected_text'].astype(str)
    return train

def read_test():
    test=pd.read_csv('../input/tweet-sentiment-extraction/test.csv')
    test['text']=test['text'].astype(str)
    return test

def read_submission():
    test=pd.read_csv('../input/tweet-sentiment-extraction/sample_submission.csv')
    return test
    
train_df = read_train()
test_df = read_test()
submission_df = read_submission()


# In[ ]:


train_df.sentiment.value_counts(dropna=False)


# In[ ]:


def jaccard(str1, str2): 
    a = set(str(str1).lower().split()) 
    b = set(str(str2).lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


# # Data preproccesing

# In[ ]:


MAX_LEN = 96
PATH = '../input/bertweet-base-transformers/'
sentiment_id = {'positive': 1809, 'negative': 3392, 'neutral': 14058}


# In[ ]:


ct = train_df.shape[0]
input_ids = np.ones((ct,MAX_LEN),dtype='int32')
attention_mask = np.zeros((ct,MAX_LEN),dtype='int32')
token_type_ids = np.zeros((ct,MAX_LEN),dtype='int32')
start_tokens = np.zeros((ct,MAX_LEN),dtype='int32')
end_tokens = np.zeros((ct,MAX_LEN),dtype='int32')

for k in range(train_df.shape[0]):
    
    # FIND OVERLAP
    text1 = " "+" ".join(train_df.loc[k,'text'].split())
    text2 = " ".join(train_df.loc[k,'selected_text'].split())
    idx = text1.find(text2)
    chars = np.zeros((len(text1)))
    chars[idx:idx+len(text2)]=1
    if text1[idx-1]==' ': chars[idx-1] = 1 
    enc = tokenizer.encode(text1) 
    
    # ID_OFFSETS
    # From: https://www.kaggle.com/nandhuelan/bertweet-first-look (comments)
    offsets = []; idx=0
    for t in enc:
        w = tokenizer.decode_id([t])
        if text1[text1.find(w,idx)-1] == " ":
            idx+=1
            offsets.append((idx,idx+len(w)))
            idx += len(w)
        else:
            offsets.append((idx,idx+len(w)))
            idx += len(w)

    # START END TOKENS
    toks = []
    for i,(a,b) in enumerate(offsets):
        sm = np.sum(chars[a:b])
        if sm>0: toks.append(i) 
        
    s_tok = sentiment_id[train_df.loc[k,'sentiment']]
    if len(enc)<92:
        input_ids[k,:len(enc)+5] = [0] + enc + [2,2] + [s_tok] + [2]
        attention_mask[k,:len(enc)+5] = 1
        if len(toks)>0:
            start_tokens[k,toks[0]+1] = 1
            end_tokens[k,toks[-1]+1] = 1        
    if len(enc)>91:
        input_ids[k,:96] = [0] + enc[:91] + [2,2] + [s_tok] + [2]
        attention_mask[k,:96] = 1        
        if len(toks)>0:
            start_tokens[k,toks[0]+1] = 1
            end_tokens[k,96-1] = 1


# In[ ]:


ct = test_df.shape[0]
input_ids_t = np.ones((ct,MAX_LEN),dtype='int32')
attention_mask_t = np.zeros((ct,MAX_LEN),dtype='int32')
token_type_ids_t = np.zeros((ct,MAX_LEN),dtype='int32')

for k in range(test_df.shape[0]):        
    # INPUT_IDS
    text1 = " "+" ".join(test_df.loc[k,'text'].split())
    enc = tokenizer.encode(text1)                
    s_tok = sentiment_id[test_df.loc[k,'sentiment']]    
    if len(enc)<92:
        input_ids_t[k,:len(enc)+5] = [0] + enc + [2,2] + [s_tok] + [2]
        attention_mask_t[k,:len(enc)+5] = 1
    if len(enc)>91:
        input_ids_t[k,:96] = [0] + enc[:91] + [2,2] + [s_tok] + [2]
        attention_mask_t[k,:96] = 1  


# How good does our process work?

# In[ ]:


all=[]
count=0
for k in range(train_df.shape[0]):    
    a = np.argmax(start_tokens[k,])
    b = np.argmax(end_tokens[k,])
    text1 = " "+" ".join(train_df.loc[k,'text'].split())
    enc = tokenizer.encode(text1)
    st = tokenizer.decode_id_nospace(enc[a-1:b])
    st = st.replace('<unk>','')
    all.append(jaccard(st,train_df.loc[k,'selected_text']))
print('>>>> Jaccard =',np.mean(all))


# In[ ]:


improve_jacc_review = False


# In[ ]:


if improve_jacc_review == True:
    all=[]
    count=0
    for k in range(train_df.shape[0]):    
        a = np.argmax(start_tokens[k,])
        b = np.argmax(end_tokens[k,])
        text1 = " "+" ".join(train_df.loc[k,'text'].split())
        enc = tokenizer.encode(text1)
        st = tokenizer.decode_id_nospace(enc[a-1:b])
        st = st.replace('<unk>','')
        if jaccard(st,train_df.loc[k,'selected_text'])<.3:
            print(k)
            print(st)
            print(train_df.loc[k,'selected_text'])
            print()
        all.append(jaccard(st,train_df.loc[k,'selected_text']))
    print('>>>> Jaccard =',np.mean(all))


# # Model

# In[ ]:


def scheduler(epoch):
    return 5e-5 * 0.2**epoch


# In[ ]:


def build_model():
    ids = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)
    att = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)
    tok = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)

    config = RobertaConfig.from_pretrained(PATH+'config.json')
    bert_model = TFRobertaModel.from_pretrained(PATH+'model.bin',config=config,from_pt=True)
    x = bert_model(ids,attention_mask=att,token_type_ids=tok)

    x1 = tf.keras.layers.Conv1D(128, 2,padding='same')(x[0])
    x1 = tf.keras.layers.BatchNormalization()(x1)
    x1 = tf.keras.layers.LeakyReLU()(x1)
    x1 = tf.keras.layers.Conv1D(64, 2,padding='same')(x1)
    x1 = tf.keras.layers.Dense(1)(x1)
    x1 = tf.keras.layers.Flatten()(x1)
    x1 = tf.keras.layers.Activation('softmax')(x1)
    
    x2 = tf.keras.layers.Conv1D(128, 2, padding='same')(x[0])
    x2 = tf.keras.layers.BatchNormalization()(x2)
    x2 = tf.keras.layers.LeakyReLU()(x2)
    x2 = tf.keras.layers.Conv1D(64, 2, padding='same')(x2)
    x2 = tf.keras.layers.Dense(1)(x2)
    x2 = tf.keras.layers.Flatten()(x2)
    x2 = tf.keras.layers.Activation('softmax')(x2)
    
    model = tf.keras.models.Model(inputs=[ids, att, tok], outputs=[x1,x2])
    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)    
    model.compile(loss='binary_crossentropy', optimizer=optimizer)

    return model


# # Train

# In[ ]:


#5-fold CV
n_splits = 5


# In[ ]:


#This will loop over more than one seed and average all folds from all seeds together
n_seeds = 1


# In[ ]:


#Set equal to False if you already have model trained and just want to generate predictions. You'll need to save the model weights to input>pre-trained-model.
trainModel=False


# In[ ]:


if trainModel==True:

    for x in range(n_seeds): 
        
        jac = []; VER='v0'; DISPLAY=1 # USE display=1 FOR INTERACTIVE
        oof_start = np.zeros((input_ids.shape[0],MAX_LEN))
        oof_end = np.zeros((input_ids.shape[0],MAX_LEN))
        preds_start = np.zeros((input_ids_t.shape[0],MAX_LEN))
        preds_end = np.zeros((input_ids_t.shape[0],MAX_LEN))

        skf = StratifiedKFold(n_splits=n_splits,shuffle=True,random_state=777+x)
        for fold,(idxT,idxV) in enumerate(skf.split(input_ids,train_df.sentiment.values)):
            
            print('#'*25)
            print('### FOLD %i'%(fold+1))
            print('#'*25)

            K.clear_session()
            model = build_model()

            reduce_lr = tf.keras.callbacks.LearningRateScheduler(scheduler)
            
            sv = tf.keras.callbacks.ModelCheckpoint(
                '%s-roberta-%i-%x.h5'%(VER,fold,x), monitor='val_loss', verbose=1, save_best_only=True,
                save_weights_only=True, mode='auto', save_freq='epoch')

            hist = model.fit([input_ids[idxT,], attention_mask[idxT,], token_type_ids[idxT,]], [start_tokens[idxT,], end_tokens[idxT,]], 
                epochs=3, batch_size=8, verbose=DISPLAY, callbacks=[sv, reduce_lr],
                validation_data=([input_ids[idxV,],attention_mask[idxV,],token_type_ids[idxV,]], 
                [start_tokens[idxV,], end_tokens[idxV,]]))

            print('Loading model...')
            model.load_weights('%s-roberta-%i-%x.h5'%(VER,fold,x))

            print('Predicting OOF...')
            oof_start[idxV,],oof_end[idxV,] = model.predict([input_ids[idxV,],attention_mask[idxV,],token_type_ids[idxV,]],verbose=DISPLAY)

            print('Predicting Test...')
            preds = model.predict([input_ids_t,attention_mask_t,token_type_ids_t],verbose=DISPLAY)
            preds_start += preds[0]/(n_splits*n_seeds)
            preds_end += preds[1]/(n_splits*n_seeds)

            # DISPLAY FOLD JACCARD
            all = []
            for k in idxV:
                a = np.argmax(oof_start[k,])
                b = np.argmax(oof_end[k,])
                if a>b: 
                    text1 = " "+" ".join(train_df.loc[k,'text'].split())
                    enc = tokenizer.encode(text1)                   
                    st = tokenizer.decode_id_nospace(enc[a-1:a+3])
                else:
                    text1 = " "+" ".join(train_df.loc[k,'text'].split())
                    enc = tokenizer.encode(text1)
                    st = tokenizer.decode_id_nospace(enc[a-1:b])
                st = st.replace('<unk>','')
                all.append(jaccard(st,train_df.loc[k,'selected_text']))
            jac.append(np.mean(all))
            print('>>>> FOLD %i Jaccard ='%(fold+1),np.mean(all))
            print()
        
        print('>>>> OVERALL 5Fold CV Jaccard =',np.mean(jac))


# # Inference

# In[ ]:


if trainModel==False:
        
    DISPLAY=1
    
    for x in range(n_seeds): 
        
        jac = []
    
        oof_start = np.zeros((input_ids.shape[0],MAX_LEN))
        oof_end = np.zeros((input_ids.shape[0],MAX_LEN))
    
        preds_start = np.zeros((input_ids_t.shape[0],MAX_LEN))
        preds_end = np.zeros((input_ids_t.shape[0],MAX_LEN))
        
        print('#'*70)
        print('### SEED %x'%(x+1))
        print('#'*70)

        skf = StratifiedKFold(n_splits=n_splits,shuffle=True,random_state=777+x)
        for fold,(idxT,idxV) in enumerate(skf.split(input_ids,train_df.sentiment.values)):  
            
            print('#'*25)
            print('### MODEL %i'%(fold+1))
            print('#'*25)

            K.clear_session()
            model = build_model()
            model.load_weights('../input/bertweet-files/v0-roberta-%i-%x.h5'%(fold,x))

            print('Predicting Test...')
            preds = model.predict([input_ids_t,attention_mask_t,token_type_ids_t],verbose=DISPLAY)
            preds_start += preds[0]/(n_splits*n_seeds)
            preds_end += preds[1]/(n_splits*n_seeds)
            
            print('Predicting OOF...')
            oof_start[idxV,],oof_end[idxV,] = model.predict([input_ids[idxV,],attention_mask[idxV,],token_type_ids[idxV,]],verbose=DISPLAY)
            
            # DISPLAY FOLD JACCARD
            all = []; pos = []; neg = []; nue = []
            for k in idxV:
                a = np.argmax(oof_start[k,])
                b = np.argmax(oof_end[k,])
                if a>b: 
                    text1 = " "+" ".join(train_df.loc[k,'text'].split())
                    enc = tokenizer.encode(text1)
                    st = tokenizer.decode_id_nospace(enc[a-1:a+3])
                else:
                    text1 = " "+" ".join(train_df.loc[k,'text'].split())
                    enc = tokenizer.encode(text1)
                    st = tokenizer.decode_id_nospace(enc[a-1:b])
                st = st.replace('<unk>','')                                              
                if train_df.loc[k,'sentiment']=='positive':
                    pos.append(jaccard(st,train_df.loc[k,'selected_text']))
                if train_df.loc[k,'sentiment']=='negative':
                    neg.append(jaccard(st,train_df.loc[k,'selected_text']))
                if train_df.loc[k,'sentiment']=='neutral':
                    st = text1
                    nue.append(jaccard(st,train_df.loc[k,'selected_text']))
                all.append(jaccard(st,train_df.loc[k,'selected_text']))  
            jac.append(np.mean(all))
            print('>>>> FOLD %i Jaccard ='%(fold+1),np.mean(all))
            print('>>>> FOLD %i Neutral Jaccard ='%(fold+1),np.mean(nue))
            print('>>>> FOLD %i Positive Jaccard ='%(fold+1),np.mean(pos))
            print('>>>> FOLD %i Negative Jaccard ='%(fold+1),np.mean(neg))                 
            print()
            
        print('>>>> OVERALL 5Fold CV Jaccard =',np.mean(jac))


# # Test predictions

# In[ ]:


all = []
for k in range(input_ids_t.shape[0]):
    a = np.argmax(preds_start[k,])
    b = np.argmax(preds_end[k,])
    if a>b: 
        text1 = " "+" ".join(test_df.loc[k,'text'].split())
        enc = tokenizer.encode(text1)
        st = tokenizer.decode_id_nospace(enc[a-1:a+3])
    else:
        text1 = " "+" ".join(test_df.loc[k,'text'].split())
        enc = tokenizer.encode(text1)
        st = tokenizer.decode_id_nospace(enc[a-1:b])
    st = st.replace('<unk>','')
    if test_df.loc[k,'sentiment']=='neutral':
        st = text1
    all.append(st)


# In[ ]:


test_df['selected_text'] = all
test_df[['textID','selected_text']].to_csv('submission.csv',index=False)

