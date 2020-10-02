#!/usr/bin/env python
# coding: utf-8

# # TensorFlow roBERTa Starter - LB 0.705
# This notebook is a TensorFlow template for solving Kaggle's Tweet Sentiment Extraction competition as a question and answer roBERTa formulation. In this notebook, we show how to tokenize the data, create question answer targets, and how to build a custom question answer head for roBERTa in TensorFlow. Note that HuggingFace transformers don't have a `TFRobertaForQuestionAnswering` so we must make our own from `TFRobertaModel`. This notebook can achieve LB 0.715 with some modifications. Have fun experimenting!
# 
# You can also run this code offline and it will save the best model weights during each of the 5 folds of training. Upload those weights to a private Kaggle dataset and attach to this notebook. Then you can run this notebook with the line `model.fit()` commented out, and this notebook will instead load your offline models. It will use your offline models to predict oof and predict test. Hence this notebook can easily be converted to an inference notebook. An inference notebook is advantageous because it will only take 10 minutes to commit and submit instead of 2 hours. Better to train 2 hours offline separately.

# # Load Libraries, Data, Tokenizer
# We will use HuggingFace transformers [here][1]
# 
# [1]: https://huggingface.co/transformers/

# In[ ]:


import pandas as pd, numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from sklearn.model_selection import StratifiedKFold
from transformers import *
import tokenizers
print('TF version',tf.__version__)


# In[ ]:


MAX_LEN = 96
PATH = '../input/tf-roberta/'
tokenizer = tokenizers.ByteLevelBPETokenizer(
    vocab_file=PATH+'vocab-roberta-base.json', 
    merges_file=PATH+'merges-roberta-base.txt', 
    lowercase=True,
    add_prefix_space=True
)
sentiment_id = {'positive': 1313, 'negative': 2430, 'neutral': 7974}
train = pd.read_csv('../input/tweet-sentiment-extraction/train.csv').fillna('')
train.head()


# In[ ]:


#train_original = train.copy(deep=True)
#train = train_original.sample(100)


# In[ ]:


train["sentiment"].value_counts()


# In[ ]:


# Get all sentiments
all_sentiments = train["sentiment"].unique()


# In[ ]:


# Get training data
train_dict = dict()
train_dict_index = dict()
for sentiment in all_sentiments:
    train_dict[sentiment] = train[train["sentiment"]==sentiment]
    train_dict_index[sentiment] = list(train_dict[sentiment].index)
    train_dict[sentiment] = train_dict[sentiment].reset_index(drop=True)


# In[ ]:


def get_training_data(train):
    
    ct = train.shape[0]
    input_ids = np.ones((ct,MAX_LEN),dtype='int32')
    attention_mask = np.zeros((ct,MAX_LEN),dtype='int32')
    token_type_ids = np.zeros((ct,MAX_LEN),dtype='int32')
    start_tokens = np.zeros((ct,MAX_LEN),dtype='int32')
    end_tokens = np.zeros((ct,MAX_LEN),dtype='int32')

    for k in range(train.shape[0]):

        # FIND OVERLAP
        text1 = " "+" ".join(train.loc[k,'text'].split())
        text2 = " ".join(train.loc[k,'selected_text'].split())
        idx = text1.find(text2)
        chars = np.zeros((len(text1)))
        chars[idx:idx+len(text2)]=1
        if text1[idx-1]==' ': chars[idx-1] = 1 
        enc = tokenizer.encode(text1) 

        # ID_OFFSETS
        offsets = []; idx=0
        for t in enc.ids:
            w = tokenizer.decode([t])
            offsets.append((idx,idx+len(w)))
            idx += len(w)

        # START END TOKENS
        toks = []
        for i,(a,b) in enumerate(offsets):
            sm = np.sum(chars[a:b])
            if sm>0: toks.append(i) 

        s_tok = sentiment_id[train.loc[k,'sentiment']]
        input_ids[k,:len(enc.ids)+5] = [0] + enc.ids + [2,2] + [s_tok] + [2]
        attention_mask[k,:len(enc.ids)+5] = 1
        if len(toks)>0:
            start_tokens[k,toks[0]+1] = 1
            end_tokens[k,toks[-1]+1] = 1
    
    return input_ids, attention_mask, token_type_ids, start_tokens, end_tokens


# In[ ]:


# Get preprocessed training data
train_data_dict = dict()
for sentiment in all_sentiments:
    train_data_dict[sentiment] = get_training_data(train_dict[sentiment])


# In[ ]:


test = pd.read_csv('../input/tweet-sentiment-extraction/test.csv').fillna('')


# In[ ]:


# Get test data
test_dict = dict()
test_dict_index = dict()
for sentiment in all_sentiments:
    test_dict[sentiment] = test[test["sentiment"]==sentiment]
    test_dict_index[sentiment] = list(test_dict[sentiment].index)
    test_dict[sentiment] = test_dict[sentiment].reset_index(drop=True)


# In[ ]:


def get_test_data(test):
    ct = test.shape[0]
    input_ids_t = np.ones((ct,MAX_LEN),dtype='int32')
    attention_mask_t = np.zeros((ct,MAX_LEN),dtype='int32')
    token_type_ids_t = np.zeros((ct,MAX_LEN),dtype='int32')

    for k in range(test.shape[0]):

        # INPUT_IDS
        text1 = " "+" ".join(test.loc[k,'text'].split())
        enc = tokenizer.encode(text1)                
        s_tok = sentiment_id[test.loc[k,'sentiment']]
        input_ids_t[k,:len(enc.ids)+5] = [0] + enc.ids + [2,2] + [s_tok] + [2]
        attention_mask_t[k,:len(enc.ids)+5] = 1
    return input_ids_t, attention_mask_t, token_type_ids_t


# In[ ]:


# Get preprocessed test data
test_data_dict = dict()
for sentiment in all_sentiments:
    test_data_dict[sentiment] = get_test_data(test_dict[sentiment])


# In[ ]:


# Build model
def build_model():
    ids = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)
    att = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)
    tok = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)

    config = RobertaConfig.from_pretrained(PATH+'config-roberta-base.json')
    bert_model = TFRobertaModel.from_pretrained(PATH+'pretrained-roberta-base.h5',config=config)
    x = bert_model(ids,attention_mask=att,token_type_ids=tok)
    
    x1 = tf.keras.layers.Dropout(0.1)(x[0]) 
    x1 = tf.keras.layers.Conv1D(1,1)(x1)
    x1 = tf.keras.layers.Flatten()(x1)
    x1 = tf.keras.layers.Activation('softmax')(x1)
    
    x2 = tf.keras.layers.Dropout(0.1)(x[0]) 
    x2 = tf.keras.layers.Conv1D(1,1)(x2)
    x2 = tf.keras.layers.Flatten()(x2)
    x2 = tf.keras.layers.Activation('softmax')(x2)

    model = tf.keras.models.Model(inputs=[ids, att, tok], outputs=[x1,x2])
    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)

    return model


# In[ ]:


# Define Jaccard similarity metric
def jaccard(str1, str2): 
    a = set(str1.lower().split()) 
    b = set(str2.lower().split())
    if (len(a)==0) & (len(b)==0): return 0.5
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


# In[ ]:


def find_a_b(start_array, end_array):
    #print(start_array)
    #print(end_array)
    a = np.argmax(start_array)
    b = np.argmax(end_array)
    if a<b:
        return a,b
    else:
        if a<len(start_array)-1:
            end_array_copy = np.array(end_array)
            end_array_copy[b]=0.0
            return find_a_b(start_array, end_array_copy)
        else: 
            if b<len(start_array)-1:
                start_array_copy = np.array(start_array)
                start_array_copy[a]=0.0
                return find_a_b(start_array_copy, end_array)
            else:
                return 0, len(end_array)-1


# In[ ]:


def train_model_for_sentiment(train_data_dict, test_data_dict, sentiment, train_dict):
    
    input_ids, attention_mask, token_type_ids, start_tokens, end_tokens = train_data_dict[sentiment]
    input_ids_t, attention_mask_t, token_type_ids_t = test_data_dict[sentiment]
    train = train_dict[sentiment]
    
    jac = []; VER='v1'; DISPLAY=1 # USE display=1 FOR INTERACTIVE
    oof_start = np.zeros((input_ids.shape[0],MAX_LEN))
    oof_end = np.zeros((input_ids.shape[0],MAX_LEN))
    preds_start = np.zeros((input_ids_t.shape[0],MAX_LEN))
    preds_end = np.zeros((input_ids_t.shape[0],MAX_LEN))

    skf = StratifiedKFold(n_splits=5,shuffle=True,random_state=777)
    print('#'*25)
    print('### SENTIMENT %s'%(sentiment))
    print('#'*25)
    for fold,(idxT,idxV) in enumerate(skf.split(input_ids,train.sentiment.values)):

        print('#'*25)
        print('### FOLD %i'%(fold+1))
        print('#'*25)

        K.clear_session()
        model = build_model()

        sv = tf.keras.callbacks.ModelCheckpoint(
            '%s-roberta-%i-%s.h5'%(VER,fold,sentiment), monitor='val_loss', verbose=1, save_best_only=True,
            save_weights_only=True, mode='auto', save_freq='epoch')

        model.fit([input_ids[idxT,], attention_mask[idxT,], token_type_ids[idxT,]], [start_tokens[idxT,], end_tokens[idxT,]], 
            epochs=3, batch_size=32, verbose=DISPLAY, callbacks=[sv],
            validation_data=([input_ids[idxV,],attention_mask[idxV,],token_type_ids[idxV,]], 
            [start_tokens[idxV,], end_tokens[idxV,]]))

        print('Loading model...')
        model.load_weights('%s-roberta-%i-%s.h5'%(VER,fold,sentiment))

        print('Predicting OOF...')
        oof_start[idxV,],oof_end[idxV,] = model.predict([input_ids[idxV,],attention_mask[idxV,],token_type_ids[idxV,]],verbose=DISPLAY)

        print('Predicting Test...')
        preds = model.predict([input_ids_t,attention_mask_t,token_type_ids_t],verbose=DISPLAY)
        preds_start += preds[0]/skf.n_splits
        preds_end += preds[1]/skf.n_splits

        # DISPLAY FOLD JACCARD
        all = []
        for k in idxV:
            #a = np.argmax(oof_start[k,])
            #b = np.argmax(oof_end[k,])
            #if a>b: 
                # st = train.loc[k,'text'] # IMPROVE CV/LB with better choice here
            #else:
            a,b = find_a_b(oof_start[k,], oof_end[k,])
            text1 = " "+" ".join(train.loc[k,'text'].split())
            enc = tokenizer.encode(text1)
            st = tokenizer.decode(enc.ids[a-1:b])
            all.append(jaccard(st,train.loc[k,'selected_text']))
        jac.append(np.mean(all))
        print('>>>> FOLD %i Jaccard ='%(fold+1),np.mean(all))
        print()
    print('>>>> OVERALL 5Fold CV Jaccard =',np.mean(jac))    
    return preds_start, preds_end


# In[ ]:


def get_string_predictions(test_data_dict, sentiment, predictions, test_dict):
    preds_start, preds_end = predictions[sentiment]
    input_ids_t, attention_mask_t, token_type_ids_t = test_data_dict[sentiment]
    test = test_dict[sentiment]
    
    all = []
    for k in range(input_ids_t.shape[0]):
        a = np.argmax(preds_start[k,])
        b = np.argmax(preds_end[k,])
        if a>b: 
            st = test.loc[k,'text']
        else:
            text1 = " "+" ".join(test.loc[k,'text'].split())
            enc = tokenizer.encode(text1)
            st = tokenizer.decode(enc.ids[a-1:b])
        all.append(st)
    return all


# In[ ]:


def get_string_predictions_neutral(test_dict, sentiment):
    test = test_dict[sentiment]
    # Do nothing for neutral sentiment? Some sentences are slightly different, may be better to train a model
    return list(test["text"])


# In[ ]:


# Train and transform output into  predictions
predictions = dict()
string_predictions = dict()
for sentiment in all_sentiments:
    if not sentiment == "neutral":
        predictions[sentiment] = train_model_for_sentiment(train_data_dict, test_data_dict, sentiment, train_dict)
        string_predictions[sentiment] = get_string_predictions(test_data_dict, sentiment, predictions, test_dict)
    else:
        string_predictions[sentiment] = get_string_predictions_neutral(test_dict, "neutral")


# In[ ]:


# Reordering
concat_string_predictions = []
concat_index = []
for sentiment in all_sentiments:
    concat_string_predictions += string_predictions[sentiment]
    concat_index += test_dict_index[sentiment]
sorted_concat_string_predictions = sorted(list(zip(concat_index, concat_string_predictions)), key=lambda x:x[0])
sorted_concat_string_predictions_list = list(zip(*sorted_concat_string_predictions))[1]


# In[ ]:


test['selected_text'] = sorted_concat_string_predictions_list
test[['textID','selected_text']].to_csv('submission.csv',index=False)
pd.set_option('max_colwidth', 60)
test.sample(25)


# ### GET PREDICTIONS ON OOF

# In[ ]:


def get_predictions_on_oof_for_sentiment(train_data_dict, test_data_dict, sentiment, train_dict):
    
    input_ids, attention_mask, token_type_ids, start_tokens, end_tokens = train_data_dict[sentiment]
    input_ids_t, attention_mask_t, token_type_ids_t = test_data_dict[sentiment]
    train = train_dict[sentiment]
    
    # PATH to saved weights
    #PATH_SAVED_WEIGHTS = '../input/twitter-comp-roberta-version-9/'
    PATH_SAVED_WEIGHTS = ''
    
    # Initialise similarity and prediction oof
    train["prediction_oof"] = ""
    train["jaccard_sim"] = np.nan
    
    jac = []; VER='v1'; DISPLAY=1 # USE display=1 FOR INTERACTIVE
    oof_start = np.zeros((input_ids.shape[0],MAX_LEN))
    oof_end = np.zeros((input_ids.shape[0],MAX_LEN))
    preds_start = np.zeros((input_ids_t.shape[0],MAX_LEN))
    preds_end = np.zeros((input_ids_t.shape[0],MAX_LEN))

    skf = StratifiedKFold(n_splits=5,shuffle=True,random_state=777)
    print('#'*25)
    print('### SENTIMENT %s'%(sentiment))
    print('#'*25)
    for fold,(idxT,idxV) in enumerate(skf.split(input_ids,train.sentiment.values)):

        print('#'*25)
        print('### FOLD %i'%(fold+1))
        print('#'*25)

        K.clear_session()
        model = build_model()

        #sv = tf.keras.callbacks.ModelCheckpoint(
        #    '%s-roberta-%i-%s.h5'%(VER,fold,sentiment), monitor='val_loss', verbose=1, save_best_only=True,
        #    save_weights_only=True, mode='auto', save_freq='epoch')
        
        # NO NEED TO TRAIN THE MODEL, JUS LOAD THE WEIGHTS
        #model.fit([input_ids[idxT,], attention_mask[idxT,], token_type_ids[idxT,]], [start_tokens[idxT,], end_tokens[idxT,]], 
        #    epochs=3, batch_size=32, verbose=DISPLAY, callbacks=[sv],
        #    validation_data=([input_ids[idxV,],attention_mask[idxV,],token_type_ids[idxV,]], 
        #    [start_tokens[idxV,], end_tokens[idxV,]]))

        print('Loading model...')
        model.load_weights(PATH_SAVED_WEIGHTS+'%s-roberta-%i-%s.h5'%(VER,fold,sentiment))

        print('Predicting OOF...')
        oof_start[idxV,],oof_end[idxV,] = model.predict([input_ids[idxV,],attention_mask[idxV,],token_type_ids[idxV,]],verbose=DISPLAY)

        #print('Predicting Test...')
        #preds = model.predict([input_ids_t,attention_mask_t,token_type_ids_t],verbose=DISPLAY)
        #preds_start += preds[0]/skf.n_splits
        #preds_end += preds[1]/skf.n_splits

        # DISPLAY FOLD JACCARD
        all = []
        for k in idxV:
            a = np.argmax(oof_start[k,])
            b = np.argmax(oof_end[k,])
            if a>b: 
                st = train.loc[k,'text'] # IMPROVE CV/LB with better choice here
            else:
            #a,b = find_a_b(oof_start[k,], oof_end[k,])
                text1 = " "+" ".join(train.loc[k,'text'].split())
                enc = tokenizer.encode(text1)
                st = tokenizer.decode(enc.ids[a-1:b])
            jaccard_sim = jaccard(st,train.loc[k,'selected_text'])
            train.at[k, "jaccard_sim"] = jaccard_sim
            train.at[k, "prediction_oof"] = st
            all.append(jaccard_sim)
            
        jac.append(np.mean(all))
        print('>>>> FOLD %i Jaccard ='%(fold+1),np.mean(all))
        print()
        
    print('>>>> OVERALL 5Fold CV Jaccard =',np.mean(jac))    
    return train


# In[ ]:


oof_predictions = get_predictions_on_oof_for_sentiment(train_data_dict, test_data_dict, "positive", train_dict)


# In[ ]:


pd.set_option('display.max_colwidth', -1)
oof_predictions.sample(20)


# In[ ]:


oof_predictions["len_selected_text"] = oof_predictions["selected_text"].map(lambda x:str(x).strip()).map(len)
oof_predictions["len_prediction_oof"] = oof_predictions["prediction_oof"].map(lambda x:str(x).strip()).map(len)


# In[ ]:


oof_predictions["num_words_text"] = oof_predictions["text"].map(lambda x:len(x.split()))
oof_predictions["num_words_selected_text"] = oof_predictions["selected_text"].map(lambda x:len(x.split()))
oof_predictions["num_words_prediction_oof"] = oof_predictions["prediction_oof"].map(lambda x:len(x.split()))


# In[ ]:


import matplotlib.pyplot as plt
plt.figure(figsize=(20, 5))
_ = plt.hist(oof_predictions["len_selected_text"], alpha=0.5, bins=np.arange(0, 140, 1), label='len_selected_text')
_ = plt.hist(oof_predictions["len_prediction_oof"], alpha=0.5, bins=np.arange(0, 140, 1), label='len_prediction_oof')
plt.legend()


# In[ ]:


#oof_predictions[oof_predictions["len_selected_text"]==3][["text", "selected_text", "len_selected_text","prediction_oof", "len_prediction_oof","sentiment", "jaccard_sim"]].sample(10)


# In[ ]:


_ = plt.hist(oof_predictions["num_words_selected_text"], alpha=0.5, bins=np.arange(0, 25, 1), label='num_words_selected_text')
_ = plt.hist(oof_predictions["num_words_prediction_oof"], alpha=0.5, bins=np.arange(0, 25, 1), label='num_words_prediction_oof')
plt.legend()


# In[ ]:


fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (15, 5))

oof_predictions["num_words_text_rand"] = oof_predictions["text"].map(lambda x:len(x.split()) + np.random.normal(0, 0.3, 1)[0])
oof_predictions["num_words_prediction_oof_rand"] = oof_predictions["prediction_oof"].map(lambda x:len(x.split())+ np.random.normal(0, 0.3, 1)[0])
ax1.scatter(oof_predictions["num_words_text_rand"], oof_predictions["num_words_prediction_oof_rand"], s=0.01)

oof_predictions["num_words_text_rand"] = oof_predictions["text"].map(lambda x:len(x.split()) + np.random.normal(0, 0.3, 1)[0])
oof_predictions["num_words_selected_text_rand"] = oof_predictions["selected_text"].map(lambda x:len(x.split())+ np.random.normal(0, 0.3, 1)[0])
ax2.scatter(oof_predictions["num_words_text_rand"], oof_predictions["num_words_selected_text_rand"], s=0.01)


# In[ ]:





# In[ ]:




