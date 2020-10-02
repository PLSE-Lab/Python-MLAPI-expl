#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Importing data sets
import pandas as pd
train_data = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/train.csv')
test_data = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/test.csv') 


# In[ ]:


# Training data details
train_data.info()
train_data.head(20)


# In[ ]:


# Testing data details
test_data.info()
test_data.head(20)


# In this competition, the aim is to identify/extract "selected_text" from the "text" field of the test data set. Only after that, we will be able to cross-check and verify whether sentiment extracted in "sentiment" column of the test data set is correct or not. 

# In[ ]:


# Row counts where missing value is present in Train data
print(train_data.notnull().sum())
print(train_data.isnull().sum())


# There is one row in the training data set which has its "text" and "selected text" missing. We can discard that.

# In[ ]:


train_data.dropna(axis = 0,inplace=True)


# In[ ]:


# Row counts where missing value is present in Test data
print(test_data.notnull().sum())
print(test_data.isnull().sum())


# In[ ]:


# plot frequency of positive, negative and neutral sentiments in Train Data
from matplotlib import pyplot as plt
count_sentiments = pd.value_counts(train_data['sentiment'], sort=True)
count_sentiments.plot(kind='bar', color=(['green','red','orange']), alpha=0.8, rot=0)
plt.title("Distribution of Sentiment Types in Train Data")
plt.xticks(range(3), ['positive', 'negative', 'neutral'])
plt.xlabel("Sentiment Type")
plt.ylabel("Frequency")
plt.show()


# In[ ]:


# plot frequency of positive, negative and neutral sentiments in Test Data
from matplotlib import pyplot as plt
count_sentiments_te = pd.value_counts(test_data['sentiment'], sort=True)
count_sentiments_te.plot(kind='bar', color=(['green','red','orange']), alpha=0.8, rot=0)
plt.title("Distribution of Sentiment Types in Test Data")
plt.xticks(range(3), ['positive', 'negative', 'neutral'])
plt.xlabel("Sentiment Type")
plt.ylabel("Frequency")
plt.show()


# In both train and test datasets, no. of positive tweets are higher than no. of negative and neutral tweets. 

# In[ ]:


# Removes punctuation from text. Convert entire text to lower case.
import string
def remove_punctuation(text):
    no_punct = "".join([c for c in text if c not in string.punctuation])
    return no_punct

train_data['s_text_clean'] = train_data['selected_text'].apply(str).apply(lambda x: remove_punctuation(x.lower()))
train_data.head(20)


# In[ ]:


# Breaks up entire string into a list of words based on a pattern specified by the Regular Expression
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')  
train_data['s_text_tokens'] = train_data['s_text_clean'].apply(str).apply(lambda x: tokenizer.tokenize(x))
train_data.head(20)


# In[ ]:


# Remove stopwords
from nltk.corpus import stopwords
def remove_stopwords(text):
    words = [w for w in text if (w not in stopwords.words('english') or w not in 'im')]
    return words

train_data['s_text_tokens_NOTstop'] = train_data['s_text_tokens'].apply(lambda x: remove_stopwords(x))
train_data.head(20)


# In[ ]:


# Lemmatization
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
def word_lemmatizer(text):
    lem_text = [lemmatizer.lemmatize(i) for i in text]
    return lem_text

train_data['s_text_lemma'] = train_data['s_text_tokens_NOTstop'].apply(lambda x: word_lemmatizer(x))
train_data.head(20)


# In[ ]:


# Stemming
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

def word_stemmer(text):
    stem_text = " ".join([stemmer.stem(i) for i in text])
    return stem_text

train_data['s_text_stem'] = train_data['s_text_lemma'].apply(lambda x: word_stemmer(x))
train_data.head(20)


# In[ ]:


from wordcloud import WordCloud, STOPWORDS
stop_w = set(STOPWORDS)

sentiment=['positive','neutral','negative']
fig, a = plt.subplots(1,3, figsize=(20,20))
for i,s in enumerate(sentiment):   
    total_token = ''
    total_token +=' '.join(train_data.loc[train_data['sentiment']==s,'s_text_stem'])
    if (s == 'positive'):
        w_cloud = WordCloud(width=1200, height=1200, background_color='green', stopwords = stop_w, min_font_size=12).generate(total_token)
    if (s == 'neutral'):
        w_cloud = WordCloud(width=1200, height=1200, background_color='orange', stopwords = stop_w, min_font_size=12).generate(total_token)
    if (s == 'negative'):
        w_cloud = WordCloud(width=1200, height=1200, background_color='red', stopwords = stop_w, min_font_size=12).generate(total_token)
    a[i].imshow(w_cloud, interpolation = 'bilinear')  
    a[i].set_title(s)
    a[i].axis('off')


# In[ ]:


import seaborn as sns

def unique_words_analysis(df):
    fig,ax = plt.subplots(1,3, figsize=(16,4))
    for i,s in enumerate(sentiment):
        new = train_data[train_data['sentiment']==s]['s_text_stem'].map(lambda x: len(set(x.split())))
        if (s =='positive'):
            sns.distplot(new.values, ax = ax[i], color='green', rug=True)
        if (s =='neutral'):
            sns.distplot(new.values, ax = ax[i], color='orange', rug=True)
        if (s =='negative'):
            sns.distplot(new.values, ax = ax[i], color='red', rug=True)
        ax[i].set_title(s)
    fig.suptitle('Distribution of number of unique words')
    fig.show()

unique_words_analysis(train_data)


# We observe that both positive and negative tweets' no. of unique words follow almost the similar pattern of distribution (positively skewed). Though neutral tweets also follow a positively skewed distribution, it has a more wide spread as compared to the spread of other two types.

# In[ ]:


# Segregating positive, negative, neutral sentiment data
positive_train = train_data[train_data['sentiment']=='positive']
neutral_train = train_data[train_data['sentiment']=='neutral']
negative_train = train_data[train_data['sentiment']=='negative']


# In[ ]:


# Common Word frequency analysis for positive text
from nltk.probability import FreqDist
import pandas as pd
import cufflinks as cf
cf.go_offline()
cf.set_config_file(offline=False, world_readable=True)

fdist_pos = FreqDist(positive_train['s_text_stem'])
top_twen_pos = fdist_pos.most_common(20)
#top_ten_pos

df1 = pd.DataFrame(top_twen_pos, columns = ['Text' , 'count'])
df1.groupby('Text').sum()['count'].sort_values(ascending=False).iplot(kind='bar', yTitle='Count', color='green', linecolor='black', title='Top 20 Common Words in positive text',orientation='v')


# In[ ]:


# Common Word frequency analysis for neutral text

fdist_neu = FreqDist(neutral_train['s_text_stem'])
top_twen_neu = fdist_neu.most_common(20)

df2 = pd.DataFrame(top_twen_neu, columns = ['Text' , 'count'])
df2.groupby('Text').sum()['count'].sort_values(ascending=False).iplot(kind='bar', yTitle='Count', color='orange', linecolor='black', title='Top 20 Common Words in neutral text',orientation='v')


# In[ ]:


# Common Word frequency analysis for negative text

fdist_neg = FreqDist(negative_train['s_text_stem'])
top_twen_neg = fdist_neg.most_common(20)

df3 = pd.DataFrame(top_twen_neg, columns = ['Text' , 'count'])
df3.groupby('Text').sum()['count'].sort_values(ascending=False).iplot(kind='bar', yTitle='Count', color='red', linecolor='black', title='Top 20 Common Words in negative text',orientation='v')


# Exploratory data analysis ends here. Now we begin the advanced part where we will transform data set into [RoBERTa](http://arxiv.org/pdf/1907.11692.pdf) format. [BERT](http://arxiv.org/pdf/1810.04805.pdf) is a masked language model. It requires: "start_tokens", "end_tokens". These tokens pad the inputs. It also uses "attention_masks" to avoid performing attention on padding token indices. "0" is used for the tokens which are masked; "1" is used for the tokens which are not masked.

# In[ ]:


#BPE (Byte Pair Encoding) tokenizer is used for tokenizing text
import tokenizers
import numpy as np
max_len = 128

tokenizer = tokenizers.ByteLevelBPETokenizer(
            vocab_file = '/kaggle/input/roberta-base/vocab.json',
            merges_file = '/kaggle/input/roberta-base/merges.txt',
            lowercase =True,
            add_prefix_space=True
)

sentiment_id = {'positive':tokenizer.encode('positive').ids[0], 
                'negative':tokenizer.encode('negative').ids[0], 
                'neutral':tokenizer.encode('neutral').ids[0]}

train_data.reset_index(inplace=True)

# input data formating for training
tot_tw = train_data.shape[0]

input_ids = np.ones((tot_tw, max_len), dtype='int32')
attention_mask = np.zeros((tot_tw, max_len), dtype='int32')
token_type_ids = np.zeros((tot_tw, max_len), dtype='int32')
start_mask = np.zeros((tot_tw, max_len), dtype='int32')
end_mask = np.zeros((tot_tw, max_len), dtype='int32')

for i in range(tot_tw):
    set1 = " "+" ".join(train_data.loc[i,'text'].split())
    set2 = " ".join(train_data.loc[i,'selected_text'].split())
    idx = set1.find(set2)
    set2_loc = np.zeros((len(set1)))
    set2_loc[idx:idx+len(set2)]=1
    if set1[idx-1]==" ":
        set2_loc[idx-1]=1
  
    enc_set1 = tokenizer.encode(set1)

    selected_text_token_idx=[]
    for k,(a,b) in enumerate(enc_set1.offsets):
        sm = np.sum(set2_loc[a:b]) 
        if sm > 0:
            selected_text_token_idx.append(k)

    senti_token = sentiment_id[train_data.loc[i,'sentiment']]
    input_ids[i,:len(enc_set1.ids)+5] = [0]+enc_set1.ids+[2,2]+[senti_token]+[2] 
    attention_mask[i,:len(enc_set1.ids)+5]=1

    if len(selected_text_token_idx) > 0:
        start_mask[i,selected_text_token_idx[0]+1]=1
        end_mask[i, selected_text_token_idx[-1]+1]=1


# In[ ]:


# Categorical Cross Entropy with Label Smoothing
# Label Smoothing is done to enhance accuracy

def custom_loss(y_true, y_pred):
    loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred, from_logits = False, label_smoothing = 0.20)
    loss = tf.reduce_mean(loss)
    return loss


# In[ ]:


import tensorflow as tf
import tensorflow.keras.backend as K
from sklearn.model_selection import StratifiedKFold
from transformers import *
import tokenizers
from keras.layers import Dense, Flatten, Conv1D, Dropout, Input
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping

def build_model():
        ids = tf.keras.layers.Input((max_len,), dtype=tf.int32)
        att = tf.keras.layers.Input((max_len,), dtype=tf.int32)
        tok =  tf.keras.layers.Input((max_len,), dtype=tf.int32) 

        config_path = RobertaConfig.from_pretrained('/kaggle/input/tf-roberta/config-roberta-base.json')
        roberta_model = TFRobertaModel.from_pretrained('/kaggle/input/tf-roberta/pretrained-roberta-base.h5', config=config_path)
        x = roberta_model(ids, attention_mask = att, token_type_ids=tok)
        
        x1 = tf.keras.layers.Dropout(0.05)(x[0])
        x1 = tf.keras.layers.Conv1D(128, 2,padding='same')(x1) 
        #128 is the no. of filters; 2 is the kernel size of each filter
        x1 = tf.keras.layers.LeakyReLU()(x1)
        x1 = tf.keras.layers.Conv1D(64, 2,padding='same')(x1)
        x1 = tf.keras.layers.Dense(1)(x1)
        x1 = tf.keras.layers.Flatten()(x1)
        x1 = tf.keras.layers.Activation('softmax')(x1)
    
        x2 = tf.keras.layers.Dropout(0.05)(x[0]) 
        x2 = tf.keras.layers.Conv1D(128, 2,padding='same')(x2)
        x2 = tf.keras.layers.LeakyReLU()(x2)
        x2 = tf.keras.layers.Conv1D(64, 2, padding='same')(x2)
        x2 = tf.keras.layers.Dense(1)(x2)
        x2 = tf.keras.layers.Flatten()(x2)
        x2 = tf.keras.layers.Activation('softmax')(x2)


        model = tf.keras.models.Model(inputs=[ids, att, tok], outputs=[x1,x2])
        optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5) 
        model.compile(loss=custom_loss, optimizer=optimizer)

        return model


# In[ ]:


#input data formating for testing

tot_test_tw = test_data.shape[0]

input_ids_t = np.ones((tot_test_tw,max_len), dtype='int32')
attention_mask_t = np.zeros((tot_test_tw,max_len), dtype='int32')
token_type_ids_t = np.zeros((tot_test_tw,max_len), dtype='int32')

for i in range(tot_test_tw):
    set1 = " "+" ".join(test_data.loc[i,'text'].split())
    enc_set1 = tokenizer.encode(set1)

    s_token = sentiment_id[test_data.loc[i,'sentiment']]
    input_ids_t[i,:len(enc_set1.ids)+5]=[0]+enc_set1.ids+[2,2]+[s_token]+[2]
    attention_mask_t[i,:len(enc_set1.ids)+5]=1


# In[ ]:


from keras.callbacks import ModelCheckpoint, EarlyStopping
from transformers import TFRobertaModel
import tensorflow.keras.backend as K
from sklearn.model_selection import StratifiedKFold

pred_start= np.zeros((input_ids_t.shape[0],max_len))
pred_end= np.zeros((input_ids_t.shape[0],max_len))

for i in range(2):
    print('--'*20)
    print('-- MODEL %i --'%(i+1))
    print('--'*20)
    K.clear_session()
    model = build_model()
    model.load_weights('/kaggle/input/model4/v4-roberta-%i.h5'%(i+3))
    pred = model.predict([input_ids_t,attention_mask_t,token_type_ids_t],verbose=1)
    pred_start = pred_start + (pred[0]/2)
    pred_end = pred_end + (pred[1]/2)


# In[ ]:


all = []
for k in range(input_ids_t.shape[0]):
    a = np.argmax(pred_start[k,])
    b = np.argmax(pred_end[k,])
    if a>b: 
        st = test_data.loc[k,'text'] 
    else:
        text1 = " "+" ".join(test_data.loc[k,'text'].split())
        enc = tokenizer.encode(text1)
        st = tokenizer.decode(enc.ids[a-1:b])
    all.append(st)
test_data['selected_text']=all
test_data.head(20)


# In[ ]:


test_data[['textID','selected_text']].to_csv('submission.csv', index=False)
print("Submission successful")


# In[ ]:


submission_df = pd.read_csv("submission.csv")
submission_df.head(10)

