#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import cohen_kappa_score as kappa_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from skopt import gp_minimize
from skopt.space import Real, Integer
from sklearn.preprocessing import StandardScaler
kappa_scorer = make_scorer(kappa_score)

import os
print(os.listdir("../input"))
print(os.listdir("../input/petfinder-adoption-prediction/"))
print(os.listdir("../input/petfinder-adoption-prediction/train"))
print(os.listdir("../input/petfinder-adoption-prediction/test"))


# In[ ]:


train_df = pd.read_csv("../input/petfinder-adoption-prediction/train/train.csv")
test_df = pd.read_csv("../input/petfinder-adoption-prediction/test/test.csv")


# In[ ]:


train_df.head()


# In[ ]:


cat_cols = ['Type','Age','Breed1','Breed2', 'Gender', 'Color1', 'Color2', 'Color3', 'MaturitySize', 
          'FurLength', 'Vaccinated', 'Dewormed', 'Sterilized','Health', 'Quantity','State','VideoAmt','PhotoAmt']


# In[ ]:


num_cols = ['Fee']


# In[ ]:


text_cols = ['Description']


# ## Handling categorical columns

# In[ ]:


embed_sizes = [len(train_df[col].unique()) + 1 for col in cat_cols]


# ## Handling numerical columns

# In[ ]:


print('scaling num_cols')
for col in num_cols:
    print('scaling {}'.format(col))
    col_mean = train_df[col].mean()
    train_df[col].fillna(col_mean, inplace=True)
    test_df[col].fillna(col_mean, inplace=True)
    scaler = StandardScaler()
    train_df[col] = scaler.fit_transform(train_df[col].values.reshape(-1, 1))
    test_df[col] = scaler.transform(test_df[col].values.reshape(-1, 1))


# ## Handling text columns

# In[ ]:


from tqdm import tqdm
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


# In[ ]:


print('getting embeddings')
def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in tqdm(open('../input/fasttext-english-word-vectors-including-subwords/wiki-news-300d-1M-subword.vec')))


# In[ ]:


num_words = 20000
maxlen = 80
embed_size = 300


# In[ ]:


train_df['Description'] = train_df['Description'].astype(str).fillna('no text')
test_df['Description'] = test_df['Description'].astype(str).fillna('no text')


# In[ ]:


print("   Fitting tokenizer...")
tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(train_df['Description'].values.tolist())


# In[ ]:


train_df['Description'] = tokenizer.texts_to_sequences(train_df['Description'])
test_df['Description'] = tokenizer.texts_to_sequences(test_df['Description'])


# In[ ]:


word_index = tokenizer.word_index
nb_words = min(num_words, len(word_index))
embedding_matrix = np.zeros((nb_words, embed_size))
for word, i in word_index.items():
    if i >= num_words: continue
    try:
        embedding_vector = embeddings_index[word]
    except KeyError:
        embedding_vector = None
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector


# In[ ]:


def get_input_features(df):
    X = {'description':pad_sequences(df['Description'], maxlen=maxlen)}
    X['numerical'] = np.array(df[num_cols])
    for cat in cat_cols:
        X[cat] = np.array(df[cat])
    return X


# ## Define NN Model

# In[ ]:


from keras.layers import Input, Embedding, Concatenate, Flatten, Dense, Dropout, BatchNormalization, CuDNNLSTM, SpatialDropout1D
from keras.layers import Bidirectional, GlobalAveragePooling1D, GlobalMaxPool1D
from keras.models import Model
from keras.optimizers import  Adam

categorical_inputs = []
for cat in cat_cols:
    categorical_inputs.append(Input(shape=[1], name=cat))

categorical_embeddings = []
for i, cat in enumerate(cat_cols):
    categorical_embeddings.append(
        Embedding(embed_sizes[i], 10)(categorical_inputs[i]))

categorical_logits = Concatenate()([Flatten()(cat_emb) for cat_emb in categorical_embeddings])
categorical_logits = Dense(256, activation = 'relu')(categorical_logits)


numerical_inputs = Input(shape=[len(num_cols)], name='numerical')
numerical_logits = numerical_inputs
numerical_logits = BatchNormalization()(numerical_logits)
numerical_logits = Dense(128, activation = 'relu')(numerical_logits)

text_inp = Input(shape=[maxlen], name='description')
text_embed = Embedding(nb_words, embed_size, weights=[embedding_matrix], trainable=False)(text_inp)
text_logits = SpatialDropout1D(0.2)(text_embed)
text_logits = Bidirectional(CuDNNLSTM(64, return_sequences=True))(text_logits)
avg_pool = GlobalAveragePooling1D()(text_logits)
max_pool = GlobalMaxPool1D()(text_logits)
text_logits = Concatenate()([avg_pool, max_pool])

x = Concatenate()([categorical_logits, text_logits, numerical_logits])
x = BatchNormalization()(x)

x = Dense(128, activation = 'relu')(x)
x = Dropout(0.3)(x)
out = Dense(1, activation = 'sigmoid')(x)

model = Model(inputs=[text_inp] + categorical_inputs + [numerical_inputs],outputs=out)
model.compile(optimizer=Adam(lr = 0.0001), loss = 'mse')


# In[ ]:


from sklearn.model_selection import train_test_split

tr_df, val_df = train_test_split(train_df, test_size = 0.2, random_state = 23)


# In[ ]:


# from keras.utils.np_utils import to_categorical


# In[ ]:


tr_df['AdoptionSpeed'].values.shape


# In[ ]:


y_train = tr_df['AdoptionSpeed'].values / 4
y_valid = val_df['AdoptionSpeed'].values / 4


# In[ ]:


#for i, l in enumerate(tr_df['AdoptionSpeed'].values):
#    y_train[i,l] = 1
#for i, l in enumerate(val_df['AdoptionSpeed'].values):
#    y_valid[i,l] = 1


# In[ ]:


X_train = get_input_features(tr_df)
X_valid = get_input_features(val_df)
X_test = get_input_features(test_df)


# In[ ]:


hist = model.fit(X_train, y_train, validation_data = (X_valid,y_valid), batch_size = 256, epochs = 10, verbose = True)


# In[ ]:


y_pred = model.predict(X_test)[:,0]


# In[ ]:


y_pred.shape


# In[ ]:


y_pred = np.round(y_pred * 4).astype(int)


# In[ ]:


y_pred[:10]


# In[ ]:


#y_pred2 = np.argmax(y_pred,axis = 1)
#y_pred2.shape


# In[ ]:



submission_df = pd.DataFrame(data={"PetID":test_df["PetID"], "AdoptionSpeed":y_pred})
submission_df.to_csv("submission.csv", index=False)


# In[ ]:


submission_df['AdoptionSpeed'].mean(), train_df['AdoptionSpeed'].mean()


# In[ ]:


submission_df.head(20)


# In[ ]:




