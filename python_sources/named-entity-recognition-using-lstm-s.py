#!/usr/bin/env python
# coding: utf-8

# ### Task 1: Project Overview and Import Modules

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
np.random.seed(0)
plt.style.use("ggplot")

import tensorflow as tf
print('Tensorflow version:', tf.__version__)
print('GPU detected:', tf.config.list_physical_devices('GPU'))


# ### Task 2: Load and Explore the NER Dataset

# In[ ]:


data = pd.read_csv(r"../input/clientdat/UTPBatchModified_final.csv",encoding  ='ISO-8859-1')
for i in range(1,18):
    data=data.drop(['ATTRIBUTE_NAME_'+str(i)], axis = 1)
data=data.drop(['NOUN'], axis = 1)
data=data.drop(['MFR_NAME_1'], axis = 1)
data=data.drop(['COMMENTS'], axis = 1)
data=data.drop(['STATUS (CLEANSED/HOLD/ENRICHED)'], axis = 1)
data=data.drop(['Mfr/Vendor_Remarks'], axis = 1)


# In[ ]:


for i in range(0,(len(data))):
    if (data['STANDARDIZED_VALUE_2'].loc[i][-2:]!='IN' and data['STANDARDIZED_VALUE_2'].loc[i]!='-'):
        data['STANDARDIZED_VALUE_2'].loc[i]=data['STANDARDIZED_VALUE_2'].loc[i]+' IN'
    elif(data['STANDARDIZED_VALUE_2'].loc[i]== '-'):
        data['STANDARDIZED_VALUE_2'].loc[i]=data['STANDARDIZED_VALUE_2'].loc[i]+' IN'
    elif (data['STANDARDIZED_VALUE_2'].loc[i][-2:]=='IN'):
        data['STANDARDIZED_VALUE_2'].loc[i]=data['STANDARDIZED_VALUE_2'].loc[i][:-2]+'-IN'
    elif (data['STANDARDIZED_VALUE_2'].loc[i][-2:]=='MM'):
        data['STANDARDIZED_VALUE_2'].loc[i]=data['STANDARDIZED_VALUE_2'].loc[i][:-2]+'-MM'
        
for i in range(0,(len(data))):
    if (data['STANDARDIZED_VALUE_10'].loc[i][-2:]!='IN' and data['STANDARDIZED_VALUE_10'].loc[i]!='-'):
        data['STANDARDIZED_VALUE_10'].loc[i]=data['STANDARDIZED_VALUE_10'].loc[i]+'_IN'
    elif(data['STANDARDIZED_VALUE_10'].loc[i]== '-'):
        data['STANDARDIZED_VALUE_10'].loc[i]='_IN'
    elif (data['STANDARDIZED_VALUE_10'].loc[i][-2:]=='IN'):
        data['STANDARDIZED_VALUE_10'].loc[i]=data['STANDARDIZED_VALUE_10'].loc[i][:-2]+'_IN'
    elif (data['STANDARDIZED_VALUE_10'].loc[i][-2:]=='MM'):
        data['STANDARDIZED_VALUE_10'].loc[i]=data['STANDARDIZED_VALUE_2'].loc[i][:-2]+'_MM'


# In[ ]:


data['STANDARDIZED_VALUE_10']


# In[ ]:


test_data=pd.DataFrame()
for i in range(0,len(data)):
    data1 = data.iloc[i].reset_index()
    for j in range(len(data.iloc[i])):
        if(data1[i][j]=='-'):
            data1=data1.drop(j)
    t_data=pd.DataFrame([{'Words' : 'wrd', 'Tags' : 'tg'}])
    t_data=t_data['Words'].append((data1))
    t_data=t_data.drop(0)
    t_data=t_data.drop(1)
    t_data=t_data.rename(columns={'index': 'Tags', i :'Words'})
    t_data=t_data.reset_index()[['Tags','Words']]
    l = [(i+1) for s in range(len(t_data))]
    Sentence=pd.Series(l)
    t_data['Sentence #']=Sentence
    test_data=test_data.append(t_data)
test_data=test_data.reset_index()[['Tags','Words','Sentence #']]
test_data['Tags']=test_data['Tags'].replace({'MFR_NAME_1':'MFR_NAME'})
test_data.shape    


# In[ ]:


test_data.drop_duplicates(subset ="Words",keep = 'first', inplace = True)
df2 = pd.DataFrame({"Tags":['O'], 
                    "Words":['None'], 'Sentence #':[1007]})
test_data=test_data.append(df2)
test_data=test_data.reset_index()[['Tags','Words','Sentence #']]


# In[ ]:


words = list(set(test_data["Words"].values))
words.append("ENDPAD")
num_words = len(words)
tags = list(set(test_data["Tags"].values))
num_tags = len(tags)
print("Unique words in corpus:", test_data['Words'].nunique())
print("Unique tags in corpus:", test_data['Tags'].nunique())


# ### Task 3: Retrieve Sentences and Corresponsing Tags

# In[ ]:


class SentenceGetter(object):
    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, t) for w, t in zip(s["Words"].values.tolist(), s["Tags"].values.tolist())]
        self.grouped = self.data.groupby("Sentence #").apply(agg_func)
        self.sentences = [s for s in self.grouped]
    
    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None


# In[ ]:


getter = SentenceGetter(test_data)
sentences = getter.sentences


# ### Task 4: Define Mappings between Sentences and Tags

# In[ ]:


word2idx = {w: i + 1 for i, w in enumerate(words)}
tag2idx = {t: i for i, t in enumerate(tags)}


# ### Task 5: Padding Input Sentences and Creating Train/Test Splits

# In[ ]:


maxlen = max([len(s) for s in sentences])
print ('Maximum sentence length:', maxlen)


# In[ ]:


plt.hist([len(s) for s in sentences], bins=50)
plt.show()


# In[ ]:


from tensorflow.keras.preprocessing.sequence import pad_sequences

max_len = 20

X = [[word2idx[w[0]] for w in s] for s in sentences]


y = [[tag2idx[w[1]] for w in s] for s in sentences]


# In[ ]:


X = pad_sequences(maxlen=max_len, sequences=X, padding="post", value=num_words-1)
y = pad_sequences(maxlen=max_len, sequences=y, padding="post", value=tag2idx["O"])


# In[ ]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


# ### Task 6: Build and Compile a Bidirectional LSTM Model

# In[ ]:


from tensorflow.keras import Model, Input
from tensorflow.keras.layers import LSTM, Embedding, Dense
from tensorflow.keras.layers import TimeDistributed, SpatialDropout1D, Bidirectional


# In[ ]:


input_word = Input(shape=(max_len,))
model = Embedding(input_dim=num_words, output_dim=20, input_length=max_len)(input_word)
model = SpatialDropout1D(0.1)(model)
model = Bidirectional(LSTM(units=100, return_sequences=True, recurrent_dropout=0.1))(model)
out = TimeDistributed(Dense(num_tags, activation="softmax"))(model)
model = Model(input_word, out)
model.summary()


# In[ ]:


model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])


# ### Task 7: Train the Model

# In[ ]:


from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


# In[ ]:


get_ipython().run_cell_magic('time', '', '\nchkpt = ModelCheckpoint("model_weights.h5", monitor=\'val_loss\',verbose=1, save_best_only=True, save_weights_only=True, mode=\'min\')\nearly_stopping = EarlyStopping(monitor=\'val_accuracy\', min_delta=0, patience=1, verbose=0, mode=\'max\', baseline=None, restore_best_weights=False)\nhistory = model.fit(x=x_train,y=y_train,validation_data=(x_test,y_test),batch_size=32,epochs=20,verbose=1)')


# ### Task 8: Evaluate Named Entity Recognition Model

# In[ ]:


model.evaluate(x_test, y_test)


# In[ ]:


i = np.random.randint(0, x_test.shape[0]) #659
p = model.predict(np.array([x_test[i]]))
p = np.argmax(p, axis=-1)
y_true = y_test[i]
print("{:15}{:5}\t {}\n".format("Word", "True", "Pred"))
print("-" *30)
for w, true, pred in zip(x_test[i], y_true, p[0]):
    print("{:15}{}\t{}".format(words[w-1], tags[true], tags[pred]))


# In[ ]:




