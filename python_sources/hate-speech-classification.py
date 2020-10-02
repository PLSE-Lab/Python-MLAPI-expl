#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences, sequence
from keras.models import Sequential
from keras.layers import Dense, Flatten, CuDNNLSTM, LSTM, Conv1D, MaxPooling1D, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras import optimizers,metrics,layers
import matplotlib.pyplot as plt
## Plot
# NLTK
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

# Other
import re
import string
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE


# In[ ]:


w2vdf = pd.read_csv('../input/word2vec.csv')


# In[ ]:


w2vdf.head()


# In[ ]:


vocab = ['unk']
g = dict()
g['unk'] = np.array([0]*300)
for i in range(w2vdf.shape[0]):
    g[w2vdf.iloc[i,1]] = np.fromstring(w2vdf.iloc[i,2][1:-1],dtype = float,sep = ' ')
    vocab.append(w2vdf.iloc[i,1])
#print(g)


# In[ ]:


datadf = pd.read_csv('../input/data.csv')
datadf = datadf.iloc[:, 1:]
datadf.columns = ['text', 'label']
df = datadf
print(df.head(20))
df.shape


# In[ ]:


labels = [df.at[i,'label'] for i in range(df.shape[0])]


# In[ ]:


def cnfmatrix(y_test,results):
    fp = 0.0
    fn = 0.0
    tp = 0.0
    tn = 0.0
    t = 0.0
    n = 0.0
    results.shape
    for i in range(results.shape[0]):
        if y_test[i]==1 and results[i]==1:
            tp+=1
            t+=1
        elif y_test[i]==1 and results[i]==0:
            fn+=1
            t+=1
        elif y_test[i]==0 and results[i]==1:
            fp+=1
            n+=1
        elif y_test[i]==0 and results[i]==0:
            tn+=1
            n+=1
    print(tp/results.shape[0],fp/results.shape[0])
    print(fn/results.shape[0],tn/results.shape[0])
    Precision  = tp/(tp+fp)
    Recall = tp/(tp+fn)
    print("Precision: ",Precision,"Recall: ",Recall)
    f1score = (2*Precision*Recall)/(Precision+Recall)
    print("f1score: ",f1score)
    print("accuracy: ",(tp+tn)/results.shape[0])
    print("hate_acc: ", (tp)/t)
    print("non_hate_acc: ", (tn)/n)


# In[ ]:


def clean_text(text):
    
    ## Remove puncuation
    text = text.translate(string.punctuation)
    
    ## Convert words to lower case and split them
    text = text.lower().split()
    
    ## Remove stop words
#     stops = set(stopwords.words("english"))
#     text = [w for w in text if not w in stops and len(w) >= 3]
    
    text = " ".join(text)

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    #text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", "", text)
    text = re.sub(r"\/", "", text)
    text = re.sub(r"\^", "", text)
    text = re.sub(r"\+", "", text)
    text = re.sub(r"\-", "", text)
    text = re.sub(r"\=", "", text)
    text = re.sub(r"'", " ", text)
    
    return text


# In[ ]:


df.head()


# In[ ]:


df.iloc[:, 1]


# In[ ]:


data = []
print(data)
for i in range(df.shape[0]):
    b = df.at[i,'text'].split()
    p = []
    for j in b:
        #print(pdict[j])
        p.append(g[j])
    p = p + [g['unk']]*(50-len(p))
    data.append(p)
labels = np.array(labels)
labels.shape
ts = 7000
#print(data[:10])


# In[ ]:


from sklearn import preprocessing
data = np.array(data,dtype=float)
temp = np.zeros(data.shape)

for i in range(data.shape[0]):
        temp[i] = preprocessing.normalize(data[i])
#print(temp[0][0])
datab = np.array([[vocab.index(j) for j in df.iloc[i,0].split()] for i in range(df.shape[0])])
datab = sequence.pad_sequences(datab, maxlen=50)
xb_train = datab[:ts]
xb_test = datab[ts:]
x_train,y_train = temp[:ts],labels[:ts]
x_test,y_test = temp[ts:],labels[ts:]
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


# In[ ]:


xb_train[-1]


# In[ ]:


from keras.wrappers.scikit_learn import KerasRegressor,KerasClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
embedding_matrix = np.array([g[i] for i in vocab])
def the_models():
    model = Sequential()
    model.add(Embedding(len(vocab), 300, input_length=50, trainable=True))
    #model.add(layers.Dense(50,activation = 'linear'))
    model.add(CuDNNLSTM(50,return_sequences=True))
    model.add(CuDNNLSTM(50))
    model.add(layers.Dense(20,activation = 'relu'))
    model.add(layers.Dense(1,activation = 'sigmoid'))
    #sgd = optimizers.SGD(lr=0.0001, decay=0.001, momentum=0.0, nesterov=True)
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    return model


#ann_estimator = KerasClassifier(build_fn = the_models, epochs=10, batch_size=20, verbose=True)
#boosted_ann = AdaBoostClassifier(base_estimator=ann_estimator, n_estimators = 1)

#the_models()


# In[ ]:


model = the_models()
history = model.fit(xb_train,y_train,epochs =8,batch_size=100)
#boosted_ann.fit(xb_train, y_train)


# In[ ]:


#history = model.fit(x_train,y_train,batch_size=None,epochs=100,validation_data=None)


# In[ ]:


predictions = model.predict_classes(xb_test)
#results = results.tolist()
results = predictions
print(len(results))


# In[ ]:


# o = [x for x in range(len(results[:30]))]
# plt.scatter(o,results[:30],c='r')
# plt.scatter(o,y_test.tolist()[:30],c = 'g')


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import mpld3
mpld3.enable_notebook()
plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
plt.show()


# In[ ]:


cnfmatrix(y_test,np.array(results))


# In[ ]:




