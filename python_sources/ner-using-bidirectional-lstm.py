#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# 
# **Importing the dataset for named entity recognition model**

# In[ ]:


# dframe = pd.read_csv("../input/entity-annotated-corpus/ner.csv", encoding = "ISO-8859-1", error_bad_lines=False)
f=open("../input/wikiner-data/aij-wikiner-fr-wp2.txt", "r")
data=[]
contents =f.readlines()
indx=0
for x in contents:
    words=x.split()
#     print(len(words))
    for i in range(len(words)):
        
        tags=words[i].split("|")
#         print(len(tags))
        data.append([indx,tags[0],tags[2]])
#     print(words)
    indx+=1
wikiner_data= pd.DataFrame(data, columns=['sentence', 'word', 'tag'])


# In[ ]:


wikiner_data.head(20)


# In[ ]:


wikiner_data.head()


# > **Create list of list of tuples to differentiate each sentence from each other**

# In[ ]:


class SentenceGetter(object):
    
    def __init__(self, dataset):
        self.n_sent = 1
        self.dataset = dataset
        self.empty = False
        agg_func = lambda s: [(w, t) for w, t in zip(s["word"].values.tolist(),
                                                       
                                                        s["tag"].values.tolist())]
        self.grouped = self.dataset.groupby("sentence").apply(agg_func)
        self.sentences = [s for s in self.grouped]
    
    def get_next(self):
        try:
            s = self.grouped["sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None


# In[ ]:


getter = SentenceGetter(wikiner_data)


# In[ ]:


sentences = getter.sentences


# In[ ]:


maxlen = max([len(s) for s in sentences])
print ('Maximum sequence length:', maxlen)


# In[ ]:


# Check how long sentences are so that we can pad them
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use("ggplot")


# In[ ]:


plt.hist([len(s) for s in sentences], bins=50)
plt.show()


# In[ ]:


words = list(set(wikiner_data["word"].values))
words.append("ENDPAD")


# In[ ]:


n_words = len(words); n_words


# In[ ]:


tags = list(set(wikiner_data["tag"].values))


# In[ ]:


n_tags = len(tags); n_tags


# **Converting words to numbers and numbers to words**

# In[ ]:


word2idx = {w: i for i, w in enumerate(words)}
tag2idx = {t: i for i, t in enumerate(tags)}


# In[ ]:


from keras.preprocessing.sequence import pad_sequences
X = [[word2idx[w[0]] for w in s] for s in sentences]


# In[ ]:


X = pad_sequences(maxlen=242, sequences=X, padding="post",value=n_words - 1)


# In[ ]:


y = [[tag2idx[w[1]] for w in s] for s in sentences]


# In[ ]:


y = pad_sequences(maxlen=242, sequences=y, padding="post", value=tag2idx["O"])


# In[ ]:


from keras.utils import to_categorical
y = [to_categorical(i, num_classes=n_tags) for i in y]


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[ ]:


from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional


# In[ ]:


get_ipython().system('pip install git+https://www.github.com/keras-team/keras-contrib.git')


# In[ ]:


from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
from keras_contrib.layers import CRF
input = Input(shape=(242,))
model = Embedding(input_dim=n_words + 1, output_dim=20,
                  input_length=242, mask_zero=True)(input)  # 20-dim embedding
model = Bidirectional(LSTM(units=50, return_sequences=True,
                           recurrent_dropout=0.1))(model)  # variational biLSTM
model = TimeDistributed(Dense(50, activation="relu"))(model)  # a dense layer as suggested by neuralNer
crf = CRF(n_tags)  # CRF layer
out = crf(model)  # output
model = Model(input, out)


# In[ ]:


model.compile(optimizer="rmsprop", loss=crf.loss_function, metrics=[crf.accuracy])
model.summary()


# In[ ]:


history = model.fit(X_train, np.array(y_train), batch_size=32, epochs=5, validation_split=0.2, verbose=1)


# In[ ]:


# i = 1
# p = model.predict(np.array([X_test[i]]))
# p = np.argmax(p, axis=-1)
# print("{:15} ({:5}): {}".format("Word", "True", "Pred"))
# for w,pred in zip(X_test[i],p[0]):
#     print("{:15}: {}".format(words[w],tags[pred]))


# In[ ]:


idx2tag = {i: w for w, i in tag2idx.items()}

def pred2label(pred):
    out = []
    for pred_i in pred:
        out_i = []
        for p in pred_i:
            p_i = np.argmax(p)
            out_i.append(idx2tag[p_i].replace("PAD", "O"))
        out.append(out_i)
    return out
test_pred = model.predict(X_test, verbose=1)   
pred_labels = pred2label(test_pred)
test_labels = pred2label(y_test)


# In[ ]:


get_ipython().system('pip install seqeval')


# In[ ]:


from seqeval.metrics import precision_score, recall_score, f1_score, classification_report


# In[ ]:


print("F1-score: {:.1%}".format(f1_score(test_labels, pred_labels)))


# In[ ]:


get_ipython().system('pip install sklearn_crfsuite')


# In[ ]:


from  sklearn_crfsuite.metrics import flat_classification_report  
report = flat_classification_report(y_pred=pred_labels, y_true=test_labels)
print(report)

