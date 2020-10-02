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

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# This notebook shows how to build and run Google sentencepiece package and tokenize the encoded text.

# # build sentencepiece
# 
# Note: Internet must be enabled in kernel environment's settings for this step.
# 
# Download sentencepiece's source code and build the package.

# In[ ]:


get_ipython().run_cell_magic('bash', '-e', "if ! [[ -f ./spm_train ]]; then\n  wget https://github.com/google/sentencepiece/archive/v0.1.8.zip\n  echo '8799f4983608897e8eb3370385eda149180d309c7276db939f955d6507d53846  v0.1.8.zip' | sha256sum -c\n  unzip v0.1.8.zip\n  conda install -y cmake pkg-config\n  export SENTENCEPIECE_HOME=$(pwd)/sentencepiece\n  export PKG_CONFIG_PATH=${SENTENCEPIECE_HOME}/lib/pkgconfig\n  (cd sentencepiece-0.1.8 && mkdir -p build)\n  (cd sentencepiece-0.1.8/build && cmake -DCMAKE_INSTALL_PREFIX=${SENTENCEPIECE_HOME} ..  && make -j4 && make install)\n  (cd sentencepiece-0.1.8/python && python setup.py install)\n  rm -rf sentencepiece-0.1.8 v0.1.8.zip\nfi")


# # Prepare input

# In[ ]:


def read_train_text(filename='../input/train.csv'):
    return pd.read_csv(filename)

def write_cipher_text(texts, filename='spm_train.txt'):
    with open(filename, 'w',encoding='utf-8') as f:
        for text in texts:
            f.write(text + "\n")

train_df = read_train_text()
test_df = read_train_text(filename='../input/test.csv')
ciphertexts = list(train_df.ciphertext.values) + list(test_df.ciphertext.values)
write_cipher_text(ciphertexts)


# # Train SentencePieceModel

# In[ ]:


import sentencepiece as spm
spm.SentencePieceTrainer.Train(
        '--input=' + os.path.join('spm_train.txt') +
        ' --model_prefix=train --vocab_size=1000')


# In[ ]:


def encode_ciphertext(ciphertext):
    sp = spm.SentencePieceProcessor()
    sp.Load('train.model')
    encodedtext = []
    for text in ciphertext:
        encodedtext.append(sp.encode_as_ids(text))
    return encodedtext

train_encoded = encode_ciphertext(train_df.ciphertext)
test_encoded = encode_ciphertext(test_df.ciphertext)


# # Train Random Forest Model

# In[ ]:


from collections import defaultdict, Counter

word_counter = defaultdict(int)
for text in train_encoded + test_encoded:
    counter = Counter(text)
    for l,c in counter.items():
        word_counter[l] += c


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


# In[ ]:


def build_df(word_counter, df, encodedtext):
    keys = list(word_counter.keys())    
    rows = []
    for rowid, row in df.iterrows():
        counter = Counter(encodedtext[rowid])
        entry = [counter.get(k, 0) for k in keys]
        entry += [row['difficulty']]
        if 'target' in row:
            entry += [row['target']]
        rows.append(entry)
    return pd.DataFrame(rows)

train = build_df(word_counter, train_df, train_encoded)
test = build_df(word_counter, test_df, test_encoded)


# In[ ]:


X = train.iloc[:, :-1]
Y = train.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)


# In[ ]:


rf = RandomForestClassifier(n_estimators=1000, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
acc = np.sum(y_pred == y_test) / len(y_test)
print(acc)


# # Make submission

# In[ ]:


rf.fit(X, Y)
y_pred = rf.predict(test)
submission = pd.DataFrame(test_df.Id, columns=['Id'])
submission['Predicted'] = y_pred
submission.to_csv('submission.csv', index=False)


# In[ ]:




