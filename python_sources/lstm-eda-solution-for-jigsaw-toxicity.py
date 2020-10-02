#!/usr/bin/env python
# coding: utf-8

# <img src="https://storage.googleapis.com/kaggle-datasets/179715/404258/toxicity.png?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1557355694&Signature=YiAIra%2B04%2FeJCV8RCLMlgp3jsMsorIG4xJx4Fc54FXVK9MjrV6ZqVzrktUXwfc0iK2BfPCvV12DBY0C7B9GycwuM7Vnt%2BgZP8LF8I2afZLJKgjPNVAY7iQBkXCdISstzhn%2FrrTS5EV0M2qrLr2V8SxSakUsNrnXE354VC5jQrVjCbjTQo0cfbGZYg9akO7LS9D5StlmlEgYVMP2VpyYzZ90oPB%2FbW3uvpgh0APnzEfHrjxxPDo3fY0Tc3hHvNZOVBqh%2B9CQkJOaL7wkrjzzxsI0nXNMnaxpJdMdMtBCI9tYDM6eG3Yqn07qoGNVk3UKVsMOqbE%2F36W0kcpSKmGomiA%3D%3D" width=1900px height=400px/>

# # Jigsaw Unintended Bias in Toxicity Classification
# <h5>Detect toxicity across a diverse range of conversations</h5>
# <h3> Kernel description: </h3>
# In this competition the intention is to detect toxic comments and minimize unintended model bias, for that we to optimize a metric designed to measure unintended bias
# 

# # Table of Contents:
# 
# **1. [Problem Definition](#id1)** <br>
# **2. [Data Background](#id2)** <br>
# **3. [Load the Dataset](#id3)** <br>
# **4. [Data Pre-processing](#id4)** <br>
# **5. [Model](#id5)** <br>
# **6. [Visualization and Analysis of Results](#id6)** <br>
# **7. [Submittion](#id7)** <br>
# **8. [References](#ref)** <br>

# <a id="id1"></a> <br> 
# # **1. Problem Definition:** 
# 
# When the Conversation AI team first built toxicity models, they found that the models incorrectly learned to associate the names of frequently attacked identities with toxicity. Models predicted a high likelihood of toxicity for comments containing those identities (e.g. "gay"), even when those comments were not actually toxic (such as "I am a gay woman"). This happens because training data was pulled from available sources where, unfortunately, certain identities are overwhelmingly referred to in offensive ways. Training a model from data with these imbalances risks simply mirroring those biases back to users.
# 
# In this competition, our challenge is to build a model that recognizes toxicity and minimizes this type of unintended bias with respect to mentions of identities.

# <a id="id2"></a> <br> 
# # **2. Data Background:** 
# At the end of 2017 the Civil Comments platform shut down and chose make their ~2m public comments from their platform available in a lasting open archive so that researchers could understand and improve civility in online conversations for years to come. Jigsaw sponsored this effort and extended annotation of this data by human raters for various toxic conversational attributes.
# 
# In the data supplied for this competition, the text of the individual comment is found in the comment_text column. Each comment in Train has a toxicity label (**target**), and models should predict the **target toxicity** for the Test data. This attribute (and all others) are fractional values which represent the fraction of human raters who believed the attribute applied to the given comment.
# 
# ** Very Importante! **
# <p style="color:red;">For evaluation, test set examples with target >= 0.5 will be considered to be in the positive class (toxic).</p>
# 
# The data also has several additional toxicity subtype attributes. Models do not need to predict these attributes for the competition, they are included as an additional avenue for research. Subtype attributes are:
# 
# * severe_toxicity
# * obscene
# * threat
# * insult
# * identity_attack
# * sexual_explicit

# Additionally, a subset of comments have been labelled with a variety of identity attributes, representing the identities that are mentioned in the comment. The columns corresponding to identity attributes are listed below. **Only identities with more than 500 examples in the test set (combined public and private) will be included in the evaluation calculation**.
# 
# *Identities with MORE than 500 examples*
# 
# * **male**
# * **female**
# * **psychiatric_or_mental_illness**
# * **homosexual_gay_or_lesbian**
# * **christian**
# * **jewish**
# * **muslim**
# * **black**
# * **white**
# 
# *Identities with LESS than 500 examples*
# * transgender
# * other_gender
# * heterosexual
# * bisexual
# * other_sexual_orientation
# * hindu
# * buddhist
# * atheist
# * other_religion
# * asian
# * latino
# * other_race_or_ethnicity
# * physical_disability
# * intellectual_or_learning_disability
# * other_disability

# <h3>*Examples*</h3>
# Here are a few examples of comments and their associated toxicity and identity labels. Label values range from 0.0 - 1.0 represented the fraction of raters who believed the label fit the comment.
# 
# **Comment:** i'm a white woman in my late 60's and believe me, they are not too crazy about me either!!<br>
# **Toxicity Labels:** All 0.0<br>
# **Identity Mention Labels:** female: 1.0, white: 1.0 (all others 0.0)<br>
# 
# **Comment:** Why would you assume that the nurses in this story were women?<br>
# **Toxicity Labels:** All 0.0<br>
# **Identity Mention Labels:** female: 0.8 (all others 0.0)<br>
# 
# **Comment:** Continue to stand strong LGBT community. Yes, indeed, you'll overcome and you have.<br>
# **Toxicity Labels:** All 0.0<br>
# **Identity Mention Labels:** homosexual_gay_or_lesbian: 0.8, bisexual: 0.6, transgender: 0.3 (all others 0.0)<br>

# <a id="id3"></a> <br> 
# # **3. Load the Dataset** 
# 

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.preprocessing import text, sequence
from keras import backend as K
from keras.models import load_model
import keras
import pickle

import os
from tqdm import tqdm
tqdm.pandas()


import warnings
warnings.filterwarnings('ignore')


# In[ ]:


import os
print(os.listdir("../input"))


# <h3>3.1 Heads of the data</h3>
# The first thing we have to  do is loading the data and take a quick look at the number of rows and collumns in dataset. This is the first contact with the data. Do this after carefully reading the background on the data and the meaning of the attibutes provided by the problem.
# 

# In[ ]:


train = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv')
test = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv')
sub = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/sample_submission.csv')


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


total_num_comments = train.shape[0]
unique_comments = train['comment_text'].nunique()


print('Train set: %d (Entries) and %d (Attributes).' % (train.shape[0], train.shape[1]))
print('Test set: %d (Entries) and %d (Attributes).' % (test.shape[0], test.shape[1]))

print('Number of Unique Comments {}'.format(unique_comments))
print('Percentage of Unique Comments %.2f%%' %( (unique_comments/total_num_comments)*100 ))


# <h3>3.2 Data Visualization</h3>
# Check Distribution of comment Lenght and Word Numbers.
# 

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid", {"axes.facecolor": ".9"})
sns.set_context("notebook", font_scale=1, rc={"lines.linewidth": 2.5})


# In[ ]:


train['comment_length'] = train['comment_text'].apply(lambda x : len(x))
plt.figure(figsize=(16,4))
sns.distplot(train['comment_length'])
plt.show()


# Most comments are short. However, there is an increase when comments arrive close to 1000, perhaps because that is the maximum number of characters allowed in the tool that the data was collected.
# 

# A similar behavior happens with the test data

# In[ ]:


'''
test['comment_length'] = test['comment_text'].apply(lambda x : len(x))
plt.figure(figsize=(16,4))
sns.distplot(test['comment_length'])
plt.show()
'''


# Let's take a look at the number of words in the attribute **comment_text**.

# In[ ]:


'''train['word_count'] = train['comment_text'].apply(lambda x : len(x.split(' ')))''
test['word_count'] = test['comment_text'].apply(lambda x : len(x.split(' ')))
bin_size = max(train['word_count'].max(), test['word_count'].max())//10
plt.figure(figsize=(20, 6))
sns.distplot(train['word_count'], bins=bin_size)
sns.distplot(test['word_count'], bins=bin_size)
plt.show()
'''


# In[ ]:


train['toxic_class'] = train['target'] >= 0.5
plt.figure(figsize=(16,4))
sns.countplot(train['toxic_class'])
plt.title('Toxic vs Non Toxic Comments')
plt.show()


# Looking at the figure "Toxic vs Non Toxic Comments" we can see that the vast majority of comments are non-toxic. Let's check if there is any relationship between the toxic comments and the size of the comments.

# Another interesting step is to know how toxic comments are distilled over time.

# In[ ]:


'''train['created_date'] = pd.to_datetime(train['created_date']).values.astype('datetime64[M]')
target_df = train.sort_values('created_date').groupby('created_date', as_index=False).agg({'id':['count'], 'target':['mean']})
target_df.columns = ['Date', 'Count', 'Toxicity Rate']'''


# In[ ]:


'''plt.figure(figsize=(16,4))
sns.lineplot(x=target_df['Date'], y=target_df['Toxicity Rate'])
plt.title('Toxicity over time')
plt.show()'''


# In a period in late 2015, the toxicity rate was low for some reason. Maybe the end of the year will make people less toxic.But, the same behavior doesn't repeat at the end of 2016.

# In[ ]:


'''plt.figure(figsize=(16,4))
sns.lineplot(x=target_df['Date'], y=target_df['Count'])
plt.title('Count of toxicity comments over time')
plt.show()'''


# <a id="id4"></a> <br> 
# # **4. Data Pre-processing** 

# In[ ]:


# Take care of dataframe memory
mem_usg = train.memory_usage().sum() / 1024**2 
print("Memory usage is: ", mem_usg, " MB")


# In[ ]:


train = train[["target", "comment_text"]]
mem_usg = train.memory_usage().sum() / 1024**2 
print("Memory usage is: ", mem_usg, " MB")


# In[ ]:


train_data = train["comment_text"]
label_data = train["target"]
test_data = test["comment_text"]
train_data.shape, label_data.shape, test_data.shape


# In[ ]:


tokenizer = text.Tokenizer()
tokenizer.fit_on_texts(list(train_data) + list(test_data))


# In[ ]:


train_data = tokenizer.texts_to_sequences(train_data)
test_data = tokenizer.texts_to_sequences(test_data)


# In[ ]:


MAX_LEN = 200
train_data = sequence.pad_sequences(train_data, maxlen=MAX_LEN)
test_data = sequence.pad_sequences(test_data, maxlen=MAX_LEN)


# In[ ]:


max_features = None


# In[ ]:


max_features = max_features or len(tokenizer.word_index) + 1
max_features


# In[ ]:


type(train_data), type(label_data.values), type(test_data)
label_data = label_data.values


# <a id="id5"></a> <br> 
# # **5. Model** 

# In[ ]:


# Keras Model
# Model Parameters
NUM_HIDDEN = 512
EMB_SIZE = 256
LABEL_SIZE = 1
MAX_FEATURES = max_features
DROP_OUT_RATE = 0.25
DENSE_ACTIVATION = "sigmoid"
NUM_EPOCHS = 1

# Optimization Parameters
BATCH_SIZE = 512
LOSS_FUNC = "binary_crossentropy"
OPTIMIZER_FUNC = "adam"
METRICS = ["accuracy"]

class LSTMModel:
    
    def __init__(self):
        self.model = self.build_graph()
        self.compile_model()
    
    def build_graph(self):
        model = keras.models.Sequential([
            keras.layers.Embedding(MAX_FEATURES, EMB_SIZE),
            keras.layers.CuDNNLSTM(NUM_HIDDEN),
            keras.layers.Dropout(rate=DROP_OUT_RATE),
            keras.layers.Dense(LABEL_SIZE, activation=DENSE_ACTIVATION)])
        return model
    
    def compile_model(self):
        self.model.compile(
            loss=LOSS_FUNC,
            optimizer=OPTIMIZER_FUNC,
            metrics=METRICS)


# In[ ]:


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import gc

X_train = train_data
y_train = label_data
X_test = test_data

KFold_N = 5
from sklearn.model_selection import KFold
splits = list( KFold(n_splits=KFold_N).split(X_train,y_train) )

from keras.callbacks import EarlyStopping, ModelCheckpoint
import keras.backend as K
import numpy as np


oof_preds = np.zeros((X_train.shape[0]))
test_preds = np.zeros((X_test.shape[0]))


# In[ ]:


for fold in range(KFold_N):
    K.clear_session()
    tr_ind, val_ind = splits[fold]
    ckpt = ModelCheckpoint(f'gru_{fold}.hdf5', save_best_only = True)
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)
    model = LSTMModel().model#build_model()
    model.fit(X_train[tr_ind],
        y_train[tr_ind]>0.5,
        batch_size=BATCH_SIZE,
        epochs=NUM_EPOCHS,
        validation_data=(X_train[val_ind], y_train[val_ind]>0.5),
        callbacks = [es,ckpt])

    oof_preds[val_ind] += model.predict(X_train[val_ind])[:,0]
    test_preds += model.predict(X_test)[:,0]
    
test_preds /= KFold_N    


# In[ ]:


from sklearn.metrics import roc_auc_score
roc_auc_score(y_train>=0.5,oof_preds)


# <a id="id7"></a> <br> 
# # **7. Submittion** 

# In[ ]:


submission = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/sample_submission.csv', index_col='id')
submission['prediction'] = test_preds
submission.reset_index(drop=False, inplace=True)
submission.head()


# In[ ]:


submission.to_csv('submission.csv', index=False)


# <a id="ref"></a> <br> 
# # **8. References** 

# <h1 style="color:red">Please, votes up if you like this Kernel.</h1>
# I get lot of plots from https://www.kaggle.com/dimitreoliveira/toxicity-bias-extensive-eda-and-bi-lstm. Up Vote his awesome work.
