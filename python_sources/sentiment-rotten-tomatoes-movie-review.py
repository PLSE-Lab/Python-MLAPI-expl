#!/usr/bin/env python
# coding: utf-8

# # Importing Required Libraries

# In[1]:


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


# In[2]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings("ignore")


# # Loading Data

# In[3]:


# Load Data
df = pd.read_csv('../input/train.tsv', delimiter='\t')
pd.set_option('display.max_colwidth', -1)
df.head()


# In[4]:


df.tail()


# In[5]:


df.info()


# In[6]:


df.shape


# In[7]:


df.describe()


# # Exploratory Data Analysis

# In[8]:


sns.countplot(df["Sentiment"]) #Examining_Sentiment


# In[9]:


df["text_length"] = df["Phrase"].apply(len)


# In[10]:


df[["Sentiment","text_length","Phrase"]].head()


# In[11]:


df["text_length"].describe()


# In[12]:


df["text_length"].hist(bins=50)


# In[13]:


g = sns.FacetGrid(df,col = "Sentiment")
g.map(plt.hist,"text_length")


# In[14]:


sns.boxenplot(x="Sentiment",y="text_length",data=df, palette="rainbow")


# In[15]:


sns.heatmap(df[["Sentiment","text_length"]].corr(),annot=True,cmap="cool",fmt = "g")


# In[16]:


#word cloud
from nltk.corpus import stopwords
from wordcloud import WordCloud


# In[17]:


text = df["Phrase"].to_string()
wordcloud = WordCloud(relative_scaling=0.5 , background_color='white',stopwords=set(stopwords.words('english'))).generate(text)
plt.figure(figsize=(12,12))
plt.imshow(wordcloud)
plt.axis("off")
plt.show()


# # Encode Categorical Variable

# In[18]:


from keras.utils import to_categorical
X = df["Phrase"]
y = to_categorical(df["Sentiment"])
num_classes = df["Sentiment"].nunique()
y


# In[19]:


# setting seed to have identical result in future run for comparisons
seed = 42
np.random.seed(seed)


# # Train Test Split
# * test_size is how much do you subset the training data into a validation set

# In[20]:


from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split(X , y , test_size = 0.3 , random_state = seed)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# # Tokenize text

# In[21]:


from keras.preprocessing.text import Tokenizer
max_features = 20000
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(X_train))
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)


# In[22]:


totalNumWords = [len(one_review) for one_review in X_train]
plt.hist(totalNumWords,bins=50)
plt.show()


# In[23]:


X_train[6]


# In[24]:


from keras.preprocessing import sequence
max_words = max(totalNumWords)
X_train = sequence.pad_sequences(X_train , maxlen = max_words)
X_test = sequence.pad_sequences(X_test , maxlen = max_words)
print(X_train.shape,X_test.shape)


# In[25]:


X_train[6]


#  # LSTM

# **Importing Libraries**

# In[26]:


import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense,Embedding,LSTM,Conv1D,MaxPooling1D
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report


# In[27]:


batch_size = 128
epochs = 2


# # LSTM MODEL

# In[28]:


def get_model(max_features , embed_dim):
    np.random.seed(seed)
    K.clear_session()
    model = Sequential()
    model.add(Embedding(max_features , embed_dim , input_length=X_train.shape[1]))
    model.add(LSTM(100 , dropout=0.2 , recurrent_dropout=0.2))
    model.add(Dense(num_classes , activation='softmax'))
    model.compile(loss = 'categorical_crossentropy' , optimizer = 'adam' , metrics = ['accuracy'])
    print(model.summary())
    return model
    


# # CNN - LSTM MODEL

# In[29]:


def get_cnn_lstm_model(max_features, embed_dim):
    np.random.seed(seed)
    K.clear_session()
    model = Sequential()
    model.add(Embedding(max_features, embed_dim, input_length=X_train.shape[1]))
    model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))    
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model


# In[30]:


def model_train(model):
    #training the model
    model_history = model.fit(X_train , y_train , validation_data = (X_test , y_test), 
                              epochs = epochs ,batch_size= batch_size,verbose = 2)
    #plotting train history
    plot_model_history(model_history)


# In[31]:


def plot_model_history(model_history):
    fig , axs = plt.subplots( 1 , 2 , figsize=(15,5))
    
    #Summarize history for accuracy
    
    axs[0].plot(range(1,len(model_history.history['acc'])+1),model_history.history['acc'])
    axs[0].plot(range(1,len(model_history.history['val_acc'])+1),model_history.history['val_acc'])
    
    axs[0].set_title("Model Accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].set_xlabel("Epoch")
    
    axs[0].set_xticks(np.arange(1,len(model_history.history['acc'])+1),len(model_history.history['acc'])/10)
    
    axs[0].legend(['train', 'val'], loc='best')
    
    #Summarize history for loss
    
    axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])
    axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])
    
    axs[1].set_title("Model Loss")
    axs[1].set_ylabel("Loss")
    axs[1].set_xlabel("Epoch")
    
    axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)
    
    axs[1].legend(['train', 'val'], loc='best')
    
    plt.show()
    
    


# In[32]:


def model_evaluate():
    #predict classes with test set
    y_pred_test = model.predict_classes(X_test , batch_size = batch_size, verbose =0)
    print("Predicted ", y_pred_test[:50])
    print("True " , np.argmax(y_test[:50],axis = 1))
    print('Accuracy:\t{:0.1f}%'.format(accuracy_score(np.argmax(y_test,axis = 1),y_pred_test)*100))
    
    #Classification Report
    print("\n")
    print(classification_report(np.argmax(y_test, axis =1),y_pred_test))
    
    #Confusion Matrix
    confmat = confusion_matrix(np.argmax(y_test , axis = 1), y_pred_test)
    fig , ax = plt.subplots(figsize=(4,4))
    ax.matshow(confmat , cmap =plt.cm.Blues , alpha = 0.3)
    
    for i in range(confmat.shape[0]):
        for j in range(confmat.shape[1]):
            ax.text( x = j , y = i , s =confmat[i,j] , va = 'center' , ha = 'center')
    
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()


# # Train The Model

# In[33]:


max_features = 20000
embed_dim =100
model = get_cnn_lstm_model(max_features,embed_dim)
model_train(model)


# # Evaluate Model With Test Set

# In[34]:


model_evaluate()


# In[ ]:





# In[ ]:




