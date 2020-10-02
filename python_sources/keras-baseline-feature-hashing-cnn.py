#!/usr/bin/env python
# coding: utf-8

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


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
resources = pd.read_csv("../input/resources.csv")
train = train.sort_values(by="project_submitted_datetime")


# In[3]:


train.columns.values
#print(test.head())
#print(resources.head())


# In[4]:


teachers_train = list(set(train.teacher_id.values))
teachers_test = list(set(test.teacher_id.values))
inter = set(teachers_train).intersection(teachers_test)


# In[5]:


print("Number teachers train : %s, Number teachers test : %s, Overlap : %s "%(len(teachers_train), len(teachers_test), len(inter)))


# In[6]:


char_cols = ['project_subject_categories', 'project_subject_subcategories',
       'project_title', 'project_essay_1', 'project_essay_2',
       'project_essay_3', 'project_essay_4', 'project_resource_summary']


# In[ ]:





# In[7]:


#https://www.kaggle.com/mmi333/beat-the-benchmark-with-one-feature
resources['total_price'] = resources.quantity * resources.price

mean_total_price = pd.DataFrame(resources.groupby('id').total_price.mean()) 
sum_total_price = pd.DataFrame(resources.groupby('id').total_price.sum()) 
count_total_price = pd.DataFrame(resources.groupby('id').total_price.count())
mean_total_price['id'] = mean_total_price.index
sum_total_price['id'] = mean_total_price.index
count_total_price['id'] = mean_total_price.index

def create_features(df):
    

    df = pd.merge(df, mean_total_price, on='id')
    df = pd.merge(df, sum_total_price, on='id')
    df = pd.merge(df, count_total_price, on='id')
    df['year'] = df.project_submitted_datetime.apply(lambda x: x.split("-")[0])
    df['month'] = df.project_submitted_datetime.apply(lambda x: x.split("-")[1])
    for col in char_cols:
        df[col] = df[col].fillna("NA")
    df['text'] = df.apply(lambda x: " ".join(x[col] for col in char_cols), axis=1)
    return df

train = create_features(train)
test = create_features(test)


# In[ ]:





# In[8]:


cat_features = ["teacher_prefix", "school_state", "year", "month", "project_grade_category", "project_subject_categories", "project_subject_subcategories"]
#"teacher_id", 
num_features = ["teacher_number_of_previously_posted_projects", "total_price_x", "total_price_y", "total_price"]
cat_features_hash = [col+"_hash" for col in cat_features]


# In[9]:


max_size=15000#0
def feature_hash(df, max_size=max_size):
    for col in cat_features:
        df[col+"_hash"] = df[col].apply(lambda x: hash(x)%max_size)
    return df


# In[10]:


train = feature_hash(train)
test = feature_hash(test)


# In[ ]:


#print(train['text'])


# In[11]:


from sklearn.preprocessing import StandardScaler
#from sklearn.feature_extraction.text import TfidfVectorizer
from keras.preprocessing import text, sequence

max_features = 50000
maxlen = 300
scaler = StandardScaler()
X_train_num = scaler.fit_transform(train[num_features])
X_test_num = scaler.transform(test[num_features])
X_train_cat = np.array(train[cat_features_hash], dtype=np.int)
X_test_cat = np.array(test[cat_features_hash], dtype=np.int)
tokenizer = text.Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(train["text"].tolist()+test["text"].tolist())
list_tokenized_train = tokenizer.texts_to_sequences(train["text"].tolist())
list_tokenized_test = tokenizer.texts_to_sequences(test["text"].tolist())
X_train_words = sequence.pad_sequences(list_tokenized_train, maxlen=maxlen)
X_test_words = sequence.pad_sequences(list_tokenized_test, maxlen=maxlen)


X_train_target = train.project_is_approved


# In[14]:


from keras.layers import Input, Dense, Embedding, Flatten, concatenate, Dropout, Convolution1D, GlobalMaxPool1D
from keras.models import Model
from keras import optimizers

def get_model():
    input_cat = Input((len(cat_features_hash), ))
    input_num = Input((len(num_features), ))
    input_words = Input((maxlen, ))
    
    x_cat = Embedding(max_size, 10)(input_cat)
    x_cat = Flatten()(x_cat)
    x_cat = Dropout(0.5)(x_cat)
    x_words = Embedding(max_features, 100)(input_words)
    x_words = Convolution1D(100, 3, activation="relu")(x_words)
    x_words = GlobalMaxPool1D()(x_words)
    x_words = Dropout(0.5)(x_words)
    
    x_cat = Dense(100, activation="relu")(x_cat)
    x_num = Dense(100, activation="relu")(input_num)
    x_num = Dropout(0.5)(x_num)
    x = concatenate([x_cat, x_num, x_words])
    x = Dense(50, activation="relu")(x)
    x = Dropout(0.5)(x)
    predictions = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=[input_cat, input_num, input_words], outputs=predictions)
    model.compile(optimizer=optimizers.Adam(0.001, decay=1e-6),
              loss='binary_crossentropy',
              metrics=['accuracy'])
    return model


# In[15]:


model = get_model()


# In[ ]:


model.fit([X_train_cat, X_train_num, X_train_words], X_train_target, validation_split=0.1,
          epochs=5, batch_size=1024)


# In[ ]:


pred_test = model.predict([X_test_cat, X_test_num, X_test_words])


# In[ ]:


test["project_is_approved"] = pred_test
test[['id', 'project_is_approved']].to_csv("baseline_submission.csv", index=False)


# In[ ]:




