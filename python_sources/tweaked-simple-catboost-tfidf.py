#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import catboost as cb
import numpy as np

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


# In[5]:


MAX_TFIDF_FEATURES = 50
stop_words = stopwords.words('russian')


# In[6]:


train_data = pd.read_csv("../input/train.csv")
test_data = pd.read_csv("../input/test.csv")
y = train_data.deal_probability.copy()


# In[7]:


y.head()


# Data preprocessing

# In[4]:


selected_columns = ["item_id", "user_id", "region", "price", "item_seq_number", "user_type", "image_top_1", "category_name", "description", "title", "param_1"]
label_column = "deal_probability"

train_labels = train_data[label_column]
train_data = train_data[selected_columns]
test_data = test_data[selected_columns]


# In[5]:


def preprocess(df):
    df["price"].fillna(train_data["price"].mean(), inplace=True)
    df["image_top_1"].fillna(train_data["image_top_1"].mode()[0], inplace=True)
    df['description'].fillna(' ', inplace=True)
    df['param_1'].fillna(' ', inplace=True)
    df['title'].fillna(' ', inplace=True)
    return df


# In[7]:


train_data = preprocess(train_data)
test_data = preprocess(test_data)


# In[8]:


train_data.head()


# Feature engineering

# In[10]:


def tfidf_vectorize(series, max_features):
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words=stop_words)
    return np.array(vectorizer.fit_transform(series).todense(), dtype=np.float16)

def feature_engineering(df):
    description_vectors = tfidf_vectorize(df['description'], MAX_TFIDF_FEATURES)
    title_vectors = tfidf_vectorize(df['title'], MAX_TFIDF_FEATURES)
    param_1_vectors = tfidf_vectorize(df['param_1'], MAX_TFIDF_FEATURES)

    for i in range(MAX_TFIDF_FEATURES):
        df.loc[:, 'title_tfidf_' + str(i)] = title_vectors[:, i]
        df.loc[:, 'description_tfidf_' + str(i)] = description_vectors[:, i]
        df.loc[:, 'param_1_tfidf_' + str(i)] = param_1_vectors[:, i]
    
    df.drop(["description", "title", "param_1"], inplace=True, axis=1)
    return df


# In[11]:


train_data = feature_engineering(train_data)
X = train_data


# In[1]:


X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.1, random_state=23)


# Train model

# In[8]:


model = cb.CatBoostRegressor(iterations=20,
                             learning_rate=0.01,
                             depth=10,
                             loss_function='RMSE',
                             eval_metric='RMSE',
                             random_seed = 23, 
                             od_type='Iter',
                             metric_period = 50,
                             od_wait=20) #how to add RMSE as an eval metri
model.fit(X_train, y_train,
          eval_set=(X_valid,y_valid),
          use_best_model=True,
          cat_features=[0, 1, 2, 4, 5, 6, 7])


# In[14]:


test_data = feature_engineering(test_data)


# Make predictions

# In[15]:


preds = model.predict(test_data)


# Make submission

# In[16]:


submission = pd.DataFrame(columns=["item_id", "deal_probability"])
submission["item_id"] = test_data["item_id"]
submission["deal_probability"] = preds
submission["deal_probability"].clip(0.0, 1.0, inplace=True)
submission.to_csv("submission.csv", index=False)

