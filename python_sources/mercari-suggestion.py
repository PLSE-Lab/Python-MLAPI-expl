#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pa
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from scipy.sparse import hstack
from sklearn.linear_model import Ridge


# In[ ]:


from nltk.corpus import stopwords
stop_words = set (stopwords.words('english'))


# In[ ]:


train=pa.read_table("../input/train.tsv")


# In[ ]:


test=pa.read_table("../input/test.tsv")


# In[ ]:


train['item_description']=train['item_description'].str.lower()


# In[ ]:


test['item_description']=test['item_description'].str.lower()


# In[ ]:


train['item_description']=train['item_description'].replace('[^a-zA-Z]', ' ', regex = True)


# In[ ]:


test['item_description']=test['item_description'].replace('[^a-zA-Z]', ' ', regex = True)


# In[ ]:


train.isnull().sum()


# In[ ]:


train["category_name"].fillna(value='missing/missing/missing', inplace=True)
train["brand_name"].fillna(value="missing", inplace=True)
train["item_description"].fillna(value="No description yet", inplace =True)


# In[ ]:


test["category_name"].fillna(value='missing/missing/missing', inplace=True)
test["brand_name"].fillna(value="missing", inplace=True)
test["item_description"].fillna(value="No description yet", inplace =True)


# In[ ]:


train['category_main']=train.category_name.str.split("/").str.get(0)
train['category_sub1']=train.category_name.str.split("/").str.get(1)
train['category_sub2']=train.category_name.str.split("/").str.get(2)


# In[ ]:


test['category_main']=test.category_name.str.split("/").str.get(0)
test['category_sub1']=test.category_name.str.split("/").str.get(1)
test['category_sub2']=test.category_name.str.split("/").str.get(2)


# In[ ]:


train.isnull().sum()


# In[ ]:


test.isnull().sum()


# In[ ]:


def stop(txt):
    words = [w for w in txt.split(" ") if not w in stop_words and len(w)>2]
    return words


# In[ ]:


train['tokens']=train['item_description'].map(lambda x:stop(x))


# In[ ]:


test['tokens']=test['item_description'].map(lambda x:stop(x))


# In[ ]:


train['desc_len']=train['tokens'].map(lambda x: len(x))


# In[ ]:


test['desc_len']=test['tokens'].map(lambda x: len(x))


# In[ ]:


train['name_len']=train['name'].map(lambda x: len(x))


# In[ ]:


test['name_len']=test['name'].map(lambda x: len(x))


# In[ ]:


train.head()


# In[ ]:


from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()


# In[ ]:


def stemm(text):
    stemmed=[stemmer.stem(w) for w in text]
    return stemmed


# In[ ]:


train['stemmed']=train['tokens'].map(lambda x: stemm(x))


# In[ ]:


test['stemmed']=test['tokens'].map(lambda x: stemm(x))


# In[ ]:


train.head()


# In[ ]:


def join(txt):
    joinedtext=' '.join(word for word in txt)
    return joinedtext


# In[ ]:


train['final_desc']=train['stemmed'].map(lambda x: join(x))


# In[ ]:


test['final_desc']=test['stemmed'].map(lambda x: join(x))


# In[ ]:


train['final_desc'].head()


# In[ ]:


test['final_desc'].head()


# In[ ]:


vectorizer = TfidfVectorizer(min_df=10)
X_tfidf = vectorizer.fit_transform(train['final_desc']) 


# In[ ]:


X_tfidf.shape


# In[ ]:


train['name'].shape


# In[ ]:


#Avectorizer = TfidfVectorizer(min_df=10)
Y_tfidf = vectorizer.transform(test['final_desc']) 


# In[ ]:


test['name'].shape


# In[ ]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
categorical_cols=['name',"brand_name","category_main","category_sub1","category_sub2"]
for col in categorical_cols:
    # taking a column from dataframe, encoding it and replacing same column in the dataframe.
    train[col] = le.fit_transform(train[col])


# In[ ]:


test.head(2)


# In[ ]:


categorical_cols=['name',"brand_name","category_main","category_sub1","category_sub2"]
for col in categorical_cols:
    # taking a column from dataframe, encoding it and replacing same column in the dataframe.
    test[col] = le.fit_transform(test[col])


# In[ ]:


train.head()


# In[ ]:


y = train['price']


# In[ ]:


train.columns


# In[ ]:


test.head(2)


# In[ ]:


train1=train.drop(train.columns[[0,3,5,7,11,13,14,15]],axis=1)


# In[ ]:


train1.head(1)


# In[ ]:


test1=test.drop(test.columns[[0,3,6,8,10,13,14]],axis=1)


# In[ ]:


test1.head(2)


# In[ ]:


X = hstack([X_tfidf,train1])


# In[ ]:


Y = hstack([Y_tfidf,test1])


# In[ ]:


clf = Ridge(alpha=20.0)


# In[ ]:


import time
start=time.clock()
clf.fit(X, y)
print(time.clock()-start)


# In[ ]:


import time
start=time.clock()
rslt=clf.predict(Y)
print(time.clock()-start)


# In[ ]:


rslt.shape


# In[ ]:


test.shape


# In[ ]:


rslt1=pa.DataFrame(rslt)


# In[ ]:


rslt1.columns=["price"]


# In[ ]:


rslt1["test_id"]=rslt1.index


# In[ ]:


rslt1.to_csv("sample_submission.csv", encoding='utf-8', index=False)

