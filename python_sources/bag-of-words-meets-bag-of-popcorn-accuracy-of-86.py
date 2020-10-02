#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd


# In[ ]:


from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
tfidf = TfidfTransformer()
cv = CountVectorizer()


# In[ ]:


from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()


# In[ ]:


from sklearn.pipeline import Pipeline


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


pipe = Pipeline([('bow',CountVectorizer()),
                 ('tfidf',TfidfTransformer()),
                 ('model', MultinomialNB())])


# In[ ]:


df = pd.read_csv('../input/labeledTrainData.tsv',sep = '\t')


# In[ ]:


X = df['review']
y= df['sentiment']


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[ ]:


pipe.fit(X_train,y_train)


# In[ ]:


predictions = pipe.predict(X_test)


# In[ ]:


from sklearn.metrics import confusion_matrix,classification_report


# In[ ]:


print(classification_report(y_test,predictions))


# In[ ]:





# In[ ]:





# In[ ]:


test_data = pd.read_csv('../input/testData.tsv',sep = '\t')


# In[ ]:


test_data.shape


# In[ ]:


Xt  = test_data['review']


# In[ ]:


Xt.shape


# In[ ]:


predictions_new = pipe.predict(Xt)


# In[ ]:


predictions_new.shape


# In[ ]:


type(predictions_new)


# In[ ]:


test_data_id = test_data['id']


# In[ ]:


test_data_id.shape


# In[ ]:


type(test_data_id)


# In[ ]:


Label=[]
for num in predictions_new:
    Label.append(num)


# In[ ]:


type(Label)


# In[ ]:


len(Label)


# In[ ]:


sentiment=pd.DataFrame({'sentiment':Label})


# In[ ]:


sentiment.head()


# In[ ]:


idx=pd.DataFrame({'id':test_data_id})


# In[ ]:


idx.head()


# In[ ]:


OUTPUT_RESULT="submission_pipeline.csv"
submission=pd.concat([idx,sentiment],axis=1)
submission.to_csv(OUTPUT_RESULT,index=False)


# In[ ]:




