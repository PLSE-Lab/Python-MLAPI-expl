#!/usr/bin/env python
# coding: utf-8

# # My Solution 

# In[ ]:


# task link: https://www.kaggle.com/c/nlp-getting-started


# In[ ]:


import pandas as pd

train = pd.read_csv('../input/nlp-getting-started/train.csv', index_col='id')
train.head()
len(train)


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train['text'], train['target'], random_state=2)


# In[ ]:


len(X_train)


# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(strip_accents='ascii', token_pattern=u'(?ui)\\b\\w*[a-z]+\\w*\\b', lowercase=True, stop_words='english')
X_train_cv = cv.fit_transform(X_train)
X_test_cv = cv.transform(X_test)


# In[ ]:


y_test.head()


# In[ ]:


print(len(y_test))
print(len(y_test.loc[y_test==1]),len(y_test.loc[y_test==1])/len(y_test))
print(len(y_test.loc[y_test==0]),len(y_test.loc[y_test==0])/len(y_test))


# ## baseline: naive base

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(strip_accents='ascii', token_pattern=u'(?ui)\\b\\w*[a-z]+\\w*\\b', lowercase=True, stop_words='english')
X_train_cv = cv.fit_transform(X_train)
X_test_cv = cv.transform(X_test)


# In[ ]:


print(len(X_test_cv.toarray()[0]))


# In[ ]:


from sklearn.naive_bayes import MultinomialNB
naive_bayes = MultinomialNB()
naive_bayes.fit(X_train_cv, y_train)
predictions = naive_bayes.predict(X_test_cv)


# In[ ]:


from sklearn.metrics import accuracy_score, precision_score, recall_score
print('Accuracy score: ', accuracy_score(y_test, predictions))
print('Precision score: ', precision_score(y_test, predictions))
print('Recall score: ', recall_score(y_test, predictions))


# ## improvement: vocab

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import train_test_split
import re


X_train, X_test, y_train, y_test = train_test_split(train['text'], train['target'], random_state=1)
def clean_text(text):
    text = re.sub(r'https?://\S+', '', text) # Remove link
    text = re.sub(r'\n',' ', text) # Remove line breaks
    text = re.sub('\s+', ' ', text).strip() # Remove leading, trailing, and extra spaces
    return text
X_train = [clean_text(x) for x in X_train]
X_test = [clean_text(x) for x in X_test]
# cv = CountVectorizer(strip_accents='ascii', token_pattern=r'#?[a-z]+\b', lowercase=True, stop_words='english')
# cv = CountVectorizer(strip_accents='ascii', token_pattern=r'(?:\b[a-z]+\b)|(?:[@])|(?:#\w+)', lowercase=True, stop_words='english')
# cv = CountVectorizer(strip_accents='ascii', token_pattern=r'(?:\b[a-z]+\b)|(?:#\w+)', lowercase=True)
# cv = CountVectorizer(strip_accents='ascii', token_pattern=r'(?:\b[a-z]+\b)', lowercase=True)
# cv = CountVectorizer(strip_accents='ascii', token_pattern=r'(?:\b[\w#@]+\b)', lowercase=True)
# cv = CountVectorizer(strip_accents='ascii', token_pattern=r'(?:\b[\w#!@]+\b)',analyzer='word',max_df=0.1, lowercase=True)
# cv = CountVectorizer(strip_accents='ascii', token_pattern=r'(?:\b[a-z#!@]+\b)',analyzer='word',max_df=0.1,max_features=3500, lowercase=True)
cv = CountVectorizer(strip_accents='ascii', token_pattern=r'(?:\b[\w#!@]+\b)',analyzer='word',max_df=0.1,max_features=3500, lowercase=True)
# cv = CountVectorizer(strip_accents='ascii', token_pattern=r'(?:\b[\w#!@]+\b)',analyzer='word', lowercase=True)
# cv = CountVectorizer(strip_accents='ascii', token_pattern=r'(?:\b[\w#]+\b)',analyzer='word',max_df=0.1, lowercase=True)

# cv = CountVectorizer(strip_accents='ascii', token_pattern=u'(?ui)\\b\\w*[a-z]+\\w*\\b', lowercase=True)

X_train_cv = cv.fit_transform(X_train)
print(X_train_cv.shape)
X_test_cv = cv.transform(X_test)
# print(len(cv.get_feature_names()))
# print(cv.get_feature_names())


# In[ ]:



from sklearn.naive_bayes import MultinomialNB
naive_bayes = MultinomialNB()
naive_bayes.fit(X_train_cv, y_train)
predictions = naive_bayes.predict(X_test_cv)
predictions_train = naive_bayes.predict(X_train_cv)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
print('Accuracy score: ', accuracy_score(y_test, predictions))
print('Accuracy score(train): ', accuracy_score(y_train, predictions_train))


# In[ ]:


# show some data samples
count = 0
for x,pred,y in zip(X_test,predictions,y_test):
    count+=1
    if pred!=y:
        print(count,x,pred,y)
        print('===')


# In[ ]:


# for submission

test = pd.read_csv('../input/nlp-getting-started/test.csv', index_col='id')
train = pd.read_csv('../input/nlp-getting-started/train.csv', index_col='id')




from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(strip_accents='ascii', token_pattern=r'(?:\b[\w#!@]+\b)',analyzer='word',max_df=0.1,max_features=3500, lowercase=True)

def clean_text(text):
    text = re.sub(r'https?://\S+', '', text) # Remove link
    text = re.sub(r'\n',' ', text) # Remove line breaks
    text = re.sub('\s+', ' ', text).strip() # Remove leading, trailing, and extra spaces
    return text
X_train = [clean_text(x) for x in train['text']]
X_test = [clean_text(x) for x in test['text']]

X_train_cv = cv.fit_transform(X_train)
X_test_cv = cv.transform(X_test)

from sklearn.naive_bayes import MultinomialNB
naive_bayes = MultinomialNB()
naive_bayes.fit(X_train_cv, train['target'])
predictions = naive_bayes.predict(X_test_cv)


# In[ ]:


predictions[:10]
test['target'] = predictions

submission_df = test[['target']]
# submission_df.head()
submission_df.to_csv('result_7_b.csv')


# ## xgboost
# 

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(train['text'], train['target'], random_state=1)
def clean_text(text):
    text = re.sub(r'https?://\S+', '', text) # Remove link
    text = re.sub(r'\n',' ', text) # Remove line breaks
    text = re.sub('\s+', ' ', text).strip() # Remove leading, trailing, and extra spaces
    return text
X_train = [clean_text(x) for x in X_train]
X_test = [clean_text(x) for x in X_test]
cv = CountVectorizer(strip_accents='ascii', token_pattern=r'(?:\b[\w#!@]+\b)',analyzer='word',max_df=0.1,max_features=3500, lowercase=True)

X_train_cv = cv.fit_transform(X_train)
print(X_train_cv.shape)
X_test_cv = cv.transform(X_test)

from xgboost import XGBClassifier

xgcls = XGBClassifier()

xgcls.fit(X_train_cv, y_train)
predictions = xgcls.predict(X_test_cv)

from sklearn.metrics import accuracy_score, precision_score, recall_score
print('Accuracy score: ', accuracy_score(y_test, predictions))


# ## textcnn/bilstm
# tried offline, works worse than naive bayes baseline.

# > ## bert

# In[ ]:


"""
a simple bert based on huggingface and implemented on my own. 
this classifier has been modified in a rush from my NER extractor, sorry if some code looks bad :(
"""
import transformers
import torch
from transformers import BertForSequenceClassification, AdamW, BertTokenizer, get_linear_schedule_with_warmup
from tqdm.notebook import trange as tnrange

class BertClassifier:
    def __init__(self,args):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased',num_labels=len(args.tags))
        self.args = args

        if self.args.device == 'gpu':
            self.model.to('cuda')
    

    def preprocess_texts(self, texts):
        ids_list = []
        for text in texts:
            ids_list.append(self.preprocess_single_text( text, return_tensor=False))
        return ids_list

    def preprocess_single_text(self, text, return_tensor=True): 
        ids = self.tokenizer.encode(text,add_special_tokens=True)[:self.args.max_seq]

        # padding
        ids = self.padding(ids, self.args.max_seq)

        if return_tensor:
            return torch.tensor([ids])
        else:
            return ids

    def padding(self, l, max_len, padding_id=0):
        l = l[:max_len]+[0]*max([max_len-len(l),0])
        return l


    def preprocess_training_data(self,texts,labels):
        if len(texts) != len(labels):
            raise Exception('training data size not agree.')
        
        res_texts = []
        res_labels = []

        for i,_ in enumerate(texts):
            test, label =  self.preprocess_single_training_data(texts[i],labels[i])
            res_texts.append(test)
            res_labels.append(label)

        return torch.tensor(res_texts), torch.tensor(res_labels)





    def preprocess_single_training_data(self,text,label):
        text = self.tokenizer.encode(text,add_special_tokens=True)
        # text = self.tokenizer.convert_tokens_to_ids(text)
        return self.padding(text,self.args.max_seq), label
    


    def train(self, data, lables):

        optimizer = AdamW(self.model.parameters(), correct_bias=False,lr=1e-5)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=self.args.num_training_steps)

        self.model.zero_grad()
        
        epochs = tnrange(self.args.epoch)
        for current_epoch in epochs:
            iterations =  tnrange(len(lables)//self.args.batch_size)
            batch = self.make_bach(data, lables, self.args.batch_size)
            for _ in iterations:
                batch_data, batch_lables = next(batch)
                self.model.train()

                batch_data, batch_lables = self.preprocess_training_data(batch_data,batch_lables)

                
                if self.args.device == 'gpu':
                    batch_data = batch_data.to('cuda')
                    batch_lables = batch_lables.to('cuda')
                loss, res = self.model(batch_data, labels=batch_lables)[:2]
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)

                optimizer.step()
                scheduler.step()
                self.model.zero_grad()

    def make_bach(self, data, lables, batch_size):
        return (
            [
                data[i*batch_size:(i+1)*batch_size], lables[i*batch_size:(i+1)*batch_size]
                ] 
        for i in range(len(data)//batch_size)
        )


    def evaluate(self,test_tensor,labels):
        predictions = self.predict(test_tensor)
        return self.evaluate_with_metrics(predictions,labels)

    
    def predict(self,test):
        test_ori = self.preprocess_texts(test)
        test_tensor = torch.tensor(test_ori)

        if self.args.device == 'gpu':
            test_tensor = test_tensor.to('cuda')

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(test_tensor)
            predictions = outputs[0]
        prediction = torch.argmax(predictions,1)
            
        return prediction


    def evaluate_with_metrics(self,predictions,labels):
        return None
        


class TestArgs:
    def __init__(self):
        self.tags = {0,1}

        self.epoch = 4
        self.batch_size = 16
        self.max_seq = 256

        # warm_up
        # 7613 5709
        self.warmup_steps = 5709*4//16//10
        self.num_training_steps = 5709*4//16

        # gradient clip
        self.max_grad_norm = 1

        self.device = 'gpu'
    
    




# In[ ]:


len(train)


# In[ ]:



# extractor = BertClassifier(TestArgs())
# print(extractor.predict(['This is a pen for you! I am becoming a god.','I am a sample.']))
# extractor.train(
#     ('This is a pen.','I love playing games.','this is good')*100,
#     ([1],[0],[1])*100,
#     )
# print(extractor.predict(('This is a pen.','I love playing games.','this is good')*10))

import re
from sklearn.model_selection import train_test_split
def clean_text(text):
    text = re.sub(r'https?://\S+', '', text) # Remove link
    text = re.sub(r'\n',' ', text) # Remove line breaks
    text = re.sub('\s+', ' ', text).strip() # Remove leading, trailing, and extra spaces
    return text


extractor = BertClassifier(TestArgs())
X_train, X_test, y_train, y_test = train_test_split(train['text'], train['target'], random_state=1)
# X_train, y_train = train['text'], train['target']

X_train = [clean_text(x) for x in X_train]
X_test = [clean_text(x) for x in X_test]


y_train = [[x] for x in y_train]
print(tuple(y_train)[:10])

extractor = BertClassifier(TestArgs())
extractor.train(tuple(X_train),tuple(y_train))


# In[ ]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
def split(x, batch):
    for i in range(0, len(x), batch):
        yield(x[i:i+batch])
batched_test = split(X_test,16)
predictions = []
for i,test_batch in enumerate(batched_test):
    predictions += extractor.predict(tuple(test_batch)).tolist()
print('Accuracy score: ', accuracy_score(y_test, predictions))
predictions_train = []
batched_train = split(X_train,16)
for i,batch in enumerate(batched_train):
    predictions_train += extractor.predict(tuple(batch)).tolist()
print('Accuracy score(train): ', accuracy_score(y_train, predictions_train))


# In[ ]:


predictions


# In[ ]:


# for submission
# accuracy varies from 81% to 82% looks OK
test = pd.read_csv('../input/nlp-getting-started/test.csv', index_col='id')
train = pd.read_csv('../input/nlp-getting-started/train.csv', index_col='id')

batched_test = split(test['text'],16)
predictions = []
for i,test_batch in enumerate(batched_test):
    predictions += extractor.predict(tuple(test_batch)).tolist()

predictions[:10]
test['target'] = predictions

submission_df = test[['target']]
# submission_df.head()
submission_df.to_csv('result_8.csv')


# The final submission is the result of BERT, which works better than any other results.

# In[ ]:




