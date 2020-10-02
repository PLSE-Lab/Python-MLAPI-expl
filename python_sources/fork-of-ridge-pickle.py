#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from scipy.sparse import hstack
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import train_test_split
import gc
from sklearn.linear_model import Ridge
import pickle


# In[ ]:


class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']


# In[ ]:


train = pd.read_csv('../input/train.csv').fillna(' ')
test = pd.read_csv('../input/test.csv').fillna(' ')
subm = pd.read_csv('../input/sample_submission.csv')


# In[ ]:


train_text = train['comment_text']
test_text = test['comment_text']
all_text = pd.concat([train_text, test_text])


# In[ ]:


word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}',
    stop_words='english',
    ngram_range=(1, 2),min_df=2,max_df=0.5,
    max_features=60000
    )
word_vectorizer.fit(all_text)
train_word_features = word_vectorizer.transform(train_text)
test_word_features = word_vectorizer.transform(test_text)


# In[ ]:


char_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='char',
    stop_words='english',
    ngram_range=(2, 6),min_df=2,max_df=0.5,
    max_features=60000
    )
char_vectorizer.fit(all_text)
train_char_features = char_vectorizer.transform(train_text)
test_char_features = char_vectorizer.transform(test_text)


# In[ ]:



train_features = hstack([train_char_features,train_word_features])
test_features = hstack([test_char_features,test_word_features])


# In[ ]:


scores = []
#submission = pd.DataFrame.from_dict({'id': test['Id']})
result = pd.DataFrame.from_dict({'id': test['Id']})
ridge_file='ridge.pckl'
ridge_model_pkl = open(ridge_file, 'wb')
for class_name in class_names:
    train_target = train[class_name]
    classifier = Ridge(alpha=29, copy_X=True, fit_intercept=True, solver='sag',
                     max_iter=150,   normalize=False, random_state=0,  tol=0.0025)
   # classifier.fit(train_features, train_target)
   # submission[class_name] = classifier.predict(test_features)
    cv_score = np.mean(cross_val_score(classifier, train_features, train_target, cv=3,
                                       scoring='roc_auc'))
    scores.append(cv_score)
    print('CV score for class {} is {}'.format(class_name, cv_score))

    classifier.fit(train_features, train_target)
    
    pickle.dump(classifier, ridge_model_pkl)
    

    #submission[class_name] = classifier.predict(test_features)
ridge_model_pkl.close()
print('Total CV score is {}'.format(np.mean(scores)))
#submission.to_csv('submission.csv', index=False)



    


# In[ ]:


ridge_model_pkl = open(ridge_file, 'rb')
ridge_model = pickle.load(ridge_model_pkl)
print ("Loaded Decision tree model :: ", ridge_model)


# In[ ]:


result = pd.DataFrame.from_dict({'id': test['Id']})
result


# In[ ]:



result = pd.DataFrame.from_dict({'id': test['Id']})
classifier=None
models = []
with open("ridge.pckl", "rb") as f:
    while True:
        try:
            models.append(pickle.load(f))
        except EOFError:
            break
i=0            
for class_name in class_names:

    train_target = train[class_name]
    
    
    result[class_name] = models[i].predict(test_features)
    i=i+1
result.to_csv('submission.csv', index=False)



    


# In[ ]:


result


# In[ ]:




