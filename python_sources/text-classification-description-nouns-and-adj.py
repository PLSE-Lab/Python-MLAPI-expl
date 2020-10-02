#!/usr/bin/env python
# coding: utf-8

# text classification try

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd

# data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import euclidean_distances
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


data_rent= pd.read_json("../input/train.json")
data_rent.head()
print(data_rent.groupby('interest_level').size())


# In[ ]:


data_rent['features']=data_rent['features'].apply(lambda x: ', '.join(x))
import nltk
from nltk.tag import pos_tag
from nltk import word_tokenize
from nltk.corpus import stopwords
stop = stopwords.words('english')
def cleaning_text(sentence):
   sentence=sentence.lower()
   sentence=re.sub('[^\w\s]',' ', sentence) #removes punctuations
   sentence=re.sub('\d+',' ', sentence) #removes digits
   cleaned=' '.join([w for w in sentence.split() if not w in stop]) # removes english stopwords
   cleaned=' '.join([w for w , pos in pos_tag(cleaned.split()) if (pos == 'NN' or pos=='JJ' or pos=='JJR' or pos=='JJS' )])
   #selecting only nouns and adjectives
   cleaned=' '.join([w for w in cleaned.split() if not len(w)<=2 ]) #removes single lettered words and digits
   cleaned=cleaned.strip()
   return cleaned
	  
data_rent['cleaned']= data_rent['description'].apply(lambda x: cleaning_text(x))
data_rent['feat_cleaned']= data_rent['features'].apply(lambda x: cleaning_text(x))
data_rent["final_feat"] = data_rent["cleaned"].map(str) +" "+data_rent["feat_cleaned"]
data_rent.head(2)


# In[ ]:


from sklearn.feature_extraction import DictVectorizer as DV
vectorizer = DV( sparse = False )
data_rent_1 = vectorizer.fit_transform(data_rent)


# In[ ]:


import numpy as np
import pandas as pd
from scipy import stats, integrate
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=True)


# In[ ]:


sns.distplot(data_rent['bedrooms'],kde=False)
sns.distplot(data_rent['bathrooms'],kde=False)


# In[ ]:


data_high=data_rent.loc[(data_rent['interest_level']=='high')]
data_medium=data_rent.loc[(data_rent['interest_level']=='medium')]
data_low=data_rent.loc[(data_rent['interest_level']=='low')]


# In[ ]:


data_high['cleaned'].apply(lambda x: pd.value_counts(x.split(" "))).sum(axis = 0)


# In[ ]:


from sklearn.model_selection import train_test_split
train, test = train_test_split(data_rent, test_size = 0.2)
print(len(train))
print(len(test))


# In[ ]:


binVectorizer = CountVectorizer(binary=True)
counts = binVectorizer.fit_transform(train['cleaned'])


# In[ ]:


from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
targets = train['interest_level'].values
classifier.fit(counts, targets)


# In[ ]:


examples = test['cleaned']
example_counts = binVectorizer.transform(examples)
predictions = classifier.predict(example_counts)
predictions_df=pd.DataFrame(predictions)
predictions_df.head(10)
actual=test['interest_level'].values
from sklearn.metrics import confusion_matrix
matrix=pd.DataFrame(confusion_matrix(actual, predictions,labels=["low", "medium", "high"]))
print(matrix)


# In[ ]:


pd.crosstab(test['interest_level'], predictions, rownames=['True'], colnames=['Predicted'], margins=True)


# In[ ]:


from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('vectorizer',  CountVectorizer()),
    ('classifier',  MultinomialNB()) ])

 


# In[ ]:


pipeline.fit(train['cleaned'].values, train['interest_level'].values)
predicts=pipeline.predict(test['cleaned'])


# In[ ]:


data_test= pd.read_json("../input/test.json")
data_test.head()

