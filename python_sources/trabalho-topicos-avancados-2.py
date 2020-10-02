#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
dataset = pd.read_csv("/kaggle/input/datasettransformado/dataset-transformado.csv", header=None)


# In[ ]:


dataset.head()


# In[ ]:


columnList = list(dataset.columns)

newColumnList = columnList[-1:]+columnList[:-1]

dfAdaboost = dataset[newColumnList]

#dfAdaboost[dfAdaboost.columns[-1]] 

dfAdaboost.columns = ['Review', 'SentimentOld']
dfAdaboost = dfAdaboost.query('SentimentOld != "Neutra"')

#dfAdaboost.rename(columns={1:'Reviews', 0:'SentimentOld'}, inplace=True)

#df.loc[df['column name'] condition, 'new column name'] = 'value if condition is met'

dfAdaboost.loc[dfAdaboost.SentimentOld == 'Positiva', 'Sentiment'] = 1
dfAdaboost.loc[dfAdaboost.SentimentOld == 'Negativa', 'Sentiment'] = -1
#dfAdaboost['Sentiment'] = (-1,1)[dfAdaboost[dfAdaboost.columns[-1]] == 'Positiva']# (1 if dfAdaboost[dfAdaboost.columns[-1]] == 'Positiva' else -1)

del dfAdaboost['SentimentOld']

dfAdaboost.head(1000)


# In[ ]:


# preparacao dos dados
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# tokenizando as sentencas
dataset[1] = [word_tokenize(word) for word in dataset[1]]
print(dataset)


# In[ ]:


# remocao de stopwords
dataset[1] = dataset[1].apply(lambda x: [item for item in x if item not in stopwords.words("english")])


# In[ ]:


print(dataset.head())


# In[ ]:


dataset[1] = [' '.join(word) for word in dataset[1]]


# In[ ]:


print(dataset.head())


# In[ ]:


from sklearn import model_selection

X_train, X_test, y_train, y_test = model_selection.train_test_split(dataset[1], 
                                                                    dataset[3], 
                                                                    test_size=0.3)


# In[ ]:




