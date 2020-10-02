#!/usr/bin/env python
# coding: utf-8

#  Importing Libraries

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn import naive_bayes, feature_extraction, metrics, model_selection
import pandas as pd
from collections import Counter

get_ipython().run_line_magic('matplotlib', 'inline')


# Loading the dataset and observing its content

# In[ ]:


data = pd.read_csv('../input/spam.csv', encoding='latin-1')
data.head(n=10)


# Here the dataset is visualized using the plot function as a bar as well as a pie diagram

# In[ ]:


count_c = pd.value_counts(data["v1"], sort = True)
count_c.plot(kind = 'bar', color = ['blue', 'orange'])
plt.show()


# In[ ]:


count_c.plot(kind = 'pie', autopct = '%1.0f%%')
plt.show()


# The analysed dataset is then has to be further analysed for its content. The words that most commonly occur in non-spam messages and spam messages. These words are extracted and put into a Data Frame format for easy viewing. The most_common function is used to remove the stop words that what so evr has no significance in telling the class to which the message belong to.

# In[ ]:


count1 = Counter(" ".join(data[data['v1']=='ham']['v2']).split()).most_common(20)
df1 = pd.DataFrame.from_dict(count1)
df1 = df1.rename(columns={0:'words in non-spam', 1:'count'})
count2 = Counter(" ".join(data[data['v1']=='spam']['v2']).split()).most_common(20)
df2 = pd.DataFrame.from_dict(count2)
df2 = df2.rename(columns={0:'words in spam',1: 'count'})


# In[ ]:


df1.plot.bar(legend = False)
y_pos = np.arange(len(df1["words in non-spam"]))
plt.xticks(y_pos, df1["words in non-spam"])
plt.title('More frequent words in non-spam messages')
plt.xlabel('words')
plt.ylabel('number')
plt.show()


# In[ ]:


df2.plot.bar(legend = False, color= 'orange')
y_pos = np.arange(len(df2["words in spam"]))
plt.xticks(y_pos, df2["words in spam"])
plt.title('More frequent words in spam messages')
plt.xlabel('words')
plt.ylabel('number')
plt.show()


# After the words in each of the respective classes has been found, the words are to be loaded in as features for the model that we build.

# In[ ]:


f = feature_extraction.text.CountVectorizer(stop_words = 'english')
X = f.fit_transform(data["v2"])
np.shape(X)


# In[ ]:


data['v1'] = data['v1'].map({'ham':0,'spam':1})
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, data['v1'], test_size = 0.33, random_state = 42)
print([np.shape(X_train),np.shape(X_test)])


# In[ ]:


list_alpha = np.arange(1/100000, 20, 0.11)
score_train = np.zeros(len(list_alpha))
score_test = np.zeros(len(list_alpha))
recall_test = np.zeros(len(list_alpha))
precision_test= np.zeros(len(list_alpha))
count = 0
for alpha in list_alpha:
    bayes = naive_bayes.MultinomialNB(alpha=alpha)
    bayes.fit(X_train, y_train)
    score_train[count] = bayes.score(X_train, y_train)
    score_test[count]= bayes.score(X_test, y_test)
    recall_test[count] = metrics.recall_score(y_test, bayes.predict(X_test))
    precision_test[count] = metrics.precision_score(y_test, bayes.predict(X_test))
    count = count + 1 


# In[ ]:


mat = np.matrix(np.c_[list_alpha, score_train, score_test, recall_test, precision_test])
data1 = pd.DataFrame(data = mat, columns = ['alpha', 'Train Accuracy', 'Test Accuracy', 'Test Recall', 'Test Precision'])
data1.head(10)


# In[ ]:


best_index = data1['Test Precision'].idxmax()
data1.iloc[best_index, :]


# In[ ]:


data1[data1['Test Precision']==1].head(n=5)


# In[ ]:


best_index = data1[data1['Test Precision']==1]['Test Accuracy'].idxmax()
bayes = naive_bayes.MultinomialNB(alpha=list_alpha[best_index])
bayes.fit(X_train, y_train)
data1.iloc[best_index, :]


# In[ ]:


m_confusion_test = metrics.confusion_matrix(y_test, bayes.predict(X_test))
pd.DataFrame(data = m_confusion_test, columns = ['Predicted 0', 'Predicted 1'], index = ['Actual 0', 'Actual 1'])


# It has been observed that multinomial naive bayes seems to work more effectively than bernoulli naive bayes.

# In[ ]:




