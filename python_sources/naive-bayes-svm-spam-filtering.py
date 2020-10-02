#!/usr/bin/env python
# coding: utf-8

# #  Spam classification with Naive Bayes and Support Vector Machines.

# - Libraries
# - Exploring the Dataset
# - Distribution spam and non-spam plots
# - Text Analytics
# - Feature Engineering
# - Predictive analysis (**Multinomial Naive Bayes and Support Vector Machines**)
# - Conclusion
# 

# ## Libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn import feature_extraction, model_selection, naive_bayes, metrics, svm
from IPython.display import Image
import warnings
warnings.filterwarnings("ignore")
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Exploring the Dataset

# In[2]:


data = pd.read_csv('../input/spam.csv', encoding='latin-1')
data.head(n=10)


# ## Distribution spam/non-spam plots

# In[3]:


count_Class=pd.value_counts(data["v1"], sort= True)
count_Class.plot(kind= 'bar', color= ["blue", "orange"])
plt.title('Bar chart')
plt.show()


# In[4]:


count_Class.plot(kind = 'pie',  autopct='%1.0f%%')
plt.title('Pie chart')
plt.ylabel('')
plt.show()


# ## Text Analytics

# We want to find the frequencies of words in the spam and non-spam messages. The words of the messages will be model features.<p>
# We use the function Counter.

# In[5]:


count1 = Counter(" ".join(data[data['v1']=='ham']["v2"]).split()).most_common(20)
df1 = pd.DataFrame.from_dict(count1)
df1 = df1.rename(columns={0: "words in non-spam", 1 : "count"})
count2 = Counter(" ".join(data[data['v1']=='spam']["v2"]).split()).most_common(20)
df2 = pd.DataFrame.from_dict(count2)
df2 = df2.rename(columns={0: "words in spam", 1 : "count_"})


# In[6]:


df1.plot.bar(legend = False)
y_pos = np.arange(len(df1["words in non-spam"]))
plt.xticks(y_pos, df1["words in non-spam"])
plt.title('More frequent words in non-spam messages')
plt.xlabel('words')
plt.ylabel('number')
plt.show()


# In[7]:


df2.plot.bar(legend = False, color = 'orange')
y_pos = np.arange(len(df2["words in spam"]))
plt.xticks(y_pos, df2["words in spam"])
plt.title('More frequent words in spam messages')
plt.xlabel('words')
plt.ylabel('number')
plt.show()


# We can see that the majority of frequent words in both classes are stop words such as 'to', 'a', 'or' and so on. <p>
# With stop words we refer to the most common words in a lenguage, there is no simgle, universal list of stop words. <p>

# ## Feature engineering

# Text preprocessing, tokenizing and filtering of stopwords are included in a high level component that is able to build a dictionary of features and transform documents to feature vectors.<p>
# **We remove the stop words in order to improve the analytics**

# In[8]:


f = feature_extraction.text.CountVectorizer(stop_words = 'english')
X = f.fit_transform(data["v2"])
np.shape(X)


# We have created more than 8400 new features. The new feature $j$ in the row $i$ is equal to 1 if the word $w_{j}$ appears in the text example $i$. It is zero if not.

# ## Predictive Analysis

# **My goal is to predict if a new sms is spam or non-spam. I assume that is much worse misclassify non-spam than misclassify an spam. (I don't want to have false positives)**
# <p>
# The reason is because I normally don't check the spam messages.<p> The two possible situations are:<p>
# 1. New spam sms in my inbox. (False negative).<p>
# OUTCOME: I delete it.<p>
# 2. New non-spam sms in my spam folder (False positive).<p>  OUTCOME: I probably don't read it. <p>
# I prefer the first option!!!

# First we transform the variable spam/non-spam into binary variable, then we split our data set in training set and test set. 

# In[9]:


data["v1"]=data["v1"].map({'spam':1,'ham':0})
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, data['v1'], test_size=0.33, random_state=42)
print([np.shape(X_train), np.shape(X_test)])


# ### Multinomial naive bayes classifier

# We train different bayes models changing the regularization parameter $\alpha$. <p>
# We evaluate the accuracy, recall and precision of the model with the test set.

# In[10]:


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


# Let's see the first 10 learning models and their metrics!

# In[11]:


matrix = np.matrix(np.c_[list_alpha, score_train, score_test, recall_test, precision_test])
models = pd.DataFrame(data = matrix, columns = 
             ['alpha', 'Train Accuracy', 'Test Accuracy', 'Test Recall', 'Test Precision'])
models.head(n=10)


# I select the model with the most test precision

# In[12]:


best_index = models['Test Precision'].idxmax()
models.iloc[best_index, :]


# **My best model does not produce any false positive, which is our goal.** <p>
# Let's see if there is more than one model with 100% precision !

# In[13]:


models[models['Test Precision']==1].head(n=5)


# Between these models with the highest possible precision, we are going to select which has more test accuracy.

# In[14]:


best_index = models[models['Test Precision']==1]['Test Accuracy'].idxmax()
bayes = naive_bayes.MultinomialNB(alpha=list_alpha[best_index])
bayes.fit(X_train, y_train)
models.iloc[best_index, :]


# #### Confusion matrix with naive bayes classifier

# In[15]:


m_confusion_test = metrics.confusion_matrix(y_test, bayes.predict(X_test))
pd.DataFrame(data = m_confusion_test, columns = ['Predicted 0', 'Predicted 1'],
            index = ['Actual 0', 'Actual 1'])


# * #### We misclassify 56 spam messages as non-spam emails whereas we don't misclassify any non-spam message.

# ### Support Vector Machine

# We are going to apply the same reasoning applying the support vector machine model with the gaussian kernel.
# 
# We train different models changing the regularization parameter C. <p>
# We evaluate the accuracy, recall and precision of the model with the test set.

# In[18]:


list_C = np.arange(500, 2000, 100) #100000
score_train = np.zeros(len(list_C))
score_test = np.zeros(len(list_C))
recall_test = np.zeros(len(list_C))
precision_test= np.zeros(len(list_C))
count = 0
for C in list_C:
    svc = svm.SVC(C=C)
    svc.fit(X_train, y_train)
    score_train[count] = svc.score(X_train, y_train)
    score_test[count]= svc.score(X_test, y_test)
    recall_test[count] = metrics.recall_score(y_test, svc.predict(X_test))
    precision_test[count] = metrics.precision_score(y_test, svc.predict(X_test))
    count = count + 1 


# Let's see the first 10 learning models and their metrics!

# In[17]:


matrix = np.matrix(np.c_[list_C, score_train, score_test, recall_test, precision_test])
models = pd.DataFrame(data = matrix, columns = 
             ['C', 'Train Accuracy', 'Test Accuracy', 'Test Recall', 'Test Precision'])
models.head(n=10)


# I select the model with the most test precision

# In[19]:


best_index = models['Test Precision'].idxmax()
models.iloc[best_index, :]


# **My best model does not produce any false positive, which is our goal.** <p>
# Let's see if there is more than one model with 100% precision !

# In[20]:


models[models['Test Precision']==1].head(n=5)


# Between these models with the highest possible precision, we are going to selct which has more test accuracy.

# In[21]:


best_index = models[models['Test Precision']==1]['Test Accuracy'].idxmax()
svc = svm.SVC(C=list_C[best_index])
svc.fit(X_train, y_train)
models.iloc[best_index, :]


# #### Confusion matrix with support vector machine classifier.

# In[22]:


m_confusion_test = metrics.confusion_matrix(y_test, svc.predict(X_test))
pd.DataFrame(data = m_confusion_test, columns = ['Predicted 0', 'Predicted 1'],
            index = ['Actual 0', 'Actual 1'])


# #### We misclassify 31 spam as non-spam messages whereas we don't misclassify any non-spam message.

# ## Conclusion

# **The best model I have found is support vector machine with 98.3% accuracy.** <p>
# **It classifies every non-spam message correctly (Model precision)** <p> 
# **It classifies the 87.7% of spam messages correctly (Model recall)**<p>
