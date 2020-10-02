#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve


# In[ ]:


data = pd.read_csv('../input/Combined_News_DJIA.csv')
data.head()


# The **Label** variable will be a **1** if the DJIA stayed the same or rose on that date or **0** if the DJIA fell on that date.
# 

# #### Train and Test Split

# Lets first join all the headlines for each row together.

# In[ ]:


combined=data.copy()
combined['Combined']=combined.iloc[:,2:27].apply(lambda row: ''.join(str(row.values)), axis=1)


# In[ ]:


train = data[data['Date'] < '2015-01-01']
test = data[data['Date'] > '2014-12-31']


# In[ ]:


print("Length of train is",len(train))
print("Length of test is", len(test))


# In[ ]:


trainheadlines = []
for row in range(0,len(train.index)):
    trainheadlines.append(' '.join(str(x) for x in train.iloc[row,2:27]))
testheadlines = []
for row in range(0,len(test.index)):
    testheadlines.append(' '.join(str(x) for x in test.iloc[row,2:27]))


# #### Simple EDA

# In[ ]:


train = combined[combined['Date'] < '2015-01-01']
test = combined[combined['Date'] > '2014-12-31']


# In[ ]:


non_decrease = train[train['Label']==1]
decrease = train[train['Label']==0]
print(len(non_decrease)/len(train))


# In[ ]:


non_decrease_test = test[test['Label']==1]
decrease_test = test[test['Label']==0]
print(len(non_decrease_test)/len(test))


# We can see that the occurrence of non-decrease situation is almost equal to that of a decrease market.

# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from wordcloud import WordCloud,STOPWORDS
import re
import nltk
from nltk.corpus import stopwords
def to_words(content): ### function to clean the words
    letters_only = re.sub("[^a-zA-Z]", " ", content) ### get only letters
    words = letters_only.lower().split()             ### lowercase       
    stops = set(stopwords.words("english"))         ### remove stopwords such as 'the', 'and' etc.         
    meaningful_words = [w for w in words if not w in stops] ### get meaningful words
    return( " ".join( meaningful_words )) 


# In[ ]:


non_decrease_word=[]
decrease_word=[]
for each in non_decrease['Combined']:
    non_decrease_word.append(to_words(each))

for each in decrease['Combined']:
    decrease_word.append(to_words(each))


# In[ ]:


wordcloud1 = WordCloud(background_color='black',
                      width=3000,
                      height=2500
                     ).generate(decrease_word[1])
plt.figure(1,figsize=(8,8))
plt.imshow(wordcloud1)
plt.axis('off')
plt.title("Words which indicate a fall in DJIA ")
plt.show()


# In[ ]:


wordcloud2 = WordCloud(background_color='green',
                      width=3000,
                      height=2500
                     ).generate(non_decrease_word[3])
plt.figure(1,figsize=(8,8))
plt.imshow(wordcloud2)
plt.axis('off')
plt.title("Words which indicate a rise/stable DJIA ")
plt.show()


# ## Text Preprocessing

# In[ ]:


example = train.iloc[3,3]
print(example)


# ##### Lower Case

# In[ ]:


example2 = example.lower()
print(example2)


# ##### Count Vectorizer

# In[ ]:


example3 = CountVectorizer().build_tokenizer()(example2)
print(example3)


# In[ ]:


pd.DataFrame([[x,example3.count(x)] for x in set(example3)], columns = ['Word', 'Count'])


# The process involved:
# 
# - Converting the headline to lowercase letters
# - Splitting the sentence into a list of words
# - Removing punctuation and meaningless words
# 

# ### Basic Model Training and Testing

# The tool we'll be using is CountVectorizer, which takes a single list of strings as input, and produces word counts for each one.

# In[ ]:


basicvectorizer = CountVectorizer()
basictrain = basicvectorizer.fit_transform(trainheadlines)
print(basictrain.shape)


# In[ ]:


testheadlines = []
for row in range(0,len(test.index)):
    testheadlines.append(' '.join(str(x) for x in test.iloc[row,2:27]))


# In[ ]:


basictest = basicvectorizer.transform(testheadlines)
print(basictest.shape)


# Our resulting table contains counts for 31,675 different words!

# **Model Fitting**

# In[ ]:


Classifiers = [
    LogisticRegression(C=0.1,solver='liblinear',max_iter=2000),
    KNeighborsClassifier(3),
    RandomForestClassifier(n_estimators=500,max_depth=9),
    ]


# In[ ]:


Accuracy=[]
Model=[]
for classifier in Classifiers:
    try:
        fit = classifier.fit(basictrain,train['Label'])
        pred = fit.predict(basictest)
        prob = fit.predict_proba(basictest)[:,1]
    except Exception:
        fit = classifier.fit(basictrain,train['Label'])
        pred = fit.predict(basictest)
        prob = fit.predict_proba(basictest)[:,1]
    accuracy = accuracy_score(pred,test['Label'])
    Accuracy.append(accuracy)
    Model.append(classifier.__class__.__name__)
    fpr, tpr, _ = roc_curve(test['Label'],prob)


# In[ ]:


df=pd.DataFrame(columns = ['Model', 'Accuracy'],index=np.arange(1, len(df)+1))
df.Model=Model
df.Accuracy=Accuracy
df


# ### Feature Extraction

# **TFIDF Model**
# 

# Lets try to improve the score with more models and feature Selection.

# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf=TfidfVectorizer()
train_text = []
test_text = []
for each in train['Combined']:
    train_text.append(to_words(each))

for each in test['Combined']:
    test_text.append(to_words(each))
train_features = tfidf.fit_transform(train_text)
test_features = tfidf.transform(test_text)


# ## Model fitting

# In[ ]:


Classifiers = [
    LogisticRegression(C=0.1,solver='liblinear',max_iter=2000),
    KNeighborsClassifier(3),
    SVC(kernel="rbf", C=0.25, probability=True),
    DecisionTreeClassifier(),
    RandomForestClassifier(n_estimators=500,max_depth=9),
    AdaBoostClassifier(),
    ]


# In[ ]:


dense_features=train_features.toarray()
dense_test= test_features.toarray()
Accuracy=[]
Model=[]
for classifier in Classifiers:
    try:
        fit = classifier.fit(train_features,train['Label'])
        pred = fit.predict(test_features)
        prob = fit.predict_proba(test_features)[:,1]
    except Exception:
        fit = classifier.fit(dense_features,train['Label'])
        pred = fit.predict(dense_test)
        prob = fit.predict_proba(dense_test)[:,1]
    accuracy = accuracy_score(pred,test['Label'])
    Accuracy.append(accuracy)
    Model.append(classifier.__class__.__name__)
    fpr, tpr, _ = roc_curve(test['Label'],prob)
    


# In[ ]:


df=pd.DataFrame(columns = ['Model', 'Accuracy'],index=np.arange(1, len(df)+1))
df.Model=Model
df.Accuracy=Accuracy
df


# As we can see, there has been a slight improvement from the previous scores.

# ## Advanced Modeling

# The technique we just used is known as a **bag-of-words** model. We essentially placed all of our headlines into a "bag" and counted the words as we pulled them out.
# 
# However,  a single word doesn't always have enough meaning by itself.
# 
# We need to consider the rest of the words in the sentence as well!

# ### N - gram model

# ##### n=2

# In[ ]:


advancedvectorizer = CountVectorizer(ngram_range=(2,2))
advancedtrain = advancedvectorizer.fit_transform(trainheadlines)
print(advancedtrain.shape)


# This time we have 366,721 unique variables representing two-word combinations!

# In[ ]:


advancedtest = advancedvectorizer.transform(testheadlines)


# In[ ]:


Accuracy=[]
Model=[]
for classifier in Classifiers:
    try:
        fit = classifier.fit(advancedtrain,train['Label'])
        pred = fit.predict(advancedtest)
        prob = fit.predict_proba(advancedtest)[:,1]
    except Exception:
        fit = classifier.fit(advancedtrain,train['Label'])
        pred = fit.predict(advancedtest)
        prob = fit.predict_proba(advancedtest)[:,1]
    accuracy = accuracy_score(pred,test['Label'])
    Accuracy.append(accuracy)
    Model.append(classifier.__class__.__name__)
    fpr, tpr, _ = roc_curve(test['Label'],prob)
    


# In[ ]:


df=pd.DataFrame(columns = ['Model', 'Accuracy'],index=np.arange(1, len(df)+1))
df.Model=Model
df.Accuracy=Accuracy
df


# We are getting much better results now and we are getting an accuracy of 56.08%.

# ##### n=3

# In[ ]:


advancedvectorizer = CountVectorizer(ngram_range=(3,3))
advancedtrain = advancedvectorizer.fit_transform(trainheadlines)
print(advancedtrain.shape)
advancedtest = advancedvectorizer.transform(testheadlines)


# This time we have 611,140 unique variables representing three-word combinations!

# In[ ]:


Accuracy=[]
Model=[]
for classifier in Classifiers:
    try:
        fit = classifier.fit(advancedtrain,train['Label'])
        pred = fit.predict(advancedtest)
        prob = fit.predict_proba(advancedtest)[:,1]
    except Exception:
        fit = classifier.fit(advancedtrain,train['Label'])
        pred = fit.predict(advancedtest)
        prob = fit.predict_proba(advancedtest)[:,1]
    accuracy = accuracy_score(pred,test['Label'])
    Accuracy.append(accuracy)
    Model.append(classifier.__class__.__name__)
    fpr, tpr, _ = roc_curve(test['Label'],prob)


# In[ ]:


df=pd.DataFrame(columns = ['Model', 'Accuracy'],index=np.arange(1, len(df)+1))
df.Model=Model
df.Accuracy=Accuracy
df


# The accuracy does not seem to increase and it looks like we have hit our maximum accuracy point at 56%.
