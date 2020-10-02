#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt


# In[ ]:


print(os.listdir("../input"))


# In[ ]:


Data = pd.read_csv("../input/amazon_alexa.tsv",sep='\t')
Data.head()


# In[ ]:


Data.groupby('rating').describe()


# ## Lets try to classify and Analyze the bad reviews through out this dataset and lets see if we could soulve some problem....

# In[ ]:


Data = Data[Data.rating!=5]
Data = Data[Data.rating!=4]


# In[ ]:


Data.head()


# In[ ]:


Data.shape


# In[ ]:


Data["index"] = range(0,409)
Data = Data.set_index("index")
Data.head()


# In[ ]:


import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize


# ## What porterStemmer actually does...

# In[ ]:


# It is a process of normalization
text2 = "Kiss kissed kisses know knowing last lasting"
stemmer = PorterStemmer()
Norm_Word= stemmer.stem(text2)
Tokens = text2.split()
" ".join(stemmer.stem(token) for token in Tokens)


# In[ ]:


STOPWORDS = set(stopwords.words('english'))
corpus=[]
for i in range(0,409):
    review = re.sub('[^a-zA-Z]', ' ', Data['verified_reviews'][i])
    review = review.lower()
    review = review.split()
    stemmer = PorterStemmer()
    review = [stemmer.stem(token) for token in review if not token in STOPWORDS]
    #contain all words that are not in stopwords dictionary
    review=' '.join(review)
    corpus.append(review)
corpus


# 
# 
# ##  Lets find the most commonly used words

# In[ ]:


words = []
for i in range(0,len(corpus)):
    words = words + (re.findall(r'\w+', corpus[i]))# words cantain all the words in the dataset
words


# In[ ]:


from collections import Counter
words_counts = Counter(words)
print(words_counts)


# In[ ]:


most_common_words = sorted(words_counts.items(), key=lambda x: x[1], reverse=True)
most_common_words


# In[ ]:


most_commmom_wordList = []
most_commmom_CountList = []
for x, y in most_common_words:
    most_commmom_wordList.append(x)
    most_commmom_CountList.append(y)


# In[ ]:


import seaborn as sns
plt.figure(figsize=(20,18))
plot = sns.barplot(np.arange(20), most_commmom_CountList[0:20])
plt.ylabel('Word Count',fontsize=20)
plt.xticks(np.arange(20), most_commmom_wordList[0:20], fontsize=20, rotation=40)
plt.title('Most Common Word used in Bad Review.', fontsize=20)
plt.show()


# # It looks like there need to be a lot of improvement in the Audio system from both hardware and software perspective especially improvement in the audio Output system.

# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
texts = ["good movie",
         "not a good movie",
         "did not like",
         "I like it",
         "good one"]
tfidf = TfidfVectorizer(min_df=2, max_df=0.5, ngram_range=(1, 2))
features = tfidf.fit_transform(texts)
tfidf.get_feature_names()


# In[ ]:


Vectorize = TfidfVectorizer(analyzer='word',stop_words='english',ngram_range=(1, 2),min_df=2)
X = Vectorize.fit_transform(corpus).toarray()
y = Data['feedback']


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score,roc_curve,auc


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=42)


# In[ ]:


model1 = RandomForestClassifier(n_estimators=200, max_features="auto")
model1.fit(x_train,y_train)


# In[ ]:


y_pred1 = model1.predict(x_test)
accuracy1 = accuracy_score(y_test,y_pred1)
print("Accuracy for RandomForest:\t"+str(accuracy1))
print("Precision for RandomForest:\t"+str(precision_score(y_test,y_pred1)))
print("Recall for RandomForest:\t"+str(recall_score(y_test,y_pred1)))


# In[ ]:


model2 = GradientBoostingClassifier(learning_rate=1.5, verbose=1, max_features='auto')
model2.fit(x_train,y_train)


# In[ ]:


y_pred2 = model2.predict(x_test)
accuracy2 = accuracy_score(y_test,y_pred2)
print("Accuracy for GradientBoosting:\t"+str(accuracy2))
print("Precision for GradientBoosting:\t"+str(precision_score(y_test,y_pred2)))
print("Recall for GradientBoosting:\t"+str(recall_score(y_test,y_pred2)))


# In[ ]:


prob_1=model1.predict_proba(x_test)
prob_1 = prob_1[:,1]# Probalility prediction for Rangomforest classifier
prob_2=model2.predict_proba(x_test)
prob_2 = prob_2[:,1]# Probalility prediction for GradientBoosting classifier


# In[ ]:


fpr1, tpr1, _ = roc_curve(y_test, prob_1)
fpr2, tpr2, _ = roc_curve(y_test, prob_2)
plt.figure(figsize=(14,12))
plt.title('Receiver Operating Characteristic')
plt.plot(fpr1, tpr1, label = 'AUC(Randomforest Classifier) = %0.3f' % auc(fpr1, tpr1))
plt.plot(fpr2, tpr2, label = 'AUC(GradientBoosting Classifier) = %0.3f' % auc(fpr2, tpr2))
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

