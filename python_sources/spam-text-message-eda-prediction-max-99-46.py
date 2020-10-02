#!/usr/bin/env python
# coding: utf-8

# # Introduction 
# We will apply machine learning to this dataset with a total of 5572 messages. We will guess that the message received as a result of this application came from a man or a woman.
# 
# 
# Content:
# 1. [Load and Check Data](#1)
# 1. [Veriable Description](#2)
# 1. [Categorical Variable](#3)
# 1. [Missing Value](#4)
# 1. [Cleaning](#5)
#     - [Regular Expression](#6)
#     - [Convert to lowercase](#7)
#     - [Split](#8)
#     - [Stopwords](#9)
#     - [Lemmatization](#10)
# 1. [Bag Of Words](#11)
# 1. [Modelling](#12)
#     - [Train Test Split](#13)
#     - [Naive Bayes](#14)
#     - [Simple Logistic Regression](#15)
#     - [KNN](#16)

# # Load and Check Data <a id="1"></a>

# In[ ]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("seaborn-whitegrid")
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv("/kaggle/input/spam-text-message-classification/SPAM text message 20170820 - Data.csv")
data.head(10)


# In[ ]:


data.info()


# In[ ]:


data.describe().T


# # Veriable Description <a id="2"></a>
# The data set consists of 2 columns.
# - Category: Indicates whether there is spam.
# - Message: Message post.

# # Categorical Variable <a id="3"></a>

# In[ ]:


def bar_plot(variable):
    var = data[variable]
    varValue = var.value_counts()
    
    # visualize
    plt.figure(figsize=(9,3))
    plt.bar(varValue.index, varValue)
    plt.xticks(varValue.index,varValue.index.values)
    plt.ylabel("Count")
    plt.title(variable)
    plt.show()
    print("{}: \n{}".format(variable,varValue))


# In[ ]:


bar_plot("Category")


# As seen above, there are 4825 normal messages and 747 spam messages.

# Let's look at the average length of normal and spam messages.

# In[ ]:


ham_message_length = []
spam_message_length = []
for i in data.values:
    if(i[0] == "ham"):
        ham_message_length.append(len(i[1]))
    else:
        spam_message_length.append(len(i[1]))
        
# average
ham_average = sum(ham_message_length)/len(ham_message_length)
spam_average = sum(spam_message_length)/len(spam_message_length)
print("ham_average: ", ham_average)
print("spam_average: ", spam_average)


# In[ ]:


plt.figure(figsize=(9,3))
plt.bar(["ham_average","spam_average"], [ham_average,spam_average])
plt.show()


# # Missing Value <a id="4"></a>
# First of all, let's check if there is a missing value.

# In[ ]:


data.isnull().sum()


# As can be seen from the above values, there is no missing data in this data set.

# ## Cleaning <a id="5"></a>
# First, to apply machine learning to the dataset, we equal normal messages to 1 and spam messages to 0.

# In[ ]:


from sklearn.model_selection import train_test_split
import re
from nltk.corpus import stopwords
import nltk as nlp
from sklearn.feature_extraction.text import CountVectorizer 


# In[ ]:


data.Category = [1 if each == "ham" else 0 for each in data.Category]
df = data
data


# ## Regular Expression <a id="6"></a>

# In[ ]:


# for example

example = data.Message[0]
print("before: ", example)

example = re.sub("[^a-zA-Z]"," ",example)
print("after:", example)


# Let's apply the above example to all data.

# In[ ]:


regularExpressionMessages = []
for message in data["Message"]:
    message = re.sub("[^a-zA-Z]"," ",message)
    regularExpressionMessages.append(message)

data["Message"] = regularExpressionMessages
data


# ## Convert to lowercase <a id="7"></a>

# In[ ]:


# for example

example = data.Message[0]
print("before: ", example)

example = example.lower()
print("after:", example)


# Let's apply the above example to all data.

# In[ ]:


lowercaseMessages = []
for message in data["Message"]:
    message = message.lower()
    lowercaseMessages.append(message)

data["Message"] = lowercaseMessages
data


# ## Split <a id="8"></a>

# In[ ]:


# for example

example = data.Message[0]
print("before: ", example)

example = nlp.word_tokenize(example)
print("after:", example)


# Let's apply the above example to all data.

# In[ ]:


splitMessages = []
for message in data["Message"]:
    message = nlp.word_tokenize(message)
    splitMessages.append(message)

data["Message"] = splitMessages
data


# ## Stopwords <a id="9"></a>
# Here it is:
# - It is to delete words that will not be useful for us while machine learning .
# - "the","in", "at" like words can be given as examples.

# In[ ]:


# for example

print("before: ", data["Message"][0] )

message1 = [message for message in data["Message"][0] if not message in set(stopwords.words("english"))]
print("after: ", message1)


# Let's apply the above example to all data.

# In[ ]:


stopwordsMessages = []
for i in data["Message"]:
    i = [message for message in i if not message in set(stopwords.words("english"))]
    stopwordsMessages.append(i)

data["Message"] = stopwordsMessages
data


# ## Lemmatization <a id="10"></a>
# Now we need to find the roots of the separated words.

# In[ ]:


# for example

example = data.Message[6]
print("before: ", example)

lemma = nlp.WordNetLemmatizer()
example = [ lemma.lemmatize(word) for word in example]
print("after:", example)


# In[ ]:


example = " ".join(example)
print("example: ", example)


# Like the example above, we can combine all the words and make them a sentence. In this way, we can now start machine learning.

# In[ ]:


joinMessages = []
for message in data["Message"]:
    message = " ".join(message)
    joinMessages.append(message)

data["Message"] = joinMessages
data


# # Bag Of Words <a id="11"></a>

# In[ ]:


# for example
sentence1 = "I am coming from school"
sentence2 = "I am coming from Istanbul today"

from sklearn.feature_extraction.text import CountVectorizer 
count_vectorizer = CountVectorizer(max_features=5, stop_words="english")
sparce_matrix = count_vectorizer.fit_transform([sentence1,sentence2]).toarray()
sparce_matrix


# Let's apply the above example to all data.

# In[ ]:


count_vectorizer = CountVectorizer(max_features=10000, stop_words="english")
sparce_matrix = count_vectorizer.fit_transform(np.array(data["Message"])).toarray()
sparce_matrix


# In[ ]:


sparce_matrix.shape


# In[ ]:


# text classification

y = data.iloc[:,0].values
x = sparce_matrix


# # Modelling <a id="12"></a>
# ## Train Test Split <a id="13"></a>

# In[ ]:


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)

print("x_train",x_train.shape)
print("x_test",x_test.shape)
print("y_train",y_train.shape)
print("y_test",y_test.shape)


# # Naive Bayes <a id="14"></a>

# In[ ]:


from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()
nb.fit(x_train,y_train)

print("Accuracy: ", nb.score(x_test,y_test)*100)


# In[ ]:


from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# # Simple Logistic Regression <a id="15"></a>

# In[ ]:


logreg = LogisticRegression()
logreg.fit(x_train,y_train)
acc_log_train = round(logreg.score(x_train,y_train)*100,2)
acc_log_test = round(logreg.score(x_test,y_test)*100,2)
print("Training Accuracy: %{}".format(acc_log_train))
print("Testing Accuracy: %{}".format(acc_log_test))


# # KNN <a id="16"></a>

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 3)


# In[ ]:


knn.fit(x_train,y_train)


# In[ ]:


print('With KNN (K=3) accuracy is: ',knn.score(x_test,y_test))


# # SVM <a id="17"></a>

# In[ ]:


from sklearn.svm import SVC

svm = SVC(random_state=1)
svm.fit(x_train,y_train)

print("accuracy of svm algo: ", svm.score(x_test,y_test)*100)


# # Decision Tree Regression <a id="18"></a>

# In[ ]:


from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(random_state=42)
dt.fit(x_train,y_train)

print("Decision Tree Score: ", dt.score(x_test,y_test)*100)


# In[ ]:




