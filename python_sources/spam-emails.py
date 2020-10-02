#!/usr/bin/env python
# coding: utf-8

# <h1 align="center">Assignment</h1>
# <h2 align="center">Faisal Akhtar</h2>
# <h2 align="center">Roll No.: 17/1409</h2>
# <p>Machine Learning - B.Sc. Hons Computer Science - Vth Semester</p>
# <p>Implement a classification/ logistic regression problem for checking whether an email is spam or not.</p>

# <h3>Libraries imported</h3>

# In[ ]:


import pandas as pd

import string
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


# <h3>Reading data from CSV</h3>
# Removing unnecessary columns<br>
# Renaming column-names to something meaningful

# In[ ]:


data = pd.read_csv("../input/sms-spam-collection-dataset/spam.csv",encoding = "latin")
data = data.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis = 1)
data = data.rename(columns = {'v1':'Spam/Ham','v2':'message'})
data.head()


# <h3>Preprocessing</h3>
# 
# **Removing punctuation and stop words**

# In[ ]:


def preprocessing_func(var):
    text = var.translate(str.maketrans('', '', string.punctuation))
    text = [word for word in text.split() if word.lower() not in stopwords.words('english')]
    return " ".join(text)

data_2 = data['message'].copy()
data_2 = data_2.apply(preprocessing_func)


# **Vectorizing the data**<br>
# Collecting each word and its frequency in each email.<br>
# The vectorization will produce a matrix.

# In[ ]:


vectorizer = TfidfVectorizer("english")
data_matrix = vectorizer.fit_transform(data_2)


# <h3>Test/Train Split</h3>
# 
# Dividing data into test-train sets, 30% and 70%

# In[ ]:


X_train, X_test, Y_train, Y_test = train_test_split(data_matrix, data['Spam/Ham'], test_size=0.3)


# <h3>Logistic Regression</h3>
# 
# **Fit the model according to "data" variable obtained from CSV.**

# In[ ]:


logistic = LogisticRegression()
logistic.fit(X_train, Y_train)


# **Logistic Regression Model metrics**

# In[ ]:


predictions = logistic.predict(X_test)

# Accuracy score metrics
acc = accuracy_score(Y_test, predictions)
print("Accuracy score : ",acc,"\nAccuracy %ge = ",acc*100,"%")

# Scatter Plot
plt.scatter(Y_test, predictions)
plt.xlabel("True Values",color='red')
plt.ylabel("Predictions",color='blue')
plt.title("Predicted vs Actual value")
plt.grid(True)
plt.show()

