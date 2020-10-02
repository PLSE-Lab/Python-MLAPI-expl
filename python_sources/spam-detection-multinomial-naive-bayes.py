#!/usr/bin/env python
# coding: utf-8

# <p style="text-align: center;"><span style="font-size:18px"><span style="color:#2f4f4f"><strong><span style="font-family:Lucida Sans Unicode,Lucida Grande,sans-serif">Spam detection - Multinomial Naive Bayes</span></strong></span></span></p>

# <p><u><span style="color:#a52a2a"><span style="font-family:Lucida Sans Unicode,Lucida Grande,sans-serif">Below are the steps that we are going to do:</span></span></u></p>
# 
# <p style="margin-left: 40px;"><span style="color:#000080"><span style="font-family:Lucida Sans Unicode,Lucida Grande,sans-serif">1. Read data from the dataset</span></span></p>
# 
# <p style="margin-left: 40px;"><span style="color:#000080"><span style="font-family:Lucida Sans Unicode,Lucida Grande,sans-serif">2. Remove the empty columns v3,v4,v5&nbsp;</span></span></p>
# 
# <p style="margin-left: 40px;"><span style="color:#000080"><span style="font-family:Lucida Sans Unicode,Lucida Grande,sans-serif">3. Remove the first row which contains v1, v2 which is not releveant to our dataset</span></span></p>
# 

# In[ ]:


import pandas as pd
spam = pd.read_csv("../input/sms-spam-collection-dataset/spam.csv", encoding = "ISO-8859-1",names=["label","message","v3","v4","v5"])
del spam["v3"]
del spam["v4"]
del spam["v5"]
spam = spam.drop(spam.index[0])


# <p><span style="color:#000080"><span style="font-family:Lucida Sans Unicode,Lucida Grande,sans-serif">Just to confirm if there are any invalid data, analyze the unique target variable( in our case the target column is the &quot;label&quot;)&nbsp;, and the result is the expected &quot;ham&quot; and &quot;spam&quot;; &nbsp;so we can proceed further</span></span></p>
# 

# In[ ]:


spam.label.unique()


# In[ ]:


import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
 


# <p><u><span style="color:#a52a2a"><span style="font-family:Lucida Sans Unicode,Lucida Grande,sans-serif">Iterate through all the messages and do the following:</span></span></u></p>
# 
# <p><span style="color:#000080"><span style="font-family:Lucida Sans Unicode,Lucida Grande,sans-serif">a) Remove all the characters except a to z and A to Z</span></span></p>
# 
# <p><span style="color:#000080"><span style="font-family:Lucida Sans Unicode,Lucida Grande,sans-serif">b) Convert all the characters to lower case so that there cant be any duplicates</span></span></p>
# 
# <p><span style="color:#000080"><span style="font-family:Lucida Sans Unicode,Lucida Grande,sans-serif">c) Split the sentences to get the list of words&nbsp;</span></span></p>
# 
# <p><span style="color:#000080"><span style="font-family:Lucida Sans Unicode,Lucida Grande,sans-serif">d) for each of the word in a Sentense, stem that (get the base word) and remove all the words if that exists in stop words &nbsp;</span></span></p>
# <p><span style="color:#000080"><span style="font-family:Lucida Sans Unicode,Lucida Grande,sans-serif">   (Stop words are like the, is, was, those, these,... e.t.c) &nbsp;</span></span></p>
# 
# <p><span style="color:#000080"><span style="font-family:Lucida Sans Unicode,Lucida Grande,sans-serif">&nbsp;e) The final output will be the messages that contains the&nbsp;result of step a to d&nbsp;</span></span></p>
# 
# <p>&nbsp; &nbsp; &nbsp;&nbsp;</p>
# 

# In[ ]:


# we are interating from index 1 because the zero index row is already deleted in previous step.
corpus = []
for i in range(1, len(spam)+1):
    review  = re.sub('[^a-zA-Z]',' ',spam['message'][i])
    review  = review.lower()
    review  = review.split()
    review  = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review  = ' '.join(review)
    corpus.append(review)
    
print(corpus[:10])  
 
    


# <p>&nbsp; <span style="color:#000080"><span style="font-family:Lucida Sans Unicode,Lucida Grande,sans-serif">Create vectorizer that takes the top 5000 most frequent words</span></span></p>

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000)
x = cv.fit_transform(corpus).toarray()


# <p><span style="color:#000080"><span style="font-family:Lucida Sans Unicode,Lucida Grande,sans-serif">The target column &quot;label&quot; contains text data which is &quot;ham&quot; or &quot;spam&quot;,&nbsp;</span></span></p>
# 
# <p><span style="color:#000080"><span style="font-family:Lucida Sans Unicode,Lucida Grande,sans-serif">we need to convert this to numerical value for processing.&nbsp;</span></span></p>
# 
# <p><span style="color:#000080"><span style="font-family:Lucida Sans Unicode,Lucida Grande,sans-serif">get_dummies will do this for us, creates two columns for each category.</span></span></p>
# 
# <p><span style="color:#000080"><span style="font-family:Lucida Sans Unicode,Lucida Grande,sans-serif">We can choose any one of them using iloc</span></span></p>
# 

# In[ ]:


y = pd.get_dummies(spam["label"])
y = y.iloc[:,1].values


# <p>&nbsp; <span style="color:#000080"><span style="font-family:Lucida Sans Unicode,Lucida Grande,sans-serif">Split the data into training and test sets</span></span></p>

# In[ ]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state=0)


# <p>&nbsp; <span style="color:#000080"><span style="font-family:Lucida Sans Unicode,Lucida Grande,sans-serif">Fit using Naive Bayes</span></span></p>

# In[ ]:


from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(x_train, y_train)


# <p>&nbsp; <span style="color:#000080"><span style="font-family:Lucida Sans Unicode,Lucida Grande,sans-serif">Use the test data to predict , check the confusion matrix and observe the accuracy</span></span></p>

# In[ ]:


y_pred = spam_detect_model.predict(x_test)

from sklearn.metrics import confusion_matrix
confusion_m = confusion_matrix(y_test,y_pred)
print(confusion_m)

from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)

