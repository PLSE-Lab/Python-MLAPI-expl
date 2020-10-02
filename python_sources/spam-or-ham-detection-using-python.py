#!/usr/bin/env python
# coding: utf-8

# # 'Spam or Ham?' classification with Python
# 
# **Welcome to this kernel!**
# 
# **If you like my work, please, leave an upvote: it will be really appreciated and it will motivate me in offering more content to the Kaggle community ! :)**
# 
# **Context**
# 
# The SMS Spam Collection is a set of SMS tagged messages that have been collected for SMS Spam research. It contains one set of SMS messages in English of 5,574 messages, tagged acording being ham (legitimate) or spam.
# You can find it on Kaggle at the following link: https://www.kaggle.com/uciml/sms-spam-collection-dataset
# 
# **Content**
# 
# The files contain one message per line. Each line is composed by two columns: v1 contains the label (ham or spam) and v2 contains the raw text.
# 
# This corpus has been collected from free or free for research sources at the Internet:
# 
# - A collection of 425 SMS spam messages was manually extracted from the Grumbletext Web site. 
# This is a UK forum in which cell phone users make public claims about SMS spam messages, most of them without reporting the very spam message received. The identification of the text of spam messages in the claims is a very hard and time-consuming task, and it involved carefully scanning hundreds of web pages. The Grumbletext Web site is http://www.grumbletext.co.uk/ . 
# 
# - A subset of 3,375 SMS randomly chosen ham messages of the NUS SMS Corpus (NSC), which is a dataset of about 10,000 legitimate messages collected for research at the Department of Computer Science at the National University of Singapore. The messages largely originate from Singaporeans and mostly from students attending the University. These messages were collected from volunteers who were made aware that their contributions were going to be made publicly available. 
# 
# - A list of 450 SMS ham messages collected from Caroline Tag's PhD Thesis available at http://etheses.bham.ac.uk/253/1/Tagg09PhD.pdf . 
# 
# - Finally, we have incorporated the SMS Spam Corpus v.0.1 Big. It has 1,002 SMS ham messages and 322 spam messages and it is public available at http://www.esp.uem.es/jmgomez/smsspamcorpus/ . 
# 
# This corpus has been used in the following academic researches:
# 
# Acknowledgements
# The original dataset can be found here. The creators would like to note that in case you find the dataset useful, please make a reference to previous paper and the web page: http://www.dt.fee.unicamp.br/~tiago/smsspamcollection/ in your papers, research, etc.
# 
# We offer a comprehensive study of this corpus in the following paper. This work presents a number of statistics, studies and baseline results for several machine learning methods.
# 
# Almeida, T.A., Gomez Hidalgo, J.M., Yamakami, A. Contributions to the Study of SMS Spam Filtering: New Collection and Results. Proceedings of the 2011 ACM Symposium on Document Engineering (DOCENG'11), Mountain View, CA, USA, 2011.

# # Let's start!
# 
# The first action to do is to import the data.

# In[ ]:


import os
print(os.listdir("../input"))


# Let's now import pandas and numpy and then load the csv using the pd.read_csv command.

# In[ ]:


import pandas as pd
import numpy as np


# In[ ]:


# Encoding the data using only the first columns: the other seems to be an issue of the data (empty)

df = pd.read_csv('../input/spam.csv', sep=',', encoding='latin-1', usecols=lambda col: col not in ["Unnamed: 2","Unnamed: 3","Unnamed: 4"])


# As you can see, there are a few commands added to the read_csv.
# 
# - **Sep**: means that we are using as a separator the comma '','' because we are working with a csv and this is how the columns are split.
# - **Encoding**: In computer technology, encoding is the process of applying a specific code, such as letters, symbols and numbers, to data for conversion purposes. In this case, I have used latin-1.
# - **Usecols**: the dataset has a few extra columns without labels that I will not use, so I used a lambda to exclude them.
# 
# Let's now use df.head(1) to review the first line of the process made above: 

# In[ ]:


df.head(1)


# Okay, but the names of the columns are not really clear, right?
# 
# Let's rename them to something more significant:

# In[ ]:


df = df.rename(columns={"v1":"label", "v2":"text"})


# Great: let's review the result:

# In[ ]:


df.head(5)


# # Word Counts with CountVectorizer
# 
# The CountVectorizer provides a simple way to both tokenize a collection of text documents and build a vocabulary of known words, but also to encode new documents using that vocabulary.
# 
# You can use it as follows:
# 
# - Create an instance of the CountVectorizer class.
# - Call the fit() function in order to learn a vocabulary from one or more documents.
# - Call the transform() function on one or more documents as needed to encode each as a vector.
# 
# An encoded vector is returned with a length of the entire vocabulary and an integer count for the number of times each word appeared in the document.
# 
# Because these vectors will contain a lot of zeros, we call them sparse. Python provides an efficient way of handling sparse vectors in the scipy.sparse package.
# 
# The vectors returned from a call to transform() will be sparse vectors, and you can transform them back to numpy arrays to look and better understand what is going on by calling the toarray() function.
# 
# Source: https://machinelearningmastery.com/prepare-text-data-machine-learning-scikit-learn/ 

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer()


# In[ ]:


# Splitting the data into training and test

from sklearn.model_selection import train_test_split


# In[ ]:


X_train,X_test,y_train,y_test = train_test_split(df["text"],df["label"], test_size = 0.2, random_state = 10)


# In[ ]:


# Fitting the CountVectorizer using the training data

vect.fit(X_train)


# In[ ]:


# Transforming the dataframes

X_train_df = vect.transform(X_train)
X_test_df = vect.transform(X_test)
type(X_train_df)


# Perfect: let's move to the machine learning part!

# # Machine Learning
# 
# Let's see if, with the simple preprocessing we did, a Logistic Regression can already fit well this dataset.

# In[ ]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train_df,y_train)


# At this point, we can proceed and make our predictions.
# 
# After the prediction, we will print the accuracy score and a classification report to review the results.

# In[ ]:


# Making predictions

prediction = dict()

prediction["Logistic"] = model.predict(X_test_df)


# In[ ]:


# Reviewing the metrics

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report


# In[ ]:


accuracy_score(y_test,prediction["Logistic"])


# In[ ]:


print(classification_report(y_test,prediction["Logistic"]))


# Great: so we have a classifier with a great precision but less recall on the spam class and the F1-score is quite good!

# **If you like my work, please, leave an upvote: it will be really appreciated and it will motivate me in offering more content to the Kaggle community ! :)**

# # Thank you for your attention!
