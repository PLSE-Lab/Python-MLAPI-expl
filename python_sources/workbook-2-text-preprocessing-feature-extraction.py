#!/usr/bin/env python
# coding: utf-8

# ## [Workbook 2](https://www.kaggle.com/sabasiddiqi/workbook-2-text-preprocessing-feature-extraction) - Text Preprocessing for Beginners - Feature Extraction
# 
# <br>**Level : **Beginner
# 
# 
# This notebook discusses **Text Data Preprocessing - Feature Extraction** for **NLP Problems** using Toxic Comment Classification Dataset. Data comprises of large number of Wikipedia comments which have been labeled by human raters for toxic behavior. <br>Data is available via following link.
# [Toxic Comment Classification](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data)
# 
#  Previous Notebook : [Workbook 1 - Text Preprocessing for Beginners - Data Cleaning](https://www.kaggle.com/sabasiddiqi/workbook-1-text-pre-processing-for-beginners)
# 
# 
# To skip the initial steps (reading data, removing nans, splitting train/test), Jump to [Feature Extraction](#jump).

# Importing required libraries

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


# Reading processed data from CSV file and saving as Pandas' Dataframe

# In[ ]:


print(os.listdir("../input/workbook-1-text-pre-processing-for-beginners/"))


# In[ ]:


train_data=pd.read_csv('../input/workbook-1-text-pre-processing-for-beginners/train_data.csv')
test_data=pd.read_csv('../input/workbook-1-text-pre-processing-for-beginners/test_data.csv')
print("Preprocessed Training Data: \n",train_data.head())
print("\n Preprocessed Test Data: \n",test_data.head())


# Checking for empty cells in data

# In[ ]:


print("Empty Comment Cells In Train: ",train_data['comment_text'].isna().sum())
print("Empty Comment Cells In Test: ",test_data['comment_text'].isna().sum())


# There are 45 and 102 empty comments in train and test after Cleaning Comments data in [Workbook 1](https://www.kaggle.com/sabasiddiqi/workbook-1-text-pre-processing-for-beginners). <br>Extracting index of empty cells to remove them in next step.

# In[ ]:


train_drop_list=train_data[train_data.iloc[:,0].isna()]
train_drop_list_idx=train_drop_list.index
test_drop_list=test_data[test_data.iloc[:,0].isna()]
test_drop_list_idx=test_drop_list.index
print("Index of Empty Comment Cells in Train: \n",train_drop_list_idx)
print("Index of Empty Comment Cells in Test : \n",test_drop_list_idx)


# Dropping/Removing Empty Cell Rows from data using index values (axis=0 is to specify rows), and verifying.

# In[ ]:


train_data_new=train_data.drop(train_drop_list_idx,axis=0)
#test_data_new=test_data.drop(test_drop_list_idx,axis=0)
test_data_new=test_data
print("Verifying - Empty Comments After Removal: ")
print("Train: ",train_data_new['comment_text'].isna().sum())
print("Test: ",test_data_new['comment_text'].isna().sum())


# In[ ]:


print("Train Data shape --  Before drop: ",train_data.shape, "After Drop: ",train_data_new.shape )
print("Test Data shape --  Before drop: ",test_data.shape, "After Drop: ",test_data_new.shape )


# Splitting preprocessed data into Train (80%) and Test (20%) by using [train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html). And separating comments and labels for Train and Test data.

# In[ ]:


#train, test = train_test_split(data_new, test_size=0.2,random_state=42)
train, test = train_data_new, test_data_new
test_comments=test.iloc[:,0]
train_comments=train.iloc[:,0]
test_labels=test.iloc[:,1:]
train_labels=train.iloc[:,1:]
print("Train Comments Shape : ",train_comments.shape)
print("Train Labels Shape :",train_labels.shape)
print("Test Comments Shape :",test_comments.shape)
#print("Test Labels Shape :",test_labels.shape)


# ##  <a id="jump">Feature Extraction - Bag of Words</a>

# When dealing with textual data, feature extraction refers to conversion of data to numerical form(features) supported by Machine Learning Algorithms.<br>  
# One way to do so is using "**Bag of Words**" , the model  involves representation of text (referred as sentence or document) as a multiset of words keeping a record of their frequency.
# A **Term Frequency Matrix** is used to keep record of frequency of words in the document(comment).
# 
# For example, lets consider these two comments.
# 
# *Comment 1: the cat jumped over the fence*<br>
# *Comment 2: the dog jumped over the wall* <br>
# 
# TF for these documents will be,
# 
# **Term frequency matrix: **
# ![](https://storage.googleapis.com/kagglesdsdata/datasets/83007/192833/Picture2.png?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1543552930&Signature=WhjJ5gcNT4Gj9tvksqIoHb87wHMoc5XnnuzMsG9g8H5rdEEdwvhiEqLye4AkDu7uawiFfDcrYNB%2B619UpzP%2FI9MQb5xDfj5jAHvYRIY4Utr5gzaGHGT1YseIWyFa9OEA6VeCm%2BZAnJ4NBXHlX2yomAt%2F864VS%2FhMaUjuagjmJGOKwrk6sAj3piOex0BXkw3BpKFm31CT8B0x9yXkRzyyhAaF5do84%2F6DqfIOQySkOochZMWcjvpJEk2qAjouqgunUNZa7ki3xuBZIywpFfDJKtE5E8IUIgvUmHDkBdaaQnbAvGsnyF6IvAlTVDM1TuFDvL5mINF13mkId0fnsIUJ0g%3D%3D)
# 
# 
# Our actual term frequency matrix size has ~ 169000 terms (words). We limit it to 10000 most frequent words.
# 

# ### Creating a Term Frequency Matrix:
# *Note: TF is only fit on training data
# 
# [CountVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html)-Convert a collection of text documents to a matrix of token counts <br>
# **fit** : to generate learning model parameters from training data<br>
# **transform** : parameters generated from fit method are applied on model to generate transformed data set.

# In[ ]:


vectorizer = CountVectorizer(analyzer = 'word',stop_words='english',max_features=10000)
train_comments_count=vectorizer.fit(train_comments).transform(train_comments)
print("Term Frequency Matrix(TF): \n",train_comments_count.toarray())
print("Verifying that TF is not empty by checking the sum ",train_comments_count.toarray().sum() )


# ## Inverse Document Frequency<br><br>
# 
# We chose word frequency here to represent text features. However, Inverse Document Frequency can be applied to Term Frequency Matrix to furthur improve our classifier.
# 
# **Term Frequency (TF)**  is a scoring of the frequency of the word in the current document, whereas <br>
# **Inverse Document Frequency (IDF)** is a scoring of how rare the word is across documents.<br>
# 
# **Why do we need to find rare words ?** Terms that appear across many comments are less discriminating. TFIDF assigns weightage to words wrt other words in document.
# 
# TF-IDF not only counts the frequency of a term in the given document), but also reflects the importance of each term to the document by penalizing frequently appearing terms(words) in most samples.
# 
# 
# To find out the IDF using Term Frequency Matrix:
# * Scale each term frequency of term** i **by **log(N/fi)** 
# where, <br>
# **N** = # of comments in Term Frequency matrix  <br>
# **fi** = # of comments term i appears in <br>
# 
# 
# 
# **Scaled matrix:** 
# 
# 
# ![](https://storage.googleapis.com/kagglesdsdata/datasets/83007/192833/Picture1.png?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1543552915&Signature=hd0c%2BFU6RJxKMYj6u4aymfVFyTFGpKAIbRofsmTC6r4MFg7Jud9a2GU7j1O5JrCFB0%2BDfXgdUAWqDL4CTepCzaJjKmFSv4ipzd65kPizjjYmSvR7G0EJ5ZX1AoVn89exl2oepn2VmMiFRtVi8L1PJ%2BqtiBiwupIBBwyVfrBWyZgujkul%2BUKmrtaeWxizVXgTAMMSbctp45pT2ALrGbYwWpStk%2BTYGqhegG353PmvPPBpUUMpGQcUiKY1JtH%2BqrLjwoGK30aC7dSspV1zC%2BDfZkskTsa4qxS1mtUcGismK%2F2OmZpWLUqZAnvEpLJ1PFtO4bfiIAyTpMtx70OVIlr52Q%3D%3D)
# 
# 

# **TF-IDF for Training Data:**
# 

# In[ ]:


tf_transformer = TfidfTransformer()
tf_transformer.fit(train_comments_count)
train_tfidf = tf_transformer.transform(train_comments_count)
print("Train TF-IDF Matrix Shape: ",train_tfidf.shape)


# **TF-IDF forTest Data:**

# In[ ]:


test_comments_count = vectorizer.transform(test_comments)
test_tfidf = tf_transformer.transform(test_comments_count)
print("Test TF-IDF Matrix Shape: ",test_tfidf.shape)


# Saving data in npz format (as sparse matrix)  to use it in different notebook, or you can continue working in the same notebook.

# In[ ]:


from scipy import sparse

train.to_csv('train.csv', index = False)
test.to_csv('test.csv', index = False)
sparse.save_npz("train_tfidf.npz", train_tfidf)
sparse.save_npz("test_tfidf.npz", test_tfidf)


# In[ ]:




