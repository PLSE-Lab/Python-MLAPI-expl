#!/usr/bin/env python
# coding: utf-8

#  # **Set up**

# In[ ]:


# Data Cleaning Libraries
import numpy as np
import pandas as pd


# In[ ]:


# Data Visulation Libraries
import matplotlib.pyplot as plt
import seaborn as sns
import plotly as py
import plotly.graph_objs as go
import cufflinks as cf
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
cf.go_offline()
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# Data Prediction Libraries
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report


# In[ ]:


# Data set (Amazon Musical Instruents Review)
AMIR = pd.read_csv('/kaggle/input/amazon-music-reviews/Musical_instruments_reviews.csv')


# In[ ]:


# AMIR dataset info.
AMIR.info()     # Eagle Eye View


# In[ ]:


# Dropping null rows
AMIR = AMIR.dropna()
AMIR.info()


# In[ ]:


# AMIR Dataset
AMIR.head()


# In[ ]:


# Length of words in each message of review text column
AMIR['Length of Words'] = AMIR['reviewText'].apply(lambda x : len(x.split()))
AMIR.rename(columns={'overall':'rating'},inplace=True)  # renaming 'overall' column with 'rating' 
AMIR.head(4)


# In[ ]:


# Total Number of Users who rated the product as per rating category
AMIR.groupby(by='rating').helpful.count()


# # **Data Visulation**

# In[ ]:


# Overall Rating with respect to Length of words in reviewtext messages
g = sns.FacetGrid(AMIR,col='rating',sharex=True)
g.map(sns.kdeplot,'Length of Words',color='red')


# In[ ]:


# Total number of people that rated the products as per rating category
go.Figure(data=[go.Pie(values=AMIR.groupby(by='rating').helpful.count(),labels=[1,2,3,4,5],
                       title='Volume received by each rating category.')])


# # **Data Prediction**
# # **Predicting whether the review text is Positive or Negative**

# In[ ]:


# Predicting whether the reviewText message is positive or negative
# Considering Rating '1,2,3' as 'Negative Review' 
# Considering Rating '4,5' as 'Positive Review'

review = {1:'Negative',2:'Negative',3:'Negative',4:'Positive',5:'Positive'}
AMIR['review'] = AMIR['rating'].map(review)
AMIR[['reviewText','rating','review']].head()


# In[ ]:


# Selecting Features & Labels
X = AMIR['reviewText']        # features
y = AMIR['review']            # labels


# In[ ]:


# Splitting data into Training Data & Test Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[ ]:


# Pipeline 
pipeline = Pipeline([
    ('Count Vectorizer',CountVectorizer()),
    ('Model',MultinomialNB())
])


# In[ ]:


# Training Data
pipeline.fit(X_train,y_train)


# In[ ]:


# Model Prediction
y_pred = pipeline.predict(X_test)


# In[ ]:


# Model Evaluation
print(confusion_matrix(y_test,y_pred))
print('\n')
print(classification_report(y_test,y_pred))

