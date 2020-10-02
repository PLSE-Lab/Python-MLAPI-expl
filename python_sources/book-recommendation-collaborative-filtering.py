#!/usr/bin/env python
# coding: utf-8

# this kernel consists of a recommendation system made from the  dataset using collaborative filtering method
# Please feel free to add suggestions
# 

# In[ ]:


import numpy as np
import pandas as pd


# In[ ]:


books = pd.read_csv('../input/books.csv')


# In[ ]:


books.info()


# In[ ]:


book = books[['book_id','authors','title']]
book.head()


# In[ ]:


book.info()


# In[ ]:


ratings = pd.read_csv('../input/ratings.csv')


# In[ ]:


ratings.info()


# In[ ]:


ratings['rating'].unique()


# In[ ]:


books_data = pd.merge(book, ratings, on='book_id')


# In[ ]:


from surprise import Reader, Dataset, SVD, evaluate,accuracy
from surprise.model_selection import train_test_split
from surprise.model_selection import KFold

reader = Reader(rating_scale=(1,5))


data = Dataset.load_from_df(ratings[['book_id', 'user_id', 'rating']], reader)
trainset, testset = train_test_split(data, test_size=.25)

#kf = KFold(n_splits=3)

algo = SVD()
algo.fit(trainset)
predictions = algo.test(testset)
accuracy.rmse(predictions, verbose=True)
#for trainset, testset in kf.split(data):

    # train and test algorithm.
    #algo.fit(trainset)
    #predictions = algo.test(testset)

    # Compute and print Root Mean Squared Error
    #accuracy.rmse(predictions, verbose=True)


# In[ ]:


def recommendation(user_id):
    user = book.copy()
    already_read = books_data[books_data['user_id'] == user_id]['book_id'].unique()
    user = user.reset_index()
    user = user[~user['book_id'].isin(already_read)]
    user['Estimate_Score']=user['book_id'].apply(lambda x: algo.predict(user_id, x).est)
    user = user.drop('book_id', axis = 1)
    user = user.sort_values('Estimate_Score', ascending=False)
    print(user.head(10))


# In[ ]:


recommendation(2)

