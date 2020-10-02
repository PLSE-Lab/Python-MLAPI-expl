#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from surprise import Dataset
from surprise import Reader

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# Finding good book reccomendations can be very hard. Literary taste is highly subjective and it can be very difficult to find others whose opinions generally align with yours. This is an issue that is simply asking to be solved with the use of a simple statistical model. However, the premiere book reccomendation website, Goodreads, uses a reccomendation function that simply shows you books that are judged similar to an individual book that you enjoyed. This is not at all the solution I'm looking for, and seems to be heavily biased towards popular books which I have already heard of. I would much prefer to see users with similar taste and look through their reviews to get a sense of what I might want to try next. Fortunately with the use of a little scraped data we can build a model of our own.

# In[ ]:


df = pd.read_csv('../input/goodbooks-10k-updated/ratings.csv')
df1 = pd.read_csv('../input/goodbooks-10k-updated/personalLibrary.csv')

print("Dataset contains {} rows and {} columns".format(df.shape[0], df.shape[1]))
print(df['user_id'].describe())

df = df.append(df1)
print("Dataset contains {} rows and {} columns".format(df.shape[0], df.shape[1]))
df['user_id'].describe()


# I've taken an existing dataset and added my own reviews to it. This leaves us with nearly six million reviews from over fifty thousand users. Now we'll go ahead and transform this into a nice matrix and get to filtering.

# In[ ]:


book_matrix = df.pivot_table(index='book_id', columns='user_id', values='rating')
book_matrix.head()


# In[ ]:


rosatiRating = book_matrix[53424]
rosatiRating.head()


# In[ ]:


similar = book_matrix.corrwith(rosatiRating)

corrUser = pd.DataFrame(similar, columns=['Correlation'])
corrUser.dropna(inplace=True)
corrUser.head()


# Here we've created a new dataset of just my reviews and compared it against the reviews of every user in the original dataset to find the highest correlation. Now we'll sort these and look at the users with the highest correlation (ignoring the values of 1 of course because these will be users with a single review in common and myself).

# In[ ]:


len(corrUser)
corrUser.sort_values(by=['Correlation'], ascending=False).head(1040)


# In[ ]:


df.loc[df['user_id'] == 15264]


# Excellent. Now lets go ahead and change these book ids into their respective book titles for a more legible exprience. Ideally we would have done this at the very beginning but unfortunately replacing all the values in a fifty-three thousand column matrix exceeds kaggle's max RAM allowance. Who knew?

# In[ ]:


stuff = pd.read_csv('../input/goodbooks-10k-updated/books.csv')

books = stuff['original_title'].tolist()
ids = stuff['book_id'].tolist()


# In[ ]:


replacements = dict(zip(ids,books))
oneguy = df.loc[df['user_id'] == 15264]
oneguy['book_id'].replace(replacements, inplace=True)
oneguy.head(12)


# And there we have it. I was meaning to get around to reading East of Eden anyway, so I suppose that settles it. This could be repeated with any of the other highly correlated users. The same techniques could be used to determine which books are most similar to each other rather than which users are most similar.

# In[ ]:


otherguy = df.loc[df['user_id'] == 402]
otherguy['book_id'].replace(replacements, inplace=True)
otherguy.head(12)


# Thanks for reading.
