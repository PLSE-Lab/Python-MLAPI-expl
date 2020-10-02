#!/usr/bin/env python
# coding: utf-8

# I went to Alhambra once. It was neat. Let's see what people say about it.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re
import itertools
import string
from nltk import word_tokenize
from nltk.corpus import stopwords
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


reviews = pd.read_csv('../input/reviews.csv')
reviews.head()


# In[ ]:


reviews.userop = reviews.userop.str.replace('\D+','')                                   .fillna(0)                                   .apply(int)
reviews.userop.head()


# In[ ]:


reviews.userop.hist();


# Need a log scale, probably.

# In[ ]:


reviews['log_userop'] = np.log10(reviews.userop).replace('-inf',0)
reviews['log_userop'].hist();


# That's much better. Now we can see that the mean number of reviews is a little under 10**2 = 100.

# In[ ]:


reviews.rating.hist();


# Alhambra is one of the most amazing things that I've ever seen in my life, so I can completely understand that the ratings skew so positively!
# 
# Let's see how a user's rating relates to the number of reviews/opinions they've previously given.

# In[ ]:


sns.violinplot(x='rating',y='log_userop',data=reviews);


# So that nicely shows that higher ratings come from users that have written more reviews. What do we make of that?
# 
# I think I had predicted the opposite pattern -- that people that have written more previous reviews would be more discerning/harder to impress, and thus would give lower reviews!
# 
# I'm not sure I have a good explanation for why the actual pattern obtains!
# 
# I'm also a little curious how rating depends on time of year. We might expect that ratings depend on how crazy busy it is, or on the weather.
# 
# We need to convert the date column to a datetime object.

# In[ ]:


reviews.date.head()


# In[ ]:


reviews.date = pd.to_datetime(reviews.date)
reviews.date.head()


# In[ ]:


reviews['year']    = reviews.date.apply(lambda x: x.year)
reviews['month']   = reviews.date.apply(lambda x: x.month)
reviews['day']     = reviews.date.apply(lambda x: x.day)
reviews['weekday'] = reviews.date.apply(lambda x: x.weekday())


# In[ ]:


reviews.year.value_counts().sort_index(ascending=True)


# I would guess Trip Advisor just became more popular over the last few years, as opposed to Alhambra suddenly becoming orders of magnitude more popular recently!

# In[ ]:


reviews.month.value_counts().sort_index(ascending=True)


# Travel is heaviest from May through October. That's a long travel season, right?

# In[ ]:


reviews.day.value_counts().sort_index(ascending=True).plot(ylim=(0,500));


# Hmm. Are those swings just noise, or are they meaningful? I'm leaning towards meaningful, but I don't know why there'd be more reviews at the beginning of the month....

# In[ ]:


reviews.weekday.value_counts().sort_index().plot(ylim=(0,1500));


# Not too much change by day of the week.
# 
# Now let's see how ratings depend on date and time.

# In[ ]:


sns.barplot(x='month',y='rating',data=reviews);


# I tried a bunch of different plots like that, and it just consistently looks like date and time don't have any (interesting) relationship with rating! It's a little surprising knowing that climate and 'tourist season' definitely have an impact on my own enjoyment of a place, but it makes sense given what we saw near the beginning of this notebook, with almost everyone giving Alhambra a 5 -- there's just not much variance in the ratings to be explained/accounted for!

# ### On to exploring the reviews themselves!

# In[ ]:


reviews.head()


# In[ ]:


reviews.ix[0]


# In[ ]:


reviews.ix[0,'quote']


# In[ ]:


reviews.ix[0,'reviewnospace']


# In[ ]:


reviews.ix[0,'titleopinion']


# So we want the `titleopinion`

# In[ ]:


stop = stopwords.words('english') + [string.punctuation]
tokenizer_helper = lambda x: [y for y in word_tokenize(x.lower()) if y not in stop]
reviews['tokenized_titleopinion'] = reviews['titleopinion'].apply(tokenizer_helper)
reviews['tokenized_titleopinion'].head()


# Now we can start looking at the words in reviews.

# In[ ]:


wordList = pd.Series(list(itertools.chain.from_iterable(reviews['tokenized_titleopinion'])))
wordList.head()


# In[ ]:


wordList.value_counts()


# In[ ]:




