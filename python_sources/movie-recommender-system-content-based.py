#!/usr/bin/env python
# coding: utf-8

# First I cleaned **"IMDB data from 2006 to 2016"**  and added some movies to this data set (some movies from Al Pachino, which was not any in this data set!) and I saved it in a new data set as "IMDBMovieData.csv". Here I imported necessary packages to read this data set as *pandas* Dataframe.

# In[ ]:


import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
import csv
from sklearn import preprocessing
import warnings
warnings.filterwarnings("ignore")
trainset = pd.read_csv("../input/imdb-data-set/IMDBMovieData.csv", encoding='latin-1')


# Then I **removed some columns** : 'Title', 'ID', 'Votes', 'Year', 'Revenue','Metascore', 'Rating','Description', 'Runtime' because I did not use them in this level of recommendation.

# In[ ]:


X = trainset.drop(['Title', 'ID', 'Votes', 'Year', 'Revenue','Metascore', 'Rating','Description', 'Runtime'], axis=1)


# 
# In this level of prediction, my program gives recommendations by **Genre , Actors** and **Director** so I got dummies from these tree columns:

# In[ ]:


features = ['Genre','Actors','Director']
for f in features:
    X_dummy = X[f].str.get_dummies(',').add_prefix(f + '.')
    X = X.drop([f], axis = 1)
    X = pd.concat((X, X_dummy), axis = 1)


# Here is an **example** of **one movie vector** after concat!!!: As you can see, this movie is in action, adventure, fantasy genre. And it has some actors and director which are not showable in this scale of output.

# In[ ]:


import csv
import pandas as pd
trainset = pd.read_csv("../input/imdb-data-set/IMDBMovieData.csv", encoding='latin-1')
X = trainset.drop(['Title', 'ID', 'Votes', 'Year', 'Revenue','Metascore', 'Rating','Description', 'Runtime'], axis=1)
features = ['Genre','Actors','Director']
for f in features:
    X_dummy = X[f].str.get_dummies(',').add_prefix(f + '.')
    X = X.drop([f], axis = 1)
    X = pd.concat((X, X_dummy), axis = 1)
print (X.loc[5])


# Then I wrote csv file from Genres, Actors and Directors name from X column (it's name is "testing") and then filled it by **giving vote** to each one *I like* or *I hate* by my measurement.

# In[ ]:


y = list(X.columns.values)
with open('testing.csv', 'w', encoding="ISO-8859-1") as test:
       write = csv.writer(test, delimiter = ",")
       for i in range(3030):
           write.writerow([y[i]])


# For example in the code bellow, you can run and see first 30 rows of my testing file with my tastes.

# In[ ]:


import pandas as pd
header = pd.read_csv("../input/votefile/testing.csv")
header.head(30)


# Now for similarity measuring I used **Cosine similarity measure** to find out which movie is more similar to my interests, by using this formula: 
# sim(x, y) = cos(rx, ry) =
# ![image.png](attachment:image.png)
# 
# But since the size of the movies are near, we can **pass up** *denominator *and just compute dot product of our interest (our vote to each actor, director and genre) to each movie vector which is filled with zero or one by existence or absence of each genre, actor and director. I kept these parameters in "sim" array.
# 
# Then I defined another array named "similar" which was built by sorting first n indexes of 'sim' array with maximum parameter value.
# 

# So till now, it is a **complete code** (without making "testing" file) which takes data in data frame then drops unnecessary columns then gets dummies from necessary columns and then it makes testing file including all genres, actors and directors for user to vote. Now I run this code with my intrest list (*testing*.csv) :

# In[ ]:


import numpy as np
import pandas as pd
import csv
from sklearn import preprocessing
import warnings
warnings.filterwarnings("ignore")

trainset = pd.read_csv("../input/imdb-data-set/IMDBMovieData.csv", encoding='latin-1')
X = trainset.drop(['Title', 'ID', 'Votes', 'Year', 'Revenue','Metascore', 'Rating','Description', 'Runtime'], axis=1)
#trainset.Revenue = X.Revenue.fillna(X.Revenue.mean())
#trainset.Metascore= X.Metascore.fillna(X.Revenue.min())
features = ['Genre','Actors','Director']
for f in features:
    X_dummy = X[f].str.get_dummies(',').add_prefix(f + '.')
    X = X.drop([f], axis = 1)
    X = pd.concat((X, X_dummy), axis = 1)

test = pd.read_csv("../input/votefile/testing.csv")
T = test.drop(['Content'], axis=1)
T = T['Vote'].fillna(0)
vote = T.values
vec = np.ones((1004,3026), dtype=np.uint8)
vec = X.values

sim = np.ones((1004,), dtype=np.complex_)
for i in range (1,1004):
    sim[i] = np.inner(vec[i],vote.transpose())

similar = sim.argsort()[::-1][:30]
for i in range (30):
    print (trainset.iloc[similar[i],1])


# And this is my favorite list based on content! and I think it is right :D But it seems that there are some problems! because the **genre** list is less than **actors**(or directors). I voted to almost all of the genres but most of the actors and directors are without vote. So this list **is more genre based**, for example I like **Keanu Reeves** and I voted to him but there were not any movie of him in this list (maybe because he has not played in my favorite genre ;D ) for solving this problem we can give higher vote to actors (if actors are more important or equal to genre) or normalize the vector of our tastes by these three concepts.

# Now I define **another recommendation based on a text analysis** which uses "[Gestalt Pattern Matching](http://collaboration.cmc.ec.gc.ca/science/rpn/biblio/ddj/Website/articles/DDJ/1988/8807/8807c/8807c.htm). To find most similar movies by their **description**. here is the code:

# In[ ]:


import re
import difflib
import pandas as pd
import numpy as np
import math
trainset = pd.read_csv("../input/imdb-data-set/IMDBMovieData.csv", encoding='latin-1')
s1 = trainset.iloc[422,3]
s1w = re.findall('\w+', s1.lower())
sim = np.ones((1004,), dtype=np.float)
for i in range (1,1004):
    if i != 422:
        s2 = trainset.iloc[i,3]
        if type(s2) == str :
            s2w = re.findall('\w+', s2.lower())
            common = set(s1w).intersection(s2w) 
            common_ratio = 100*(difflib.SequenceMatcher(None, s1w, s2w).ratio())
            sim[i] = common_ratio    
M = np.argmax(sim)
print ("your input movie is:",trainset.iloc[422,1])
print ("My suggestion for you is:",trainset.iloc[M,1])


# As you see I gave "*Harry Potter and the Deathly Hallows: Part 1*" movie and most similar movie by description analysis algorithm was the "*Harry Potter and the Deathly Hallows: Part 2*" and it seems to be right!:D
