#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# # Having fun with the simpsons data
# 
# Going throught this dataset I wonder if the amount of time a character is on screen determines the score of the episode. Maybe one of the characters is more likable/relatable than others or more funnier and that helps to the score.
# 
# **NOTE**: If you find something weird or wrong in the data feel free to tell me in the coments.
# 
# **DISCLAIMER**: I'm a rookie at data science stuff
# 
# So lets start going through the data.
# 
# ## The Data
# 
# A simple look at the data tell us that we can't use the raw data because it's distributed across several csvs so we have to modify it to our liking.
# 
# We start loading the characters, episodes and script csv.

# In[ ]:


chars = pd.read_csv("../input/simpsons_characters.csv")
episodes = pd.read_csv("../input/simpsons_episodes.csv")

#weird fix, check later
script = pd.read_csv("../input/simpsons_script_lines.csv",
                    error_bad_lines=False,
                    warn_bad_lines=False,
                    low_memory=False)


# In[ ]:


chars = chars.sort_values("id")


# In[ ]:


script = script.sort_values("episode_id")
script = script[["id","episode_id","character_id","timestamp_in_ms"]]
script = script.dropna()


# In[ ]:


episodes = episodes.sort_values("id")
episodes = episodes[["id","imdb_rating"]]
episodes.imdb_rating.unique()


# Let's create our own dataframe to store all the data.
# 
# For each episode we fill it with Zero's

# In[ ]:


columns = []
columns+=  [x for x in chars.id.values]
columns.append("score")
data = pd.DataFrame(columns = columns, index = list(script.episode_id.unique()))
data = data.fillna(0)


# We start filling the dataframe with the scripts data. (This takes a long long time)
# 
# This will add up time each character talks in every episode.

# In[ ]:


count = 0
for index, row in script.iterrows():
    #print(int(row.episode_id), int(row.character_id),int(row.timestamp_in_ms))
    try: #weird hack for poorly parse data
        data.loc[int(row.episode_id), int(row.character_id)] += int(row.timestamp_in_ms)
    except ValueError:
        try:
            print("error at index: ", index, " | Data: ", row.episode_id, row.character_id)
        except:
            print("unkown at index: ", index, " | Data: ", row.episode_id, row.character_id)
    else:
        pass #I dunno
        
print("done!") #for my mental sake


# Now let's check our data. Looks pretty good!

# In[ ]:


data.head()


# Now with the scores.
# 
# Scores for episodes are not absolutes, when someone looks at a 8.2 at imdb he usually thinks *"That's a good episode"*. So we'll think about the scores in the same discrete way.
# 
# This is how I classified it:
# 
# - From 9 to 10 it's a **very-good** episode
# - From 7 to 9 it's a **good** episode
# - From 5 to 7 it's a **regular** episode
# - From 3 to 5 it's a **bad** episode
# - From 1 to 3 it's a **very-bad** episode

# In[ ]:


for index, row in episodes.iterrows():
    try: #weird hack for poorly parse data
        discrete_score = np.NaN
        if row.imdb_rating >= 9:
            discrete_score = "very-good"
        elif row.imdb_rating < 9 and row.imdb_rating >= 7:
            discrete_score = "good"
        elif row.imdb_rating < 7 and row.imdb_rating >= 5:
            discrete_score = "regular"
        elif row.imdb_rating < 5 and row.imdb_rating >= 3:
            discrete_score = "bad"
        elif row.imdb_rating < 5 and row.imdb_rating >= 3:
            discrete_score = "very-bad"
            
        data.loc[int(row.id), "score"] = discrete_score
        
    except ValueError:
        print("error at index: ", index, " | Data: ", row.imdb_rating)


# Finally, we get rid of any unwanted data.

# In[ ]:


data = data.dropna()


# # The Prediction
# 
# Now for the good part. We'll use a svm classifier, but in theory you sould be able to use most classifiers in sktlearn.
# 
# So we import the necesary libraries.

# In[ ]:


from sklearn import svm
from sklearn.model_selection import cross_val_score
classifier = svm.SVC(gamma=0.01, C=100.)


# We prepare our data for the classifier

# In[ ]:


target = data["score"]
train = data.drop(["score"], axis=1)
print(target.shape, train.shape)


# And we cross validate it. 

# In[ ]:


scores = cross_val_score(classifier, train, target, cv = 40)


# let's check our scores!

# In[ ]:


scores


# In[ ]:


print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


# Not very good, but it was a fun side project.

# # Conclusion
# 
# Most of the work done here was getting the data ready. Thanks to sklearn doing classification it's prety straightfoward. That said, there's still a lot of work that can be done with the same data and, hopefully, get better results.
# 
# Still, this was a fun experience for a Sunday afternon.
# 
# ## What could be done but I didn't do
# 
# - Take out the "one episode characters" that have little to no spoken time in the episode, leaving only the "guest characters" in the data.
# - Change the score discrete values to ones related to the real scores (Not a bad episode but a *bad simpsons episode*)
# - Use different models and scores for prediction and comparing them with an hipotesis test.
# - I dunno
# 

# In[ ]:




