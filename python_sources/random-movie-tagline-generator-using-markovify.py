#!/usr/bin/env python
# coding: utf-8

# I will be using the markovify package for python to make an automatic tagline generator based on films of a chosen genre. I am a relative beginner to python and libraries like pandas so I'm sure this kernel will be full of less than optimal ways to do things but I am just looking to practise. Please feel free to help me out and suggest ways I could improve the tagline generator as it's still a bit rough around the edges! First lets import the required libraries and load in the data - we only need the metadata csv file for this.

# In[ ]:


import pandas as pd
import markovify as mkv
import numpy as np

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv('../input/movies_metadata.csv',low_memory = False)
df.head()


# We are only interested in films which have a tagline, let's see how many there are. 

# In[ ]:


df.tagline.isnull().value_counts()


# We can see that 25054 of the films in the dataset have a tagline, we can remove all the others. I can also tell you that there is  one film which has a tagline which is just an empty string. Although this would likely only cause a minor problem, it is still probably worth removing:

# In[ ]:


(df.tagline.str.rstrip()=='').value_counts()


# In[ ]:


df = df.drop(df[df.tagline.str.rstrip()==''].index)
df = df.dropna(subset=['tagline'])


# Next lets shrink our dataframe and have a look at the genre column to see what format the data is in.

# In[ ]:


df = df[['original_title','genres','tagline']]
df.genres.head()


# Information about the genres appears to be in the form of a list of dictionaries, with each dictionary holding two key:value pairs - an id and a name. There are some films without any genres associated and these have an empty list in their 'genres' column. We can use the following function to remove any cases of this (where 'genres' = []). It interprets the genres square as a string and limits our dataframe to cases where its length is longer than 3 (i.e. its genres column contains more than just '[]' or '[ ]'). Let me know if you have a more elegant way of doing this as I struggled a bit! 

# In[ ]:


df = df[df.genres.str.len()>2]
df= df.reset_index(drop=True)
df.head()


# The 'list of dictionaries' format is a little tricky to work with, but we can use a short function using eval and a list comprehension to generate a column which presents the genre information in a more simple format.

# In[ ]:


df.genres = df.genres.apply(lambda x: [j['name'] for j in eval(x)])
df.head()


# Now we have a neater dataframe where the genres column contains a straightforward list of the genres that each film and tagline is associated with. We can use this column to get a full list of all of the genres represented in the database.

# In[ ]:


genre_list = set()
for s in df['genres']:
    genre_list = set().union(s, genre_list)
    
genre_list = sorted(list(genre_list))
genre_list


# Next we can generate a 'genre table' in which the genres associated with each tagline is represented numerically (1 if a tagline is associated with that genre and 0 if not). 

# In[ ]:


genre_table = df[['tagline']]
for genre in genre_list:
    genre_table[genre] = df['genres'].str.join(" ").str.contains(genre).apply(lambda x:1 if x else 0)
genre_table


# We are now going to use this table to generate a 'composite tagline' - a single string containing all of the taglines associated with each genre concatenated together. This is what we will pass into the markov generator for each genre. The composite tagline is made by concatenating together all the taglines in the genre table which have a value of 1 for each genre.

# In[ ]:


new_df = pd.DataFrame(index = genre_list)
for genre in genre_list:
    new_df.loc[genre,'Composite_Tagline'] = genre_table[genre_table[genre] == 1].tagline.str.rstrip('!.,').str.cat(sep = ". ")
new_df


# We can use the markovify package and the 'mkv.Text' function to build a random text generator based on the horror tagline to see if it works. First we build the model:

# In[ ]:


text_model = mkv.Text(new_df.loc['Horror','Composite_Tagline'])


# Now we can print a few taglines to see what taglines it generates.

# In[ ]:


for i in range(5):
    print(text_model.make_sentence())


# If your results are anything like mine, they'll range from the very believable - 'The darkest day of the Year is Here.' - to the utterly absurd - 'Skinner has returned from the shadows when the monster man... fish... or devil? Don't Believe Everything You Hear.' Finally we can make a function which takes in the genre of choice and generates a chosen number of taglines based on that genre.

# In[ ]:


def random_tagline_generator(genre,taglines=1):
    text_model = mkv.Text(new_df.loc[genre,'Composite_Tagline'])
    return_string = str()
    for i in range(taglines):
        new_tagline = text_model.make_sentence()
        return_string += (new_tagline + "\n")
    return(return_string)


# Now we can use this function to automatically generate taglines for any genre of our choice!

# In[ ]:


print(random_tagline_generator('War',5))


# I hope you enjoy playing around with the tagline generator! This kernel was mostly just done for fun and for my own practise using pandas. There are lots of ways it could be improved/expanded, feel free to recommend me some!
