#!/usr/bin/env python
# coding: utf-8

# **Creating word2vec models for SouthPark characters**
# 
# We start by defining the input and output locations.

# In[ ]:


INPUT_CSV_FILE = "../input/All-seasons.csv"
OUTPUT_TEXT_FILE = "./vector.txt"
OUTPUT_JSON_FILE = "./vector.json"
print("Ready")
# Any results you write to the current directory are saved as output.


# Then we parse the csv file and store it in a Pandas Dataframe.
# The info methods gives us a small summary of the Dataframe

# In[ ]:


import pandas as pd

southpark_df = pd.read_csv(INPUT_CSV_FILE)
southpark_df.info()


# In[ ]:


unique_characters = southpark_df.Character.unique()
print("Found ",len(unique_characters)," unique characters")
print(unique_characters)


# Too many characters, let's get how many lines each has and grab the top 4 most prolific characters

# In[ ]:


grouped_by_character = southpark_df.groupby(['Character']).count().reset_index()
sorted_characters = grouped_by_character.sort_values('Line', ascending=False)
sorted_characters.head(4)


# Let's start with Eric Cartman and get all the lines he has said

# In[ ]:


CHARACTER_NAME = "Cartman"
character_lines = southpark_df.loc[southpark_df.Character == CHARACTER_NAME]["Line"]
print(character_lines.describe())
character_lines.head()


# **Tokenzing **
# 
# Now we want to split sentences to an array of lower cased words, to do that we apply a lambda to the dataframe

# In[ ]:


import string
import re
def tokenize(s) :
    lower_case = s.lower();
    without_punctuation = re.sub(r'[^\w\s]','',lower_case)
    return without_punctuation.split()

tokenized_lines = character_lines.apply(lambda row: tokenize(row))
tokenized_lines.tail(4)


# **Creating the Word2Vec model from our tokenized lines**

# In[ ]:


from gensim.models import Word2Vec
model = Word2Vec(tokenized_lines, size=100, window=5, min_count=5, workers=4)
model.wv.save_word2vec_format(OUTPUT_TEXT_FILE, binary=False)
import os
print(os.listdir('./'))


# Next we need to convert the text output to a json file

# In[ ]:


import json
def to_json(input, output): 
    f = open(input)
    v = {"vectors": {}}
    for line in f:
        w, n = line.split(" ", 1)
        v["vectors"][w] = list(map(float, n.split()))
    with open(output, "w") as out:
        json.dump(v, out)

to_json(OUTPUT_TEXT_FILE, OUTPUT_JSON_FILE)
import os
print(os.listdir('./'))


# In our ouput we now have a vector.json that we can download and use from anywhere. 
# 
# Example usage in this sandbox : https://codesandbox.io/s/32n4w2kmz5
