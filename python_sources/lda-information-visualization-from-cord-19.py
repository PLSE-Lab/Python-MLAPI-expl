#!/usr/bin/env python
# coding: utf-8

# This is a starting kernel to convert the documents into a meaningful dataframe and visualize texts and the most meaningful words from it to serve for the further analysis
# 
# 
# 
# 
# 
# 
# 

# **Generating a dataframe from the documents (json files)**

# In[ ]:


#let's imprt the necessary libraries
import numpy as np
import pandas as pd
import os
import json
import glob


# In[ ]:


# Get all the files saved into a list and then iterate over them like below to extract relevant information
# hold this information in a dataframe and then move forward from there. 


# In[ ]:


#Creating an empty dataframe with only column names to fill it with files content
df = pd.DataFrame(columns=['Doc_ID', 'Title', 'Text', 'Source'])


# In[ ]:


df


# In[ ]:


#Grabbing the files from the repositories using glob library

json_filenames = glob.glob(f'/kaggle/input/CORD-19-research-challenge/2020-03-13/**/**/*.json', recursive=True)


# In[ ]:


#Taking a look at the first 10 filenames path 


# In[ ]:


json_filenames[:10]


# In[ ]:


# Now we just iterate over the files and populate the data frame. 


# In[ ]:


def get_df(json_filenames, df):

    for file_name in json_filenames:

        row = {"Doc_ID": None, "Title": None, "Text": None, "Source": None}

        with open(file_name) as json_data:
            data = json.load(json_data)
            
            #getting the column values for this specific document
            row['Doc_ID'] = data['paper_id']
            row['Title'] = data['metadata']['title']            
            body_list = []
            for _ in range(len(data['body_text'])):
                try:
                    body_list.append(data['body_text'][_]['text'])
                except:
                    pass

            body = " ".join(body_list)
            row['Text'] = body
            
            # Now just add to the dataframe. 
            row['Source'] = file_name.split("/")[5]
            
            df = df.append(row, ignore_index=True)
    
    return df


# In[ ]:


corona_dataframe = get_df(json_filenames, df)


# In[ ]:


corona_dataframe.shape


# In[ ]:


corona_dataframe.head()


# In[ ]:


corona_dataframe.tail()


# In[ ]:


output = corona_dataframe.to_csv('kaggle_CORD-19_csv_format.csv')


# **NER extraction from Text**

# In[ ]:


# spaCy based imports
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English


# In[ ]:


nlp = spacy.load('en_core_web_lg')


# In[ ]:


#Here's we'll visualize the extraction of entities from some text in the dataframe generated previously


# In[ ]:


#NER extraction using Spacy library
doc = nlp(corona_dataframe["Text"][10])
spacy.displacy.render(doc, style='ent',jupyter=True)


# **LDA Topic modeling**

# In[ ]:


#Loading the necessary libraries
from sklearn.feature_extraction.text import CountVectorizer
# Load the regular expression library
import re
from wordcloud import WordCloud
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.simplefilter("ignore", DeprecationWarning)

# Load the LDA model from sk-learn
from sklearn.decomposition import LatentDirichletAllocation as LDA


# In[ ]:


#Let's first clean our text datas with some basic operations 
#It may be improved more and more 

#Remove punctuation
corona_dataframe['Text'] = corona_dataframe['Text'].map(lambda x: re.sub('[,\.!?]', '', x))
#Convert to lowercase
corona_dataframe['Text'] = corona_dataframe['Text'].map(lambda x: x.lower())
#Print out the first rows of papers
corona_dataframe['Text'].head()


# In[ ]:


#Let's have an idea about what reveal the titles of the papers


#Join the different processed titles together.
long_string = ','.join(list(corona_dataframe['Title'].values))
#Create a WordCloud object
wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue')
#Generate a word cloud
wordcloud.generate(long_string)
#Visualize the word cloud
wordcloud.to_image()


# In[ ]:


#Let's take a look at the distribution of the most significant words of the text corpus

#Helper function
def plot_10_most_common_words(count_data, count_vectorizer):
    words = count_vectorizer.get_feature_names()
    total_counts = np.zeros(len(words))
    for t in count_data:
        total_counts+=t.toarray()[0]
    
    count_dict = (zip(words, total_counts))
    count_dict = sorted(count_dict, key=lambda x:x[1], reverse=True)[0:10]
    words = [w[0] for w in count_dict]
    counts = [w[1] for w in count_dict]
    x_pos = np.arange(len(words)) 
    
    plt.figure(2, figsize=(15, 15/1.6180))
    plt.subplot(title='10 most common words')
    sns.set_context("notebook", font_scale=1.25, rc={"lines.linewidth": 2.5})
    sns.barplot(x_pos, counts, palette='husl')
    plt.xticks(x_pos, words, rotation=90) 
    plt.xlabel('words')
    plt.ylabel('counts')
    plt.show()
#Initialise the count vectorizer with the English stop words
count_vectorizer = CountVectorizer(stop_words='english')
#Fit and transform the processed titles
count_data = count_vectorizer.fit_transform(corona_dataframe['Text'])
#Visualise the 10 most common words
plot_10_most_common_words(count_data, count_vectorizer)


# In[ ]:


#LDA model training and results visualization
#To keep things simple, we will only tweak the number of topic parameters.
#The first 5 topics mention the most meaningful terms related in the text of all papers 
 
#Helper function
def print_topics(model, count_vectorizer, n_top_words):
    words = count_vectorizer.get_feature_names()
    for topic_idx, topic in enumerate(model.components_):
        print("\nTopic #%d:" % topic_idx)
        print(" ".join([words[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
        
#Tweak the two parameters below
number_topics = 5
number_words = 10
#Create and fit the LDA model imported from sklearn library
lda = LDA(n_components=number_topics, n_jobs=1)
lda.fit(count_data)
#Print the topics found by the LDA model
print("Topics found via LDA:")
print_topics(lda, count_vectorizer, number_words)


# In[ ]:


#I hope you enjoy it 

