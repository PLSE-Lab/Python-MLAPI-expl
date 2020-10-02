#!/usr/bin/env python
# coding: utf-8

# <h1> CORD-19 Solution Toolbox</h1>
# 
# 
# We give here a minimal toolset to explore the dataset and start performing an EDA.  
# 
# 
# We will provide the tools to:
# 
# * Browse through the files in the collections;  
# * Read content from JSON files;  
# * Bulk process JSON files to extract content in a DataFrame;  
# * Visualize text content;  
# * Visualize most frequent items in categorical features;  
# 
# 

# # Load packages
# 
# We just load the minimum packages for now.

# In[ ]:


import numpy as np
import pandas as pd

import os
import json


# # Explore the data

# In[ ]:


count = 0
file_exts = []
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        count += 1
        file_ext = filename.split(".")[-1]
        file_exts.append(file_ext)

file_ext_set = set(file_exts)

print(f"Files: {count}")
print(f"Files extensions: {file_ext_set}\n\n=====================\nFiles extension count:\n=====================")
file_ext_list = list(file_ext_set)
for fe in file_ext_list:
    fe_count = file_exts.count(fe)
    print(f"{fe}: {fe_count}")


# Let's also look to the structure of directories, to see how the data is structured high-level:

# In[ ]:


count = 0
for root, folders, filenames in os.walk('/kaggle/input'):
    print(root, folders)


# Majority of files are in json format. The files are grouped in 4 folders and 4 tar archives.
# 
# We provide some tools to explore the jsons.
# 
# ## Read a json file
# 
# 

# In[ ]:


json_folder_path = "/kaggle/input/CORD-19-research-challenge/comm_use_subset/comm_use_subset/pdf_json"
json_file_name = os.listdir(json_folder_path)[0]
print(json_file_name)
json_path = os.path.join(json_folder_path, json_file_name)

with open(json_path) as json_file:
    json_data = json.load(json_file)


# To use more easy, we can normalize the json. Here is the code.

# In[ ]:


json_data_df = pd.io.json.json_normalize(json_data)


# The json was transformed in a row in a dataframe, with the column names resulted by aggregating the succesive levels of the json structure.   Let's check the result.

# In[ ]:


json_data_df


# ## Convert the folder in a dataframe
# 
# 
# Let's process now the folder. We will create a dataset with the data from the folder. We just take a subset of data (500 samples).  For your work, just comment the line of code where the subset is declared and uncomment the line of code above, to process entire dataset.

# In[ ]:


print(f"Files in folder: {len(os.listdir(json_folder_path))}")


# In[ ]:


from tqdm import tqdm

# to process all files, uncomment the next line and comment the line below
# list_of_files = list(os.listdir(json_folder_path))
list_of_files = list(os.listdir(json_folder_path))[0:500]
pmc_custom_license_df = pd.DataFrame()

for file in tqdm(list_of_files):
    json_path = os.path.join(json_folder_path, file)
    with open(json_path) as json_file:
        json_data = json.load(json_file)
    json_data_df = pd.io.json.json_normalize(json_data)
    pmc_custom_license_df = pmc_custom_license_df.append(json_data_df)


# In[ ]:


pmc_custom_license_df.head()


# ## Extract abstract text
# 
# 
# Let's extract now abstract text from abstract column.  
# 
# Similar approach can be used to extract other parts from a dictionary-type field.
# 

# In[ ]:


pmc_custom_license_df['abstract_text'] = pmc_custom_license_df['abstract'].apply(lambda x: x[0]['text'] if x else "")


# In[ ]:


pd.set_option('display.max_colwidth', 500)
pmc_custom_license_df[['abstract', 'abstract_text']].head()


# ## Visualize data
# 
# Let's present here few useful techniques for data visualization:
# 
# * Worldclouds for text fields;
# 
# * Countplot for category-type features.
# 

# In[ ]:


from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook
get_ipython().run_line_magic('matplotlib', 'inline')
stopwords = set(STOPWORDS)

def show_wordcloud(data, title = None):
    wordcloud = WordCloud(
        background_color='white',
        stopwords=stopwords,
        max_words=200,
        max_font_size=40, 
        scale=5,
        random_state=1
    ).generate(str(data))

    fig = plt.figure(1, figsize=(15,15))
    plt.axis('off')
    if title: 
        fig.suptitle(title, fontsize=14)
        fig.subplots_adjust(top=2.3)

    plt.imshow(wordcloud)
    plt.show()


# In[ ]:


show_wordcloud(pmc_custom_license_df['abstract_text'], title = 'Comm use subset - papers abstract - frequent words  (500 samples)')


# In[ ]:


show_wordcloud(pmc_custom_license_df['bib_entries.BIBREF0.title'], title = 'Comm use subset - papers title - frequent words (500 samples)')


# In[ ]:


pmc_custom_license_df.loc[((pmc_custom_license_df['bib_entries.BIBREF0.venue']=="") | ((pmc_custom_license_df['bib_entries.BIBREF0.venue'].isna()))), 'bib_entries.BIBREF0.venue'] = "Not identified"


# In[ ]:


import seaborn as sns
def plot_count(feature, title, df, size=1, show_percents=False):
    f, ax = plt.subplots(1,1, figsize=(4*size,4))
    total = float(len(df))
    g = sns.countplot(df[feature], order = df[feature].value_counts().index[0:20], palette='Set3')
    g.set_title("Number of {}".format(title))
    if(size > 2):
        plt.xticks(rotation=90, size=10)
    if(show_percents):
        for p in ax.patches:
            height = p.get_height()
            ax.text(p.get_x()+p.get_width()/2.,
                    height + 3,
                    '{:1.2f}%'.format(100*height/total),
                    ha="center") 
    ax.set_xticklabels(ax.get_xticklabels());
    plt.show()    


# In[ ]:


plot_count('bib_entries.BIBREF0.venue', 'Comm use subset - Top 20 Journals (500 samples)', pmc_custom_license_df, 3.5)


# 
# 
# 
# If you enjoyed this content, please let your feedback in the comments area. Thank you for your kind input.
# 
# 
