#!/usr/bin/env python
# coding: utf-8

# # How to Display a Table (starter notebook)

# This is a sample submission for the following task: https://www.kaggle.com/antgoldbloom/aipowered-literature-review-csvs/tasks?taskId=823.  This submission produces only [one table](https://www.kaggle.com/covid-19-contributions#Immune%20Response).  The ideal submission will produce every table from https://www.kaggle.com/covid-19-contributions.  

# ## Task Details
# 
# Create a notebook that visualizes the tables in this dataset.  The tables should resemble the tables at https://www.kaggle.com/covid-19-contributions as closely as possible.  Feel free to make improvements such as adding scroll bars, collapsible tables, and an expandable TOC (and any other ideas that you might have).
# 
# ## Expected Submission
# 
# Our preferred format for a submission is a single Kaggle notebook that is attached to [this dataset](https://www.kaggle.com/antgoldbloom/aipowered-literature-review-csvs).  Notebooks that contain embeddable HTML elements (and similar visualizations) are also encouraged.
# 
# ## Evaluation
# 
# Accuracy (5 points)
#  - Did the participant accomplish the task?
# 
# Documentation (5 points)
#  - Is the code easy to read and reuse?
#  - Is it easy to update the code and make small changes to the tables?
# 
# Presentation (5 points)
#  - Do the tables look good?
#  - Is the document easy to navigate?

# In[ ]:


import numpy as np 
import pandas as pd 
from IPython.display import HTML


def make_clickable(url, title):
    return '<a href="{}">{}</a>'.format(url,title)
df = pd.read_csv('/kaggle/input/aipowered-literature-review-csvs/kaggle/working/Key Scientific Questions/Human immune response to COVID-19.csv',header=None)
list_of_columns = ['Date', 
                   'URL',
                   'Result',
                   'Study Type', 
                   'Measure of Evidence Strength', 
                   'Sample (n)'] # Define columns
df = df.rename(columns=df.iloc[0]).drop(df.index[0]) # Reset header
df['URL'] = df.apply(lambda x: make_clickable(x['Study Link'], x['Study']), axis=1) # Add in link
df = df[list_of_columns] # Drop extra columns
print('Human Immune Response to COVID-19')
HTML(df.to_html(escape=False))


# This was a quick attempt to reproduce the table from the following screenshot: 
# ![](https://i.imgur.com/CCGq8yX.png)
# 
# 

# In[ ]:




