#!/usr/bin/env python
# coding: utf-8

# # 02 Exploring The Article-Metadata-CSV
# 
# This notebook explores the **basics of the metadata** of all articles.  
# It focuses on the **"all_sources_metadata_2020-03-13.csv"-file**.
# 
# The notebook is a sequel to the ["01 Exploring The Folder-Structure"-notebook](https://www.kaggle.com/morrisb/01-exploring-the-folder-structure?scriptVersionId=30397035).
# 
# # Table Of Content
# 
# + [1. Import Libraries](#1)<br>
# + [2. Load The Data](#2)<br>
# + [3. Inspect The Columns One By One](#3)<br>
#  + [3.1. Column "sha"](#3.1)<br>
#  + [3.2. Column "source_x"](#3.2)<br>
#  + [3.3. Column "title"](#3.3)<br>
#  + [3.4. Column "doi"](#3.4)<br>
#  + [3.5. Column "pmcid"](#3.5)<br>
#  + [3.6. Column "pubmed_id"](#3.6)<br>
#  + [3.7. Column "license"](#3.7)<br>
#  + [3.8. Column "abstract"](#3.8)<br>
#  + [3.9. Column "publish_time"](#3.9)<br>
#  + [3.10. Column "authors"](#3.10)<br>
#  + [3.11. Column "journal"](#3.11)<br>
#  + [3.12. Column "Microsoft Academic Paper ID"](#3.12)<br>
#  + [3.13. Column "WHO #Covidence"](#3.13)<br>
#  + [3.14. Column "has_full_text"](#3.14)<br>
#  + [3.15. Summary Of The Columns](#3.15)<br>
# + [4. Interesting Combinations Of Columns](#4)<br>
#  + [4.1. Combine "publish_time" And "source_x"](#4.1)<br>
#  + [4.2. Combine "publish_time" And "license"](#4.2)<br>
#  + [4.3. Combine "publish_time" And "has_full_text"](#4.3)<br>
#  + [4.4. Combine "source_x" And "license"](#4.4)<br>
#  + [4.5. Combine "source_x" And "has_full_text"](#4.5)<br>
# + [5. Conclusion](#5)<br>
# 
# # <a id=1>1. Import Libraries</a>

# In[ ]:


# To do linear algebra
import numpy as np

# To store data
import pandas as pd

# To create interactive plots
import plotly.graph_objects as go


# # <a id=2>2. Load The Data</a>

# In[ ]:


# Read the csv file
df = pd.read_csv('../input/CORD-19-research-challenge/2020-03-13/all_sources_metadata_2020-03-13.csv')

print('{} entries are in the file.'.format(df.shape[0]))
print('{} columns are in the file.'.format(df.shape[1]))
print('\nThese are some sampled entries:')
df.sample(3)


# # <a id=3>3. Inspect The Columns One By One</a>
# 
# Here you can find a **short overview of each column** on its own.

# In[ ]:


def countUniqueAndNAN(df, column):
    '''
    Counts and prints the number of empty, filled and unique values for a column in a dataframe
    
    Input:
    df - dataframe to use
    column - column to inspect
    
    Output:
    '''
    
    
    # If there are empty values in the column
    if df[column].isna().sum():
    
        # Count the filled and empty values in the column
        tmp_dict = df[column].isna().value_counts().to_dict()

    else:
        tmp_dict = {False: len(df[column]), True: 0}
        
    print('The column "{}" has:\n\n{} filled and\n{} empty values.'.format(column, tmp_dict[False], tmp_dict[True]))


    # Count the unique values in the column
    nunique = df[column].nunique()

    print('\n{} unique values are in the column.'.format(nunique))




def interactiveBarPlot(x, y, column, title):
    '''
    Creates interactive bar plot with x and y data
    
    Input:
    x - data on x axis
    y - data on y axis
    column - column to inspect
    title- title of the plot
    n - number of most counted items to plot
    
    Output:
    '''

    bar = go.Bar(x=x, 
                 y=y, 
                 orientation='h')

    layout = go.Layout(title=title, 
                       xaxis_title='Value Count', 
                       yaxis_title='{}'.format(column))

    fig = go.Figure([bar], layout)
    fig.show()


# ## <a id=3.1>3.1. Column "sha"</a>
# 
# The column contains the sha-value (**Secure Hash Algorithm**) for the PDF document.  
# You can use the value to check if you are using the right PDF by computing the sha-value of the file on your own.

# In[ ]:


# Define the column to inspect
column = 'sha'


# Count empty, filled and unique values for the column
countUniqueAndNAN(df, column)



# Create mapper for empty and filled values
mapper = {False:'Filled Values', True:'Empty Values'}

# Count empty values
tmp_dict = df[column].isna().value_counts().to_dict()

# Split data to x- and y-axis
y, x = zip(*tmp_dict.items())

# Title of the plot
title = 'Counted empty values for the column "{}"'.format(column)

# Plot count of empty values
interactiveBarPlot(x, [mapper[i] for i in y], column, title)



# Number of most common itemsto plot
n = 20

# Get value counts
tmp_data = df[column].value_counts().head(n)

# Get the data
x = tmp_data.values
y = tmp_data.index

# Title of the plot
title = 'Counted category values for the column "{}"'.format(column) + ('' if len(x)<n else ' (Top {})'.format(n))

# Create the plot
interactiveBarPlot(x, y, column, title)


# More than half of the entries have an associated PDF/sha.
# 
# Whether the **duplicated hashes** depict hash-collisions, multiple publications of the same PDF in different journals or something different is right now unclear.
# 
# ## <a id=3.2>3.2. Column "source_x"</a>
# 
# The column contains the source in which **online repository** the article can be found.

# In[ ]:


# Define the column to inspect
column = 'source_x'


# Count empty, filled and unique values for the column
countUniqueAndNAN(df, column)



# Create mapper for empty and filled values
mapper = {False:'Filled Values', True:'Empty Values'}

# Count empty values
tmp_dict = df[column].isna().value_counts().to_dict()

# Split data to x- and y-axis
y, x = zip(*tmp_dict.items())

# Title of the plot
title = 'Counted empty values for the column "{}"'.format(column)

# Plot count of empty values
interactiveBarPlot(x, [mapper[i] for i in y], column, title)



# Number of most common itemsto plot
n = 20

# Get value counts
tmp_data = df[column].value_counts().head(n)

# Get the data
x = tmp_data.values
y = tmp_data.index

# Title of the plot
title = 'Counted category values for the column "{}"'.format(column) + ('' if len(x)<n else ' (Top {})'.format(n))

# Create the plot
interactiveBarPlot(x, y, column, title)


# For each publication the source is known.  
# Most of the articles can be found at the PMC ([PubMed Central](https://en.wikipedia.org/wiki/PubMed_Central)). The other sources are the "[Chan Zuckerberg Initiative](https://en.wikipedia.org/wiki/Chan_Zuckerberg_Initiative)", "[bioRxiv](https://en.wikipedia.org/wiki/BioRxiv)" and the "[medRxiv](https://en.wikipedia.org/wiki/MedRxiv)".
# 
# ## <a id=3.3>3.3. Column "title"</a>
# 
# The column contains the title of the publication.

# In[ ]:


# Define the column to inspect
column = 'title'


# Count empty, filled and unique values for the column
countUniqueAndNAN(df, column)



# Create mapper for empty and filled values
mapper = {False:'Filled Values', True:'Empty Values'}

# Count empty values
tmp_dict = df[column].isna().value_counts().to_dict()

# Split data to x- and y-axis
y, x = zip(*tmp_dict.items())

# Title of the plot
title = 'Counted empty values for the column "{}"'.format(column)

# Plot count of empty values
interactiveBarPlot(x, [mapper[i] for i in y], column, title)



# Number of most common itemsto plot
n = 20

# Get value counts
tmp_data = df[column].value_counts().head(n)

# Get the data
x = tmp_data.values
y = tmp_data.index

# Shorten the y-axis labels
y = [i[:30] for i in y]

# Title of the plot
title = 'Counted category values for the column "{}"'.format(column) + ('' if len(x)<n else ' (Top {})'.format(n))

# Create the plot
interactiveBarPlot(x, y, column, title)


# Since **some titles of publications have many entries**, it is possible they have common or generic titles. Another possible explanation could be the articles have been published multiple times with only few or non changes.
# 
# ## <a id=3.4>3.4. Column "doi"</a>
# 
# The column represents a "digital object identifier" and seems to be an online link to the publication.

# In[ ]:


# Define the column to inspect
column = 'doi'


# Count empty, filled and unique values for the column
countUniqueAndNAN(df, column)



# Create mapper for empty and filled values
mapper = {False:'Filled Values', True:'Empty Values'}

# Count empty values
tmp_dict = df[column].isna().value_counts().to_dict()

# Split data to x- and y-axis
y, x = zip(*tmp_dict.items())

# Title of the plot
title = 'Counted empty values for the column "{}"'.format(column)

# Plot count of empty values
interactiveBarPlot(x, [mapper[i] for i in y], column, title)



# Number of most common itemsto plot
n = 20

# Get value counts
tmp_data = df[column].value_counts().head(n)

# Get the data
x = tmp_data.values
y = tmp_data.index

# Title of the plot
title = 'Counted category values for the column "{}"'.format(column) + ('' if len(x)<n else ' (Top {})'.format(n))

# Create the plot
interactiveBarPlot(x, y, column, title)


# **Many entries reference the same digital object** on the web.  
# It has to be investigated whether this leads to many almost duplicates in the articles.
# 
# ## <a id=3.5>3.5. Column "pmcid"</a>
# 
# The column contains an ID of the PMC.

# In[ ]:


# Define the column to inspect
column = 'pmcid'


# Count empty, filled and unique values for the column
countUniqueAndNAN(df, column)



# Create mapper for empty and filled values
mapper = {False:'Filled Values', True:'Empty Values'}

# Count empty values
tmp_dict = df[column].isna().value_counts().to_dict()

# Split data to x- and y-axis
y, x = zip(*tmp_dict.items())

# Title of the plot
title = 'Counted empty values for the column "{}"'.format(column)

# Plot count of empty values
interactiveBarPlot(x, [mapper[i] for i in y], column, title)



# Number of most common itemsto plot
n = 20

# Get value counts
tmp_data = df[column].value_counts().head(n)

# Get the data
x = tmp_data.values
y = tmp_data.index

# Title of the plot
title = 'Counted category values for the column "{}"'.format(column) + ('' if len(x)<n else ' (Top {})'.format(n))

# Create the plot
interactiveBarPlot(x, y, column, title)


# The distribution seems similar to the "doi".  
# Possibly it is another variable for the same information.
# 
# ## <a id=3.6>3.6. Column "pubmed_id"</a>
# 
# The column contains the ID of the pubmed.

# In[ ]:


# Define the column to inspect
column = 'pubmed_id'


# Count empty, filled and unique values for the column
countUniqueAndNAN(df, column)



# Create mapper for empty and filled values
mapper = {False:'Filled Values', True:'Empty Values'}

# Count empty values
tmp_dict = df[column].isna().value_counts().to_dict()

# Split data to x- and y-axis
y, x = zip(*tmp_dict.items())

# Title of the plot
title = 'Counted empty values for the column "{}"'.format(column)

# Plot count of empty values
interactiveBarPlot(x, [mapper[i] for i in y], column, title)



# Number of most common itemsto plot
n = 20

# Get value counts
tmp_data = df[column].value_counts().head(n)

# Get the data
x = tmp_data.values
y = tmp_data.index

# Shorten the y-axis labels
y = ['ID: '+str(i)[:30] for i in y]

# Title of the plot
title = 'Counted category values for the column "{}"'.format(column) + ('' if len(x)<n else ' (Top {})'.format(n))

# Create the plot
interactiveBarPlot(x, y, column, title)


# The distribution seems similar to the "doi" and the "pmcid".  
# Possibly it is another variable for the same information.
# 
# ## <a id=3.7>3.7. Column "license"</a>
# 
# The column contains the license under which article has been published.

# In[ ]:


# Define the column to inspect
column = 'license'


# Count empty, filled and unique values for the column
countUniqueAndNAN(df, column)



# Create mapper for empty and filled values
mapper = {False:'Filled Values', True:'Empty Values'}

# Count empty values
tmp_dict = df[column].isna().value_counts().to_dict()

# Split data to x- and y-axis
y, x = zip(*tmp_dict.items())

# Title of the plot
title = 'Counted empty values for the column "{}"'.format(column)

# Plot count of empty values
interactiveBarPlot(x, [mapper[i] for i in y], column, title)



# Number of most common itemsto plot
n = 20

# Get value counts
tmp_data = df[column].value_counts().head(n)

# Get the data
x = tmp_data.values
y = tmp_data.index

# Title of the plot
title = 'Counted category values for the column "{}"'.format(column) + ('' if len(x)<n else ' (Top {})'.format(n))

# Create the plot
interactiveBarPlot(x, y, column, title)


# Some licenses have **different notations but the same meaning**.  
# Cleaning with lowering the letters and removing hyphens seems necessary.
# 
# ## <a id=3.8>3.8. Column "abstract"</a>
# 
# The column contains the abstracts of the publications.

# In[ ]:


# Define the column to inspect
column = 'abstract'


# Count empty, filled and unique values for the column
countUniqueAndNAN(df, column)



# Create mapper for empty and filled values
mapper = {False:'Filled Values', True:'Empty Values'}

# Count empty values
tmp_dict = df[column].isna().value_counts().to_dict()

# Split data to x- and y-axis
y, x = zip(*tmp_dict.items())

# Title of the plot
title = 'Counted empty values for the column "{}"'.format(column)

# Plot count of empty values
interactiveBarPlot(x, [mapper[i] for i in y], column, title)



# Number of most common itemsto plot
n = 20

# Get value counts
tmp_data = df[column].value_counts().head(n)

# Get the data
x = tmp_data.values
y = tmp_data.index

# Shorten the y-axis labels
y = [i[:30] for i in y]

# Title of the plot
title = 'Counted category values for the column "{}"'.format(column) + ('' if len(x)<n else ' (Top {})'.format(n))

# Create the plot
interactiveBarPlot(x, y, column, title)


# The distribution seems similar to the "doi", "pmcid" and "pubmed_id".  
# Since it is very **unlikely for different publications to have the same abstract**, it can be assumed some articles have multiple entries in the csv-file.
# 
# ## <a id=3.9>3.9. Column "publish_time"</a>
# 
# The column contains the date of the publication.

# In[ ]:


# Define the column to inspect
column = 'publish_time'


# Count empty, filled and unique values for the column
countUniqueAndNAN(df, column)



# Create mapper for empty and filled values
mapper = {False:'Filled Values', True:'Empty Values'}

# Count empty values
tmp_dict = df[column].isna().value_counts().to_dict()

# Split data to x- and y-axis
y, x = zip(*tmp_dict.items())

# Title of the plot
title = 'Counted empty values for the column "{}"'.format(column)

# Plot count of empty values
interactiveBarPlot(x, [mapper[i] for i in y], column, title)



# Convert column to datetime
df_tmp = pd.to_datetime(df['publish_time'].dropna(), errors='coerce').to_frame()

# Set and sort index
df_tmp.set_index('publish_time', inplace=True)
df_tmp.sort_index(inplace=True)

# Resample the data on month basis
df_tmp = df_tmp.resample('M').size().to_frame()

# Filter empty months out
df_tmp = df_tmp[df_tmp[0]!=0]

# Create plot
scatter = go.Scatter(x=df_tmp.index, 
                     y=df_tmp[0], 
                     mode='markers')

layout = go.Layout(title='Number of publications over time', 
                   xaxis_title='Month of publication', 
                   yaxis_title='Number of publications')

fig = go.Figure([scatter], layout)
fig.show()


# The publications seem to include the first SARS outbreak in 2002/2003 with rising numbers of publications in the following years.  
# The **peak is right now with an explosion of articles** and publications.
# 
# ## <a id=3.10**>3.10. Column "authors"</a>
# 
# The column contains a list of all authors of the publication.

# In[ ]:


# Define the column to inspect
column = 'authors'


# Count empty, filled and unique values for the column
countUniqueAndNAN(df, column)



# Create mapper for empty and filled values
mapper = {False:'Filled Values', True:'Empty Values'}

# Count empty values
tmp_dict = df[column].isna().value_counts().to_dict()

# Split data to x- and y-axis
y, x = zip(*tmp_dict.items())

# Title of the plot
title = 'Counted empty values for the column "{}"'.format(column)

# Plot count of empty values
interactiveBarPlot(x, [mapper[i] for i in y], column, title)



# Number of most common itemsto plot
n = 20

# Get value counts
tmp_data = df[column].value_counts().head(n)

# Get the data
x = tmp_data.values
y = tmp_data.index

# Shorten the y-axis labels
y = [i[:30] for i in y]

# Title of the plot
title = 'Counted category values for the column "{}"'.format(column) + ('' if len(x)<n else ' (Top {})'.format(n))

# Create the plot
interactiveBarPlot(x, y, column, title)


# The distribution of the data implies another correlation with "doi" and "abstract" since the combinations of many authors should be quite unique.
# 
# ## <a id=3.11>3.11. Column "journal"</a>
# 
# The column contains the journal in which the article has been published.

# In[ ]:


# Define the column to inspect
column = 'journal'


# Count empty, filled and unique values for the column
countUniqueAndNAN(df, column)



# Create mapper for empty and filled values
mapper = {False:'Filled Values', True:'Empty Values'}

# Count empty values
tmp_dict = df[column].isna().value_counts().to_dict()

# Split data to x- and y-axis
y, x = zip(*tmp_dict.items())

# Title of the plot
title = 'Counted empty values for the column "{}"'.format(column)

# Plot count of empty values
interactiveBarPlot(x, [mapper[i] for i in y], column, title)



# Number of most common itemsto plot
n = 20

# Get value counts
tmp_data = df[column].value_counts().head(n)

# Get the data
x = tmp_data.values
y = tmp_data.index

# Title of the plot
title = 'Counted category values for the column "{}"'.format(column) + ('' if len(x)<n else ' (Top {})'.format(n))

# Create the plot
interactiveBarPlot(x, y, column, title)


# The articles have been published in many different journals.  
# It could be interesting to investigate the articles regarding this information.
# 
# ## <a id=3.12>3.12. Column "Microsoft Academic Paper ID"</a>
# 
# This column contains an ID by Microsoft.

# In[ ]:


# Define the column to inspect
column = 'Microsoft Academic Paper ID'


# Count empty, filled and unique values for the column
countUniqueAndNAN(df, column)



# Create mapper for empty and filled values
mapper = {False:'Filled Values', True:'Empty Values'}

# Count empty values
tmp_dict = df[column].isna().value_counts().to_dict()

# Split data to x- and y-axis
y, x = zip(*tmp_dict.items())

# Title of the plot
title = 'Counted empty values for the column "{}"'.format(column)

# Plot count of empty values
interactiveBarPlot(x, [mapper[i] for i in y], column, title)



# Number of most common itemsto plot
n = 20

# Get value counts
tmp_data = df[column].value_counts().head(n)

# Get the data
x = tmp_data.values
y = tmp_data.index

# Shorten the y-axis labels
y = ['ID: '+ str(i)[:30] for i in y]

# Title of the plot
title = 'Counted category values for the column "{}"'.format(column) + ('' if len(x)<n else ' (Top {})'.format(n))

# Create the plot
interactiveBarPlot(x, y, column, title)


# Since most of the values are empty and it is another ID the column is likely to have similar significance as the other ID columns.
# 
# ## <a id=3.13>3.13. Column "WHO #Covidence"</a>
# 
# The column contains an ID by the WHO.

# In[ ]:


# Define the column to inspect
column = 'WHO #Covidence'


# Count empty, filled and unique values for the column
countUniqueAndNAN(df, column)



# Create mapper for empty and filled values
mapper = {False:'Filled Values', True:'Empty Values'}

# Count empty values
tmp_dict = df[column].isna().value_counts().to_dict()

# Split data to x- and y-axis
y, x = zip(*tmp_dict.items())

# Title of the plot
title = 'Counted empty values for the column "{}"'.format(column)

# Plot count of empty values
interactiveBarPlot(x, [mapper[i] for i in y], column, title)



# Number of most common itemsto plot
n = 20

# Get value counts
tmp_data = df[column].value_counts().head(n)

# Get the data
x = tmp_data.values
y = tmp_data.index

# Shorten the y-axis labels
y = ['WHO: '+ str(i)[:30] for i in y]

# Title of the plot
title = 'Counted category values for the column "{}"'.format(column) + ('' if len(x)<n else ' (Top {})'.format(n))

# Create the plot
interactiveBarPlot(x, y, column, title)


# Since most of the values are empty and it is another ID the column is likely to have similar significance as the other ID columns.
# 
# ## <a id=3.14>3.14. Column "has_full_text"</a>
# 
# The column contains a bool variable whether the full text is available.

# In[ ]:


# Define the column to inspect
column = 'has_full_text'


# Count empty, filled and unique values for the column
countUniqueAndNAN(df, column)



# Create mapper for empty and filled values
mapper = {False:'Filled Values', True:'Empty Values'}

# Count empty values
tmp_dict = df[column].isna().value_counts().to_dict()

# Split data to x- and y-axis
y, x = zip(*tmp_dict.items())

# Title of the plot
title = 'Counted empty values for the column "{}"'.format(column)

# Plot count of empty values
interactiveBarPlot(x, [mapper[i] for i in y], column, title)



# Number of most common itemsto plot
n = 20

# Get value counts
tmp_data = df[column].value_counts().head(n)

# Get the data
x = tmp_data.values
y = tmp_data.index

# Title of the plot
title = 'Counted category values for the column "{}"'.format(column) + ('' if len(x)<n else ' (Top {})'.format(n))

# Create the plot
interactiveBarPlot(x, y, column, title)


# For more than 13k publications the full text is available.
# 
# ## <a id=3.15>3.15. Summary Of The Columns</a>
# 
# The most important insight is, there are **multiple entries for the same publication**. This is based on the fact that many abstracts occure mulitple times in the data. This also holds good for the columns "authors", "title" and the ID-columns ("doi", "pmcid", "pubmed_id", "Microsoft Academic Paper ID", and "WHO #Covidence").
# 
# These mentioned ID-columns can possibly be used to download additional information on the publications. Further analysis of these columns seems unnecsessary.
# 
# The **"title" and the "abstract" can be used to perform a short text-analysis** to get the basic relations between words and the most important words without touching the full text.
# 
# The **"authors" can be used to create a network of collaborations**. If it is possible to find the most important publications this can be used to attach weight to the network and as a result to find the most important clusters of scientists. 
# 
# Similarly the "journal" can be used to find the most important journals with the highest value of information for the publications.
# 
# The **"publish_time" is important to order the publications** and to find subsequent conclusions based on older articles. 
# 
# # <a id=4>4. Interesting Combinations Of Columns</a>
# 
# In this paragraph multiple columns will be combined to get interesting insights.

# In[ ]:


# Copy the dataframe
df_combine = df[['source_x', 'license', 'publish_time', 'journal', 'has_full_text']].copy()

# Convert column to datetime
df_combine['publish_time'] = pd.to_datetime(df_combine['publish_time'], errors='coerce')

# Set and sort index
df_combine.set_index('publish_time', inplace=True)
df_combine.sort_index(inplace=True)

# Clean license column
df_combine['license'] = df_combine['license'].replace(np.nan, '', regex=True).apply(lambda x: x.replace('-', ' ').upper()).replace('', np.nan, regex=True)


# ## <a id=4.1>4.1. Combine "publish_time" And "source_x"</a>
# 
# Inspect when the sources have published their articles.

# In[ ]:


scatter = []

column = 'source_x'

# Iterate over all unique sources
for value in df_combine[column].unique():
    
    # Resample the data on month basis
    df_tmp = df_combine[(df_combine[column]==value) & (df_combine.index.notnull())].resample('M').size().to_frame().loc['2000':]
    
    # Filter empty months out
    df_tmp = df_tmp[df_tmp[0]!=0]
    
    # Create scatter
    scatter.append(go.Scatter(x=df_tmp.index, 
                              y=df_tmp[0],
                              name=value))


layout = go.Layout(title='Number of publications over time for the column "{}"'.format(column), 
                   xaxis_title='Month of publication', 
                   yaxis_title='Number of publications')

fig = go.Figure(scatter, layout)
fig.show()


# **Most of the available articles are from the past few years** (only data after year 2000 will be displayed).  
# 
# It has to be noted that the **CZI seems very productive these days**. Whether they publish meaningful articles or only short notices has to be investigated. Furthermore the CZI only releases the year of their publication and no more exact information.  
# The biorxiv clearly has an increase in publications, while the PMC stagnates.
# 
# ## <a id=4.2>4.2. Combine "publish_time" And "license"</a>
# 
# Inspect when the different licenses changed over time.

# In[ ]:


scatter = []

column = 'license'

# Iterate over all unique sources
for value in df_combine[column].unique():
    
    # Resample the data on month basis
    df_tmp = df_combine[(df_combine[column]==value) & (df_combine.index.notnull())].resample('M').size().to_frame().loc['2000':]
    
    # Filter empty months out
    df_tmp = df_tmp[df_tmp[0]!=0]
    
    # Create scatter
    scatter.append(go.Scatter(x=df_tmp.index, 
                              y=df_tmp[0], 
                              name=value))


layout = go.Layout(title='Number of publications over time for the column "{}"'.format(column), 
                   xaxis_title='Month of publication', 
                   yaxis_title='Number of publications')

fig = go.Figure(scatter, layout)
fig.show()


# The "CC BY" license increases over time and is the most important license in this dataset.
# 
# ## <a id=4.3>4.3. Combine "publish_time" And "has_full_text"</a>
# 
# Check if the full text availability changed over time. 

# In[ ]:


scatter = []

column = 'has_full_text'

# Iterate over all unique sources
for value in df_combine[column].unique():
    
    # Resample the data on month basis
    df_tmp = df_combine[(df_combine[column]==value) & (df_combine.index.notnull())].resample('M').size().to_frame().loc['2000':]
    
    # Filter empty months out
    df_tmp = df_tmp[df_tmp[0]!=0]
    
    # Create scatter
    scatter.append(go.Scatter(x=df_tmp.index, 
                              y=df_tmp[0], 
                              name=value))


layout = go.Layout(title='Number of publications over time for the column "{}"'.format(column), 
                   xaxis_title='Month of publication', 
                   yaxis_title='Number of publications')

fig = go.Figure(scatter, layout)
fig.show()


# The availability of the full text increases more rapidly than the missing of the text in the past few years.
# 
# ## <a id=4.4>4.4. Combine "source_x" And "license"</a>
# 
# Check the percentage of the different licenses per source.

# In[ ]:


# Create a pivot table to find the most used license by source_x
df_tmp = df_combine.pivot_table(values='has_full_text', index='license', columns='source_x', aggfunc='size', fill_value=0)

# Compute percentages
df_tmp = df_tmp / df_tmp.sum(axis=0) * 100

df_tmp


# The biorxiv and medrxiv have their own license. The most important license for the CZI and PMC is the "CC BY".
# 
# ## <a id=4.5>4.5. Combine "source_x" And "has_full_text"</a>
# 
# Check the percentage of the full text availability per source.

# In[ ]:


# Create a pivot table to find the most used license by journal
df_tmp = df_combine.pivot_table(values='license', index='has_full_text', columns='source_x', aggfunc='size', fill_value=0)

# Compute percentages
df_tmp = df_tmp / df_tmp.sum(axis=0) * 100

df_tmp


# The PMC has most of the oldest publications and the lowest full text availability.  
# The other and newer sources have a higher availabilty.
# 
# # <a id=5>5. Conclusion</a>
# 
# The dataset contains **multiple entries for the same publications**. This could lead to a skewed view if not handled in the right way.  
# 
# The **time column should be useful to get an ordering**. Maybe it is possible to find articles building upon each other and to extract deeper insights.  
# 
# The **title and abstract can be used to perform a short text analysis** without having to tame the 2GB of json-data.
# 
# The **authors can be used to create a network of collaborations**. This could be helpful to find sources to expect reliable and meaningful new insights from. 

# In[ ]:




