#!/usr/bin/env python
# coding: utf-8

# # Introduction
# Second Kernel I'm working on for Exploratory Data Analysis... so be gentle in the comment :)  
# and please don't hesitate to point where I went wrong or where I could improve. 
# 
# The whole point of building these kernels is for me to develop an ease about this part of the process and reusable tidbits of codes.  So, I'm all ears for constructive criticism.

# ## Table of Contents
# ---
# 1. [Imports and Functions](#import)  
# 1.1 [Importing the necessary librairies for this kernel](#import_librairies)  
# 1.2 [Reusable functions](#functions)  
# 1.3 [Importing the dataset into a pandas DataFrame](#import)  
# 
# 2. [High Level feel for the dataset](#feel)  
# 2.1 [Shape](#shape)  
# 2.2 [Missing Values](#missing)  
# 2.3 [Categorical Features](#cat_feats)    
# 2.4 [Numerical Features](#num_feats)    
# 2.5 [Class Imbalance](#class)   *I'm currently about here... the sections above are **mostly** completed*
# 
# 3. [Correlations](#corr)  
# 
# 4. [Comparing DNA Structures](#dna)  
# 
# 5. [Building a simple classifier](#clf)  
# 5.1 [Preprocessing](#clf_process)  
# 5.2 [Scale](#clf_scale)  
# 5.3 [Train](#clf_train)  
# 5.4 [Test](#clf_test)
# 
# 6. [Conclusion](#conclusion)
# 
# 7. [Still to be done](#tbd)

# # Imports  <a class="anchor" id="import"></a>
# ## Import necessary librairies  <a class="anchor" id="import_librairies"></a>

# In[1]:


#Imports
import numpy as np
import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns
sns.set(style="whitegrid")

import os
print(os.listdir("../input"))


# ## Reusable Functions <a class="anchor" id="functions"></a>
# I'll put here are the functions I built to help me in my EDA - feel free to reuse them

# In[2]:


def get_null_cnt(df):
    """Return pandas Series of null count encounteres in DataFrame, where index will represent df.columns"""
    null_cnt_series = df.isnull().sum()
    null_cnt_series.name = 'Null_Counts'
    return null_cnt_series

def plot_ann_barh(series, xlim=None, title=None, size=(12,6)):
    """Return axes for a barh chart from pandas Series"""
    #required imports
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    #setup default values when necessary
    if xlim == None: xlim=series.max()
    if title == None: 
        if series.name == None: title='Title is required'
        else: title=series.name
    
    #create barchart
    ax = series.plot(kind='barh', title=title, xlim=(0,xlim), figsize=size, grid=False)
    sns.despine(left=True)
    
    #add annotations
    for i in ax.patches:
        # get_width pulls left or right; get_y pushes up or down
        ax.text(i.get_width()+(xlim*0.01), i.get_y()+.38,                 str(i.get_width()), fontsize=10,
    color='dimgrey')
    
    #the invert will order the data as it is in the provided pandas Series
    plt.gca().invert_yaxis()
    
    return ax


# ## Importing the dataset and first look <a class="anchor" id="import_dataset"></a>
# We've got 2 files in this dataset that join on structureId.  
# I'd like to be able to join then at the very beginning so that the EDA will take into account the sequences that we have in the seperate csv file - we'll see if that's possible

# In[37]:


pdb = pd.read_csv('../input/pdb_data_no_dups.csv', index_col='structureId')
pdb_seq = pd.read_csv('../input/pdb_data_seq.csv', index_col='structureId', na_values=['NaN',''], keep_default_na = False)
pdb.head()


# In[25]:


# See what's the highest number of sequence that a single protein has
pdb_seq_cnt = pdb_seq['chainId'].groupby('structureId').count()
print('The MAX number of sequences for a single protein is {}'.format(pdb_seq_cnt.max()))
print('The AVERAGE number of sequences for a single protein is {}'.format(pdb_seq_cnt.mean()))


# I wanted to join both file right here and kind of "one-hot encode" the sequence on the single line for each protein... but with number of sequences over 100, I don't think it's a good idea.  
# 
# I'll revisit how to join the data together when I get to working on the classifier... or most likely before that... because I think it would be cool to do something with the sequences concatenated together.  We'll see... 

# # High Level Feel<a class="anchor" id="feel"></a>
# 
# ## Shape<a class="anchor" id="shape"></a>

# In[26]:


print('The pdb dataset has {} rows and {} columns'.format(pdb.shape[0], pdb.shape[1]))
print('The pdb_seq dataset has {} rows and {} columns'.format(pdb_seq.shape[0], pdb_seq.shape[1]))


# ## Missing Values <a class="anchor" id="missing"></a>
# Since the numbers of columns is pretty small - let's use missingno to look at a matrix visualization of missing values:

# In[6]:


msno.matrix(pdb.sample(1000))


# OK... we've got some missing values in pretty much all columns.  
# Since we're sampling some rows out of the dataset, let's see if the columns that shows up full on the matrix are actually all there

# In[27]:


# sanity check on missingno
# build null count series
pdb_null_cnt = get_null_cnt(pdb)
# plot series result
ax = plot_ann_barh(pdb_null_cnt, xlim=len(pdb), title='Count of Null values for each columns in PDB DataFrame')


# Oh... it's a chance we've added annotation !  
# We would have missed the 2 missing values in **classification** if we didn't have them, since the bar is so small
# 
# ---
# Let's see the same things on the sequence data:

# In[32]:


msno.matrix(pdb_seq.sample(1000))


# In[38]:


# sanity check on missingno
# build null count series
pdb_seq_null_cnt = get_null_cnt(pdb_seq)
# plot series result
ax = plot_ann_barh(pdb_seq_null_cnt, xlim=len(pdb_seq), title='Count of Null Values in PDB Sequence DataFrame', size=(12,2))


# Again, we have a small number of missing values in there.  
# The most important ones are probably the ones we see in **chainId** and **sequence**, why does it happen on a handful of sequences ???
# 
# ---
# The plot above did list null values until I changed the parameters in the read_csv function - hence the reason why I'm referring to null values in **chainId** above.

# In[39]:


# get index for lines with null values in chainId column
null_chain_idx = list(pdb_seq[pdb_seq['chainId'].isnull()].index)

# get count per structureId which has at least one line where chainId column is null
null_chain_cnt = pdb_seq.loc[null_chain_idx,'chainId'].groupby('structureId').count()


print(list(null_chain_idx))
print(null_chain_cnt[null_chain_cnt > 1])


# The series above is empty and the list of indices as well - but in my original import, that wasn't the case.  
# While investigating null values with the code above, I discovered that some **chainId** values are 'NA'.  This value is correct, because the **chainId** are a represented by a set of letters usually starting with 'A'.  In the case of some protein structure, the value 'NA' was set as  **chainId** to represent the step in the sequence 'N' / 'NA' / 'NB' / etc. So, I modified the read_csv function parameters to *not* consider 'NA' as a null value anymore.  After that, there isn't anymore null value in this field - which is a relief :)

# In[13]:


# get index for lines with null values in sequence column
null_seq_idx = list(pdb_seq[pdb_seq['sequence'].isnull()].index)

# get count per structureId which has at least one line where sequence column is null
null_seq_cnt = pdb_seq.loc[null_seq_idx,'chainId'].groupby('structureId').count()

print(null_seq_idx)
print(null_seq_cnt[null_seq_cnt > 1])


# The series is empty - so we've got only one record per **structureId** that have null value in the column investigated.  
# 
# However, the missing values in **sequence** have a bit more impact. It means that the sequence for these protein structure are really not available and can't be "replaced" or "imputed".  Let's see if they all belong to the same class - which could then be completely excluded from the classification exercise if that's the case.

# In[ ]:


print('Class of Protein Structure without sequence:')
class_no_seq = pdb.loc[null_seq_idx,'classification'].value_counts()
class_no_seq


# In[ ]:


print('Total count (across dataset) of classes with at least 1 protein structure without sequence:')
class_no_seq_total = pdb[pdb['classification'].isin( list( class_no_seq.index))]['classification'].value_counts()
class_no_seq_total


# OK, it's a bit hard to make out, because the list are in different orders - let's try to remedy that and look at the results side-by-side

# In[ ]:


# create Dataframe with the 2 Series we created above
class_df = pd.concat([class_no_seq, class_no_seq_total], axis=1)
class_df.columns=['no_seq_cnt', 'total_cnt']

# Create a new ratio column to understand how many of the total count don't have sequence
class_df['ratio'] = class_df.apply(lambda row: round(100* (row['no_seq_cnt'] / row['total_cnt']), 2) ,axis=1)

# Sort values and print
class_df = class_df.sort_values('ratio', ascending=False)
class_df


# Now it's easier to see that the first 4 classes can easily be treated the same depending on if we will be using the sequences in the classification or not.  
# For the other, the percentage is so low, that the sequences or even the protein structures observations could be completely removed from the dataset and it wouldn't affect the classification exercise much.

# ## Categorical Features <a class="anchor" id="cat_feats"></a>
# Now let's take a look at the categorical features we have in the dataset.

# In[ ]:


print('The number of unique values in each of PDB dataset categorical columns')
print(pdb.select_dtypes(include=object).nunique())


# Let's see...
# * classification is the class we'll be trying to predict - so nothing to do there
# * pdbxdetails seems to be a *freeform* column, my guess is that we'll have to treat this as unstructure text and see if we can get something out of it.
# * the 3 remaining ones seems to be proper categorical features
# 
# Let's look at countplots to see what kind of possibles values we're dealing with

# In[ ]:


expTech_counts = pdb['experimentalTechnique'].value_counts(dropna=False)
ax = plot_ann_barh(expTech_counts, title='CountPlot for experimentalTechnique', size=(12,8))


# In[ ]:


macroType_counts = pdb['macromoleculeType'].value_counts(dropna=False)
ax = plot_ann_barh(macroType_counts, title='CountPlot for macromoleculeType', size=(12,4))


# In[ ]:


#since this feature has over 500 values, let's have look at the top20 one only
crystMethod_counts = pdb['crystallizationMethod'].value_counts(dropna=False)[:20]
ax = plot_ann_barh(crystMethod_counts, title='CountPlot for Top20 crystallizationMethod values')


# What have we learned...
# * Each feature has one or two main value(s) that represent most of the data
# * **experimentalTechnique** has combined values - I'll have to think about how to encode them before feeding them to a classifier (maybe some kind of special one-hot-encoding)
# * **macromoleculeType** has hybrid values - same type of deal as experimentalTechnique
# * **crystallizationMethod** has a significant amount of missing value and we'll have to decide how to fill them in

# ## Numerical Features <a class="anchor" id="num_feats"></a>
# Now let's have a look at the numerical features in the dataset.

# In[ ]:


pdb.select_dtypes(exclude=object).describe().T


# Everything seems to look fine - the only thing would be that min values in **publicationYear** column (201.00).  
# That looks like a mistake, let's see if we have more columns with only 3 digits (or any other invalid year values):

# In[ ]:


#let's look at the publication year a bit closer before we look at incorrect values
pdb_pubYear = pdb['publicationYear']
print('publicationYear has {} unique values, from {} to {}.'.format(pdb_pubYear.nunique(), int(pdb_pubYear.min()), int(pdb_pubYear.max())))

# Now for the number of instance we have with incorrect years
pdb_pubYear_str = pdb_pubYear.fillna(0).astype('int').astype('str')
# 0 was added as possible value in the regex pattern since it's the value I used for filling in missing values in that column
pattern = '(0|(19|20)\d{2})'
correct_year = pdb_pubYear_str.str.match(pattern, na=False)
print('\nThe incorrect year values and their respective counts:')
pdb_pubYear_str[~correct_year].value_counts()


# No we don't... we've only got that 1 line that's wrong.  
# 
# ---
# I think we can safely replace that value with the mean **publicationYear** from all the protein structure that share the same class. Let's see if it actually falls after 2010 or on 2001 - which seems to be the more likely actual data.

# In[ ]:


# get class and then get all records in protein db with the same class
bad_year_class = pdb[~correct_year]['classification'].tolist()
bad_year_class_pubYear = pdb[pdb['classification'].isin(bad_year_class)]['publicationYear']

print('The mean publicationYear for class {} is {}'.format(bad_year_class[0], 
                                                           int(bad_year_class_pubYear.mean())
                                                          )
     )

print('The most used publicationYear for class {} is {}'.format(bad_year_class[0], 
                                                                #the next line is not pretty - but it gets the first value from
                                                                # value_counts series, since the series is always ordered by value_count
                                                                int(list(bad_year_class_pubYear.value_counts().index)[0])
                                                               )
     )


# Humm... difficult choice, but in this case, I'll go with 2011.

# In[ ]:


pdb.loc[~correct_year, 'publicationYear'] = 2011

# sanity check
pdb[~correct_year]


# ## Class Imbalance <a class="anchor" id="class"></a>
# There's a lot classes in this dataset, so we'll start by looking at the most popular ones.

# In[ ]:


print('There are {} classes in this dataset'.format(pdb['classification'].nunique()))


# In[ ]:


classes_counts = pdb['classification'].value_counts(dropna=False)[:50]
ax = plot_ann_barh(classes_counts, title='CountPlot for Top50 Classes', size=(12,10))


# Humm... there's a good distribution of these classes... I mean there's an imbalance for sure, but it's not too bad since the most popular one accounts for only 15% of the data. 
# 
# It would be interesting to see if we can build a higher class and bin these together.  
# One thing to note is that we have at least 1 unknown class - about midway of the plot above: UNKNOWN FUNCTION

# 
# # Correlations

# In[ ]:





# # Comparing DNA Structures
# The whole point of this kernel was to find out a neat way to compare DNA structures - both visually and functionally.
# 
# I want to try to find a cool way to plot structures so they can be easily compared... let's see if I can :)

# In[ ]:





# # Building a simple classifier <a class="anchor" id="clf"></a>

# In[ ]:





# ## Preprocessing the data <a class="anchor" id="clf_process"></a>

# In[ ]:





# ## Scaling features <a class="anchor" id="clf_scale"></a>

# In[ ]:





# ## Training the classifier <a class="anchor" id="clf_train"></a>

# In[ ]:





# ## Testing the classifier <a class="anchor" id="clf_test"></a>

# In[ ]:





# ## Conclusion <a class="anchor" id="conclusion"></a>

# ## Still to be done... <a class="anchor" id="tbd"></a>
# * Correlation section
# * Classifier Section
