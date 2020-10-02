#!/usr/bin/env python
# coding: utf-8

# # Jigsaw Kaggle Competition - WIP 

# ## Notebook  Content
# * [Intro and Agenda](#1)
# * [Import Packages](#2)
# * [How to use less memory while programming?](#3)
# * [EDA](#4)
#     * [References](#5)
#     * [Quick check on the testset and submission file](#6)
#     * [Training Dataset](#7)
#     * [Examples of toxic and non-toxic comments](#8)
#     * [Distribution of the Target Variable](#9)
#     * [Distribution of Additional Columns](#10)
#     * [Identity Columns](#11)
#         * [Information About the Additional Columns](#12)
#     * [Exploration of the ID-Columns](#13)
#     * [Reaction Columns](#14)
#     * [Conclusion on Additional Columns](#15)
#     * [Comment Columns](#16)
#         * [Visualize Most Common Words](#17)

# 

# <a id="1"></a> <br>
# ## Intro and Agenda

# In this kernel we aim to explore the Jigsaw data, in order to get some insights on what makes a toxic comment and what doesn't. We also want to find out whether some of the given information in the training data is relevant to our predicitons or not. A problem we hope to solve with the exploratory data analysis is the prediction of many non-toxic comments as toxic, only because they contain words that are often part of toxic comments. Maybe a deeper look into the provided data, will help us find a way to eliminate such prediction mistakes.
# 
# Next to that we want to share our techniques to import and processing the data consuming minimal RAM.
# 
# **Outline:**
#  
# - check submission file
# - check test file
# - deep dive into the training set
#  - visualize additional parameters
#  - give some examples
#  - check columns separately (toxicity, identity, id-s, reaction and comment)
#     

# <a id="2"></a> <br>
# ## Import Packages

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns 
import matplotlib.pyplot as plt 

get_ipython().run_line_magic('matplotlib', 'inline')
sns.set()

#import pydot
from IPython.display import Image

import os
print(os.listdir("../input"))

# ignore warnings
import warnings
warnings.filterwarnings("ignore")

# To release RAM import gc collector
import gc
gc.enable()


# <a id="3"></a> <br>
# ## How to use less memory while programming?

# It is quite hard to work without interruptions when you have very large amounts of data and around 12 GB of RAM memory under your disposal. The kernels on Kaggle die or run much slower due to this. In this part we want to show you some small tricks on how to use less memory, so that the kernel does not die as often and the committing takes a little less time.
# 
# 1.  **The garbage collector** 
# 
#    The garbage collector (Python module: gc) attempts to collect  garbage, or to free memory occupied by objects that are no longer in use by the program/script.
#     What you can do is to call gc.collect() after everytime you delete unnecessary data or new variables are being assigned.
#     The collector has many more methods to offer. Probably the most important one is gc.enable(), which enables the automatic garbage collection.
#     
# 2. **The %whos command** 
# 
#      Make use of the %whos command in the python notebook to see all the current variables that you are not using anymore and manually delete them with the del command. Thus has been done several times throughout this kernel. 
# 
# 3. **The reduce memory function** (Source : [LIghtGBM (goss + dart) + Parameter Tuning](https://www.kaggle.com/ashishpatel26/lightgbm-gbdt-dart-baysian-ridge-reg-lb-3-61))
# 
#       The reduce memory function **reduce_mem_usage** works similar to the lean importing of the files, it changes the datatype of columns to an appropriate one. You could use this function when you read a new csv-file.
#       
# 4. **Lean importing of files** (Source: [3 Simple Ways to Handle Large Data with Pandas](https://towardsdatascience.com/3-simple-ways-to-handle-large-data-with-pandas-d9164a3c02c1))
# 
#       Define columns and data-types with a dictionary. Python imports the data usually at highest resolution (int64, float64) which may not be always necessary. Make use of the reduce-memory-function after importing the data and check to which data type the columns are transformed (e.g. df.info()). This is the basis for the dictionary for future data imports of your CSV when rerunning the notebook.
#       For example, if we downloaded a dataset for predicting stock prices, our prices might be saved as 64 bit floating point. Since we might not really need such a long float number, we can import these columns as a float16 instead and save space in your RAM.
# 
# 5. **Delete unnecessary columns of the dataframe**
# 
#       Once you're done with your exploratory data analysis, you know which features are of use to you. At this point you can drop all the columns that you think will not improve your result in the end. Since we're talking about millions of lines in the dataframe, it is worth doing and will make a difference when it comes to using less memory.

# **Reduce Memory Function**

# In[ ]:


# Reduce memory function was taken from the kaggle following kernel:
# https://www.kaggle.com/ashishpatel26/lightgbm-gbdt-dart-baysian-ridge-reg-lb-3-61
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


# <a id="4"></a> <br>
# # EDA

# Here we will explore the training data and see if it can tell us something more about what makes a comment toxic.

# <a id="5"></a> <br>
# ## References

# * [Simple EDA Text Preprocessing - Jigsaw ](https://www.kaggle.com/nz0722/simple-eda-text-preprocessing-jigsaw)
# * [Jigsaw EDA](https://www.kaggle.com/gpreda/jigsaw-eda)
# * [Jigsaw Competition : EDA and Modeling](https://www.kaggle.com/tarunpaparaju/jigsaw-competition-eda-and-modeling)
# * [EDA - Toxicity of Identities (Updated 29/4)](https://www.kaggle.com/chewzy/eda-toxicity-of-identities-updated-29-4)
# 

# <a id="6"></a> <br>
# ## Quick check on the testset and submission file

# In[ ]:


# Read CSV-Fikes

test_data = pd.read_csv('../input/test.csv')
sample_submission = pd.read_csv('../input/sample_submission.csv' )


# In[ ]:


# Print the head of the data 
print(test_data.head())
print('\n\n')
print(sample_submission.head(10))


# As we can see the test data only includes the comment id and the comment text. The submission file contains the id of the comment and the perdiction, which is the target (the toxicity value). As we do not need these dataframes for any further exploratory data analysis, we will delete them.

# In[ ]:


# delete test data and the submission file for now
del test_data
del sample_submission
gc.collect


# <a id="7"></a> <br>
# ## Training Dataset

# In[ ]:


# Declaring a dictionary with column name and column type, 
# so that we can save memory while reading the csv-file
# the dictionary is based on the result from the memory reduce function

dtypesDict_tr = {
'id'                            :         'int32',
'target'                        :         'float16',
'severe_toxicity'               :         'float16',
'obscene'                       :         'float16',
'identity_attack'               :         'float16',
'insult'                        :         'float16',
'threat'                        :         'float16',
'asian'                         :         'float16',
'atheist'                       :         'float16',
'bisexual'                      :         'float16',
'black'                         :         'float16',
'buddhist'                      :         'float16',
'christian'                     :         'float16',
'female'                        :         'float16',
'heterosexual'                  :         'float16',
'hindu'                         :         'float16',
'homosexual_gay_or_lesbian'     :         'float16',
'intellectual_or_learning_disability':    'float16',
'jewish'                        :         'float16',
'latino'                        :         'float16',
'male'                          :         'float16',
'muslim'                        :         'float16',
'other_disability'              :         'float16',
'other_gender'                  :         'float16',
'other_race_or_ethnicity'       :         'float16',
'other_religion'                :         'float16',
'other_sexual_orientation'      :         'float16',
'physical_disability'           :         'float16',
'psychiatric_or_mental_illness' :         'float16',
'transgender'                   :         'float16',
'white'                         :         'float16',
'publication_id'                :         'int8',
'parent_id'                     :         'float32',
'article_id'                    :         'int32',
'funny'                         :         'int8',
'wow'                           :         'int8',
'sad'                           :         'int8',
'likes'                         :         'int16',
'disagree'                      :         'int16',
'sexual_explicit'               :         'float16',
'identity_annotator_count'      :         'int16',
'toxicity_annotator_count'      :         'int16'
}


# In[ ]:


# Read file to CSV  
train_data = pd.read_csv('../input/train.csv',dtype=dtypesDict_tr,parse_dates=['created_date'])  # nrows=10000000
train_data['created_date'] = pd.to_datetime(train_data['created_date']).values.astype('datetime64[M]')

gc.collect()


# In[ ]:


# Use the methos to well import the data
# reduce_mem_usage(train_data)
train_data.info()


# In[ ]:


# Look at the top of the dataset
train_data.head(3)


# In[ ]:


# Inspect the statistical summary of the dataset
train_data.describe()


# From the column summary we can already derive that most of the classification columns range between 0 and 1.  The IDs should for sure rather be seen as categorical values.  The indentity annotator and the toxicity annotator seems to have strong outliers comparing the maximum values to the 75% percentile.

# <a id="8"></a> <br>
# ## Examples of toxic and non-toxic comments

# In[ ]:


pd.options.display.max_colwidth=300
# Print the most severe comments (with target value greater than 0.8) 
# to get a feeling for the data but also on the data structure
for n, v in enumerate(train_data.loc[train_data.target>0.8, 'comment_text']):
    print(n, ': ', v)
    if n == 10:
        break


# In[ ]:


pd.options.display.max_colwidth=300
# Print the non-toxic comments 
for n, v in enumerate(train_data.loc[train_data.target==0.0, 'comment_text']):
    print(n, ': ', v)
    if n == 10:
        break


# <a id="9"></a> <br>
# ## Distribution of the Target Variable

# In[ ]:


# Plot the number of comments over time in the training dataset.
cnt_srs = train_data['created_date'].dt.date.value_counts()
cnt_srs = cnt_srs.sort_index()
plt.figure(figsize=(14,6))
sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8, color='blue')
plt.xticks(rotation='vertical')
plt.xlabel('Creation Date', fontsize=12)
plt.ylabel('Number of comments', fontsize=12)
plt.title("Number of comments over time in the train set")
plt.show()


# In[ ]:


# Plot the number of toxic comments over time in the training dataset.
cnt_srs = train_data.loc[train_data['target']>=0.5,'created_date'].dt.date.value_counts()
cnt_srs = cnt_srs.sort_index()
plt.figure(figsize=(14,6))
sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8, color='green')
plt.xticks(rotation='vertical')
plt.xlabel('Creation Date', fontsize=12)
plt.ylabel('Number of comments', fontsize=12)
plt.title("Number of comments over time in train set")
plt.show()


# In[ ]:


# Plot the average toxicity over time.

toxicity_gb_time = train_data[train_data['target']>0.5]['target'].groupby(train_data['created_date']).count()/train_data['target'].groupby(train_data['created_date']).count()
#print(toxicity_gb_time)
toxicity_gb_time = toxicity_gb_time.fillna(0)

toxicity_gb_time = toxicity_gb_time.sort_index()

plt.figure(figsize=(14,6))
#sns.lineplot(x=toxicity_gb_time.index, y=toxicity_gb_time.values, label='target')
plt.plot(toxicity_gb_time.index, toxicity_gb_time.values, marker='o', linestyle='-', linewidth=2, markersize=0)

plt.xticks(rotation=45)
plt.xlabel('Creation Date', fontsize=12)
plt.ylabel('Ratio of Toxic Comments', fontsize=12)
plt.title("Ratio of Toxic Comments over Time")
plt.show()


# The time frames that havea ratio of toxic comments of 0, can be explained by the very low number of comments during these periods.This can also be seen on the graph that shows the number of comments over time.

# In[ ]:


#Plot distribution of the target variable
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
ax1.hist(train_data.target, bins=2)
ax1.set_title('Distribution of the target variable')
ax1.set_ylabel('count')

sns.distplot(train_data.target, ax=ax2)
#ax2.hist(train_data.target)
ax2.set_title('Distribution of the target variabl')
ax2.set_ylabel('count')

plt.show()


# The major amount of comments is clearly non toxic (~95%) according to the training data. When exploring the distribution of the target variable results, it looks sort of categorical. The reason for that is the method how toxicity was evaluated. A single comment was evaluated by up to 5 voters saying it is toxic (1) or non-toxic (0). Calculating the average out of these votes explains the distribution of the toxicity variable. Considering this nature of distribution, both trees and neural networks seem to be a more resonable choice.  Also Multinomial regression can be tried.  Logistic regression and only work with a binary input variable, creating a propability distribution, which can be used to decide on the binary classification.

# <a id="10"></a> <br>
# ## Distribution of Additional Columns

# The training data contains additional information around the comment, which may help to explain toxicity. The testset however does not include this data. We will thus decide based on the EDA if we further investigate methods to include the data into prediction, or if we remove the additional columns for additional model creation.

# In[ ]:


# Analyse the partition of toxicity classification against the full dataset
# When we interpret the accuracy we need to consider the fact that the number
# of toxic vs. non-toxic comments is not equally balanced

columns = ['target','severe_toxicity', 'obscene',
       'identity_attack', 'insult', 'threat']
# Create series with the number of comments that only contain toxicity values greater than 0.5
toxic_crit = train_data.loc[:, columns]
toxic_crit = toxic_crit>0.5
toxic_crit = toxic_crit.sum()
toxic_crit = toxic_crit.sort_values(ascending = False)
gc.collect()


# In[ ]:


# Plot the distribution of the comments with toxicity values greater than 0.5
plt.figure(figsize=(14,6))
sns.barplot(toxic_crit.index, toxic_crit.values, alpha=0.8, color='green')
plt.xticks(rotation=45)
plt.xlabel('Criterias', fontsize=12)
plt.ylabel('Number of occurences', fontsize=12)
plt.title("Toxicity annotation greater 0.5")
plt.show()


# In[ ]:


# Plot the distribution of the toxicity and identity annotator count
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
sns.boxplot(train_data.toxicity_annotator_count, ax=ax1)
ax1.set_title('Distribution of the toxicity_annotator_count')
ax1.set_ylabel('count')
sns.boxplot(train_data.identity_annotator_count, ax=ax2)
#ax2.hist(train_data.identity_annotator_count)
ax2.set_title('Distribution of the identity_annotator_count')
ax2.set_ylabel('count')
plt.show()


# In[ ]:


import matplotlib.gridspec as gridspec

# Create 3x2 sub plots
gs = gridspec.GridSpec(3, 2)

fig = plt.figure(figsize=(14, 18))
ax1 = fig.add_subplot(gs[0, 0]) # row 0, col 0
ax2 = fig.add_subplot(gs[0, 1]) # row 0, col 1
ax3 = fig.add_subplot(gs[1, 0]) # row 1, col 0
ax4 = fig.add_subplot(gs[1, 1]) # row 1, col 1
ax5 = fig.add_subplot(gs[2, 0]) # row 2, col 0

sns.distplot(train_data['severe_toxicity'],kde=False, hist=True, bins=30, label='severe_toxicity', ax=ax1)
ax1.set_title('Dist. of the severe_toxicity')
sns.distplot(train_data['obscene'],kde=False, hist=True, bins=30, label='obscene', ax=ax2)
ax2.set_title('Dist. of the obscene')
sns.distplot(train_data['identity_attack'],kde=False, hist=True, bins=30, label='identity_attack', ax=ax3)
ax3.set_title('Dist. of the identity_attack')
sns.distplot(train_data['insult'],kde=False, hist=True, bins=30, label='insult', ax=ax4)
ax4.set_title('Dist. of the insult')
sns.distplot(train_data['threat'],kde=False, hist=True, bins=30, label='threat', ax=ax5)
ax5.set_title('Dist. of the threat')

plt.show()


# The vast majority of the comments have a value a little greater than 0 in any of the toxicity features.
# 
# What we can notice is that insult has more values above 0 than any of the other toxicity features. Maybe this is just because it is easier to insult without intent, whereas threats, identity attackts and obscenity tend to always have an intent behind.

# <a id="11"></a> <br>
# ## Identity Columns

# <a id="12"></a> <br>
# ### Information About the Additional Columns

# In[ ]:


id_attr = ['male','female','transgender','other_gender','heterosexual','homosexual_gay_or_lesbian','bisexual', 'other_sexual_orientation', 'christian','jewish','muslim','hindu','buddhist',
   'atheist','other_religion','black','white', 'asian','latino','other_race_or_ethnicity','physical_disability','intellectual_or_learning_disability','psychiatric_or_mental_illness','other_disability']


# In addition to the labels described above, the dataset also provides metadata from Jigsaw's annotation: 
# toxicity_annotator_count and identity_annotator_count and metadata from Civil Comments: 
# 
# created_date, publication_id, parent_id, article_id, rating, funny, wow, sad, likes, disagree. 
# 
# Civil Comments' label rating is the civility rating Civil Comments users gave the comment.

# In[ ]:


# Analyse the partition of toxicity classification against the full dataset
# When we interpret the accuracy we need to consider the fact that the number
# of toxic vs. non-toxic comments is not equally balanced

other_attr = train_data.loc[:, id_attr]
other_attr = other_attr>0.5
other_attr = other_attr.sum()
other_attr = other_attr.sort_values(ascending = False)


# In[ ]:


#cnt_srs = test_df['first_active_month'].dt.date.value_counts()
#cnt_srs = cnt_srs.sort_index()

# Plot the distribution of the comments with identity values greater than 0.5
plt.figure(figsize=(14,6))
sns.barplot(other_attr.index, other_attr.values, alpha=0.8, color='green')
plt.xticks(rotation='vertical')
plt.xlabel('Criterias', fontsize=12)
plt.ylabel('Number of occurences', fontsize=12)
plt.title("Distribution of other attribute annotation greater 0.5")
plt.show()


# The identity features with the highest occurrencies are: female, male, black, christian, muslim and heterosexual_gay_or_lesbian. Maybe because these are attributes most commonly discussed in the media. 

# In[ ]:


# How many entries (comments) have identity features that are all 0 or NaN?
person_cat = train_data
person_cat_nan = person_cat[(np.isnan(person_cat['asian']) | (person_cat['asian'] == 0.0))
                            & (np.isnan(person_cat['atheist']) | (person_cat['atheist'] == 0.0)) 
                            & (np.isnan(person_cat['bisexual']) | (person_cat['bisexual'] == 0.0)) 
                            & (np.isnan(person_cat['black']) | (person_cat['black'] == 0.0)) 
                            & (np.isnan(person_cat['buddhist']) | (person_cat['buddhist'] == 0.0)) 
                            & (np.isnan(person_cat['christian']) | (person_cat['christian'] == 0.0)) 
                            & (np.isnan(person_cat['female']) | (person_cat['female'] == 0.0)) 
                            & (np.isnan(person_cat['heterosexual']) | (person_cat['heterosexual'] == 0.0)) 
                            & (np.isnan(person_cat['hindu']) | (person_cat['hindu'] == 0.0)) 
                            & (np.isnan(person_cat['homosexual_gay_or_lesbian']) | (person_cat['homosexual_gay_or_lesbian'] == 0.0)) 
                            & (np.isnan(person_cat['intellectual_or_learning_disability']) | (person_cat['intellectual_or_learning_disability'] == 0.0)) 
                            & (np.isnan(person_cat['jewish']) | (person_cat['jewish'] == 0.0)) 
                            & (np.isnan(person_cat['latino']) | (person_cat['latino'] == 0.0)) 
                            & (np.isnan(person_cat['male']) | (person_cat['male'] == 0.0)) 
                            & (np.isnan(person_cat['muslim']) | (person_cat['muslim'] == 0.0)) 
                            & (np.isnan(person_cat['other_disability']) | (person_cat['other_disability'] == 0.0)) 
                            & (np.isnan(person_cat['other_gender']) | (person_cat['other_gender'] == 0.0)) 
                            & (np.isnan(person_cat['other_race_or_ethnicity']) | (person_cat['other_race_or_ethnicity'] == 0.0)) 
                            & (np.isnan(person_cat['other_religion']) | (person_cat['other_religion'] == 0.0)) 
                            & (np.isnan(person_cat['other_sexual_orientation']) | (person_cat['other_sexual_orientation'] == 0.0)) 
                            & (np.isnan(person_cat['physical_disability']) | (person_cat['physical_disability'] == 0.0))
                            & (np.isnan(person_cat['psychiatric_or_mental_illness']) | (person_cat['psychiatric_or_mental_illness'] == 0.0)) 
                            & (np.isnan(person_cat['transgender']) | (person_cat['transgender'] == 0.0)) 
                            & (np.isnan(person_cat['white']) | (person_cat['white'] == 0.0))]

print("Apparently only " + str((person_cat_nan.shape[0] / train_data.shape[0])*100) + " % of the comments do not have a value greater than 0 in any of the identity columns")


# In[ ]:


# How many of the comments that do not have a value greater than 0 in any of the identity columns,
# have a value greater than 0.5 for toxicity? 
print("" + str((len(person_cat_nan[person_cat_nan['target']>0.5]['target'])/person_cat_nan.shape[0])*100) + " % of the comments that do not have a value greater than 0 in any of the identity columns, are toxic.")


# In[ ]:


# How many of the comments that do not have a value greater than 0 in any of the identity columns,
# have a value greater than 0.5 for identity_attack? 
len(person_cat_nan[person_cat_nan['identity_attack']>0.5]['identity_attack'])

print("Only " + str((len(person_cat_nan[person_cat_nan['identity_attack']>0.5]['identity_attack'])/person_cat_nan.shape[0])*100) + " % of the comments that do not have a value greater than 0 in any of the identity columns,\nhave a value greater than 0.5 for identity_attack.")


# In[ ]:


del person_cat_nan
gc.collect()


# In[ ]:


person_cat = person_cat[((person_cat['asian'] > 0.0))# & person_cat[person_cat.asian.notnull()])
                        | ((person_cat['atheist'] > 0.0))
                        | (person_cat['bisexual'] > 0.0) 
                        | (person_cat['black'] > 0.0)
                        | (person_cat['buddhist'] > 0.0)
                        | (person_cat['christian'] > 0.0)
                        | (person_cat['female'] > 0.0)
                        | (person_cat['heterosexual'] > 0.0)
                        | (person_cat['hindu'] > 0.0)
                        | (person_cat['homosexual_gay_or_lesbian'] > 0.0)
                        | (person_cat['intellectual_or_learning_disability'] > 0.0)
                        | (person_cat['jewish'] > 0.0)
                        | (person_cat['latino'] > 0.0)
                        | (person_cat['male'] > 0.0)
                        | (person_cat['muslim'] > 0.0)
                        | (person_cat['other_disability'] > 0.0)
                        | (person_cat['other_gender'] > 0.0)
                        | (person_cat['other_race_or_ethnicity'] > 0.0)
                        | (person_cat['other_religion'] > 0.0)
                        | (person_cat['other_sexual_orientation'] > 0.0)
                        | (person_cat['physical_disability'] > 0.0)
                        | (person_cat['psychiatric_or_mental_illness'] > 0.0)
                        | (person_cat['transgender'] > 0.0)
                        | (person_cat['white'] > 0.0)]

person_cat.shape
print("" + str((person_cat.shape[0] / train_data.shape[0])*100) + " % of the comments have a value greater than 0 in at least one of the identity columns")


# In[ ]:


print("" + str(((len(person_cat[person_cat['target']>0.5]['target']) / person_cat.shape[0])*100)) + " % of the comments that have a value greater than 0 in any of the identity columns, are toxic.")


# As we can see not so many of the entries have values in the indentity columns. These are about 12 % of all the entries and almost 10 % of these entries are toxic. Out of the columns that do not have values greater than 0 in any of the identity columns, only 5 % of them are toxic and 0.05 % have are market as an identity attack (value > 0.5). Even though these ratios seem small, they are meaningful numbers considering that the amount of toxic comments compared to non-toxic ones is small.  

# In[ ]:


# Explanation
# categories = ['target']+list(train_data)[slice(8,32)]
# SUM(x*y)/COUNT(identity_col > 0)
# categories.iloc[:, 1:] multiply all the identity columns  
# categories.iloc[:, 0]  with the target column
# categories.iloc[:, 1:].multiply(categories.iloc[:, 0], axis="index").sum()  create the sum
# categories.iloc[:, 1:][categories.iloc[:, 1:]>0].count()    devide by all the number of the identity values that are bigger than 0

# Percent of toxic comments related to different identities, 
# using target and popolation amount of each identity as weights:

categories = train_data.loc[:, ['target']+list(train_data)[slice(8,32)]].dropna() # take the column target and all the categorizing columns
# categories.iloc[:, 0] --> target column
# categories.iloc[:, 1] --> all identity columns
weighted_toxic = categories.iloc[:, 1:].multiply(categories.iloc[:, 0], axis="index").sum()/categories.iloc[:, 1:][categories.iloc[:, 1:]>0].count()
weighted_toxic = weighted_toxic.sort_values(ascending=False)
plt.figure(figsize=(10,8))
sns.set(font_scale=1)
ax = sns.barplot(x = weighted_toxic.values, y = weighted_toxic.index, saturation=0.5, alpha=0.99)
plt.ylabel('Categories')
plt.xlabel('Weighted Toxic')
plt.show()


# In[ ]:


# Percent of identity_attack related to different identities, 
# using target and popolation amount of each identity as weights:

categories = train_data.loc[:, ['identity_attack']+list(train_data)[slice(8,32)]] #.dropna() # take the column target and all the categorizing columns
# categories.iloc[:, 0] --> identity_attack column
# categories.iloc[:, 1] --> all identity columns
weighted_identity_attack = categories.iloc[:, 1:].multiply(categories.iloc[:, 0], axis="index").sum()/categories.iloc[:, 1:][categories.iloc[:, 1:]>0].count()
weighted_identity_attack = weighted_identity_attack.sort_values(ascending=False)
plt.figure(figsize=(10,8))
sns.set(font_scale=1)
ax = sns.barplot(x = weighted_identity_attack.values, y = weighted_identity_attack.index, saturation=0.5, alpha=0.99)
plt.ylabel('categories')
plt.xlabel('weighted identity_attack')
plt.show()


# As we can see the percent of toxic comments related to different identities is higher for  the identities: black, homosexual_gay_or_lesbian, white, muslim, jewish, atheist and so on. Maybe this is due to the fact that these identities are more common in society and as they are getting more media attention. 

# In[ ]:


#Let's see if there are correlations between the identity columns in train_data and the target:

corrs = np.abs((person_cat.loc[:, ['target']+list(person_cat)[slice(8,32)]]).corr())
ordered_cols = (corrs).sum().sort_values().index
np.fill_diagonal(corrs.values, 0)
plt.figure(figsize=[8,8])
plt.imshow(corrs.loc[ordered_cols, ordered_cols], cmap='plasma', vmin=0, vmax=1)
plt.colorbar(shrink=0.7)
plt.xticks(range(corrs.shape[0]), list(ordered_cols), size=16, rotation=90)
plt.yticks(range(corrs.shape[0]), list(ordered_cols), size=16)
plt.title('Heat map of coefficients of correlation between identity categories and target', fontsize=17)
plt.show()


# As we can see on the heatmap, there is no significant correlation between the columns. There might nonetheless be a very small correlations between some identity column.

# In[ ]:


# Distribution of race and ethnicity

# Create 3x2 sub plots
import matplotlib.gridspec as gridspec

gs = gridspec.GridSpec(3, 2)

fig = plt.figure(figsize=(14, 18))
ax1 = fig.add_subplot(gs[0, 0]) # row 0, col 0
ax2 = fig.add_subplot(gs[0, 1]) # row 0, col 1
ax3 = fig.add_subplot(gs[1, 0]) # row 1, col 0
ax4 = fig.add_subplot(gs[1, 1]) # row 1, col 1
ax5 = fig.add_subplot(gs[2, 0]) # row 2, col 0
ax6 = fig.add_subplot(gs[2, 1]) # row 2, col 1

sns.distplot(person_cat['asian'],kde=False, hist=True, bins=30, label='asian', ax=ax1)
ax1.set_title('Dist. of the asian')
sns.distplot(person_cat['black'],kde=False, hist=True, bins=30, label='black', ax=ax2)
ax2.set_title('Dist. of the black')
sns.distplot(person_cat['jewish'],kde=False, hist=True, bins=30, label='jewish', ax=ax3)
ax3.set_title('Dist. of the jewish')
sns.distplot(person_cat['latino'],kde=False, hist=True, bins=30, label='latino', ax=ax4)
ax4.set_title('Dist. of the atino')
sns.distplot(person_cat['other_race_or_ethnicity'],kde=False, hist=True, bins=30, label='other_race_or_ethnicity', ax=ax5)
ax5.set_title('Dist. of the other_race_or_ethnicity')
sns.distplot(person_cat['other_race_or_ethnicity'],kde=False, hist=True, bins=30, label='other_race_or_ethnicity', ax=ax6)
ax6.set_title('Dist. of the other_race_or_ethnicity')

plt.show()


# Even in this case where the comments have a value of 0 or a little higher than 0 in at least one of the identity columns, the majority of the comments have a value a little greater than 0 in the ethnicity features. It should be considered which value is high enough to influence our results.

# In[ ]:


# Plot ethnicity features over time.

# Extract month and year from created_date and aggregate

# create dataframe with all the identity columns, target and creation date
withdate = train_data.loc[:, ['created_date', 'target']+list(train_data)[slice(8,32)]].dropna() 
# weight the identity scores, dividing each value by the sum of the whole column
raceweighted = withdate.iloc[:, 2:]/withdate.iloc[:, 2:].sum()  
# Multipy the raceweight columns witht the target
race_target_weighted = raceweighted.multiply(withdate.iloc[:, 1], axis="index")
# Create created_date column
race_target_weighted['created_date'] = pd.to_datetime(withdate['created_date']).values.astype('datetime64[M]')
# Group by the creation date
weighted_demo = race_target_weighted.groupby(['created_date']).sum().sort_index()


# In[ ]:


plt.figure(figsize=(14,6))
#sns.lineplot(x=toxicity_gb_time.index, y=toxicity_gb_time.values, label='target')
plt.plot(weighted_demo.index, weighted_demo['white'], marker='o', linestyle='-', linewidth=2, markersize=0)
plt.plot(weighted_demo.index, weighted_demo['asian'], marker='o', linestyle='-', linewidth=2, markersize=0)
plt.plot(weighted_demo.index, weighted_demo['black'], marker='o', linestyle='-', linewidth=2, markersize=0)
plt.plot(weighted_demo.index, weighted_demo['jewish'], marker='o', linestyle='-', linewidth=2, markersize=0)
plt.plot(weighted_demo.index, weighted_demo['latino'], marker='o', linestyle='-', linewidth=2, markersize=0)
plt.plot(weighted_demo.index, weighted_demo['other_race_or_ethnicity'], marker='o', linestyle='-', linewidth=2, markersize=0)

plt.xticks(rotation=45)
plt.xlabel('Creation Date', fontsize=12)
#plt.ylabel('Ratio of Toxic Comments', fontsize=12)
plt.title("Time Series Toxicity & Race")
plt.legend()
plt.show()


# The graph shows the identity scores of the comments over time of the ethnicitiy groups. White and Black seem to be the groups with the higher scores in these columns. 

# In[ ]:


del toxic_crit, fig,ax1, ax2, other_attr, person_cat
del weighted_identity_attack
del weighted_toxic
del race_target_weighted
del raceweighted
gc.collect


# In[ ]:


# Print out all the variables in use
# %whos


# <a id="13"></a> <br>
# ## Exploration of the ID-Columns

# In[ ]:


train_ids = train_data[['created_date','id', 'publication_id', 'parent_id', 'article_id', 'target']]
train_ids.head(10)


# In[ ]:


# Check missing values
# Only parent_id seems to have missing values
train_ids.isnull().sum()


# Overalll the id columns are well maintained, however the parent ID has quite some missing values. We will thus exclude it from our EDA.

# In[ ]:


# HOW MANY BAD COMMENTS DIVIDED BY ALL COMMENTS PER PUBLICATION_ID (RATIO)
plt.figure(figsize=(14,6))
# Group by publication_id and count the number of comments with toxcicty (over 0.5) per publication_id group.
y = train_data[train_data['target']>0.5]['target'].groupby(train_data['publication_id']).count()/train_data['target'].groupby(train_data['publication_id']).count()
# Plot the values in descending order 
sns.barplot(y.index, y.sort_values(ascending =False), color='green')
plt.ylabel('Number of Toxic Comments', fontsize=12)
plt.title("Distribution of Toxic Comments over publication_id")


# Looking at the distribution of the share of toxic versus non-toxic comments, we can see that there are some publications where the share of toxic comments is considered to be much higher than for ofther publications. This information may be a helpful feature for toxic vs. non toxic calssification.

# In[ ]:


# Convert the data type of 'created date' column
#train_data['created_date'] = pd.to_datetime(train_data['created_date']).values.astype('datetime64[M]')
#train_data['created_date'].head()


# In[ ]:


# Plot the distribution of the ID-s over time 
plt.figure(figsize=(16, 6))
sns.lineplot(x='created_date', y='id', label='IDs', data=train_data)
plt.title('Distribution of IDs over years')


# As we can see in the year there is a boost of ID-s at certain points in time. This is probably due to political/social events and is also related to the number of comments since each comment has a unique id.

# <a id="14"></a> <br>
# ## Reaction Columns

# In the following some exploratory data analysis will be performed. This part only looks at the following columns "target", "wow", "funny", "sad" and "like". Therefore the data frame is filtered to only contain the mentioned columns.

# In[ ]:


train_eda = train_data.loc[:, ["target", "sad", "wow", "funny", "likes", "disagree"]]
train_eda[train_eda['target']>0].head()


# Inspecting the head() of the first comments that also comments considered as toxic got some 'likes' but no 'dislikes', which is surprising in a way. Le's inspect the correlation.

# 
# 
# In the following the data frame is described. As the table contains rather large values there can't be drawn clear conclusions from this.
# 

# In[ ]:


train_eda.describe()


# 
# 
# There are no missing values in these columns.
# 

# In[ ]:


train_eda.isnull().sum()


# A heatmap is created to take a look at the pairwise correlations. The strongest relations exists between "sad" and "disagree" which can be explained by the fact that the content of comments that people disagree with probably make them sad as well. 
# Looking at the correlation to the target variable the correlation is very weak. As the data is not coming from Facebook, Instagram or similar platforms where up- and downvoting is popular. This can be an explanation for a) the low number of votes and b) the weak correlation. 
# 
# Overall,any correlation is below 0.3 and therefore it is not really relevant for our further analysis/seems to not really help for the classification of toxicity.

# In[ ]:


colormap = plt.cm.RdBu
plt.figure(figsize=(12,12))
sns.heatmap(train_eda.corr(), linewidths=0.1, vmax=1.0, vmin=-1., square=True, cmap=colormap, linecolor='white', annot=True)
plt.title('Pair-wise correlation')


# Another way to visualize the observations made above is a pairplot. Again the same observations can be made.

# In[ ]:


eda_wow = train_eda[train_eda["wow"] >= 1]
eda_wow.drop(["sad", "disagree", "funny", "likes"], axis=1).describe()


# In[ ]:


eda_funny = train_eda[train_eda["funny"] >= 1]
eda_funny.drop(["sad", "wow", "disagree", "likes"], axis=1).describe()


# In[ ]:


eda_sad = train_eda[train_eda["sad"] >= 1]
eda_sad.drop(["disagree", "wow", "funny", "likes"], axis=1).describe()


# In[ ]:


eda_likes = train_eda[train_eda["likes"] >= 1]
eda_likes.drop(["sad", "wow", "funny", "disagree"], axis=1).describe()


# In[ ]:


eda_disagree = train_eda[train_eda["disagree"] >= 1]
eda_disagree.drop(["sad", "wow", "funny", "likes"], axis=1).describe()


# In[ ]:


del columns, corrs, eda_likes, eda_sad, eda_wow, eda_funny
del eda_disagree
del train_eda
gc.collect


# <a id="15"></a> <br>
# ## Conclusion on Additional Columns

# Overall, it seems that the additional data, coming along with the target variable and the comment, seems not really contributing to predict the toxicity. We thus decide on excluding the information for the following preprocessing and modeling. This does not mean that for other datasets from other platforms will lead to the same conclusions, here we can well imagine that the reaction fields as well as publication ids may help to predict. For sure considering publication id in a model must be well considered as it may raise troubles regarding neutrality.

# <a id="16"></a> <br>
# ## Comment Column

# In[ ]:


get_ipython().system('pip install nltk')
import nltk
import re

# Import the English language class
from spacy.lang.en import English

nltk.download('punkt')
nltk.download('stopwords')

# Import Counter
from collections import Counter
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
stop_words = stopwords.words('english')


get_ipython().system('pip install gensim')
from nltk.corpus import stopwords  
import gensim 
from gensim.utils import simple_preprocess 
from gensim.parsing.preprocessing import STOPWORDS 
from nltk.stem import WordNetLemmatizer, SnowballStemmer 
from nltk.tokenize import word_tokenize


# <a id="17"></a> <br>
# ### Visualize Most Common Words

# In[ ]:


# Tokenize the comment text of toxic comments
tok_comments = [word_tokenize(com) for com in train_data[train_data['target']>0.5]['comment_text']]


# In[ ]:


# Remove stopwords
tokens = [[w for w in s if (w not in stop_words) & (len(w)>2)] for s in tok_comments]


# In[ ]:


from nltk.probability import FreqDist


# In[ ]:


#plot the most frequent words
tokens = np.array([np.array(s) for s in tokens])
fdist = FreqDist(np.concatenate(tokens))
fdist.plot(30,cumulative=False)
plt.show()


# In[ ]:


#plot the most frequent word pairs
from nltk import bigrams, ngrams
bigrams_tokens = bigrams(np.concatenate(tokens))
fdist_bigrams = FreqDist(list(bigrams_tokens))
fdist_bigrams.plot(30,cumulative=False)
plt.show()


# The above graphs visualize the most common words and word-groups in the toxic comments in an unfiltered way (words such as "the", are also included). As we can see some of the words, such as Trump, United States, country and so on hint at political disagreements.

# In[ ]:


def print_top_20_words(comment_text):
  
  # delete the numbers from the string
  #comment_text = re.sub(r'\d+', '', comment_text)

  # Create the nlp object
  nlp = English()

  # Process a text
  doc = nlp(comment_text)

  # Print the document text
  # print(doc.text)

  # Tokenize the article: tokens
  tokens = word_tokenize(doc.text)

  # Convert the tokens into lowercase: lower_tokens
  lower_tokens = [t.lower() for t in tokens]

  # Remove stopwords
  
  lower_tokens = [word for word in lower_tokens if word not in stopwords.words('english')]
  lower_tokens = [word.lower() for word in lower_tokens if word.isalpha()]


  # Create a Counter with the lowercase tokens: bow_simple
  bow_simple = Counter(lower_tokens)
  
  # Print the 20 most common tokens
  # print(bow_simple.most_common(10))
  d = dict()
  for elem in bow_simple.most_common(20):
    print(elem[1], elem[0])
    d[elem[0]] = elem[1]
    
  return d


# In[ ]:


downsampled_train_data = pd.read_csv('../input/train.csv',dtype=dtypesDict_tr,parse_dates=['created_date'], nrows=200000)


# In[ ]:


# Turn the whole comment_text column into list of strings (text)
comment_text = downsampled_train_data[downsampled_train_data['target'] >0.5]['comment_text'].to_string()

# List of top 20 most used words in comments with toxicity (target) higher than 0.7
# Work in chunks because nlp cannot process more than 1 Million characters
first_chunk = print_top_20_words(comment_text[:(len(comment_text)//4)])  # // devision returns a natural number without the rest
print('\n\n')
second_chunk = print_top_20_words(comment_text[(len(comment_text)//4+1):(2*len(comment_text)//4)])
print('\n\n')
third_chunk = print_top_20_words(comment_text[(2*len(comment_text)//4+1):(3*len(comment_text)//4)])
print('\n\n')
fourth_chunk = print_top_20_words(comment_text[(3*len(comment_text)//4+1):])


# In[ ]:


# Build an average out of the 2 dictionaries

newd1 = {}
for key in first_chunk.keys():
    for key2 in second_chunk.keys():
        if key in second_chunk.keys():
            newd1[key] = int(first_chunk.get(key)) + int(second_chunk.get(key))
        else:
            newd1[key] = first_chunk.get(key)
        if key2 not in first_chunk.keys():
            newd1[key2] = second_chunk.get(key2)

newd2 = {}   
for key in third_chunk.keys():
    for key2 in fourth_chunk.keys():
        if key in fourth_chunk.keys():
            newd2[key] = int(third_chunk.get(key)) + int(fourth_chunk.get(key))
        else:
            newd2[key] = third_chunk.get(key)
        if key2 not in third_chunk.keys():
            newd2[key2] = fourth_chunk.get(key2)

newd = {}
for key in newd1.keys():
    for key2 in newd2.keys():
        if key in newd2.keys():
            newd[key] = int(newd1.get(key)) + int(newd2.get(key))
        else:
            newd[key] = newd1.get(key)
        if key2 not in newd1.keys():
            newd[key2] = newd2.get(key2)


# Sort dictionary numerically descending
print("Top 20 most used words in comments with toxicity (target) higher than 0.5:")
i = 0
for key, value in sorted(newd.items(), key=lambda item: item[1], reverse=True):
    if i < 20:
        print("%s: %s" % (key, value))
    i = i + 1


# In[ ]:


# List of top 20 most used words in comments with identity attack higher than 0.5
comment_text = downsampled_train_data[downsampled_train_data['identity_attack'] >0.5]['comment_text'].to_string()
print("Top 20 most used words in comments with identity attack higher than 0.5:")
dict_identity_attack = print_top_20_words(comment_text)


# In[ ]:


from wordcloud import WordCloud ,STOPWORDS
from PIL import Image


# In[ ]:


def toxicwordcloud(dict1, subset=train_data[train_data.target>0.5], title = "Words Frequented"):
    stopword=set(STOPWORDS)
    wc= WordCloud(background_color="black",max_words=4000,stopwords=stopword)
    wc.generate(" ".join(list(dict1.keys())))
    plt.figure(figsize=(8,8))
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    plt.title(title, fontsize=20)
    plt.imshow(wc.recolor(colormap= 'gist_earth' , random_state=244), alpha=0.98)


# In[ ]:


toxicwordcloud(newd)


# In[ ]:


toxicwordcloud(dict_identity_attack)


# In[ ]:


del downsampled_train_data
gc.collect()


# After tokenizing the text of toxic comments, we see that many of the most used words, also refer to ethnic groups, religions, sex, sexual orientation, political views and so on. Hence, the toxic comments seem like an insult to different identity groups.

# **Comment length**

# The comment length has a pretty big variety of values, starting from very short ones to much longer ones (around 200 characters).

# In[ ]:


words = train_data["comment_text"].apply(lambda x: len(x) - len(''.join(x.split())) + 1)

train_data['words'] = words
words_toxic = train_data.loc[(train_data['words']<200)&(train_data['target'] >0.7)]['words']
words_identity_attack = train_data.loc[(train_data['words']<200)&(train_data['target'] >0.7)&(train_data['identity_attack'] >0.7)]['words']
words_threat = train_data.loc[(train_data['words']<200)&(train_data['target'] >0.7)&(train_data['threat'] >0.7)]['words']
words_nontoxic = train_data.loc[(train_data['words']<200)&(train_data['target'] <0.3)]['words']

sns.set()
plt.figure(figsize=(12,6))
plt.title("Comment Length (words)")
sns.distplot(words_toxic,kde=True,hist=False, bins=120, label='toxic')
sns.distplot(words_identity_attack,kde=True,hist=False, bins=120, label='identity_attack')
sns.distplot(words_threat,kde=True,hist=False, bins=120, label='threat')
sns.distplot(words_nontoxic,kde=True,hist=False, bins=120, label='nontoxic')
plt.legend(); plt.show()


# In[ ]:


# Create new features
train_data['total_length'] = train_data['comment_text'].apply(len)
train_data['capitals'] = train_data['comment_text'].apply(lambda comment: sum(1 for c in comment if c.isupper()))
train_data['num_exclamation_marks'] = train_data['comment_text'].apply(lambda comment: comment.count('!'))
train_data['num_question_marks'] = train_data['comment_text'].apply(lambda comment: comment.count('?'))
train_data['num_symbols'] = train_data['comment_text'].apply(lambda comment: sum(comment.count(w) for w in '*&$%'))
train_data['num_words'] = train_data['comment_text'].apply(lambda comment: len(comment.split()))
train_data['num_smilies'] = train_data['comment_text'].apply(lambda comment: sum(comment.count(w) for w in (':-)', ':)', ';-)', ';)')))


# In[ ]:


# Correlations between new features and the toxicity features
features = ('total_length', 'capitals', 'num_exclamation_marks','num_question_marks', 'num_symbols', 'num_words', 'num_smilies')
columns = ('target', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat')
rows = [{c:train_data[f].corr(train_data[c]) for c in columns} for f in features]
train_correlations = pd.DataFrame(rows, index=features)


# **Correlations between new features and the toxicity features**

# In[ ]:


train_correlations


# As we can see there is no significant correlation between the new features and the toxicity features. Maybe if we take the number of emojis as well, these values will show something different. 
