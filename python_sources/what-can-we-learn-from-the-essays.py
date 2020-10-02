#!/usr/bin/env python
# coding: utf-8

# # Motivation
# - What can we learn about the types of projects that get approved for DonorsChoose from the teacher submitted essays?
# - Can teachers submitting projects to DonorChoose gain any insights about what to submit to increase their chances of approval?
# - Are there any particular topics that stand out in the projects that get approved?
# - What about essay length? Vocabulary? Spelling errors?

# # Contents

# PART I
# - Clean Text Data
# - Consolidate Resources (Description & Price)
# 
# PART II
# - Explore Rejected Project Proposals
# - A Tangent: Impact of Resource Cost
# - Vocabulary Length and Diversity
# - Full Corpus Tokens Exploration
# - Token Frequency
# - Top 20 Tokens
# 
# Note: Every section in the second part has a **~ SUMMARY~** section that summarizes the results. Whenever it isn't obvious I have included a legend, but I kept the colors consistent with blue representing the accepted proposals and orange representing the rejected proposals.

# ---

# # PART I

# #### Modules

# In[1]:


# Import initial modules
import numpy as np 
import pandas as pd 
import seaborn as sns
import warnings

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

warnings.filterwarnings('ignore')


# #### Load Files

# In[2]:


import os
file_path = '../input/'

# # # Running the notebook locally:
# # Change Directory
# os.chdir('/Users/paula/Dropbox/15. Kaggle/DonorsChoose/Data')
# print(os.getcwd()) 
# file_path = ''

# Load the data into multiple dataframes
train_full = pd.read_csv(file_path + 'train.csv')
test = pd.read_csv(file_path + 'test.csv')
resources = pd.read_csv(file_path + 'resources.csv')
submission = pd.read_csv(file_path + 'sample_submission.csv')


# #### Smaller Sample Size

# In[3]:


# Reduce the dataframe for fast compute time (10% of the entire df)
sample_size = round(len(train_full)*.10)
train = train_full.sample(sample_size, random_state = 1)


# ## Clean Text Data

# #### Text Function
#     (1) Create a combined column of all the other text columns
#     (2) Remove any special characters from the text column

# In[4]:


def clean_input_data(df, combined_col_name, text_column_list, 
                     character_remove_list, character_replace_list, replace_numbers):
    
    # (1) Combine all text data into one column, separated by spaces
    
    # Remove NaN's in columns
    df.replace(np.nan, '', regex=True, inplace = True)
    
    # Join all text columns into one
    df[combined_col_name] = ''
    for t in range(0, len(text_column_list)):
        df[combined_col_name] += df[text_column_list[t]] + ' '
    # Replace all numbers with a space 
    if replace_numbers == 1:
        df[combined_col_name] = df[combined_col_name].replace('\d+', ' ', regex=True)

    
    # (2) Remove special characters from the text data
    for i in range(0,len(character_remove_list)):
        df[combined_col_name] = df[combined_col_name].apply(
            lambda x: x.replace(character_remove_list[i], character_replace_list[i])
        )


# ## Consolidate Resources
# Combine the resources df with the main train df, since the resources df has extra text data we might want to include in our model.

# In[5]:


resources.head()


# #### Combine Resource Decription

# In[6]:


# Reset index to the id
res = resources.set_index('id')
res['description'] = res['description'].astype(str)

# Join all the resource descriptions into one string, with a space
get_ipython().run_line_magic('time', "rs = res.groupby('id')['description'].apply(lambda x: ' '.join(x))")

# Convert rs into a DataFrame
df_rs = (pd.DataFrame(rs)).reset_index()
df_rs.head()


# #### Combine Resource Prices

# In[7]:


df_rsp = pd.DataFrame(res.groupby('id')['price'].sum()).reset_index()
df_rsp.head()


# #### Join Aggregated Resources DF's to Main DF

# In[8]:


# Join the aggregated resources definitions to the main df
df = pd.merge(train, df_rs, on='id',  how='left')

# Join the aggregated resources definitions to the main df
df = pd.merge(df, df_rsp, on='id',  how='left')


# ## Apply Text Clean Up Function

# #### List of all text columns
# - project_essay_1
# - project_essay_2
# - project_essay_3
# - project_essay_4 
# - project_resource_summary
# - project_title
# - description

# #### Input Variables

# In[9]:


combined_col_name = 'full_essay'
text_column_list = ['project_essay_1', 'project_essay_2', 'project_essay_3', 'project_essay_4']
character_remove_list = ['\\r', '\\n', '\\']
character_replace_list = [' ', ' ', ' ']
replace_numbers = 1


# #### Create New Text Column

# In[10]:


# Apply the function to create the cleaned DF and additional columns
clean_input_data(df, combined_col_name, text_column_list, 
                 character_remove_list, character_replace_list, replace_numbers)

# Check the first entry in the dataframe
df.head(1)


# #### Percentage of Approved Projects

# In[11]:


round(df['project_is_approved'].sum()*100/len(df['project_is_approved']),2)


# # PART II

# ## Explore the Text
# Since most projects get accepted, focus on the the essays that get rejected (hypothesis: these proposals will have shorter sentences, grammar mistakes, overall shorter responses, etc.)

# #### Rejected Proposal IDs

# In[12]:


ix_list = df.index[df['project_is_approved']==0].tolist()


# #### EXPLORE: Full Df

# In[13]:


# Set the index of the proposal to explore
ix_num = 2341

df.loc[[ix_list[ix_num]]]


# #### EXPLORE: New Text Column
# Goal: Get an understanding of what is being requested.

# In[14]:


df[df['project_is_approved']==0]['full_essay'][ix_list[ix_num]]


# #### EXPLORE: Resource Summary
# Goal: See if there are any insights that arrise from just reading some resources summaries of accepted and rejected projects.

# In[15]:


df[df['project_is_approved']==0]['project_resource_summary'][ix_list[ix_num]]


# #### EXPLORE: Resource Description
# Goal: What are the item descriptions?

# In[16]:


df[df['project_is_approved']==0]['description'][ix_list[ix_num]]


# #### ~ SUMMARY~
# 
# Some of the things I noticed: 
# - Lying: Resource summary does not match requested items p157746 
# 
# Would love to undertand why certain projects got rejected: 
# - p099141 ask for Osmo Coding Set $50 index = 54
# 
# - p070091 ask for books for kindergarders $200 index = 30
# 
# - p007150 ask for counting games $400 index = 100
# 
# - p009107 ask for cameras $400 index 1000
# 
# What is the impact of the cost of the resources asked for?

# ## A Tangent: Cost

# #### Function Explore Stats by Column

# In[17]:


def describe_by_approval(col_name):
    R = pd.DataFrame(df[df['project_is_approved'] == 0][col_name].describe())
    R[col_name] = round(R[col_name],1)
    R.rename(columns ={col_name: ('Rejected - ' + col_name) }, inplace = True) 

    A = pd.DataFrame(df[df['project_is_approved'] == 1][col_name].describe())
    A[col_name] = round(A[col_name],1)
    A.rename(columns ={col_name: ('Accepted - ' + col_name) }, inplace = True) 
    RA = pd.concat([R, A], axis =1, join = "inner")
    display(RA)


# #### DISTRIBUTION - Total Resource Cost per Proposal

# In[18]:


describe_by_approval('price')


# #### Violin Plots

# In[19]:


# Violin Chart Inputs
y1 = 'price'
y2 = 'price'
data1 = df
data2 = df[df['price'] < 1000]
T1 = 'All Proposal Prices'
T2 = 'Proposals less than $1,000'

# Set Dimension and Color
fig, axes = plt.subplots(1, 2, figsize=(10, 6))
color_set = ['sandybrown','cornflowerblue']

# Graph 1
sns.violinplot(x='project_is_approved', y=y1, data=data1, cut = 0, palette=color_set, ax=axes[0])
# Graph 2
sns.violinplot(x='project_is_approved', y=y2, data=data2, cut = 0, palette=color_set, ax=axes[1])

axes[0].set_title(T1)
axes[1].set_title(T2)

plt.show()


# #### ~ SUMMARY~
# 
# Interestingly, there were some pretty expensive projects that got approved. However, as the earlier exploration eluded to, there seems to be a higher price tag for proposals that get rejected. The median for rejected proposals is $63 dollars more than the accepted proposals. The 75th percentile has an even greater difference. As you can see from the above charts, the approved proposal density plot skews more towards 0, indicating that more projects are cheaper. 

# ## Vocabulary Length and Diversity
# Are essays that are shorter more likely to be rejected? What about the variety of vocab terms?

# #### CountVectorizer

# In[20]:


text = df.full_essay

# Count the token only once from each document (proxy for vocab variety)
from sklearn.feature_extraction.text import CountVectorizer


# #### Document Term Matrix

# In[21]:


# Appearance of unique tokens
vect = CountVectorizer(binary=True)
dtm_token_unique = vect.fit_transform(text)

# Count all appearances of tokens
vect = CountVectorizer()
dtm_token_all = vect.fit_transform(text)


# #### Count Tokens

# In[22]:


# Count the total number of tokens
tu = pd.DataFrame(dtm_token_unique.sum(axis=1))
tu.columns = ['count_unique_tokens']
df = pd.concat([df, tu], axis = 1, join = "inner")


ta = pd.DataFrame(dtm_token_all.sum(axis=1))
ta.columns = ['count_all_tokens']
df = pd.concat([df, ta], axis = 1, join = "inner")


# #### DISTRIBUTION - Unique Tokens per Proposal

# In[23]:


describe_by_approval('count_unique_tokens')


# #### DISTRIBUTION - Total Tokens per Poposal

# In[24]:


describe_by_approval('count_all_tokens')


# In[25]:


# Violin Chart Inputs
y1 = 'count_unique_tokens'
y2 = 'count_all_tokens'
data1 = df
data2 = df
T1 = 'Unique Tokens Per Proposal'
T2 = 'Total Tokens per Proposal'

# Set Dimension and Color
fig, axes = plt.subplots(1, 2, figsize=(10, 6))
color_set = ['sandybrown','cornflowerblue']

# Graph 1
sns.violinplot(x='project_is_approved', y=y1, data=data1, cut = 0, palette=color_set, ax=axes[0])
# Graph 2
sns.violinplot(x='project_is_approved', y=y2, data=data2, cut = 0, palette=color_set, ax=axes[1])

axes[0].set_title(T1)
axes[1].set_title(T2)

plt.show()


# #### ~ SUMMARY~
# 
# As expected, the approved proposals had a greater variety of tokens, with the median above the rejected proposals (126 vs 132). In our sample, the accepted proposals had a minimum of 70 unique tokens. In terms of essay word length, the accepted proposals also tended to be longer. 

# ## Full Corpus Tokens Exploration

# #### Full Data Set

# In[26]:


# Clean and combine the essay columns
text_column_list = ['project_essay_1', 'project_essay_2', 'project_essay_3', 'project_essay_4']

clean_input_data(train_full, combined_col_name, text_column_list, 
                 character_remove_list, character_replace_list, replace_numbers)


# #### Larger Rejected Proposal Sample 
# Since the data is so heavily skewed towards accepted proposals, I wanted to get more vocabulary about the rejected proposals (learn more about the text variety/patterns). However, I would still like to keep the run time fairly short, so I am going to keep 10% of the accepted proposals and just increase the sample size for rejected projects. 

# In[27]:


# Accepted Proposals 10% Sample Size
accept_ss = round(len(train_full[train_full['project_is_approved'] == 1])*.10)

# What proportion of the rejected df would I need to equal the same amount from the accepted?
# Create equal sized dataframes
ac_df = train_full[train_full['project_is_approved'] == 1].sample(accept_ss, random_state = 1)
re_df = train_full[train_full['project_is_approved'] == 0].sample(accept_ss, random_state = 1)

train2 = pd.concat([ac_df, re_df])


# #### Function: Tokens and Model Parems

# In[28]:


from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()

def vectorize_and_get_model_params(data, text_column, output_column):
    vect = CountVectorizer(binary=True, stop_words = 'english', analyzer="word")
    X_dtm = vect.fit_transform(data[text_column])
    print('features:', X_dtm.shape[1])    
    model.fit(X_dtm, data[output_column])
    token_list = vect.get_feature_names()


# In[29]:


# curious about the variety of vocab?
vectorize_and_get_model_params(ac_df, 'full_essay', 'project_is_approved')
vectorize_and_get_model_params(re_df, 'full_essay', 'project_is_approved')


# In[31]:


data = train2
text_column = 'full_essay'
output_column = 'project_is_approved'

vect = CountVectorizer(binary=True, stop_words = 'english', analyzer="word")
X_dtm = vect.fit_transform(data[text_column])
print('features:', X_dtm.shape[1])    
model.fit(X_dtm, data[output_column])
token_list = vect.get_feature_names()


# #### ~ SUMMARY~
# 
# In terms of shear number of different tokens, the accepted essay corpus had almost the same amount as the rejected corpus. However, given the feature count of the tokens for both, there is about 7,000 tokens that are NOT shared between the two corpuses.
# 
# 
# Questions: Is there a concentration of terms in one corpus or is the vocabulary pretty dispersed? If one has a higher concentration, some pattern can be pulled out. However, if one corpus is more evenly spaced, then there might be more randomness and therefore harder to draw a a conclusion since words aren't as frequently repeated across documents.

# ### Token Popularity - Prep

# #### Number of Tokens

# In[32]:


# number of times each token appears across all feature flag = 0 (in this case Rejected Project)
classA_token_count = model.feature_count_[0, :]
# number of times each token appears across all ClassB  (Accepted Project)
classB_token_count = model.feature_count_[1, :]


# #### Dataframe of Tokens per Class Type

# In[34]:


# create a DataFrame of tokens with their separate counts
tokens = pd.DataFrame({'token': token_list, 
                       'classA: Rejected':classA_token_count, 
                       'classB: Accepted':classB_token_count}).set_index('token')
# Extract the index column into its own column
tokens = tokens.reset_index()


# #### Standardize Column Names

# In[35]:


classA_name = 'classA_norm'
classA_count = 'classA: Rejected'

classB_name = 'classB_norm'
classB_count = 'classB: Accepted'


# #### Normalized Frequency Count & Ratio between Corpuses (Class Types)

# In[36]:


# Convert the Rejected and Accepted Class counts into normalized frequencies
tokens[classA_name] = tokens[classA_count]  / model.class_count_[0]
tokens[classB_name] = tokens[classB_count]  / model.class_count_[1]

# Add 1 to Rejected and Accepted Class counts to avoid dividing by 0
A_plus = (tokens[classA_count] + 1)  / model.class_count_[0]
B_plus = (tokens[classB_count] + 1 )  / model.class_count_[1]

# Calculate the ratio of Accepted-to-Rejected for each token
tokens['classB_ratio'] = tokens[classB_name] / A_plus
tokens['classB_ratio'][(tokens[classA_name])>0] = tokens[classB_name] / tokens[classA_name]

tokens['classA_ratio_extra'] = tokens[classA_name] / B_plus
tokens['classA_ratio_extra'][(tokens[classB_name])>0] = tokens[classA_name] / tokens[classB_name]


# #### Proportion of Tokens within Each Corpus

# In[37]:


## Get proportions (true, without adding 1)
tokens['classA_proportion'] = 0
total_projects_classA = model.class_count_[0]
tokens['classA_proportion'][(tokens[classA_count])>0] = (tokens[classA_count]) / total_projects_classA

tokens['classB_proportion'] = 0
total_projects_classB = model.class_count_[1]
tokens['classB_proportion'][(tokens[classB_count])>0] = (tokens[classB_count]) / total_projects_classB

# Subtract the proportions
tokens['proportion_diff'] = tokens['classB_proportion'] - tokens['classA_proportion']


# ## Token Frequency

# In[38]:


sns.set(rc={'figure.figsize':(14,8)})

sns.kdeplot(tokens['classB_proportion'], color='cornflowerblue')
sns.kdeplot(tokens['classA_proportion'], color='sandybrown')

plt.show()


# #### ~ SUMMARY ~
# 
# Most terms appear infrequently in each corpus, however, the accepted proposal corpus has slightly more repititon of terms (as can be seen from the blue peak around 0.02).

# In[39]:


# What tokens appear more than 25% of both the corpuses and are a min length of 2?
type_best = tokens[((tokens['classB_proportion'] >= 0.25) | (tokens['classA_proportion'] >= 0.25)  ) 
                   & (tokens['token'].str.len() > 2)
      ].sort_values('classB_ratio', ascending=False)


# #### Visualize Ratio of Accepted Proposal Tokens to Rejected Tokens

# In[42]:


from pylab import suptitle, yticks

main_var = 'classB_ratio'
secondary_var = 'classA_ratio_extra'
type_best = type_best.sort_values(main_var, ascending = True)

val1 = type_best[main_var].tolist()
x_max = max(type_best[main_var].max(), type_best[secondary_var].max())
val2 = type_best[secondary_var].tolist()

bars = type_best['token'].tolist()
pos = np.arange(len(val1))


fig, axes = plt.subplots(ncols=2, sharey=True)
fig.set_size_inches(7, 11)
axes[0].barh(pos,val1, align='center', color='cornflowerblue', label = 'Accepted')
axes[1].barh(pos,val2, align='center', color='sandybrown', label = 'Rejected')

yticks(pos, bars, fontsize = 14)

axes[0].yaxis.set_tick_params(labelsize=14)
axes[0].xaxis.set_tick_params(labelsize=14)
axes[1].xaxis.set_tick_params(labelsize=14)


axes[0].set_xlim(0, x_max+.1*x_max)
axes[0].invert_xaxis()
axes[1].set_xlim(0, x_max+.1*x_max)


title_string = "Ratio of Tokens between Corpus Types"
suptitle(title_string, fontsize=18, fontweight='bold')

axes[0].legend(bbox_to_anchor=(0.1, 1.0, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
axes[1].legend(bbox_to_anchor=(0.1, 1.0, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)

plt.show()


# #### ~ SUMMARY ~
# 
# There aren't that many frequent terms (as seen in the previous chart) and those words that repeat, repeat for both corpuses. Tokens: `reading` and `technology` are a couple of tokens that seem to appear more frequently in Accepted proposals (though these words appear in both corpuses in over 25% of essays). One token that jumps out as appearing more in one corpus and not the other is the token `material`. Perhaps there is a focus on general materials in the rejected proposals while the accepted proposals focus on talking about materials less?

# ## Top 20 Tokens

# ### Accepted Proposal Popular Tokens
# Appear at least in more than 0.5% of the documents. Sorted by ratio of appearances in the Accepted Proposals versus the Rejected Proposals.

# In[43]:


type_best = tokens[(tokens['classB_proportion'] > 0.005) & (tokens['token'].str.len() > 2)
      ].sort_values('classB_ratio', ascending=False).head(50)

type_best[['token', 'classA_norm', 'classB_norm', 'classB_ratio']].head(20)


# ### Rejected Proposal Popular Tokens
# Appear at least in more than 0.5% of the documents. Sorted by ratio of appearances in the Rejected Proposals versus the Accepted Proposals.

# In[44]:


type_best = tokens[(tokens['classA_proportion'] > 0.005) & (tokens['token'].str.len() > 2)
      ].sort_values('classA_ratio_extra', ascending=False).head(50)

type_best[['token', 'classA_norm', 'classB_norm', 'classA_ratio_extra']].head(20)


# #### ~ SUMMARY ~
# 
# There is an interesting pattern that seems to appear in the Accepted Proposal corpus which is not as distinct in the Rejected Proposals. The Accepted proposals have many tokens that are related to **furniture/movement** (hokki - type of chair, bouncy, stools, wobble, durable, stool, rug), **computer/technology** (subscription, chromebooks, minis, download, websites) and **literacy** (which has many once you look past the top 20). 
# 
# The patterns for the the Rejected Proposals are less clear, with certain themes coming up such as **food** (cooking, cook, burning, obesity) and **behavior related** (senses, manipulative/s, stimulation, regulate). But these are less unique to this corpus (as can be seen by the ratio column). 
# 

# # Future Work
# - Use LDA to automatically detect topics and see how they compare to the ones I was able to pick out.
# - Encorporte some of the findings from this notebook such as document complexity, length and material cost my naiveBayes model.

# Thank you for making it this far. Feedback is highly welcome and appreciated :)
