#!/usr/bin/env python
# coding: utf-8

# *This EDA in being developed. Please, keep coming for updates and feel free to comment feedbacks and suggestions. If it helps you, please, consider upvoting :)*
# 
# ### **1. INTRODUCTION**
# A year ago, a simmilar competition was hosted by Jigsaw at kaggle with the goal of classifying toxic comments. This year, a new challenge was added to that task: How to remove bias from that classification?
# 
# To illustrate this problem, compare the comments: `You are so gay.` and `I am gay`. Both use the word gay, but most of the people would consider only one of them as toxic, depending on the context.
# 
# **Why is this competition so important for the real world and so hard to deal with?** <br>
# The outcome of our analysis is the type of algorithm that companies will use to define what is free speech and what shouldn't be tolerated in a discussion. This challenge actually starts with how the training dataset was produced: Multiple people (annotators) read thousands of comments and defined if those comments were offensive or not. Where is the trick? They disagreed in many of them. The solution? Average the answer and whatever is above 0.5 is considered offensive. 
# 
# Let's see a real example from the dataset. Someone said: `Is there any such thing as a 'gay trans woman '?  Technically?  Theoretically, for example, a female, once you become your born sex, anatomically, you will be/continue to be attracted to women and not necessarily gay women`
# 
# Now, is this a toxic comment? Well, it's target value was 0.39, which means it was almost labeled as toxic.
# 
# If not even humans are sure about it, then how can we properly train an algorithm to deal with this kind of situation? I hope you find out and I encourage you in your journey to analyse the data, not only from a statistical point of view, but also from a social one.
# 
# In this kernel we will understand the data, have some ideas about how to approach the problem and how to create new features. 

# ### **2. DIFFERENCES BETWEEN TRAIN AND TEST DATASETS**
# 
# #### **2.1) THE TRAIN DATASET**
# 
# There are 45 columns for each row. Let's understand them better.
# 
# **ID, target and comment_text**: <br>
# 'id', 'target', 'comment_text'
# 
# **Main indicators of toxicity (6):** <br>
# main_indicators = ['severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat', 'sexual_explicit']
# 
# **Identity columns (24):**<br>
# identity_columns = ['asian', 'atheist', 'bisexual', 'black', 'buddhist', 'christian',
#                     'female', 'heterosexual', 'hindu', 'homosexual_gay_or_lesbian',
#                     'intellectual_or_learning_disability', 'jewish', 'latino', 'male',
#                     'muslim', 'other_disability', 'other_gender', 'other_race_or_ethnicity',
#                     'other_religion', 'other_sexual_orientation', 'physical_disability',
#                     'psychiatric_or_mental_illness', 'transgender', 'white']<br>
# 
# *Notice: The final evaluation will only have data about identities with more than 500 examples in the test + train dataset, which are: ['male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish', 'muslim', 'white', 'black', 'psychiatric_or_mental_illness']<br>*
# 
# **Comment's properties (5)**':<br>
# comment_properties = ['created_date', 'publication_id', 'parent_id', 'article_id', 'rating']
# 
# **Reactions (5)**:<br>
# reactions = ['funny', 'wow', 'sad', 'likes', 'disagree']
# 
# **Annotators (2)**:<br>
# annotators = ['identity_annotator_count', 'toxicity_annotator_count']
# 
# Let's take a look at the first 10 rows

# In[ ]:


#Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#import datasets
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")


# In[ ]:


#Separate columns into groups
ID_target_comment_text = ['id', 'target', 'comment_text']

main_indicators = ['severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat', 'sexual_explicit']

identity_columns = ['asian', 'atheist', 'bisexual', 'black', 'buddhist', 'christian',
                    'female', 'heterosexual', 'hindu', 'homosexual_gay_or_lesbian',
                    'intellectual_or_learning_disability', 'jewish', 'latino', 'male',
                    'muslim', 'other_disability', 'other_gender', 'other_race_or_ethnicity',
                    'other_religion', 'other_sexual_orientation', 'physical_disability',
                    'psychiatric_or_mental_illness', 'transgender', 'white']

# Only identities with more than 500 examples in the test set (combined public and private)
# will be included in the evaluation calculation. 
main_identities = ['male', 'female', 'homosexual_gay_or_lesbian',
                    'christian', 'jewish', 'muslim', 'white', 'black',
                    'psychiatric_or_mental_illness']

comment_properties = ['created_date', 'publication_id', 'parent_id', 'article_id']

reactions = ['funny', 'wow', 'sad', 'likes', 'disagree', 'rating']

annotators = ['identity_annotator_count', 'toxicity_annotator_count']


# In[ ]:


train.head(10)


# Lets check the percentage of missing values and unique values in our columns.

# In[ ]:


#Find missing values
nan_dict = dict()
for column in train.columns:
    nan_dict[column] = train[column].isna().sum()/len(train[column])

#Find unique values
unique_values_dict = dict()
for column in train.columns:
    unique_values_dict[column] = len(train[column].unique())/train[column].count()

columns = list(unique_values_dict.keys())
y_pos = np.arange(len(columns))
unique_percentage = list(unique_values_dict.values())
nan_percentage = list(nan_dict.values())

#fig, ax = plt.subplots(figsize=(8, 10))

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 10), sharey=True)
ax1.barh(y_pos, nan_percentage, color='green', ecolor='black')
ax2.barh(y_pos, unique_percentage, color='green', ecolor='black')

ax1.set_title('NaN percentage in train dataset')
ax2.set_title('Unique values percentage in train dataset')

ax1.set_xlabel('nan_percentage percentage')
ax2.set_xlabel('unique_values percentage')

ax1.set_yticks(y_pos)
ax1.set_yticklabels(columns)
ax1.invert_yaxis()


plt.show()


# **Missing values:** <br>
# - Identity columns will have missing values when a comment didn't mention any identity. We can replace them with 0.<br>
# - Parent_id will probably have missing values when it is an original topic. We can also replace them with 0.<br>
# 
# **Unique values**:
# - Some of the comments are repeated. Let's check them later and see if they are spam;<br>
# - There are very few article_ids, probably because the comments were extracted from a limited set of articles;<br>
# - Every comment has an unique ID;<br>
# - Date also constains milliseconds, so it is expected that we will see almost all comment_dates being unique.
# 
# 

# ### **2.2. EXAMPLES OF COMMENTS**
# 
# **WARNING**: Please, consider that this is a competition about finding toxicity and these comments might be offensive to you or someone.

# In[ ]:


for column in main_indicators + main_identities:
    print('-'*5, column, '-'*5, '\n')
    comment, target, column_value = train[['comment_text', 'target', column]][train[column] == train[column].max()].iloc[0]
    print('target:', target)
    print(str(column)+':', column_value)
    print(comment, '\n')


# #### **2.3. THE TEST DATASET**
# 
# The test and train datasets are very different. In the test dataset we have access only to the comments. It seems that one of the strategies that can be used is to train a model to categorize the data, and then train these categories to predict the toxicity.

# In[ ]:


test.head()


# ### **3. FEATURE ENGINEERING AND PRE-PROCESSING**
# 
# Some of the original and engineered features will hardly help us to predict the toxicity bias, not because the data is useless, but because it isn't available on the test dataset. For example, it would be a very difficult task to estimate the date of the comments just by its content. It would also be hard to estimate how many likes a comment would receive. However, in this kernel we will not drop those columns in case someone find a way to use them.
# 
# #### **New features and data pre-processing**:
# **Bigger identity groups**:<br>
# - For example, 'atheist', 'buddhist', 'christian', 'hindu', 'jewish', 'muslim', 'other_religion' can all be classified under religion toxicity. Let create new bigger groups: is_religion_related, is_gender_related, is_sexuality_related, is_ethinicity_related, is_mental_disability_related
# 
# **Likes ratio**:<br>
# - Maybe the absolute value of disagreement doesn't mean much, but the ratio between disagreement/likes do. We will calculate those ratios for all the reactions.
# 
# **Comments properties**:<br>
# - date_created: We will express this in terms of days past since the earliest comment;<br>
# - has_parent_id: Column indicating if a variable has parent_id, since almost 50% of the data don't have it;<br>
# - rating: rejected and approved substituted for 0 and 1;<br>
# 
# **Toxicity related to identity**:<br>
# - is_identity_related: If any of the identity_columns have a value > 0<br>
# - in_main_identity_related: If any of the main (as described in the instructions of this competition) identity_columns have a value. <br>
# - identity_degree: How many identity_columns have a value > 0<br>
# - identity_weight: The sum of values in identity_columns.<br>
# 
# **Filling NaN**:
# - All NaN will be filled with 0. It will affect the identity groups, that are classified as NaN if no identity is mentioned in the comment, and the parent_id. All comments without parent_id will belong to the same parent_id = 0. The data as provided doesn't have a parent_id = 0.

# In[ ]:


#BIGGER IDENTITY GROUPS
#This is how I would classify the columns. You might desagree. Feel free to change.
#Physical disability and other disability have a single category, so it doesn't need to be grouped.
religion_columns = ['atheist', 'buddhist', 'christian', 'hindu', 'jewish', 'muslim', 'other_religion']
gender_columns = ['male', 'female']
sexuality_columns = ['heterosexual', 'homosexual_gay_or_lesbian', 'other_gender', 'other_sexual_orientation', 'transgender']
ethinicity_columns = ['black', 'latino', 'white', 'asian', 'other_race_or_ethnicity']
mental_disability_columns = ['intellectual_or_learning_disability', 'psychiatric_or_mental_illness']

train['is_religion_related'] = (train[religion_columns] > 0).sum(axis=1)
train['is_gender_related'] = (train[gender_columns] > 0).sum(axis=1)
train['is_sexuality_related'] = (train[sexuality_columns] > 0).sum(axis=1)
train['is_ethinicity_related'] = (train[ethinicity_columns] > 0).sum(axis=1)
train['is_mental_disability_related'] = (train[mental_disability_columns] > 0).sum(axis=1)

#LIKES RATIO
pd.options.mode.chained_assignment = None  # desible copy warning - default='warn'
train['disagree_to_likes'] = 0
train['funny_to_likes'] = 0
train['wow_to_likes'] = 0
train['sad_to_likes'] = 0
train['all_to_likes'] = 0
train['disagree_to_likes'][train['likes'] > 0] = train['disagree'][train['likes'] > 0] / train['likes'][train['likes'] > 0]
train['funny_to_likes'][train['likes'] > 0] = train['funny'][train['likes'] > 0] / train['likes'][train['likes'] > 0]
train['wow_to_likes'][train['likes'] > 0] = train['wow'][train['likes'] > 0] / train['likes'][train['likes'] > 0]
train['sad_to_likes'][train['likes'] > 0] = train['sad'][train['likes'] > 0] /train['likes'][train['likes'] > 0]
train['all_to_likes'][train['likes'] > 0] = train[['disagree', 'funny', 'wow', 'sad']][train['likes'] > 0].sum(axis = 1) / train['likes'][train['likes'] > 0]

#COMMENTS PROPERTIES
#rating
train['rating'] = train['rating'].apply(lambda x: 1 if x =='approved' else 0)

#has_parent_id
train['has_parent_id'] = train['parent_id'].apply(lambda x: 1 if x > 0 else 0)

#date
train['created_date'] = pd.to_datetime(train['created_date'])
earliest_date = train['created_date'].min()
train['created_date'] = train['created_date'].apply(lambda x: (x - earliest_date).days)

#TOXICITY RELATED TO IDENTITY

#identity_degree
train['identity_degree'] = (train[identity_columns] > 0).sum(axis=1)

#identity_weight
train['identity_weight'] = train[identity_columns].sum(axis=1)

#is_identity_related
train['is_identity_related'] = train['identity_degree'].apply(lambda x: 1 if x>0 else 0)

#is_main_identity_related
train['in_main_identity_related'] = 0
for identity in main_identities:
    train['in_main_identity_related'] += train[identity].apply(lambda x: 1 if x>0 else 0)

train['in_main_identity_related'] = train['in_main_identity_related'].apply(lambda x: 1 if x >0 else 0)

#FILLING NaN
train.fillna(0, inplace = True)


# ### **3.1. CORRELATIONS**
# 
# Below we will see if there is any strong correlations between our features and target. In sections ahead we will analyse more in deepth the correlations between target and characteristcs of the comments, like comment length and words frequency. For now, we will limit ourselves to the existing features.
# 
# #### **3.1.1. HEAT MAP**

# In[ ]:


#CORRELATION HEAT MAP
columns = [column for column in train.columns if column not in ['id', 'comment_text', 'created_date']]

colormap = plt.cm.RdBu
plt.figure(figsize=(14,12))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(train[columns].astype(float).corr(),linewidths=0.1,vmax=1.0, 
            square=True, cmap=colormap, linecolor='white', annot=False)


# #### **3.1.2. CORRELATION MATRIX**

# In[ ]:


#If you want to see the correlation matrix, uncomment the code below
correlation_matrix = train[columns].corr()
#correlation_matrix


# #### **3.1.3. TARGET CORRELATION PLOT**

# In[ ]:


#Plots target correlation
target_correlation = correlation_matrix['target'].copy()
target_correlation = target_correlation.sort_values(ascending = False)

plt.rcdefaults()
fig, ax = plt.subplots(figsize=(4, 10))

# Example data
columns = list(target_correlation.index)
y_pos = np.arange(len(columns))
nan_percentage = list(target_correlation.values)

ax.barh(y_pos, nan_percentage, color='green', ecolor='black')
ax.set_yticks(y_pos)
ax.set_yticklabels(columns, fontsize = 9)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Correlation')
ax.set_title('Target correlation')
plt.show()


# From the graphs above we can clearly see that some features might be useful to predict the toxicity of a comment. Let's dig deeper.
# 
# ### **3.2. MAIN INDICATORS**
# 
# Below we will do some experiments with which threshold value for the main indicators best capture a toxic comment. For example, we could assume that any value of insult > 0.5 indicates that the comment is toxic. But what if 0.6 is a better threshold?

# In[ ]:


columns_to_make_binary = ['target'] + main_indicators

train_binary = train[columns_to_make_binary].copy()

binary_threshold = 0.5
for column in columns_to_make_binary:
    train_binary[column] = train_binary[column].apply(lambda x: 1 if x >= binary_threshold else 0)

fig, ax = plt.subplots(figsize=(10, 4))
for column in main_indicators:
    ax.bar(column, (train_binary[column] == 1).sum())

ax.set_title('Main indicators occurrencies')
plt.tight_layout()
plt.show()


# In[ ]:


#FIRST GRAPH
train_binary = train[columns_to_make_binary].copy()

train_binary['target'] = train_binary['target'].apply(lambda x: 1 if x >= 0.5 else 0)

all_positives = (train_binary['target'] == 1).sum()

true_positive_dict = dict()
false_negative_dict = dict()

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 10))

for column in main_indicators:
    true_positive_dict[column] = []
    false_negative_dict[column] = []

    for binary_threshold in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        train_binary[column] = train[column].apply(lambda x: 1 if x >= binary_threshold else 0)
        all_negatives = (train_binary['target'][train[column] > 0] == 0).sum()
        
        true_positives = (train_binary[column][train_binary['target'] == 1] == 1).sum()
        false_positive = (train_binary[column][(train_binary['target'] == 0) & (train[column] > 0)] == 1).sum()
        
        true_positive_dict[column].append(true_positives/all_positives)
        false_negative_dict[column].append(false_positive/all_negatives)
    
        ax1.annotate(binary_threshold, (false_positive/all_negatives, true_positives/all_positives))
    
    ax1.plot(false_negative_dict[column], true_positive_dict[column], label = column)
    
ax1.set_ylabel('True Positive Rate', fontsize=12)
ax1.set_xlabel('False Positive Rate', fontsize=12)
    

#SECOND GRAPH
train_temp = train[columns_to_make_binary][train['in_main_identity_related'] > 0].copy()

train_binary = train_temp.copy()
train_binary['target'] = train_binary['target'].apply(lambda x: 1 if x >= 0.5 else 0)

all_positives = (train_binary['target'] == 1).sum()

true_positive_dict = dict()
false_negative_dict = dict()

for column in main_indicators:
    true_positive_dict[column] = []
    false_negative_dict[column] = []

    for binary_threshold in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        train_binary[column] = train[column].apply(lambda x: 1 if x >= binary_threshold else 0)
        all_negatives = (train_binary['target'][train_temp[column] > 0] == 0).sum()
        
        true_positives = (train_binary[column][train_binary['target'] == 1] == 1).sum()
        false_positive = (train_binary[column][(train_binary['target'] == 0) & (train_temp[column] > 0)] == 1).sum()

        true_positive_dict[column].append(true_positives/all_positives)
        false_negative_dict[column].append(false_positive/all_negatives)
        ax2.annotate(binary_threshold, (false_positive/all_negatives, true_positives/all_positives))
            
    ax2.plot(false_negative_dict[column], true_positive_dict[column], label = column)


ax1.set_title('Main indicators ROC-AUC all comments')
ax2.set_title('Main Indicators ROC-AUC comments with main identities')
ax2.set_xlabel('False Positive Rate', fontsize=12)
ax2.legend(bbox_to_anchor=(1.10, 1), loc=2, borderaxespad=0.)    
plt.show()


# Insult is not only a good indicator of toxicity, but it is also the most frequent one among the main indicators. The graph shows that using 0.5 as a rule of thumb to classify something as toxic or not seems to be a good strategy, since the ROC-AUC curves have their inflection point around there.

# 

# ### **3.3. IDENTITY, REACTIONS AND ANNOTATORS**
# 
# Below we will average the values of these features for every 10% bin of the target. Let's see if these graphs tell us anything.

# In[1]:


train_binned = train.copy()
train_binned['target_binned'] = pd.cut(train_binned['target'], bins = 10)
train_binned = train_binned.groupby('target_binned').mean().copy()

identity_big_groups = ['is_identity_related', 'is_religion_related', 'is_gender_related', 'is_sexuality_related',
            'is_ethinicity_related', 'is_mental_disability_related', 'identity_degree',
           'identity_weight']

likes_ratio = ['disagree_to_likes', 'funny_to_likes', 'wow_to_likes', 'sad_to_likes', 'all_to_likes', 'rating']
titles = ['Identity Big Groups', 'Main Identities', 'Reactions', 'Likes Ratio', 'Annotators']
for columns, title in zip([identity_big_groups, main_identities, reactions, likes_ratio, annotators], titles):
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    ax1.hist(train['target'], color = 'grey', bins = np.array(range(11))/10, label = 'Target Histogram')
    ax1.set_ylabel('Target Frequency', fontsize=12)
    for i in range(len(columns)):
        ax2.plot(train_binned['target'], train_binned[columns[i]], label = columns[i])
    ax1.legend(bbox_to_anchor=(1.15, 1), loc=2, borderaxespad=0.)
    ax2.legend(bbox_to_anchor=(1.15, 0.9), loc=2, borderaxespad=0.)
    ax2.set_ylabel('Feature Value', fontsize=12)
    ax1.set_xlabel('Target', fontsize=12)
    fig.suptitle(title, fontsize=20)
    plt.show()


# ### **3.3.1. AN INTERESTING SURPRISE**
# 
# **Identity**: <br>
# The data shows such an interesting behavior. The peak of the identity average values are usualy when target = 0.5. Here is my hypothesis about it: <br>
# - A lot of comments are not toxic (target = 0) and a few of them have an identity mentioned (identity_degree < 0.3)
# - Very few comments are toxic (target = 1) and a few of them have an identity mentioned (identity_degree < 0.3).
# - But there are some comments, where identity is mentioned more often (identity_degree > 0.5), and it is NOT clear at all if they are toxic or not (target = 0.5). Here is where the challenge seems to be.
# 
# What is also interesting about identity_degree having a higher value in the target = 0.5 region is that it indicates that situations where there are two identities together, like "white male", are more complex to classify as toxic.
# 
# **Likes ratio and rating**: <br>
# Reactions and like ratio don't seem to be good indicators of a comment being toxic. The comment rating, however, seems to matter. 
# 
# **Annotators**: <br>
# It seems that identity annotators are way less frequent than toxicity annotator. However, on toxicity, some comments were evaluated by hundreds of them. It means that some targets are more trustable than others and should have more weight in our model. <br>
# 
# ### **3.4. FEATURES SUMMARY:** <br>
# 
# Those graphs were great to show how big of a challenge this competition offers. It appears to be relatively easy to classify as toxic a comment containing words related to insults, but extremely dificult for humans to say if a comment is toxic solely based on identity.

# In[ ]:


train_binary = train.copy()

columns_to_make_binary = ['target'] + main_indicators + main_identities

for column in columns_to_make_binary:
    train_binary[column] = train_binary[column].apply(lambda x: 1 if x>= 0.5 else 0)
    
main_col="target"
corr_mats=[]
for other_col in columns_to_make_binary:
    if other_col == 'target':
        continue
    confusion_matrix = pd.crosstab(train_binary[main_col], train_binary[other_col])
    corr_mats.append(confusion_matrix)
out = pd.concat(corr_mats,axis=1,keys=columns_to_make_binary[1:])

#cell highlighting
out = out.style.highlight_min(axis=0)
out.columns.names = [None, 'Classification']
out


# If this kernel gave you any new ideas, please, consider upvoting. Thanks in advance! :)
# 
# # TO BE CONTINUED...
