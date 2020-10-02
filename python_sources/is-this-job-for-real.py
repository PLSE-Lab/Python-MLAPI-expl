#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd
import gc


# # Introduction
# 
# In this exercise, we will explore a dataset, composed of 17,200 real and 800 fake jobpostings, try to see if we are able to use machine learning techniques to glean meaningful insight from it and construct models that are able to accurately predict whether a previously unseen job posting is real or not.
# 
# First, let's read in the data and see what we are dealing with.

# In[ ]:


init_df = pd.read_csv('../input/real-or-fake-fake-jobposting-prediction/fake_job_postings.csv')
init_df.head()


# From a glance at the head of the dataset, we find the following features:
# * *job_id*             : Intuitively, this is just the unique ID for every single entry in the dataset. 
# * *title*              : The job title or position. Most likely unique for each entry.
# * *location*           : The job's location in format: Country, State, City. 
# * *department*         : What department in the organization, the job is part of. Most likely unique for each posting
# * *salary_range*       : The salary range for the position. From an initial glance of the head, we see its blank; However, in subsequent analysis, we see that it is in format MIN-MAX
# * *company_profile*    : An overview of the company.
# * *description*        : An overview of the job description.
# * *requirements*       : An overview of the preferred experience and requirements necessary to perform the job.
# * *benefits*           : An overview of what benefits this job/company offers.
# * *telecommuting*      : Whether this job is telecommuting (1) or not (0).
# * *has_company_logo*   : Whether this job posting has the company logo or not.
# * *has_questions*      : Whether this job posting has a questionaire attached to it.
# * *employment_type*    : Whether this job is hourly, part-time, full-time, etc.
# * *required_education* : What degree is necessary: None, BS, MS, etc.
# * *industry*           : What industry this job is in: Fashion, IT, etc. Could possibly unique for each job.
# * *function*           : Keyword for the job. Could possible unique for each job.
# * *fraudulent*         : Target label. 0 if real job post. 1 if fake job post.
# 
# Although we have a lot of information in this dataset, we may not be able to make use of all of it, especially when it comes to a certain feature's redundancy or irrelevance. For example,the job title of a post may most likely have nothing to do whether a job posting is fraudulent or not. We may know this, but our models may not and may end up viewing it as more significant than it actually is; So we may end up having to drop that feature.
# 
# More on that later. 
# 
# 

# # Data Cleaning
# 
# First, let's take a step in understanding and, subsequently, cleaning our dataset by taking a look at how many unique values are in each column.

# In[ ]:


init_df.nunique()


# We see that there are some columns that may be too difficult to deal with or reconcile. We drop the following columns for the following respective reasons:
# * The *job_id* feature because there is a unique value for every single entry and we can not get any useful information from this.
# * The *job_title* feature because it is probably irrelevant, especially when it is notorious that a lot of job titles are superfluous and fluff.
# * The *location* feature because we do not want our model to care about location and keep performance generalized for a job anywhere in the world.
# * The *department* because we feel it will be like *job_title,* where the department would vary too much by posting and not be meaningful enough to draw anything from.
# * The *industry* function for the same reasons as *job_title* and *department.*
# * The *function* function for the same reasons as *job_title* and *department.*

# In[ ]:


drop_columns = ['job_id', 'title', 'location', 'department', 'industry', 'function']

proc_df = init_df.drop(drop_columns, axis=1)

del init_df
gc.collect()


# When we reviewed the head of our dataframe, we also saw that a lot of different feature values were missing. Let's explore just how many missing values there are for each feature:

# In[ ]:


proc_df.isnull().sum()


# We see that the *employment_type*, *required_experience* and *required_education* columns have a significant amout of missing values and only have 5, 7 and 13 unique object/categorical values, respectively. As these are manageable amounts, we replace any missing values with a new object/category value of *Unknown.*

# In[ ]:


cat_columns = ['employment_type', 'required_experience', 'required_education']

for col in cat_columns:
    proc_df[col].fillna("Unknown", inplace=True)


# As the *company_profile*, *description*, *requirements* and *benefits* features contain string values of various lengths where the underlying information needs to be uncovered with text/language analysis, we will fill any empty values with a blank space, ' ', instead of 'Unknown'. This is so that we do the text analysis, we only do it on what text is already there, not text we introduce, and blank space will just be ignored during said analysis.

# In[ ]:


text_columns = ['company_profile', 'description', 'requirements', 'benefits']

proc_df = proc_df.dropna(subset=text_columns, how='all')

for col in text_columns:
    proc_df[col].fillna(' ', inplace=True)
    


# We also saw how in the header entries, the *salary_range* values were all empty. Before we reconcile the null values, we take a look at what the actual values are by looking at the first few unique possibilities.

# In[ ]:


unique_salary = proc_df['salary_range'].unique()
print(unique_salary[0:5])


# Thankfully the *salary_range* feature contains meaningful and consistent features, in the format of **Min Range**-**Max Range**. From a review of all the unique values (not shown), we also see how alot of the ranges share some overlap, are not consistent and contain incorrect values (such as Dec, Oct, etc). To reconcile these inconsistencies, we choose to:
# 
# * First split the 'Min-Max' string to get the discrete Min and Max values.
# * Store the Min and Max values in the new *salary_range_min* and *salary_range_max* features, respectively.
# * Replace any null and non-numeric values in new features with '-1'
# * Drop the original *salary_range* feature.

# In[ ]:


new = proc_df['salary_range'].str.split("-", n = 1, expand = True) 

proc_df['salary_range_min']= new[0]
proc_df['salary_range_max']= new[1]

proc_df['salary_range_min'].fillna('-1', inplace=True)
proc_df['salary_range_max'].fillna('-1', inplace=True)

def remove_string(x):
    if not x.isnumeric(): 
        val = '-1'
    else:
        val = x
    return val

proc_df['salary_range_min'] = proc_df['salary_range_min'].apply(lambda x: remove_string(x))
proc_df['salary_range_max'] = proc_df['salary_range_max'].apply(lambda x: remove_string(x))

proc_df.drop('salary_range', axis=1, inplace = True) 


# # Exploratory Data Analysis
# 
# Now that we're finished with cleaning/touching up our data, for now, we can move on to some EDA to get a better idea of data. First, we take a look at how many counts of real and fake posts there are, in relation to the top unique values of a feature.

# In[ ]:


cat_eda_columns = ['telecommuting', 'has_company_logo', 'has_questions', 'employment_type', 'required_experience', 'required_education']

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec 

grid = gridspec.GridSpec(5, 2, wspace=0.5, hspace=0.5) 
plt.figure(figsize=(15,25)) 

for n, col in enumerate(proc_df[cat_eda_columns]): 
    ax = plt.subplot(grid[n]) 
    sns.countplot(x=col, data=proc_df, hue='fraudulent', palette='Set2', order=proc_df[col].value_counts().iloc[:5].index) 
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(f'{col} Distribution by Target', fontsize=15) 
    xlabels = ax.get_xticklabels() 
    ax.set_xticklabels(xlabels,  fontsize=10)
    plt.legend(fontsize=8)
    plt.xticks(rotation=30) 
    total = len(proc_df)
    sizes=[] 
    for p in ax.patches: 
        height = p.get_height()
        sizes.append(height)
        ax.text(p.get_x()+p.get_width()/2.,
                height + 3,
                '{:1.2f}%'.format(height/total*100),
                ha="center", fontsize=10) 
    
plt.show()


# Not surprisingly, there are significantly fewer fraudulent posts per feature value for each value because there are significantly fewer fraudulent posts in the dataset (800 vs ~17,000). Even so, we get the idea that fraudulent posts match closely real posts. We see that:
# * Fraudulent posts are mostly not posted as telecommuting ones, like real posts.
# * Fraudulent posts mostly do not contain a company logo, unlike real posts.
# * Fraudulent posts have an equal mix of either having a questionnaire or not; like real posts.
# * Fraudulent posts are mostly for full-time "positions," like real posts.
# * Fraudulent posts, like real posts, also do not specify the required experience and education necessary.
# 
# As we are realists, we can see that fake posts try to tick off all the main boxes of real posts to try to match them as closely as possible and entice unsuspecting people into their nefarious, insidious schemes. 
# 
# Even though some red flags pop up in this overview of determining a job posting's validity with the above markers, we feel they might not be enough especially when the dataset is so skewed.
# 
# We try to see if maybe there are some more significant markers for validity in the structure of the meat-and-potatoes of the job posting by looking at the word counts of each write-up section.

# In[ ]:


text_cols = ['company_profile', 'description', 'requirements', 'benefits']

for col in text_cols:
    fig,(ax1,ax2)= plt.subplots(ncols=2, figsize=(10, 2.5), dpi=100)
    num=proc_df[proc_df["fraudulent"]==1][col].str.split().map(lambda x: len(x))
    ax1.hist(num,bins = 20,color='orangered')
    ax1.set_title('Fake Post')
    num=proc_df[proc_df["fraudulent"]==0][col].str.split().map(lambda x: len(x))
    ax2.hist(num, bins = 20)
    ax2.set_title('Real Post')
    fig.suptitle(f'Words in {col}')
    plt.show()


# Interestingly enough word counts of both posts match a similar distribution for all sections except the company profile one! We can guess intuitively that a fake job posting will try to entice the individual with the description ("easy, fun hours!"), requirements ("none!"), and benefits ("full!"), hoping that they will not look into the company, which doesn't exist and is using some boilerplate "professional" website.

# # Data Preprocessing
# 
# Now that we've got some EDA done and a better grasp of our data, we move on to some more preprocessing. 
# 
# Although we found that a fake post has significantly fewer words in the company profile section than a real post, we do not want to use the count of words to determine whether a post is real or not; We want to try to use the content (and maybe some metadata features) of the overall job post to make that determination! So, we elect to concatenate the *company_profile*, *description*, *requirements*, and *benefits* features into one feature to contain all of the text in all of the sections.
# 
# We do this and verify it below:

# In[ ]:


text_cols = ['company_profile', 'description', 'requirements', 'benefits']

proc_df['aggr_post'] = proc_df[text_cols].apply(lambda x: ' '.join(x), axis=1)
proc_df.drop(text_cols, axis=1, inplace=True)

proc_df.head()

print(proc_df.loc[0, 'aggr_post'])


# We end up consolidating everything in the *aggr_post* feature and everything seems to look good! Now we are going to do two things:
# 
# 1) Remove any rows that contain job posts that aren't in english to keep things simple.
# 
# 2) *Normalize* our new feature, by cleaning it up by removing extraneous fluff like brackets, links, punctuation, capitalization, etc to turn it into essentially an excerpt from a Cormac McCarthy novel.

# In[ ]:


import langid

def detect_lang(x):
    code,_ = langid.classify(x)
    
    return code

proc_df = proc_df[proc_df['aggr_post'].apply(lambda x: detect_lang(x) == 'en')]

proc_df.head()


# In[ ]:


import re
import string

def clean_text(text):
    text = text.lower()                                              # make the text lowercase
    text = re.sub('\[.*?\]', '', text)                               # remove text in brackets
    text = re.sub('http?://\S+|www\.\S+', '', text)                  # remove links
    text = re.sub('https?://\S+|www\.\S+', '', text)                 # remove links
    text = re.sub('<.*?>+', '', text)                                # remove HTML stuff
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)  # get rid of punctuation
    text = re.sub('\n', '', text)                                    # remove line breaks
    #text = re.sub('\w*\d\w*', '', text)                             # remove anything with numbers, if you want
    #text = re.sub(r'[^\x00-\x7F]+',' ', text)                       # remove unicode
    return text

proc_df['aggr_post'] = proc_df['aggr_post'].apply(lambda x: clean_text(x))

proc_df.head()


# Now that we have clean, uniform text in our new feature, we take the additional step of removing stopwords from our job posts. Essentially, when we do our text analysis and feed it to our model, we don't want our model to make a prediction based on how many words like "and", "the", "or", etc there are!
# 
# So first, we tokenize each *aggr_post* value, remove stopword tokens, join the stopword tokens back and repopulate our values!

# In[ ]:


import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
stop_words = stopwords.words('english')

def remove_stopwords(text):
    words = [w for w in text if w not in stop_words]
    return words

def combine_text(list_of_text):
    combined_text = ' '.join(list_of_text)
    return combined_text

proc_df['aggr_post'] = proc_df['aggr_post'].apply(lambda x: tokenizer.tokenize(x))
proc_df['aggr_post'] = proc_df['aggr_post'].apply(lambda x : remove_stopwords(x))
proc_df['aggr_post'] = proc_df['aggr_post'].apply(lambda x : combine_text(x))

proc_df.head()


# # Training/Validation/Testing Set Preparation
# 
# Now that we've cleaned up our dataset to the best of our ability, we are able to move on to splitting up our dataset into our necessary training, validation and testing sets for model building.
# 
# As we know that in this dataset, the number of real job posts greatly outnumber the number of fraudulent posts. To overcome any bias/skew we may run into with this disparity, we create a final working dataframe with a downsampled number of real job posts (specifically 3,000).

# In[ ]:


random_state = 42

real_df = proc_df[proc_df['fraudulent']==0].copy()
fake_df = proc_df[proc_df['fraudulent']==1].copy()

real_sampled_df = real_df.sample(n=3000, random_state=random_state)

final_df = pd.concat([real_sampled_df, fake_df], axis=0)


# Now that we have our final dataset, we get rid of our unused ones to keep memory free.

# In[ ]:


del proc_df
del real_df
del real_sampled_df
del fake_df

gc.collect()


# To ensure we have enough samples of fraudulent job postings in our training, validation, testing sets, we do splits on the distinct values and rejoin them for our final sets.
# 
# To make our validation sets, we do a split on the overall sets to make a buffer set (along with a testing set) and then do another split on the buffer set (resulting in a training set and a validation set).

# In[ ]:


from sklearn.model_selection import train_test_split

seed_state = 315
random_state = 42

real_df = final_df[final_df['fraudulent']==0]
fake_df = final_df[final_df['fraudulent']==1]

y_real = real_df['fraudulent'].copy()
x_real = real_df.drop(['fraudulent'], axis=1)

y_fake = fake_df['fraudulent'].copy()
x_fake = fake_df.drop(['fraudulent'], axis=1)

x_real_tv, x_real_test, y_real_tv, y_real_test = train_test_split(x_real, y_real, test_size=0.3, random_state=seed_state)
x_real_train, x_real_val, y_real_train, y_real_val = train_test_split(x_real_tv, y_real_tv, test_size=0.2, random_state=seed_state)

x_fake_tv, x_fake_test, y_fake_tv, y_fake_test = train_test_split(x_fake, y_fake, test_size=0.3, random_state=seed_state)
x_fake_train, x_fake_val, y_fake_train, y_fake_val = train_test_split(x_fake_tv, y_fake_tv, test_size=0.2, random_state=seed_state)

x_train = pd.concat([x_real_train, x_fake_train])
y_train = pd.concat([y_real_train, y_fake_train])

x_val = pd.concat([x_real_val, x_fake_val])
y_val = pd.concat([y_real_val, y_fake_val])

x_test = pd.concat([x_real_test, x_fake_test])
y_test = pd.concat([y_real_test, y_fake_test])


# # Model Building
# 
# As we have a significant amount of information in our dataset with the metadata features and overall text of our job posts, we will take a two pronged approach. We will construct three models:
# * A linear regression model to make predictions solely on the text content of a job post.
# * A random forest model to make predictions solely on the metadata features of a job post.
# * A final linear regression model to use the predictions, of the first two models, together to make a final determination. We do this to see if we are able to improve our predictions.
# 
# So, we will split our training, validation, and testing sets into two more of each so that one group just has the text feature, and the other has the rest of the features.

# In[ ]:


x_train_post = x_train['aggr_post'].copy()
x_val_post = x_val['aggr_post'].copy()
x_test_post = x_test['aggr_post'].copy()

x_train_cat = x_train.drop(['aggr_post'], axis=1)
x_val_cat = x_val.drop(['aggr_post'], axis=1)
x_test_cat = x_test.drop(['aggr_post'], axis=1)


# To help visualize our results, I include a nifty confusion matrix function that I've been using in my recent notebooks.

# In[ ]:


import seaborn as sns
from sklearn.metrics import confusion_matrix, mean_absolute_error, make_scorer 

# Showing Confusion Matrix
# Thanks to https://www.kaggle.com/marcovasquez/basic-nlp-with-tensorflow-and-wordcloud
def plot_cm(y_true, y_pred, title):
    figsize=(14,14)
    y_pred = y_pred.astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true))
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    cm = pd.DataFrame(cm, index=np.unique(y_true), columns=np.unique(y_true))
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    fig, ax = plt.subplots(figsize=figsize)
    plt.title(title)
    sns.heatmap(cm, cmap= "YlGnBu", annot=annot, fmt='', ax=ax)


# ## Linear Regression Model 1 : Job Post Content
# 

# As we will be looking at the content of each job post, we will first vectorize our text and use that as our training input for our Linear Regression Model.

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, mean_absolute_error, make_scorer 

count_vectorizer = CountVectorizer()
x_train_post_vec = count_vectorizer.fit_transform(x_train_post)
x_val_post_vec = count_vectorizer.transform(x_val_post)
x_test_post_vec = count_vectorizer.transform(x_test_post) 

lr_post = LogisticRegression(C=0.1, solver='lbfgs', max_iter=2000, verbose=0, n_jobs=-1)
lr_post.fit(x_train_post_vec, y_train)


# In[ ]:


weights = lr_post.coef_
abs_weights = np.abs(weights)


# In[ ]:


lr_post_val_preds = lr_post.predict(x_val_post_vec)

f1_score(y_val, lr_post_val_preds, average = 'macro')
plot_cm(y_val, lr_post_val_preds, 'Confusion Matrix: LR Validation Set Predictions ')


# In[ ]:


lr_post_test_preds = lr_post.predict(x_test_post_vec)

f1_score(y_test, lr_post_test_preds, average = 'macro')
plot_cm(y_test, lr_post_test_preds, 'Confusion Matrix: LR Test Set Predictions ')


# ## Random Forest Classifier : All Other Features

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

le_employment = LabelEncoder()
le_experience = LabelEncoder()
le_education  = LabelEncoder()

x_train_cat['employment_type'] = le_employment.fit_transform(x_train_cat['employment_type'])
x_val_cat['employment_type'] = le_employment.transform(x_val_cat['employment_type'])
x_test_cat['employment_type'] = le_employment.transform(x_test_cat['employment_type'])

x_train_cat['required_experience'] = le_experience.fit_transform(x_train_cat['required_experience'])
x_val_cat['required_experience'] = le_experience.transform(x_val_cat['required_experience'])
x_test_cat['required_experience'] = le_experience.transform(x_test_cat['required_experience'])

x_train_cat['required_education'] = le_education.fit_transform(x_train_cat['required_education'])
x_val_cat['required_education'] = le_education.transform(x_val_cat['required_education'])
x_test_cat['required_education'] = le_education.transform(x_test_cat['required_education'])

rf_cat = RandomForestClassifier(n_estimators=2000,bootstrap=True)
rf_cat.fit(x_train_cat, y_train)


# In[ ]:


rf_cat_val_pred = rf_cat.predict(x_val_cat)

f1_score(y_val, rf_cat_val_pred.round(), average = 'macro')
plot_cm(y_val, rf_cat_val_pred.round(), 'Confusion Matrix: RF Validation Set Predictions ')


# In[ ]:


rf_cat_test_pred = rf_cat.predict(x_test_cat)

f1_score(y_test, rf_cat_test_pred.round(), average = 'macro')
plot_cm(y_test, rf_cat_test_pred.round(), 'Confusion Matrix: RF Test Set Predictions ')


# ## Final Aggregate Model

# In[ ]:


aggregate_val = pd.DataFrame()
aggregate_val['post_preds'] = lr_post_val_preds
aggregate_val['cat_preds'] = rf_cat_val_pred
aggregate_val.head()


# In[ ]:


aggregate_test = pd.DataFrame()
aggregate_test['post_preds'] = lr_post_test_preds
aggregate_test['cat_preds'] = rf_cat_test_pred
aggregate_test.head()


# In[ ]:


lr_final = LogisticRegression(C=0.1, solver='lbfgs', max_iter=2000, verbose=0, n_jobs=-1)
lr_final.fit(aggregate_val, y_val)

lr_final_preds = lr_final.predict(aggregate_test)

f1_score(y_test, lr_final_preds, average = 'macro')
plot_cm(y_test, lr_final_preds, 'Confusion Matrix: Aggregate Model Final Predictions ')

