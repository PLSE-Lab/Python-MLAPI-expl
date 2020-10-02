#!/usr/bin/env python
# coding: utf-8

# **Word Vectors and Features**
# 
# I have been wanting to try out a model that combines word vectors with feature engineering. I also wanted my model to use KFold CV with a validation set within each fold. And, finally, I wanted to build it myself, as much as possible, in order to strengthen my writing and thinking in python.
# 
# This isn't the prettiest kernel but I hope that it's useful for those less proficient than I am. For those farther along, please suggest improvements!
# 
# I will add more annotations, explanations, modify parameters and make changes over the course of the competition. Stay tuned...

# In[1]:


import numpy as np
import pandas as pd
import random
from sklearn import *
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from scipy.sparse import hstack, vstack, csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import xgboost as xgb
import nltk
import re
import gc

# This kernel is my attempt to accomplish a personal
# goal, "from scratch." For a clean example that achieves
# the same thing (and more), see t35khan's kernel:
# https://www.kaggle.com/t35khan/tfidf-driven-xgboost


# **Load the Files**
# 
# Two of the columns in **train** and **test** seem to have values that cause pandas some confusion. I have set them to "object" type on import.
# 
# We carve out the target variable, called "project_is_approved," from **train** right away. Drop the target column from **train**.
# 
# Read in the sample submission file. We'll use the IDs from that file later.
# 
# Finally, fill in NA values for text columns with "unk" for "unknown." The string "unk" isn't a word but it will add information to your word vectors : )

# In[2]:


train = pd.read_csv('../input/train.csv', dtype={"project_essay_3": object, "project_essay_4": object})
target = train['project_is_approved']
train = train.drop('project_is_approved', axis=1)

test = pd.read_csv('../input/test.csv', dtype={"project_essay_3": object, "project_essay_4": object})

sub = pd.read_csv('../input/sample_submission.csv')

resources = pd.read_csv('../input/resources.csv')

train.fillna(('unk'), inplace=True) 
test.fillna(('unk'), inplace=True)


# **Feature Engineering**
# 
# The first step in feature engineering is to turn labels, or categorical values, into integers. Generally, XGBoost does better with label encoding than with one hot encoding.
# 
# @opanichev provided a great piece of code so let's use it!

# In[3]:


# Feature engineering (before encoding)
# Thanks to @coronate
# https://www.kaggle.com/coronate/donorschoose-exploratory-analysis
# Clever approach to feature engineering but no improvement
#genderDictionary = {"Ms.": "Female", "Mrs.":"Female", "Mr.":"Male", "Teacher":"Neutral", "Dr.":"Neutral", np.nan:"Neutral"}
#train["gender"] = train.teacher_prefix.map(genderDictionary)
#test["gender"] = test.teacher_prefix.map(genderDictionary)

#titleDictionary = {"Ms.": "Na", "Mrs.":"Na", "Mr.":"Na", "Teacher":"Teacher", "Dr.":"Dr.", np.nan:"Na"}
#train["title"] = train.teacher_prefix.map(titleDictionary)
#test["title"] = test.teacher_prefix.map(titleDictionary)


# In[4]:


# Thanks to opanichev for saving me a few minutes
# https://www.kaggle.com/opanichev/lightgbm-and-tf-idf-starter/code

# Label encoding

df_all = pd.concat([train, test], axis=0)

cols = [
    'teacher_id', 
    'teacher_prefix', 
    'school_state', 
    'project_grade_category', 
    'project_subject_categories', 
    'project_subject_subcategories'
]

for c in tqdm(cols):
    le = LabelEncoder()
    le.fit(df_all[c].astype(str))
    train[c] = le.transform(train[c].astype(str))
    test[c] = le.transform(test[c].astype(str))
    
del df_all; gc.collect()


# When it comes to feature engineering, get creative! What metrics can we create that may provide more signal to the model than noise?

# In[5]:


# Feature engineering

# Date and time
train['project_submitted_datetime'] = pd.to_datetime(train['project_submitted_datetime'])
test['project_submitted_datetime'] = pd.to_datetime(test['project_submitted_datetime'])

# Date as int may contain some ordinal value
train['datetime_int'] = train['project_submitted_datetime'].apply(lambda x: x.strftime('%Y%m%d')).astype(int)
test['datetime_int'] = test['project_submitted_datetime'].apply(lambda x: x.strftime('%Y%m%d')).astype(int)

# Date parts

train['datetime_day'] = train['project_submitted_datetime'].dt.day
train['datetime_dow'] = train['project_submitted_datetime'].dt.dayofweek
train['datetime_year'] = train['project_submitted_datetime'].dt.year
train['datetime_month'] = train['project_submitted_datetime'].dt.month
train['datetime_hour'] = train['project_submitted_datetime'].dt.hour
train = train.drop('project_submitted_datetime', axis=1)

test['datetime_day'] = test['project_submitted_datetime'].dt.day
test['datetime_dow'] = test['project_submitted_datetime'].dt.dayofweek
test['datetime_year'] = test['project_submitted_datetime'].dt.year
test['datetime_month'] = test['project_submitted_datetime'].dt.month
test['datetime_hour'] = test['project_submitted_datetime'].dt.hour
test = test.drop('project_submitted_datetime', axis=1)

# Essay length
train['e1_length'] = train['project_essay_1'].apply(len)
test['e1_length'] = train['project_essay_1'].apply(len)

train['e2_length'] = train['project_essay_2'].apply(len)
test['e2_length'] = train['project_essay_2'].apply(len)

# Has more than 2 essays?
train['has_gt2_essays'] = train['project_essay_3'].apply(lambda x: 0 if x == 'unk' else 1)
test['has_gt2_essays'] = test['project_essay_3'].apply(lambda x: 0 if x == 'unk' else 1)


# Let's create some features from the numerical columns in the **resources.csv** file. You could try merging the descriptive text into the content that gets vectorized but we'll leave that behind for now.

# In[6]:


# Combine resources file
# Thanks, the1owl! 
# https://www.kaggle.com/the1owl/the-choice-is-yours

resources['resources_total'] = resources['quantity'] * resources['price']

dfr = resources.groupby(['id'], as_index=False)[['resources_total']].sum()
train = pd.merge(train, dfr, how='left', on='id').fillna(-1)
test = pd.merge(test, dfr, how='left', on='id').fillna(-1)

dfr = resources.groupby(['id'], as_index=False)[['resources_total']].mean()
dfr = dfr.rename(columns={'resources_total':'resources_total_mean'})
train = pd.merge(train, dfr, how='left', on='id').fillna(-1)
test = pd.merge(test, dfr, how='left', on='id').fillna(-1)

dfr = resources.groupby(['id'], as_index=False)[['quantity']].count()
dfr = dfr.rename(columns={'quantity':'resources_quantity_count'})
train = pd.merge(train, dfr, how='left', on='id').fillna(-1)
test = pd.merge(test, dfr, how='left', on='id').fillna(-1)

dfr = resources.groupby(['id'], as_index=False)[['quantity']].sum()
dfr = dfr.rename(columns={'quantity':'resources_quantity_sum'})
train = pd.merge(train, dfr, how='left', on='id').fillna(-1)
test = pd.merge(test, dfr, how='left', on='id').fillna(-1)

# We're done with IDs for now
train = train.drop('id', axis=1)
test = test.drop('id', axis=1)


# Concatenate text columns. This may not be optimal but it is efficient.

# In[7]:


# Thanks to opanichev for saving me a few minutes
# https://www.kaggle.com/opanichev/lightgbm-and-tf-idf-starter/code

train['project_essay'] = train.apply(lambda row: ' '.join([
    str(row['project_title']),
    str(row['project_essay_1']), 
    str(row['project_essay_2']), 
    str(row['project_essay_3']),
    str(row['project_essay_4']),
    str(row['project_resource_summary'])]), axis=1)
test['project_essay'] = test.apply(lambda row: ' '.join([
    str(row['project_title']),
    str(row['project_essay_1']), 
    str(row['project_essay_2']), 
    str(row['project_essay_3']),
    str(row['project_essay_4']),
    str(row['project_resource_summary'])]), axis=1)

train = train.drop([
    'project_title',
    'project_essay_1', 
    'project_essay_2', 
    'project_essay_3', 
    'project_essay_4',
    'project_resource_summary'], axis=1)
test = test.drop([
    'project_title',
    'project_essay_1', 
    'project_essay_2', 
    'project_essay_3', 
    'project_essay_4',
    'project_resource_summary'], axis=1)


# How are we doing with our training data transformations and feature engineering?

# In[8]:


train.head()


# As you can see, our data is now mostly numeric. Let's transform the concatenated text into numbers, too.
# 
# Next, we'll clean the text up a bit and [lemmatize](https://en.wikipedia.org/wiki/Lemmatisation) it. This step improves local CV by about .0025.

# In[9]:


w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()

def prep_text(text):
    text = text.strip().lower()
    text = re.sub('\r','', text) # \r\n returns
    text = re.sub('\n','', text) # \r\n returns
    text = re.sub('\W+',' ', text)
    text = re.sub(' i m ',' i\'m ', text)
    text = re.sub('n t ','n\'t ', text)
    text = [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)]
    return text

train['project_essay'] = train['project_essay'].apply(lambda x: prep_text(x))
test['project_essay'] = test['project_essay'].apply(lambda x: prep_text(x))


# Here's what our new text objects look like:

# In[10]:


# Note that stop words are handled by the TFIDF vectorzer, below
train['project_essay'][0:20]


# I won't try to explain [TFIDF](http://https://en.wikipedia.org/wiki/Tf%E2%80%93idf) or text vectorization in general. Follow the link in the comment below, and the links from the link, to learn more.

# In[11]:


# Learn more about NLP from Abishek: 
# https://www.kaggle.com/abhishek/approaching-almost-any-nlp-problem-on-kaggle
tfv = TfidfVectorizer(norm='l2', min_df=0,  max_features=8000, 
            strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}',
            ngram_range=(1,2), use_idf=True, smooth_idf=False, sublinear_tf=True,
            stop_words = 'english')


# Should we fit on **train** and **test** individually, to avoid leakyness? Maybe, but I'm not going to do it that way.
# 
# Note that **hstack** is the method by which we combine our engineered features with the TFIDF vectorized text.

# In[12]:


train_text = train['project_essay'].apply(lambda x: ' '.join(x))
test_text = test['project_essay'].apply(lambda x: ' '.join(x))

# Fitting tfidf on train + test might be leaky
tfv.fit(list(train_text.values) + list(test_text.values))
train_tfv = tfv.transform(train_text)
test_tfv = tfv.transform(test_text)


# In[13]:


# Combine text vectors and features
feat_train = train.drop('project_essay', axis=1)
feat_test = test.drop('project_essay', axis=1)

feat_train = csr_matrix(feat_train.values)
feat_test = csr_matrix(feat_test.values)

X_train_stack = hstack([feat_train, train_tfv[0:feat_train.shape[0]]])
X_test_stack = hstack([feat_test, test_tfv[0:feat_test.shape[0]]])

print('Train shape: ', X_train_stack.shape, '\n\nTest Shape: ', X_test_stack.shape)

del train, test, train_tfv, test_tfv; gc.collect()


# In[14]:


seed = 28 # Get your own seed


# In[15]:


K = 3 # How many folds do you want? 
kf = KFold(n_splits = K, random_state = seed, shuffle = True)


# **The XGBoost Model**
# 
# We'll set up arrays to capture our CV scores and the predictions made agains the **test** set. Those will be blended to give us more robust predictions and to prevent overfitting.

# In[16]:


cv_scores = []
xgb_preds = []

for train_index, test_index in kf.split(X_train_stack):
    
    # Split out a validation set
    X_train, X_valid, y_train, y_valid = train_test_split(X_train_stack, target, test_size=0.25, random_state=random.seed(seed))
    
    # params are tuned with kaggle kernels in mind
    xgb_params = {'eta': 0.165, 
                  'max_depth': 5, 
                  'subsample': 0.825, 
                  'colsample_bytree': 0.825, 
                  'objective': 'binary:logistic', 
                  'eval_metric': 'auc', 
                  'seed': seed
                 }
    
    d_train = xgb.DMatrix(X_train, y_train)
    d_valid = xgb.DMatrix(X_valid, y_valid)
    d_test = xgb.DMatrix(X_test_stack)
    
    watchlist = [(d_train, 'train'), (d_valid, 'valid')]
    model = xgb.train(xgb_params, d_train, 300, watchlist, verbose_eval=30, early_stopping_rounds=10)
    cv_scores.append(float(model.attributes()['best_score']))
    xgb_pred = model.predict(d_test)
    xgb_preds.append(list(xgb_pred))
    
    del X_train, X_valid, y_train, y_valid, d_train, d_valid, d_test; gc.collect()


# I like to get a look at the CV scores for each fold. Keep track of the average CV score to determine how parameter changes affect your model.

# In[17]:


print(cv_scores)
print(np.mean(cv_scores))


# Blend predictions...

# In[18]:


x_preds=[]
for i in range(len(xgb_preds[0])):
    sum=0
    for j in range(K):
        sum+=xgb_preds[j][i]
    x_preds.append(sum / K)


# ... and then take a peek to see how they compare to your last effort or other models. 

# In[19]:


# Peek at first 10 predictions
x_preds[0:10]


# In[20]:


# XGB preds
x_preds = pd.DataFrame(x_preds)
x_preds.columns = ['project_is_approved']


# **Combine and Save**

# In[21]:


submid = sub['id']

submission = pd.concat([submid, x_preds], axis=1)

submission.to_csv('xgb_submission.csv', index=False)

