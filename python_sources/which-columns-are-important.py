#!/usr/bin/env python
# coding: utf-8

# # Instant Gratification
# ## EDA by category
# There are many public kernels demonstrating that the dataset may be best treated as 512 different datasets. Furthermore, many of the numeric columns are of little value for most of the categories, but for each category a different c40 numeric variables contain most (possibly all) of the predictive value. These 'important' columns are identified by having a higher standard deviation than the other columns.
# 
# In this kernel, I will select three categories and perform EDA for the data that relates to these categories. I will also look to fit a model (K-Nearest Neighbours and SVMs seem to perform well for this dataset) to this data three times:
# - once using all numeric feature columns
# - once using just the c40 'important' feature columns'
# - once using the 'unimportant' feature columns.
# 
# If the 'unimportant' numeric columns have no predictive value, then the fitted models should perform no better than random chance.

# In[ ]:


from IPython.display import HTML

HTML('''<script>
code_show=true; 
function code_toggle() {
 if (code_show){
 $('div.input').hide();
 } else {
 $('div.input').show();
 }
 code_show = !code_show
} 
$( document ).ready(code_toggle);
</script>
<form action="javascript:code_toggle()"><input type="submit" value="Click here to toggle on/off the raw code."></form>''')


# ### Set up libraries and import data

# In[ ]:


import time
t_Start = time.time() # start the clock


# In[ ]:


# Import libraries

import numpy as np
import pandas as pd
from IPython.display import display
pd.options.display.max_columns = None # Displays all columns and when showing dataframes
import warnings
warnings.filterwarnings("ignore") # Hide warnings
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import time
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score
from collections import Counter
from sklearn.linear_model import LogisticRegression


# In[ ]:


# Import the data

t0 = time.time()

train = pd.read_csv('../input/train.csv')
#test = pd.read_csv('../input/test.csv') Don't import test dataset as anticipate only performing EDA in this kernel and not making submission

t1 = time.time()

print('Data imported - time taken %3.1f seconds' % (t1-t0))


# In[ ]:


# Redo the dataframe index

train = train.set_index('id')


# ### Select three categories for analysis
# I will randomly select three categories to analyse and treat these as defining three separate training datasets.

# In[ ]:


# Select three categories at random
np.random.seed(42)
cat = np.random.randint(0,512,3)
print('I will analyse categories %i, %i and %i' % (cat[0],cat[1],cat[2]))


# In[ ]:


# Create data subsets
train_1 = train[train['wheezy-copper-turtle-magic']==cat[0]].drop('wheezy-copper-turtle-magic',axis=1)
train_2 = train[train['wheezy-copper-turtle-magic']==cat[1]].drop('wheezy-copper-turtle-magic',axis=1)
train_3 = train[train['wheezy-copper-turtle-magic']==cat[2]].drop('wheezy-copper-turtle-magic',axis=1)


# ### Identifying the 'important' columns for each category

# In[ ]:


# plot the standard deviation
fig, ax = plt.subplots(1, 3, figsize=(15, 3));
ax[0].hist(train_1.describe().loc['std'],bins=100);
ax[0].set_title('Distribution of sd - category A');
ax[0].set_xlabel('Standard Deviation');
ax[0].set_ylabel('Count');
ax[1].hist(train_2.describe().loc['std'],bins=100);
ax[1].set_title('Distribution of sd - category B');
ax[1].set_xlabel('Standard Deviation');
ax[1].set_ylabel('Count');
ax[2].hist(train_3.describe().loc['std'],bins=100);
ax[2].set_title('Distribution of sd - category C');
ax[2].set_xlabel('Standard Deviation');
ax[2].set_ylabel('Count');


# As expected, there are a number of columns with a standard deviation around 3.75 and a greater number with standard deviation around 1.0. Let's look at which columns are in the first group.

# In[ ]:


count = 0
count += sum(train_1.describe().loc['std'].between(2,3))
count += sum(train_2.describe().loc['std'].between(2,3))
count += sum(train_3.describe().loc['std'].between(2,3))

print('There are %i columns with standard deviation between 2 and 3' % count)


# For the three examples selected, none of the columns have a standard deviation between 2 and 3. Therefore, will use 2.5 as the dividing line.

# The columns with a standard deviation above 2.5 for the first datasets is:

# In[ ]:


important_cols1 = train_1.columns[train_1.describe().loc['std']>2.5]
important_cols2 = train_2.columns[train_2.describe().loc['std']>2.5]
important_cols3 = train_3.columns[train_3.describe().loc['std']>2.5]


# In[ ]:


print('There are %i, %i and %i \'important\' columns for each of the three categories' % (len(important_cols1),len(important_cols2),len(important_cols3)))


# In[ ]:


all_names = {}
for col in train.columns:
    col_split_list = col.split("-")
    for i in range(len(col_split_list)):
        if col_split_list[i] in all_names:
            all_names[col_split_list[i]] += 1/(4*(len(train.columns)-1))
        else:
            all_names[col_split_list[i]] = 1/(4*(len(train.columns)-1))            

names_1={}
for col in important_cols1:
    col_split_list = col.split("-")
    for i in range(len(col_split_list)):
        if col_split_list[i] in names_1:
            names_1[col_split_list[i]] += 1/(4*len(important_cols1))
        else:
            names_1[col_split_list[i]] = 1/(4*len(important_cols1))

names_2={}
for col in important_cols2:
    col_split_list = col.split("-")
    for i in range(len(col_split_list)):
        if col_split_list[i] in names_2:
            names_2[col_split_list[i]] += 1/(4*len(important_cols1))
        else:
            names_2[col_split_list[i]] = 1/(4*len(important_cols1))

names_3={}
for col in important_cols3:
    col_split_list = col.split("-")
    for i in range(len(col_split_list)):
        if col_split_list[i] in names_3:
            names_3[col_split_list[i]] += 1/(4*len(important_cols1))
        else:
            names_3[col_split_list[i]] = 1/(4*len(important_cols1))
            
k = Counter(all_names)
high = k.most_common(10)
high_labels=[]
high_values=[]
for i in range(10):
    high_labels.append(high[i][0])
    high_values.append(high[i][1])

k1 = Counter(names_1)
high1 = k1.most_common(10)
high1_labels=[]
high1_values=[]
for i in range(10):
    high1_labels.append(high1[i][0])
    high1_values.append(high1[i][1])

k2 = Counter(names_2)
high2 = k2.most_common(10)
high2_labels=[]
high2_values=[]
for i in range(10):
    high2_labels.append(high2[i][0])
    high2_values.append(high2[i][1])

k3 = Counter(names_3)
high3 = k3.most_common(10)
high3_labels=[]
high3_values=[]
for i in range(10):
    high3_labels.append(high3[i][0])
    high3_values.append(high3[i][1])
    

fig, ax = plt.subplots(2, 2, figsize=(15, 15));
ax[0,0].bar(range(10), high_values, align='center');
ax[0,1].bar(range(10), high1_values, align='center');
ax[0,0].set_title('Ten most frequent words in column names - all data')
ax[0,0].set_xticks(range(10));
ax[0,0].set_xticklabels(high_labels);
for tick in ax[0,0].get_xticklabels():
    tick.set_rotation(45)
ax[0,1].set_title('Ten most frequent words in column names - category A')
ax[0,1].set_xticks(range(10));
ax[0,1].set_xticklabels(high1_labels);
for tick in ax[0,1].get_xticklabels():
    tick.set_rotation(45)
ax[1,0].bar(range(10), high2_values, align='center');
ax[1,1].bar(range(10), high3_values, align='center');
ax[1,0].set_title('Ten most frequent words in column names - category B')
ax[1,0].set_xticks(range(10));
ax[1,0].set_xticklabels(high2_labels);
for tick in ax[1,0].get_xticklabels():
    tick.set_rotation(45)
ax[1,1].set_title('Ten most frequent words in column names - category C')
ax[1,1].set_xticks(range(10));
ax[1,1].set_xticklabels(high3_labels);
for tick in ax[1,1].get_xticklabels():
    tick.set_rotation(45)


# Whilst different column names occur most frequently in the total dataset and for the important columns, I can see no obvious pattern in the column names for the three categories analysed.

# ### Fit a model for each category to all the columns
# If the important columns are the only ones that matter for a single category, then:
# - a model using just these columns should perform similarly (or better) than a model using all the numerical columns.
# - a model using the unimportant columns should perform no better than chance.
# 
# Note: Even if the above hold true, the unimportant columns may have value if there are interactions between categories.

# In[ ]:


# For each category, create three training arrays and results array

X_train1_all = train_1.drop('target',axis=1)
X_train1_imp = X_train1_all[important_cols1]
X_train1_unimp = X_train1_all.drop(important_cols1,axis=1)
y_train1 = train_1['target']

X_train2_all = train_2.drop('target',axis=1)
X_train2_imp = X_train2_all[important_cols2]
X_train2_unimp = X_train2_all.drop(important_cols2,axis=1)
y_train2 = train_2['target']

X_train3_all = train_3.drop('target',axis=1)
X_train3_imp = X_train3_all[important_cols3]
X_train3_unimp = X_train3_all.drop(important_cols3,axis=1)
y_train3 = train_3['target']

# For each category, train three models.
# Use SVM as has been shown to perform fairly well

svm1_all = SVC(probability=True,kernel='poly',degree=4,gamma='auto').fit(X_train1_all,y_train1)
svm1_imp = SVC(probability=True,kernel='poly',degree=4,gamma='auto').fit(X_train1_imp,y_train1)
svm1_unimp = SVC(probability=True,kernel='poly',degree=4,gamma='auto').fit(X_train1_unimp,y_train1)

svm2_all = SVC(probability=True,kernel='poly',degree=4,gamma='auto').fit(X_train2_all,y_train2)
svm2_imp = SVC(probability=True,kernel='poly',degree=4,gamma='auto').fit(X_train2_imp,y_train2)
svm2_unimp = SVC(probability=True,kernel='poly',degree=4,gamma='auto').fit(X_train2_unimp,y_train2)

svm3_all = SVC(probability=True,kernel='poly',degree=4,gamma='auto').fit(X_train3_all,y_train3)
svm3_imp = SVC(probability=True,kernel='poly',degree=4,gamma='auto').fit(X_train3_imp,y_train3)
svm3_unimp = SVC(probability=True,kernel='poly',degree=4,gamma='auto').fit(X_train3_unimp,y_train3)

print('All models trained')


# In[ ]:


# Work out the AUC score for each model using 5-fold cross validation
all_auc_1 = cross_val_score(svm1_all, X_train1_all, y_train1, scoring='roc_auc', cv=5).mean()
imp_auc_1 = cross_val_score(svm1_imp, X_train1_imp, y_train1, scoring='roc_auc', cv=5).mean()
unimp_auc_1 = cross_val_score(svm1_unimp, X_train1_unimp, y_train1, scoring='roc_auc', cv=5).mean()

all_auc_2 = cross_val_score(svm2_all, X_train2_all, y_train2, scoring='roc_auc', cv=5).mean()
imp_auc_2 = cross_val_score(svm2_imp, X_train2_imp, y_train2, scoring='roc_auc', cv=5).mean()
unimp_auc_2 = cross_val_score(svm2_unimp, X_train2_unimp, y_train2, scoring='roc_auc', cv=5).mean()

all_auc_3 = cross_val_score(svm3_all, X_train3_all, y_train3, scoring='roc_auc', cv=5).mean()
imp_auc_3 = cross_val_score(svm3_imp, X_train3_imp, y_train3, scoring='roc_auc', cv=5).mean()
unimp_auc_3 = cross_val_score(svm3_unimp, X_train3_unimp, y_train3, scoring='roc_auc', cv=5).mean()

cat1 = [all_auc_1,imp_auc_1,unimp_auc_1]
cat2 = [all_auc_2,imp_auc_2,unimp_auc_2]
cat3 = [all_auc_3,imp_auc_3,unimp_auc_3]

results=[cat1,cat2,cat3]

plt.figure(figsize=(20, 10));
plt.plot(results);
plt.legend(['Using all features', 'Using only important features', 'Using only unimportant features']);
plt.title('AUC score for each model');
plt.xlabel('Category');
plt.xticks(range(3),cat);


# There does seem to be some value in the unimportant columns, although the vast majority of the information is contained in the important columns. To check this, I will create the above results for more categories.

# In[ ]:


t0 = time.time()

cat_to_test = 100 # set the number of categories to test - 10 takes around one minute to run
cat_temp = np.sort(np.random.randint(0,512,cat_to_test)) # pick the categories at random - re-order for graph below

results = np.zeros((cat_to_test,3))
i = 0

for cat in cat_temp:
    # Filter training data just for category i and identify 'important' columns
    train_temp = train[train['wheezy-copper-turtle-magic']==cat].drop('wheezy-copper-turtle-magic',axis=1)
    important_cols_temp = train_temp.columns[train_temp.describe().loc['std']>2.5]
    
    # Set up X and y datasets
    X_train_temp_all = train_temp.drop('target',axis=1)
    X_train_temp_imp = X_train_temp_all[important_cols_temp]
    X_train_temp_unimp = X_train_temp_all.drop(important_cols_temp,axis=1)
    y_train_temp = train_temp['target']
    
    # Train models
    svm_temp_all = SVC(probability=True,kernel='poly',degree=4,gamma='auto').fit(X_train_temp_all,y_train_temp)
    svm_temp_imp = SVC(probability=True,kernel='poly',degree=4,gamma='auto').fit(X_train_temp_imp,y_train_temp)
    svm_temp_unimp = SVC(probability=True,kernel='poly',degree=4,gamma='auto').fit(X_train_temp_unimp,y_train_temp)
    
    # Work out the AUC score for each model using 5-fold cross validation
    all_auc = cross_val_score(svm_temp_all, X_train_temp_all, y_train_temp, scoring='roc_auc', cv=5).mean()
    imp_auc = cross_val_score(svm_temp_imp, X_train_temp_imp, y_train_temp, scoring='roc_auc', cv=5).mean()
    unimp_auc = cross_val_score(svm_temp_unimp, X_train_temp_unimp, y_train_temp, scoring='roc_auc', cv=5).mean()
    
    # Add results to array
    res_temp = [all_auc,imp_auc,unimp_auc]
    results[i,:] = res_temp
    i += 1
    
    if i%50 == 0:
        print('Running category %i out of %i' % ((i+1),cat_to_test))

# Plot the results
plt.figure(figsize=(20, 10));
plt.plot(results);
plt.legend(['Using all features', 'Using only important features', 'Using only unimportant features']);
plt.title('AUC score for each model');
plt.xlabel('Category');
plt.xticks(range(cat_to_test),cat_temp);

t1 = time.time()
print('Total run time = %i minutes and %3.1f seconds' % ((t1-t0)//60,(t1-t0)%60))


# In[ ]:


# Print the average AUC score across all categories
print('Models including all columns have an average AUC score of %3.3f' % np.mean(results[:,0]))
print('Models including just important columns have an average AUC score of %3.3f' % np.mean(results[:,1]))
print('Models including just unimportant columns have an average AUC score of %3.3f' % np.mean(results[:,2]))


# The models with all numercial columns included score higher than those using just the important columns. However, models using just the unimportant columns do not score any better than chance. This suggests the unimportant features only have value when used in conjunction with the important columns. On their own, they have no value.
# 
# To see if this result is genuine (or if due to dimensionality or regularisation as suggested by Chris Deott in the comments), I will replace the values in the unimportant columns with samples from a standard Normal distribution and see if the results are replicated.

# In[ ]:


t0 = time.time()

cat_to_test = 100 # set the number of categories to test - 10 takes around one minute to run
cat_temp = np.sort(np.random.randint(0,512,cat_to_test)) # pick the categories at random - re-order for graph below

results = np.zeros((cat_to_test,3))
i = 0

for cat in cat_temp:
    # Filter training data just for category i and identify 'important' columns
    train_temp = train[train['wheezy-copper-turtle-magic']==cat].drop('wheezy-copper-turtle-magic',axis=1)
    important_cols_temp = train_temp.columns[train_temp.describe().loc['std']>2.5]
    
    # Set up X and y datasets
    X_train_temp_all = train_temp.drop('target',axis=1)
    X_train_temp_imp = X_train_temp_all[important_cols_temp]
    X_train_temp_unimp = X_train_temp_all.drop(important_cols_temp,axis=1)
    
    # Replace values in unimportant columns with random Normal samples
    m = np.shape(X_train_temp_all)[0]
    unimportant_cols = X_train_temp_unimp.columns
    for col in unimportant_cols:
        X_train_temp_all[col] = np.random.normal(size=m)
   
    y_train_temp = train_temp['target']
    
    # Train models
    svm_temp_all = SVC(probability=True,kernel='poly',degree=4,gamma='auto').fit(X_train_temp_all,y_train_temp)
    svm_temp_imp = SVC(probability=True,kernel='poly',degree=4,gamma='auto').fit(X_train_temp_imp,y_train_temp)
    svm_temp_unimp = SVC(probability=True,kernel='poly',degree=4,gamma='auto').fit(X_train_temp_unimp,y_train_temp)
    
    # Work out the AUC score for each model using 5-fold cross validation
    all_auc = cross_val_score(svm_temp_all, X_train_temp_all, y_train_temp, scoring='roc_auc', cv=5).mean()
    imp_auc = cross_val_score(svm_temp_imp, X_train_temp_imp, y_train_temp, scoring='roc_auc', cv=5).mean()
    unimp_auc = cross_val_score(svm_temp_unimp, X_train_temp_unimp, y_train_temp, scoring='roc_auc', cv=5).mean()
    
    # Add results to array
    res_temp = [all_auc,imp_auc,unimp_auc]
    results[i,:] = res_temp
    i += 1
    
    if i%50 == 0:
        print('Running category %i out of %i' % ((i+1),cat_to_test))

# Plot the results
plt.figure(figsize=(20, 10));
plt.plot(results);
plt.legend(['Adding random features', 'Using only important features', 'Using only unimportant features']);
plt.title('AUC score for each model');
plt.xlabel('Category');
plt.xticks(range(cat_to_test),cat_temp);

t1 = time.time()
print('Total run time = %i minutes and %3.1f seconds' % ((t1-t0)//60,(t1-t0)%60))


# In[ ]:


# Print the average AUC score across all categories
print('Models including random features have an average AUC score of %3.3f' % np.mean(results[:,0]))
print('Models including just important columns have an average AUC score of %3.3f' % np.mean(results[:,1]))
print('Models including just unimportant columns have an average AUC score of %3.3f' % np.mean(results[:,2]))


# The AUC score is the same if the useless columns are re-populated with noise. Therefore, it seems likely these have no value for each category and should be removed to avoid over-fitting.

# ### Important column names
# In spite of the brief analysis above, I do wonder if there is something in the column names of the important columns. In particular, I wonder:
# - if there are a minimum number that are important and some are actually noise
# - if multiple columns could be re-combined to create a single feature which was predictive across categories.
# 
# This theory relies on there being some information in the names. The fourth word seems the most promising (primarily because there are relatively few unique words) so I will concentrate here.

# In[ ]:


# Create a dataframe of the counts of important and unimportant columns by the fourth word
t0 = time.time()
# For each category:
# - identify the important columns
# - count the unique column names
# - add this to a dictionary
# - create a dataframe with this information for each category

# Create a dictionary of all fourth words (with count of zero):
words_dict={}

for col in train.columns.drop('target'):
    words_dict[col.split("-")[3]] = 0

    
results = pd.DataFrame.from_dict(words_dict,orient='index')

for i in range(512):
    # Filter training data just for category i and identify 'important' columns
    train_temp = train[train['wheezy-copper-turtle-magic']==i].drop('wheezy-copper-turtle-magic',axis=1)
    important_cols_temp = train_temp.columns[train_temp.describe().loc['std']>2.5]
    
    # Create a list of the fourth word for each important column
    words_temp = []
    
    for col in important_cols_temp:
        words_temp.append(col.split("-")[3])
    
    # Create a dictionary of counts
    for item in words_dict.keys():
        if item in words_temp:
            words_dict[item] = words_temp.count(item)
        else:
            words_dict[item] = 0
    
    results[i] = pd.DataFrame.from_dict(words_dict,orient='index')
    
t1 = time.time()
print('Total run time = %i minutes and %3.1f seconds' % ((t1-t0)//60,(t1-t0)%60))    


# In[ ]:


print('The number of important columns for each category ranges from %i to %i' % (results.sum(axis=0).min(),results.sum(axis=0).max()))


# In[ ]:


# Show the number of categories with x important columns
x = np.unique(results.sum(axis=0),return_counts=True)[0]
y = np.unique(results.sum(axis=0),return_counts=True)[1]
plt.figure(figsize=(20, 10));
plt.bar(x,y);
plt.title('Number of important features per category');
plt.xlabel('Number of important features');
plt.ylabel('Count');


# There is no obvious distribution to the number of important features within a category. It is possible this could be a sample from a random uniform integer distribution between 33 and 47.

# In[ ]:



least_common = results.sum(axis=1).sort_values().head(2).tail(1).index[0]
least_count = results.sum(axis=1).sort_values()[least_common]
most_common = results.sum(axis=1).sort_values().tail(1).index[0]
most_count = results.sum(axis=1).sort_values()[most_common]

print('The least common word ,other than magic, is %s (%i times) and the most common word is %s (%i times)' % (least_common,least_count,most_common,most_count))


# In[ ]:


plt.figure(figsize=(20, 10));
plt.title('Frequency per category');
plt.xlabel('Fourth column word');
plt.xticks(range(len(results.index)),results.index,rotation=45);
plt.plot(results.max(axis=1),marker='*',label='Maximum frequency');
plt.plot(results.mean(axis=1),marker='*',label='Average frequency');
plt.plot(results.min(axis=1),marker='*',label='Minimum frequency');
plt.legend();


# For every word, there is at least one categoy where that word is totally absent. Also, for every word, there is at least one category where that word occurs between 4 and 8 times (other than magic).
# 
# Let's try removing the following words:
# - distraction
# - noise
# - discard
# - dummy

# In[ ]:


results_2 = results.drop(['distraction','noise','discard','dummy'])
print('The number of important columns ranges from %i to %i' % (results_2.sum(axis=0).min(),results_2.sum(axis=0).max()))


# There is one category with at least 10 important features with these four names. However, there is at least one column with none of these. This doesn't seem to help create a list of truly important columns. Let's fit a different model to the three categories to analyse and look at feature importance (feature importance only works for SVM if linear kernel is used).

# In[ ]:


# Fit logistic regression models

lr_imp_1 = LogisticRegression().fit(X_train1_imp,y_train1)
lr_imp_2 = LogisticRegression().fit(X_train2_imp,y_train2)
lr_imp_3 = LogisticRegression().fit(X_train3_imp,y_train3)

#Test the AUC to check the models have some predictive ability

auc_1 = cross_val_score(lr_imp_1, X_train1_imp, y_train1, scoring='roc_auc', cv=5).mean()
auc_2 = cross_val_score(lr_imp_2, X_train2_imp, y_train2, scoring='roc_auc', cv=5).mean()
auc_3 = cross_val_score(lr_imp_3, X_train3_imp, y_train3, scoring='roc_auc', cv=5).mean()

print('The AUC score for the three categories is %3.3f, %3.3f and %3.3f' % (auc_1,auc_2,auc_3))


# In[ ]:


# Feature importance for each category
feature_importance_1 = abs(lr_imp_1.coef_[0])
feature_importance_1 = 100.0 * (feature_importance_1 / feature_importance_1.max())
sorted_idx_1 = np.argsort(feature_importance_1)
pos_1 = np.arange(sorted_idx_1.shape[0]) + .5

feature_importance_2 = abs(lr_imp_2.coef_[0])
feature_importance_2 = 100.0 * (feature_importance_2 / feature_importance_2.max())
sorted_idx_2 = np.argsort(feature_importance_2)
pos_2 = np.arange(sorted_idx_2.shape[0]) + .5

feature_importance_3 = abs(lr_imp_3.coef_[0])
feature_importance_3 = 100.0 * (feature_importance_3 / feature_importance_3.max())
sorted_idx_3 = np.argsort(feature_importance_3)
pos_3 = np.arange(sorted_idx_3.shape[0]) + .5

fig, ax = plt.subplots(1, 3, figsize=(15, 20));
ax[0].barh(pos_1, feature_importance_1[sorted_idx_1], align='center',alpha=0.3);
ytick_labels_1 = np.array(X_train1_imp.columns)[sorted_idx_1]
ax[0].set_xlabel('Relative Feature Importance');
ax[0].set_title('Feature importance for category A');
ax[0].set_yticklabels('')
for i, v in enumerate(ytick_labels_1):
    ax[0].text(5, i + .25, str(v), color='black', fontweight='bold')

ax[1].barh(pos_2, feature_importance_2[sorted_idx_2], align='center',alpha=0.3);
ytick_labels_2 = np.array(X_train2_imp.columns)[sorted_idx_2]
ax[1].set_xlabel('Relative Feature Importance');
ax[1].set_title('Feature importance for category B');
ax[1].set_yticklabels('')
for i, v in enumerate(ytick_labels_2):
    ax[1].text(5, i + .25, str(v), color='black', fontweight='bold')

ax[2].barh(pos_3, feature_importance_3[sorted_idx_3], align='center',alpha=0.3);
ytick_labels_3 = np.array(X_train3_imp.columns)[sorted_idx_3]
ax[2].set_xlabel('Relative Feature Importance');
ax[2].set_title('Feature importance for category C');
ax[2].set_yticklabels('')
for i, v in enumerate(ytick_labels_3):
    ax[2].text(5, i + .25, str(v), color='black', fontweight='bold')


# I can see no pattern in the features which are important nor those which are unimportant. I can only conclude that the column names are meaningless (other than the 'magic' column).
# Let's try creating a database for all words and seeing if any are more common (or less common) in important features.

# In[ ]:


# Create a dataframe of the counts of important and unimportant columns by each word in the column names
t0 = time.time()

# Create a dictionary of all words (with count of zero):
words_dict={}
unimp_dict={}

for j in range(4):
    for col in train.columns.drop('target'):
        words_dict[col.split("-")[j]] = 0
        unimp_dict[col.split("-")[j]] = 0

imp_results = pd.DataFrame.from_dict(words_dict,orient='index')
unimp_results = pd.DataFrame.from_dict(words_dict,orient='index')

for i in range(512):
    # Filter training data just for category i and identify 'important' columns
    train_temp = train[train['wheezy-copper-turtle-magic']==i].drop(['wheezy-copper-turtle-magic','target'],axis=1)
    important_cols_temp = train_temp.columns[train_temp.describe().loc['std']>2.5]
    unimportant_cols_temp = train_temp.columns[train_temp.describe().loc['std']<2.5]   
    # Create a list of the fourth word for each important column
    words_temp = []
    unimp_temp=[]
    for j in range(4):
        for col in important_cols_temp:
            words_temp.append(col.split("-")[j])
        for col in unimportant_cols_temp:
            unimp_temp.append(col.split("-")[j])
    
    # Create a dictionary of counts
    for item in words_dict.keys():
        if item in words_temp:
            words_dict[item] = words_temp.count(item)
        else:
            words_dict[item] = 0
    imp_results[i] = pd.DataFrame.from_dict(words_dict,orient='index')
    
    # Create a dictionary of counts
    for item in unimp_dict.keys():
        if item in unimp_temp:
            unimp_dict[item] = unimp_temp.count(item)
        else:
            unimp_dict[item] = 0
    
    unimp_results[i] = pd.DataFrame.from_dict(unimp_dict,orient='index')

t1 = time.time()
print('Total run time = %i minutes and %3.1f seconds' % ((t1-t0)//60,(t1-t0)%60))    


# In[ ]:


freq = (np.sum(imp_results,axis=1)/(np.sum(imp_results,axis=1)+np.sum(unimp_results,axis=1))).rename('freq')
count = (imp_results[0]+unimp_results[0]).rename('count')
summary = pd.concat([freq,count],axis=1).sort_values(by='freq',ascending=True).drop('magic',axis=0)

fig = plt.figure(figsize=(20,10))

cx0 = fig.add_subplot(121)
cx1 = cx0.twinx()
cx2 = plt.subplot(122)
cx3 = cx2.twinx()

summary.head(20)['freq'].plot(ax=cx0)
summary.head(20)['count'].plot(ax=cx1, kind='bar', secondary_y=True,alpha=0.3)
summary.tail(20)['freq'].plot(ax=cx2)
summary.tail(20)['count'].plot(ax=cx3, kind='bar', secondary_y=True,alpha=0.3)

cx0.set_title('The twenty words which appear least often (by proportion) in important columns');
cx2.set_title('The twenty words which appear most often (by proportion) in important columns');
cx0.set_ylabel('Proportion word appears in important column');
cx0.set_xticklabels(summary.head(20).index,rotation=45);
cx1.set_ylabel('Total times word appears');
cx2.set_ylabel('Proportion word appears in important column');
cx3.set_ylabel('Total times word appears');
cx2.set_xticklabels(summary.tail(20).index,rotation=45);


# In[ ]:


plt.figure(figsize=(20, 10));
plt.title('Distributon of frequencies');
plt.xlabel('Frequency as important column');
plt.hist(summary['freq'],bins=75);


# There are no obvious trends or anomalies that I can see (although 'cat' has a slight gap to second place in terms of being important most frequently). However, many words are only present in one column. Let's look again just at the fourth word, which is more often repeated.

# In[ ]:


fourth_word = {}

for col in train.columns.drop('target'):
    fourth_word[col.split("-")[j]] = 0

fourth_wrd = []
for key in fourth_word.keys():
    fourth_wrd.append(key)
    
summary_2 = summary.loc[fourth_wrd].sort_values(by='freq')


# In[ ]:


fig = plt.figure(figsize=(25,10))

cx0 = fig.add_subplot(121)
cx1 = cx0.twinx()

summary_2['freq'].plot(ax=cx0)
summary_2['count'].plot(ax=cx1, kind='bar', secondary_y=True,alpha=0.3)

cx0.set_title('The fourth words and how often they appear in important columns');
cx0.set_ylabel('Proportion word appears in important column');
cx0.set_xticklabels(summary_2.index,rotation=45);
cx1.set_ylabel('Total times word appears');


# Still no obvious conclusions to be drawn!! Dummy is the least often important but distraction is the most often important.

# ### Conclusions
# - All of the information for each category is contained within a small subset of the feature columns. The other columns have no value, unless they provide information in respect of interactions between categories.
# - The column names do not seem to have any use, other than identifying the 'magic' column

# In[ ]:


t_End = time.time()
print('Total run time = %i minutes and %3.1f seconds' % ((t_End-t_Start)%60,(t_End-t_Start)//60))

