#!/usr/bin/env python
# coding: utf-8

# ## UMUC DATA 650 Kaggle

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

train = pd.read_csv('../input/cars-train.csv')
val = pd.read_csv('../input/cars-test.csv')

train.head()


# In[ ]:


train = train.drop(['car.id'], axis=1)
val = val.drop(['car.id'], axis=1)


# In[ ]:


x_train = train.drop(['class'], axis = 1)
y_train = train['class']
x_val = val.drop(['class'], axis = 1)
y_val = val['class']


# In[ ]:


train.describe(include='all')


# From above, we can see there is no missing data to worry about.  Cool!

# **Below are histograms for each variable in the training data**

# In[ ]:


def plot_all_hists_df(df):
    size = df.shape[1]
    col_num = 4
    row_num = int(np.ceil(size/4))
    fig = plt.figure(figsize=(9,5))
    for i, name in enumerate(df):
        ax=fig.add_subplot(row_num,col_num,i+1)
        plt.hist(list(df[name]))
        ax.set_title(name)
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.show()

plot_all_hists_df(train)


# **Now histograms for our validation data**

# In[ ]:


plot_all_hists_df(val)


# In[ ]:


print('Total number of each times each category of class shows up training set:')
print(train['class'].value_counts())
print('*****')
print('Total number of each times each category of class shows up validation set:')
print(val['class'].value_counts())


# We'll need this to normalize a histogram later on, so below we save it to a list.

# In[ ]:


class_counts_train = [680, 216, 38, 36]
class_counts_val = [227, 72, 13, 12]


# So all dependent variables seem to be relatively evenly distributed and the target variable is heavily skewed towards unacceptable.  Having such evenly distributed independent variables but skewed dependent variables does not look good at first glance but it could also work out well.  Also, there are only 4\*4\*4\*3\*3\*3\*3\*4 = 5184 total combinations of *all* variables and only only 1728 total combinations of all independent variables.  Since we have 970 total rows of data, if each row has a unique combination of dependent variables and each unique combination to dependent variables maps to a single unique independent variable, then a trained model should be able to achieve high accuracy on the validation and test sets due to the fact that we *know* for sure over half of all possible data points and there is no ambiguity in the data.  The below code calculates the total number of unique combinations of independent variables in our data.
# 

# In[ ]:


total_tuples = set()
for index, row in train.iterrows():
    total_tuples.add((row[0],row[1],row[2],row[3],row[4],row[5]))
print('Total number of unique combinations of independent variables: {}'.format(len(total_tuples)))


# Since there are 970 total rows of data, the above tells us that all 970 rows of data for our independent variables (so, excluding the 'class' variable) are all unique.  So, despite the imbalanced data set it should be possible to achieve 100% accuracy on the training set just using decision trees--although such a model will likely be very overfitting.  It gives hope that we can get a decent, generalizable model from this data, however, especially since the histogram plots all looks relatively evenly distributed outside of the target variable.
# 
# Below we create a copy of our train/validation dataframes where we use integers to order ever variables categories from least to most.  Since there is a natural ordering to *all* our categorical variables, we can preserve their relationship to each other by giving them integer values ranking them from lowest to highest.  This is done to give a natural ordering to our next histogram charts and also for model building as there are a ton of classifiers in sklearn that require ordinal variables for the target variable to even train.

# In[ ]:


train_num = train.replace({'unacc': 0, 'acc':1, 'good':2, 'vgood': 3, 'low':0, 'med':1, 'high':2, 'vhigh':3, 'small':0, 'big':2, "2":0, "3":1, "4":2, "more":3, "5more":3})
val_num = val.replace({'unacc': 0, 'acc':1, 'good':2, 'vgood': 3, 'low':0, 'med':1, 'high':2, 'vhigh':3, 'small':0, 'big':2, "2":0, "3":1, "4":2, "more":3, "5more":3})


# In[ ]:


train_num.head()


# Next we pair the values of each variable with it's corresponding class value so that we can get an idea of how each variable affects class

# In[ ]:


varXclass = list()
depvar_name = [('class X '+name) for name in train_num]
name = [x for x in train_num]
depvar_name = depvar_name[:6]
for i in range(6): varXclass.append(list())
for index, row in train_num.iterrows():
    for i in range(6):
        varXclass[i].append('class: '+str(row[6])+' X '+name[i]+' '+str(row[i]))


# In[ ]:


for i, x in enumerate(varXclass): 
    print(depvar_name[i]+' unique combos: {}'.format(len(set(x))))
    varXclass[i].sort()


# In[ ]:


def combos_to_hist(list_of_sets, names):
    fig = plt.figure(figsize=(15,10))
    num_cols = 4
    num_rows = int(np.ceil(len(list_of_sets)/4))
    for i, sets in enumerate(list_of_sets):
        ax=fig.add_subplot(num_rows,num_cols,i+1)
        plt.hist(sets)
        plt.xticks(fontsize = 12,rotation='vertical')
        ax.set_title(names[i])
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.show()
    
combos_to_hist(varXclass, depvar_name)


# From these histograms, it looks like there is a relationship between our independent variables and the class variable.  However, the class variable is extremely skewed so it might be better to normalize the counts in the histogram by their respective target variable count.

# In[ ]:


def freq_dict(in_list):
    return {i:in_list.count(i) for i in set(in_list)}

freq_dict_list=[]

for combo in varXclass:
    combo_freq_dict = freq_dict(combo)
    for pair in combo_freq_dict:
        norm = class_counts_train[int(pair[7])]
        combo_freq_dict[pair] = combo_freq_dict[pair]/norm
    freq_dict_list.append(combo_freq_dict)

def dict_to_two_lists(input_dict):
    key_list = []
    value_list = []
    for key, value in input_dict.items():
        key_list.append(key)
        value_list.append(value)
    return key_list, value_list

def freq_dict_list_hist(input_list, names):
    size = len(input_list)
    col_num = 4
    row_num = int(np.ceil(size/4))
    fig = plt.figure(figsize=(15,10))
    for i, freq_dict in enumerate(input_list):
        ax=fig.add_subplot(row_num,col_num,i+1)
        key_list, value_list = dict_to_two_lists(freq_dict)
        plt.bar(key_list, value_list)
        plt.xticks(fontsize = 12,rotation='vertical')
        ax.set_title(names[i])
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.show()
    
freq_dict_list_hist(freq_dict_list, depvar_name)


# **Ouch.**  Let's hope there is a more distinct relationship between multiple combinations of the independent variables and their class.  Next we'll get into actually building and testing models.

# In[ ]:


x_train = pd.get_dummies(x_train)
y_train = pd.get_dummies(y_train)
x_val = pd.get_dummies(x_val)
y_val = pd.get_dummies(y_val)

x_train_num = train_num.drop(['class'], axis=1)
y_train_num = train_num['class']
x_val_num = val_num.drop(['class'], axis=1)
y_val_num = val_num['class']


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

clf = RandomForestClassifier(n_estimators = 500)
clf.fit(x_train, y_train)
print(classification_report(y_train, clf.predict(x_train)))
print(classification_report(y_val, clf.predict(x_val)))
print(accuracy_score(y_val, clf.predict(x_val)))


# In[ ]:


clf = RandomForestClassifier(n_estimators = 500)
clf.fit(x_train_num, y_train_num)
print(classification_report(y_train_num, clf.predict(x_train_num)))
print(classification_report(y_val_num, clf.predict(x_val_num)))
print(accuracy_score(y_val_num, clf.predict(x_val_num)))


# In[ ]:


from sklearn.naive_bayes import MultinomialNB

clf = MultinomialNB()
clf.fit(x_train, y_train_num)
print(classification_report(y_train_num, clf.predict(x_train)))
print(classification_report(y_val_num, clf.predict(x_val)))
print(accuracy_score(y_val_num, clf.predict(x_val)))


# In[ ]:


from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()
clf.fit(x_train, y_train_num)
print(classification_report(y_train_num, clf.predict(x_train)))
print(classification_report(y_val_num, clf.predict(x_val)))
print(accuracy_score(y_val_num, clf.predict(x_val)))


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier

clf = KNeighborsClassifier()
clf.fit(x_train, y_train)
print(classification_report(y_train, clf.predict(x_train)))
print(classification_report(y_val, clf.predict(x_val)))
print(accuracy_score(y_val, clf.predict(x_val)))


# In[ ]:


clf = KNeighborsClassifier()
clf.fit(x_train_num, y_train_num)
print(classification_report(y_train_num, clf.predict(x_train_num)))
print(classification_report(y_val_num, clf.predict(x_val_num)))
print(accuracy_score(y_val_num, clf.predict(x_val_num)))


# In[ ]:


from xgboost import XGBClassifier

clf = XGBClassifier(n_estimators = 1000)
clf.fit(x_train_num, y_train_num)
print(classification_report(y_train_num, clf.predict(x_train_num)))
print(classification_report(y_val_num, clf.predict(x_val_num)))
print(accuracy_score(y_val_num, clf.predict(x_val_num)))


# Welp.  I guess that's it boys.  Not sure how I'll do much better than that.  Guess I'll tune the parameters below just to maximize it before submitting.

# In[ ]:


n_range = []
acc_score = []
for n in range(100, 3000, 100):
    n_range.append(n)
    clf = XGBClassifier(n_estimators = n)
    clf.fit(x_train_num, y_train_num)
    acc_score.append(accuracy_score(y_val_num, clf.predict(x_val_num)))

plt.title("N estimators to accuracy")
plt.plot(n_range, acc_score, ls='-', marker='o', color='red', label='One-hot encoded')
plt.legend()


# In[ ]:


from xgboost import XGBClassifier

clf = XGBClassifier(n_estimators = 650)
clf.fit(x_train_num, y_train_num)
print(classification_report(y_train_num, clf.predict(x_train_num)))
print(classification_report(y_val_num, clf.predict(x_val_num)))
print(accuracy_score(y_val_num, clf.predict(x_val_num)))


# LOL.  I was going to draft a few models in keras for fun but that just seems stupid now.  My intuition before starting this was that numerically encoding the categorical variables and applying an ensemble decision tree model (my guess was random forest but I guess gradient boosted takes the cake this time) would work well but 100% accuracy on the test set is pretty ridiculous.

# In[ ]:


test = pd.read_csv('../input/cars-final-prediction.csv')
test.head()


# In[ ]:


test_num = test.replace({'low':0, 'med':1, 'high':2, 'vhigh':3, 'small':0, 'big':2, "2":0, "3":1, "4":2, "more":3, "5more":3})
test_num.head()


# In[ ]:


test_num['num_predictions']=clf.predict(test_num.drop(['car.id'], axis=1))


# In[ ]:


test_num['class']=test_num['num_predictions'].replace({0:'unacc', 1:'acc', 2:'good', 3:'vgood'})
test_num.head()


# In[ ]:


comp_output=test_num[['car.id', 'class']]
comp_output.head()


# In[ ]:


plt.hist(list(comp_output['class']))
comp_output['class'].value_counts()


# It looks pretty good.  Decently similar distribution to the test set with 'good' and 'vgood' slightly underrepresented.  Since those are the two with the least data, I'm guessing this model is misclassifying a few of them.

# In[47]:


comp_output.to_csv('cars-submission.csv', index=False)


# In[ ]:




