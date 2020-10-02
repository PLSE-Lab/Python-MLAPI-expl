#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import random
from math import sqrt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression


# In[2]:


df_training = pd.read_csv('../input/train.csv')


# In[3]:


df_test = pd.read_csv('../input/test.csv')


# ## Initial data preparation

# In[4]:


len(df_training)


# In[5]:


len(df_test)


# In[6]:


df_training.head()


# In[7]:


df_training.columns


# * PassengerID: row id
# * Survived: target variable (1 = yes, 0 = no)
# * Pclass: ticket class
# * Name: name
# * Sex: sex
# * Age: age
# * SibSp: number of spouses or siblings aboard
# * Parch: number of parents or children aboard
# * Ticket: ticket number
# * Fare: ticket fare
# * Cabin: assigned cabin number
# * Embarked: port from which they embarked (C = Cherbourg, Q = Queenstown, S = Southampton)
# 
# It may seem that to use the Ticket and Cabin features we need to transform the data.
# 
# The Cabin feature is composed of a letter (which correlates to the class) and a number (the specific cabin). It may be interesting to split this feature into two: a categorical feature that may be very correlated to Pclass (CabinClass) and a numerical feature (CabinNumber) that specifies the approximate position from the front to the back of the ship. The problem is that according to Titanic Deck plans there is not a direct relation between the cabin number and distance from the front. It would be useful to use the cabin number to split cabins insto front, middle and back cabins. And also in left and right. However, it is hard to find a good deck plan that indicates the actual positions of the cabins. I will keep cabin number anyway because it may indicate proximity.

# In[8]:


df_training['Cabin'].unique()


# With respect to ticket number, the optional prefix (TicketPrefix) indicates issuing office and the number (TicketNumber) can be compared for equality (sharing a cabin) or for closeness (people with cabins that are close to each other.)

# In[9]:


df_training['Ticket'].values


# I don't see how to use TicketNumber and CabinNumber as proximity features, so I will stick to TicketPrefix and CabinClass.

# In[10]:


def process_ticket(df):
    df['TicketPrefix'] = df['Ticket']
    df.loc[df['Ticket'].notnull(), 'TicketPrefix'] = df['Ticket'].apply(lambda x: x.split(' ')[0] 
                                                                                  if len(x.split(' ')) > 1
                                                                                  else 'NUMBER')
    
process_ticket(df_training)
process_ticket(df_test)


# In[11]:


df_training[['Ticket', 'TicketPrefix']].head()


# In[12]:


# For cabin I keep the first letter. There are multiple instances of rows having multiple assigned cabins. In these cases
# the first letter is the same for all the assigned cabins, except in two cases in which we have:
# F GXX
# In this case, for simplicity, I decided to keep the F letter
def process_cabin(df):
    df['CabinClass'] = df['Cabin']
    df.loc[df['Cabin'].notnull(), 'CabinClass'] = df['Cabin'].apply(lambda x: str(x)[0])
    
process_cabin(df_training)
process_cabin(df_test)


# In[13]:


df_training[['Cabin', 'CabinClass']].head()


# In[14]:


dependent = 'Survived'
categorical = ['Pclass', 'Sex', 'TicketPrefix', 'CabinClass', 'Embarked']
numerical = ['Age', 'SibSp', 'Parch', 'Fare']


# ## Initial exploration
# 
# We must take into account that there are missing values.
# 
# Looking at numerical variables first.

# In[15]:


kwargs = dict(histtype = 'stepfilled', alpha = 0.3, density = True, ec = 'k')

for n in numerical:
    df = df_training[df_training[n].notnull()]
    x = df[n].values
    y = df[dependent].values
    
    fig, ax = plt.subplots(1, 2)
    (_, bins, _) = ax[0].hist(x, **kwargs)
    ax[0].set_title(n)
    
    x_0 = x[np.where(y == 0)]
    x_1 = x[np.where(y == 1)]
    ax[1].hist(x_0, **kwargs, bins = bins)
    ax[1].hist(x_1, **kwargs, bins = bins)
    ax[1].legend(['no', 'yes'])
    ax[1].set_title(n + ' vs. survived')
    
    fig.set_figwidth(16)


# It seems that all the numerical features may provide useful information in predicting the dependent variable:
# 
# * Younger passengers are more likely to survive
# * Passengers with not too few or too many embarked siblings/spouses are more likely to survive
# * Passengers are more likely to survive if they embarked with parents/children
# * Cheaper fares are less likely to survive.
# 
# Let's take a look at the categorical features now.

# In[16]:


for c in categorical:
    df = df_training[df_training[c].notnull()]
    
    fig, ax = plt.subplots(1, 2)
    freqs = df[c].value_counts()
    labels = freqs.keys()
    ax[0].bar(range(len(labels)), freqs.values, alpha = 0.3)
    ax[0].set_xticks(range(len(labels)))
    ax[0].set_xticklabels(labels, rotation = 'vertical')
    ax[0].set_title(c)
    
    freqs_01 = df.groupby('Survived')[c].value_counts()
    ax[1].bar(range(len(labels)), freqs_01[0][labels].values, alpha = 0.3)
    ax[1].bar(range(len(labels)), freqs_01[1][labels].values, bottom = freqs_01[0][labels].values, alpha = 0.3)
    ax[1].set_xticks(range(len(labels)))
    ax[1].set_xticklabels(labels, rotation = 'vertical')
    ax[1].legend(['no', 'yes'])
    ax[1].set_title(c + ' vs. survived')
    
    fig.set_figwidth(16)


# Most of the categorical features seem to also provide information about survival likelihood. For instance, it is more likely to survive if you are a woman, or if your cabin prefix is not T. Many of the passengers with ticket class = 1 did not survived.
# 
# ## Imputing missing values
# 
# Let's take a look at the proportion of missing data. Some of the fare values are zero, but we decided not to assume that this is bogus data. I am assumming that these 17 passengers travelled with a zero fare for an explainable reason.

# In[17]:


def test_missing():
    for col in numerical + categorical:
        if col in categorical:
            missing = df_training[df_training[col].isna()]
        else:
            missing = df_training[(df_training[col].isna()) | 
                                  (df_training[col].apply(lambda x: type(x) == str))]
        proportion = len(missing) / len(df_training) * 100
        print(col + ': ' + str(proportion) + '%')


# In[18]:


test_missing()


# We have two categorical variables (CabinClass and Embarked) and one numerical variable (age) with missing values. I am going to assign a new value 'Missing' to the case of the missing values for the categorical variables. For the imputation of the numerical variable I am going to go for something simple and just use the median imputation.

# In[19]:


# Categorical variables
for c in ['CabinClass', 'Embarked']:
    df_training.loc[df_training[c].isna(), c] = 'None'
    df_test.loc[df_training[c].isna(), c] = 'None'


# In[20]:


# Numerical variable
imputed = df_training[np.isreal(df_training['Age'])]['Age'].median()
df_training.loc[(df_training['Age'].isna()) | (~np.isreal(df_training['Age'])), 'Age'] = imputed
df_test.loc[(df_test['Age'].isna()) | (~np.isreal(df_test['Age'])), 'Age'] = imputed

# It turns out that the test data has a missing fare
imputed = df_training[np.isreal(df_training['Fare'])]['Fare'].median()
df_test.loc[(df_test['Fare'].isna()) | (~np.isreal(df_test['Fare'])), 'Fare'] = imputed


# In[21]:


test_missing()


# ## Correlation between variables
# 
# We calculate pearson correlation in order to determine whether we should remove any variable.

# In[22]:


features = categorical + numerical

fig, ax = plt.subplots(6, 6)

plots = 0
for i in range(len(features)):
    for j in range(i + 1, len(features)):
        row = int(plots / 6)
        col = plots % 6

        def categorical_to_numerical(f):
            if features[f] in numerical:
                values_f = df_training[features[f]]
            else:
                values = df_training[features[f]].unique()
                values_f = df_training[features[f]].values.copy()
                for v in range(len(values)):
                    values_f[np.where(values_f == values[v])] = v
            
            return values_f
        
        values_i = categorical_to_numerical(i)
        values_j = categorical_to_numerical(j)
        
        cor = ((values_i - values_i.mean()) * (values_j - values_j.mean()) /               ((len(values_i) - 1) * values_i.std() * values_j.std())).sum()
            
        ax[row][col].scatter(values_i, values_j, alpha = 0.5)
        
        ax[row][col].set_xlabel(features[i])
        ax[row][col].set_ylabel(features[j])
        ax[row][col].set_title('cor = ' + '%.2f' % cor)
        
        if features[i] in categorical:
            values = df_training[features[i]].unique().tolist()
            ax[row][col].set_xticks(range(len(values)))
            ax[row][col].set_xticklabels(values, rotation = 'vertical')
        if features[j] in categorical:
            values = df_training[features[j]].unique().tolist()
            ax[row][col].set_yticks(range(len(values)))
            ax[row][col].set_yticklabels(values)

        plots = plots + 1
        
fig.set_figwidth(16)
fig.set_figheight(16)
plt.tight_layout()


# I don't observe any strong correlation. I cannot observe any obvious outlier either.
# 
# ## Dummy variables
# 
# Transforming categorical variables into dummy variables (we create k-1 new binary variables for each categorical variable, where k is the number of values of that categorical variable).

# In[23]:


new_categorical = []
for c in categorical:
    values = df_training[c].unique()[:-1]
    for v in values:
        name = c + '_' + str(v)
        df_training[name] = (df_training[c] == v).astype(int)
        df_test[name] = (df_test[c] == v).astype(int)
        new_categorical.append(name)
    df_training = df_training.drop(c, axis = 1)
    df_test = df_test.drop(c, axis = 1)


# In[24]:


print(len(categorical + numerical))


# In[25]:


variables = new_categorical + numerical
print(len(variables))


# After this step our training dataset contains 60 variables instead of 9.
# 
# ## Standardise
# 
# We want to keep the correlation between variables. Therefore, we use standardisation instead of normalisation. This step is not necessary for some machine learning algorithms, but can help others to converge much faster and also to prevent bias in those machine learning algorithms based on the Euclidean distance.

# In[26]:


# Keeping these values to transform the test dataset
statistics = pd.concat((df_training.mean(), df_training.std()), axis = 1)
statistics.columns = ['mean', 'std']
statistics.head()


# In[27]:


for c in variables:
    mean = statistics.loc[c, 'mean']
    std = statistics.loc[c, 'std']
    df_training[c] = (df_training[c] - mean) /  std
    df_test[c] = (df_test[c] - mean) /  std


# In[28]:


df_training[variables].head()


# In[29]:


# Removing columns
c = ['Name', 'Ticket', 'Cabin']
df_training = df_training.drop(c, axis = 1)
df_test = df_test.drop(c, axis = 1)


# ## Class imbalance
# 
# Finally we test whether the training set has a class imbalance problem.

# In[30]:


print(str((df_training.Survived == 1).sum()) + ' rows have Survived = 1')
print(str((df_training.Survived == 0).sum()) + ' rows have Survived = 0')


# There's some imbalance in the data, but does not seem to extreme. I decided not to oversample the minority class.
# 
# ## Logistic regression with forward selection
# 
# My best logistic regression results were obtained after applying forward selection-based feature selection, Using the regularisation parameter didn't help. Results were evaluated using RMSE and 10-fold cross validation.

# In[31]:


random.seed(0)


# In[32]:


# generating sets for 10-fold cross validation
indexes = list(range(len(df_training)))
random.shuffle(indexes)
folds = []
for i in range(10):
    folds.append([])
for i in range(len(indexes)):
    folds[i % 10].append(indexes[i])


# In[33]:


def produce_training_test_set(df_training, train_indexes, test_indexes, column_indexes):
    columns = df_training.columns[column_indexes]
    datasets = {}
    datasets['X_train'] = df_training.iloc[train_indexes][columns].values
    datasets['X_test'] = df_training.iloc[test_indexes][columns].values
    datasets['y_train'] = df_training.iloc[train_indexes]['Survived'].values
    datasets['y_test'] = df_training.iloc[test_indexes]['Survived'].values
    
    return datasets


# In[34]:


def evaluate(datasets, C = None):
    if C is None:
        C = 1
    logreg = LogisticRegression(solver = 'lbfgs', C = C)
    logreg.fit(datasets['X_train'], datasets['y_train'])
    y_pred = logreg.predict(datasets['X_test'])
    return sqrt(np.sum(np.power(np.array(y_pred) - np.array(datasets['y_test']), 2)))


# In[35]:


def k_fold_cross_validation(df_training, folds, column_indexes, C = None):
    error = 0
    
    for k in range(10):
        train_indexes = []
        for j in range(10):
            if j == k:
                test_indexes = folds[j]
            else:
                train_indexes = train_indexes + folds[j]
                
        datasets = produce_training_test_set(df_training, train_indexes, test_indexes, column_indexes)
        
        error = error + evaluate(datasets, C)
        
    return error / 10.0


# In[36]:


# RMSE if we use all the features
column_indexes = list(range(2, 62))
k_fold_cross_validation(df_training, folds, column_indexes)


# In[37]:


# Forward selection
pending = list(range(2, 62))
model = []
min_error = sys.float_info.max
while len(pending) > 0:
    
    prev_error = min_error
    min_error = sys.float_info.max
    
    for i in pending:
        new_model = model + [i]
        error = k_fold_cross_validation(df_training, folds, new_model)
        
        if error < min_error:
            min_error = error
            best_model = new_model
            feature = i
            
    if min_error < prev_error:
        print('Selecting feature ' + df_training.columns[feature] + ' - error decreased to ' + str(min_error))
        model = best_model
        pending.remove(feature)
    else:
        print('END')
        break


# In[38]:


model_forward = model
columns = df_training.columns[model_forward]
X_train = df_training[columns].values
X_test = df_test[columns].values
y_train = df_training['Survived'].values


# In[39]:


logreg = LogisticRegression(solver = 'lbfgs')


# In[40]:


logreg.fit(X_train, y_train)


# In[41]:


y_test = logreg.predict(X_test)


# In[42]:


submission = df_test.copy()
submission['Survived'] = y_test
submission = submission[['PassengerId', 'Survived']]


# In[43]:


submission.head()


# In[44]:


submission.to_csv('logistic_regression_forward_selection.csv', index = False)

