#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import glob
import os
import re

df = pd.read_csv('../input/woodbine_horses.csv')

# drop null values
df.dropna(how='all', subset=['jockey', 'sex', 'speed_rating'], inplace=True)

df_money = df[['race_date', 'name', 'claim_price', 'win_payoff', 'show_payoff', 'place_payoff', 'dollar_odds']]
df.drop(['finish_time', 'track', 'program_number', 'claim_price', 'show_payoff', 'place_payoff', 'card_id', 'breed', 'track_conditions', 'weather', 'distance', 'race_type'], inplace=True, axis=1)

print(df.head())


# In[ ]:


print(df.isnull().sum())


# 

# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# uneven dist of dollar odds
sns.distplot(df['dollar_odds'], kde = False, color = 'b')


# In[ ]:


sns.distplot(df['age'], kde = False, color = 'b')


# In[ ]:


sns.distplot(df['weight'], kde = False, color = 'b')


# In[ ]:


# Geldings are by far the majority "sex" in races
sns.countplot(df['sex'], palette="Greens_d");


# In[ ]:


# winners as percentage function
def win_pct_by_feature(feature_column, min_num_threshold=6):
    winners_by_feature = df[df['finish_position'] == 1][['name',feature_column]].groupby(feature_column).agg(['count'])
    winners_by_feature = winners_by_feature[winners_by_feature > min_num_threshold].dropna()
    
    feature_count = df[['name',feature_column]].groupby(feature_column).agg(['count'])
    pct_winners_by_feature = winners_by_feature/feature_count[feature_count.index.isin(winners_by_feature.index)]
    
    pct_winners_by_feature.columns = ['win_percentage']
    pct_winners_by_feature = pct_winners_by_feature.sort_values(by='win_percentage',ascending=False)
    return pct_winners_by_feature


# In[ ]:


# Find the win percentage by weight

win_pct_by_weight = win_pct_by_feature('weight')

ax = win_pct_by_weight.plot(kind='bar')
ax.set_ylabel("Win Percentage")


# In[ ]:


# Find the win percentage by trainer

pct_winners_by_trainer = win_pct_by_feature('trainer')

ax = pct_winners_by_trainer.plot(kind='bar')
ax.set_ylabel("Win Percentage")


# In[ ]:


# Find the win percentage by owner

pct_winners_by_owner = win_pct_by_feature('owner')

ax = pct_winners_by_owner.plot(kind='bar')
ax.set_ylabel("Win Percentage")


# In[ ]:


# Find the win percentage by jockey

pct_winners_by_jockey = win_pct_by_feature('jockey')

ax = pct_winners_by_jockey.plot(kind='bar')
ax.set_ylabel("Win Percentage")


# We can see from the above graphs that outline percent winners based on [jockey, owner, weight, trainer] a few things
# - The best jockey wins about 2.5x more than the worst
# - The best owner wins about 2x more than the worst
# - The best trainer wins about 2x more than the worst
# - That weight doesn't seem to have an affect on winning

# 

# 

# In[ ]:


from sklearn import preprocessing

# min max scale when we have numerical values with consistent range
def minmax_scale_skewed_numerical(feature):
    minmax_scale = preprocessing.MinMaxScaler(feature_range=(0, 1), copy=True)
    scaled = minmax_scale.fit_transform(df[feature].values.reshape(-1,1))
    return scaled

# z-score standardized scale
def standarize_scale_skewed_numerical(feature):
    std_scale = preprocessing.StandardScaler()
    scaled = std_scale.fit_transform(df[feature].values.reshape(-1,1))
    return scaled


# In[ ]:


# standardize numerical features

df['weight'] = standarize_scale_skewed_numerical('weight')
df['dollar_odds'] = standarize_scale_skewed_numerical('dollar_odds')
df['age'] = standarize_scale_skewed_numerical('age')
df['speed_rating'] = minmax_scale_skewed_numerical('speed_rating')


# In[ ]:


NUMERICAL_FEATURES = ['weight', 'age', 'speed_rating', 'dollar_odds', 'start_position', 'post_position']
CATEGORICAL_FEATURES = ['sex', 'jockey', 'trainer', 'owner']
PRED = ['win']

# one hot encod categorical columns
for feature in CATEGORICAL_FEATURES:
    one_hot = pd.get_dummies(df[feature])
    one_hot.columns = [x.lower().replace(',', '').replace('.', '').replace('\'', '').replace('"', '').replace(' ', '_') for x in one_hot.columns]
    one_hot.columns = feature + '_' + one_hot.columns
    df = df.join(one_hot)
    df.drop(feature, inplace=True, axis=1)


# In[ ]:


# new column based on win
df['win'] = False
df.loc[df['finish_position'] == 1,'win'] = True

# new column based on placing
df['place'] = False
df.loc[df['finish_position'].isin([1, 2, 3]),'place'] = True

df.drop(['finish_position', 'meds_and_equip', 'win_payoff', 'name', 'win', 'race_date'], inplace=True, axis=1)
df[['start_position']] = df[['start_position']].astype(int)


# 

# In[ ]:


from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

X = df.ix[:, df.columns != 'place']
Y = df['place']

# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

kf = KFold(n_splits=2)
kf.get_n_splits(X)

for train_index, test_index in kf.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X.values[train_index], X.values[test_index]
    y_train, y_test = Y.values[train_index], Y.values[test_index]


# In[ ]:


from sklearn import metrics
from sklearn import tree

model = tree.DecisionTreeClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# summarize the fit of the model
print(metrics.classification_report(y_test, y_pred))


# In[ ]:


import itertools
import numpy as np
import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Reds):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 1.4
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[ ]:


from sklearn.metrics import confusion_matrix

class_names = Y.unique()

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()

