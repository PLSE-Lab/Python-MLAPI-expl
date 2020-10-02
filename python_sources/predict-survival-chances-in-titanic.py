#!/usr/bin/env python
# coding: utf-8

# ## Import libs

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings


# In[ ]:


warnings.filterwarnings("ignore")


# ## Import datasets

# In[ ]:


train_data = pd.read_csv('../input/train.csv')
train_data.head()


# In[ ]:


test_data = pd.read_csv('../input/test.csv')
test_data.head()


# ## Check datasets

# In[ ]:


def check_NaN(df, name):
    '''
    Check whether a Pandas DataFrame has NaN value in it.
    
    Input(s):
        df: Pandas DataFrame.
        name: String, the data's column name.
    
    Return(s):
        String, represents how many NaN in the column, or none.
    '''
    cnt = df[name].isna().sum()
    if cnt > 0:
        return 'The {} column have {} NaN value!'.format(name, cnt)
    else:
        return 'The {} column do not have NaN value.'.format(name)


# In[ ]:


def show_count_bar(df, name):
    '''
    Plot DataFrame with count bar.
    
    Input(s):
        df: Pandas DataFrame.
        name: String, the data's column name.
    
    Return(s):
        None.
    '''
    x1, y1 = np.unique(np.array(df[name][df[name].notna()]), return_counts=True)
    x2, y2 = np.unique(np.array(df[name][df[name].notna() & df['Survived'] == 1]), return_counts=True)
    
    x1 = np.array(x1).astype('str')
    x2 = np.array(x2).astype('str')
    
    plt.bar(x1, y1, label='Dead')
    plt.bar(x2, y2, label='Survived')
    plt.xlabel(name)
    plt.ylabel('Count')
    plt.legend()
    plt.show()


# In[ ]:


def show_hist(df, name):
    '''
    Plot DataFrame with hist.
    
    Input(s):
        df: Pandas DataFrame.
        name: String, the data's column name.
    
    Return(s):
        None.
    '''
    plt.hist(df[name][df[name].notna()], label='Dead')
    plt.hist(df[name][df[name].notna() & df['Survived'] == 1], label='Survived')
    plt.xlabel(name)
    plt.ylabel('Count')
    plt.legend()
    plt.show()


# In[ ]:


sns.set(style="white", palette="muted", color_codes=True)


# In[ ]:


show_count_bar(train_data, 'Pclass')

print(check_NaN(train_data, 'Pclass'))


# In[ ]:


show_count_bar(train_data, 'Sex')

print(check_NaN(train_data, 'Sex'))


# In[ ]:


show_hist(train_data, 'Age')

print(check_NaN(train_data, 'Age'))


# In[ ]:


show_count_bar(train_data, 'SibSp')

print(check_NaN(train_data, 'SibSp'))


# In[ ]:


show_count_bar(train_data, 'Parch')

print(check_NaN(train_data, 'Parch'))


# In[ ]:


show_count_bar(train_data, 'Embarked')

print(check_NaN(train_data, 'Embarked'))


# In[ ]:


show_hist(train_data, 'Fare')

print(check_NaN(train_data, 'Fare'))


# ## Process datasets

# ### Process train data

# #### One-Hot
# 
# Make `Pclass`, `Sex` and `Embarked` One-hot.

# In[ ]:


def get_one_hot(array):
    return np.array((array['Pclass'] == 1, array['Pclass'] == 2,
                    array['Pclass'] == 3, array['Sex'] == 'male',
                    array['Sex'] == 'female', array['SibSp'],
                    array['Parch'], array['Fare'],
                    array['Embarked'] == 'C', array['Embarked'] == 'Q',
                    array['Embarked'] == 'S')).swapaxes(0, 1).astype('float32')


# In[ ]:


x_train = train_data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
x_train.head()


# In[ ]:


x_train = get_one_hot(x_train)

x_train[:10]


# In[ ]:


y_train = np.array(train_data['Survived'])

y_train[:10]


# ### Process test data

# In[ ]:


x_test = test_data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]

x_test.head()


# In[ ]:


x_test = get_one_hot(x_test)

x_test[:10]


# ## Train the stacking model

# In[ ]:


from mlxtend.classifier import StackingCVClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer


# In[ ]:


imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imp = imp.fit(x_train)
x_train_imp = imp.transform(x_train)


# In[ ]:


clf1 = RandomForestClassifier()
clf2 = GradientBoostingClassifier()
lr = LogisticRegression()

sclf = StackingCVClassifier(classifiers=[clf1, clf2], meta_classifier=lr)


# In[ ]:


param_test = {'randomforestclassifier__n_estimators': [10, 120],
              'randomforestclassifier__max_depth': [2, 15],
              'gradientboostingclassifier__n_estimators': [10, 120],
              'gradientboostingclassifier__max_depth': [2, 15],
              'gradientboostingclassifier__learning_rate' : [0.01, 0.1],
              'meta_classifier__C': [0.1, 10.0]}


# In[ ]:


sclf.fit(x_train_imp, y_train)


# ## Show feature importance

# In[ ]:


x = ['Pclass:1', 'Pclass:2', 'Pclass:3', 'male', 'female', 'SibSp', 'Parch', 'Fare', 'Embarked:C', 'Embarked:Q', 'Embarked:S']

plt.figure(figsize=(16, 9))
plt.title('Importance for the classification')
plt.bar(x, sclf.clfs_[0].feature_importances_)
plt.show()


# ## Predict test data

# In[ ]:


x_test_imp = imp.transform(x_test)


# In[ ]:


data = np.array([np.array(test_data['PassengerId']), sclf.predict(x_test_imp)]).swapaxes(0, 1)

results = pd.DataFrame(data, columns=['PassengerId', 'Survived'])
results.set_index('PassengerId', inplace=True)

results.head()


# In[ ]:


results.to_csv('predict.csv')

