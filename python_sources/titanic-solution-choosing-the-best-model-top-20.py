#!/usr/bin/env python
# coding: utf-8

# # Choosing the best model for titanic dataset.
# We will make some visualization and data preparation. After we'll fit some models, fine tune them and choose the best one. 
# ### We will try the folowing  models:
# * GradientBoostingClassifier
# * LogisticRegression
# * SGDClassifier
# * GaussianNB
# * KNeighborsClassifier
# * DecisionTreeClassifier
# * RandomForestClassifier
# * SVC

# First of all we need to upload data from .csv files.

# In[ ]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
np.random.seed(42)


# In[ ]:


train_data = pd.read_csv('/kaggle/input/titanic/train.csv', index_col=0)
test_data = pd.read_csv('/kaggle/input/titanic/test.csv',index_col=0)
train_data.head()


# Now we need to separate features and labeles as it is supervised learning task.

# In[ ]:


X = train_data.drop(columns=['Survived'])
y = train_data.Survived


# Now we can take a quick look at the data.

# In[ ]:


print(X.info())
print(test_data.info())


# Here we can see that we have some columns with missing data.
# * in train set: Age, Cabin and Embarked
# * in test set: Age, Cabin and Ticket
# 
# We will process them later.

# # 1. Visualization

# We will see how survival depends on some features.

# ### 1.1 Age

# In[ ]:


age_groups = train_data.Age // 5 + 1
age_groups.fillna(0, inplace=True)

num_people = train_data.shape[0]

deceased_age_group = age_groups[train_data.Survived==0].value_counts()
survived_age_group = age_groups[train_data.Survived==1].value_counts()
width = 0.8


inds = np.sort(deceased_age_group.index.values)
plt.figure(figsize=(10, 8))

age_groups_deceased_bar = plt.bar(inds, deceased_age_group[inds].values / num_people, width)
age_groups_survived_bar = plt.bar(inds, survived_age_group[inds].values / num_people, width,
                                  bottom=deceased_age_group[inds].values / num_people)

plt.title('Percentage of Passengers by Age Group', fontdict={'fontsize':20})

age_groups_ticks = ['None'] + ['{}-{}'.format(int(i * 5), int((i + 1)* 5)) for i in inds[0:]]

plt.xticks(inds, age_groups_ticks)
plt.legend((age_groups_deceased_bar[0], age_groups_survived_bar[0]), ('Deceased', 'Survived'))

plt.show()


# As we can see, more than half of children survived. We can't say the same about adults.

# In[ ]:


train_data[train_data.Age <= 15].Survived.value_counts(normalize=True)


# ### 1.2 Gender

# In[ ]:


deceased_gender = train_data.Sex[train_data.Survived==0].value_counts()
survived_gender = train_data.Sex[train_data.Survived==1].value_counts()


inds = ['male', 'female']

plt.figure(figsize=(8, 6))

male_plt = plt.bar(inds, deceased_gender.values / num_people, width)
female_plt = plt.bar(inds, survived_gender[inds].values / num_people,
                     width, bottom=deceased_gender[inds].values / num_people)

plt.title('Percentage of Passengers by Gender', fontdict={'fontsize':20})

plt.xticks(inds, ('Men', 'Women'))
plt.legend((male_plt[0], female_plt[0]), ('Deceased', 'Survived'))

plt.show()


# More than half of women survived but we have opposite situation with men.

# In[ ]:


print(train_data[train_data.Sex == 'male'].Survived.value_counts(normalize=True))
print('-'*40)
print(train_data[train_data.Sex == 'female'].Survived.value_counts(normalize=True))


# So that means that gender is quite a strong feature.

# ### 1.3 Class

# In[ ]:


deceased_class = train_data.Pclass[train_data.Survived==0].value_counts()
survived_class = train_data.Pclass[train_data.Survived==1].value_counts()


inds = [1, 2, 3]

plt.figure(figsize=(8, 6))

male_plt = plt.bar(inds, deceased_class[inds].values / num_people, width)
female_plt = plt.bar(inds, survived_class[inds].values / num_people,
                     width, bottom=deceased_class[inds].values / num_people)

plt.title('Percentage of Passengers by Class', fontdict={'fontsize':20})

plt.xticks(inds, ('First', 'Second', 'Third'))
plt.legend((male_plt[0], female_plt[0]), ('Deceased', 'Survived'))

plt.show()


# We see that the worse the class, the lower the percentage of surviving people.

# In[ ]:


train_data.groupby('Pclass').Survived.value_counts(normalize=True, sort=False)


# # 2. Data preparation

# First of all we see that we have a lot of missing values in the Age column. And as it is an important feature, we should deal with it.
# We know the names of all people on titanic so we can get a "rank" of every person(Mr, Miss, Dr, Rev and etc.) and fill missing age values by means of other people ages with the same rank.

# In[ ]:


ranks = X['Name'].str.extract(r'\b(\w+)\.') #getting ranks using regex
ranks[0].value_counts()


# Here can be a problem that for some ranks we have only one or two persons so it is impossible to calculate a mean value for them.

# In[ ]:


ranks[train_data.Age.isna()][0].value_counts()


# But as we see for ranks with missing values we have other people to work with.

# In[ ]:


missing_age_ranks = ranks[train_data.Age.isna()][0].value_counts().index.values
missing_age_ranks


# In[ ]:


age_na_fills = {}
for rank in missing_age_ranks:
    age_na_fills[rank] = round(train_data[(ranks == rank).values].Age.mean())
age_na_fills


# We got values to replace nan with. And we need a function to replace missing age values with defined dict, so we can use this function again. If we have no such a rank in our dict we return 29 as it is mean age value.

# In[ ]:


def fill_age_na(data, age_na_fills, ranks):
    data['Age'] = data.apply(lambda row: age_na_fills.get(ranks[row.name], 29) if np.isnan(row['Age']) else row['Age'], axis=1)


# Also we have two columns with categorical values - Sex and Embarked. And some columns that we can drop. And as I want to create a Pipline for data preparation, I need to realize transformers to process columns I want to drop and one for age column.

# In[ ]:


from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder 
from sklearn.impute import SimpleImputer
class AgeFillNa(BaseEstimator, TransformerMixin):
    def __init__(self, age_na_fills=None):
        self.age_na_fills = age_na_fills # dict {rank:mean_age}
        
    def fit(self, X, y=None):
        return self
  
    def transform(self, X):
        output = X.copy()
        ranks = X['Name'].str.extract(r'\b(\w+)\.')
        
        if self.age_na_fills is not None:
            fill_age_na(output, self.age_na_fills, ranks[0])
        else:
            output.Age = output.Age.fillna(output.Age.mean())
        
        return output

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns_drop):
        self.columns_drop = columns_drop #list of column to drop

    def fit(self, X, y=None):
        return self
  
    def transform(self, X):
        return X.drop(columns=self.columns_drop)
    
class MultiColumnLabelEncoder(BaseEstimator, TransformerMixin):
    
    def __init__(self, columns=None):
        self.columns = columns # list of column to encode
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        output = X.copy()
        
        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col].astype(str))
        else:
            for colname, col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        
        return output

columns_to_drop = ['Ticket', 'Cabin', 'Name']
columns_to_encode = ['Sex', 'Embarked']


# Now we can create a pipeline to automate a whole preparation process.

# In[ ]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

prepare_pipeline = Pipeline([
        ('age_fill_na', AgeFillNa(age_na_fills)), # processes missing age values
        ('selector', DataFrameSelector(columns_to_drop)), # drops some columns
        ('label_enc', MultiColumnLabelEncoder(columns=columns_to_encode)),# processes categorical columns
        ('imputer', SimpleImputer(strategy="mean")), #process missing values
        ('std_scaler', StandardScaler()), # standardization
    ])


# # 3. Model selection

# Now it's time to choose a model to work with. We will try some of scikit-learn classification models.

# In[ ]:


from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

X = prepare_pipeline.fit_transform(train_data.drop(columns=['Survived']))


# Firstly we will try all models without any tuning. Just to see how they work. Than we will choose some of them and fine tune selected models.

# In[ ]:


from sklearn.metrics import f1_score
empt_models = [GradientBoostingClassifier(), LogisticRegression(solver='lbfgs'), SGDClassifier(), GaussianNB(), 
               KNeighborsClassifier(), DecisionTreeClassifier(), RandomForestClassifier(n_estimators=100), 
               SVC(gamma='auto')]

for model in empt_models:
    results = cross_val_predict(model, X, y, cv=5)
    print("model: {}, result: {:.3f}".format(model.__class__, f1_score(y, results)))


# As we can see, the best models are SVC, GradientBoostingClassifier, KNeighborsClassifier. and RandomForestClassifier Now we need to fine tune them. For this task I'll use GridSearchCV.

# ### 3.1 SVC

# In[ ]:


from sklearn.model_selection import GridSearchCV


# In[ ]:


params = [{'kernel':['linear'], 'gamma':[i / 10 for i in range(1, 100, 5)], 'C':[i / 10 for i in range(5, 30, 5)]},
          {'kernel':['rbf'], 'C':[i / 10 for i in range(5, 30, 5)]},
          {'kernel':['poly'], 'degree':np.arange(2, 10), 'C':[i / 10 for i in range(5, 30, 5)]}]

grid_search_svc = GridSearchCV(SVC(gamma='auto'), params, cv=5, scoring='neg_mean_squared_error')
grid_search_svc.fit(X, y)
grid_search_svc.best_params_


# In[ ]:


svc_model = SVC(gamma='auto', **grid_search_svc.best_params_)
svc_score = cross_val_predict(svc_model, X, y, cv=5)
f1_score(y, svc_score)


# ### 3.2 GradientBostingClassifier

# In[ ]:


params = {'learning_rate':[i/10 for i in range(3, 6)], 'n_estimators':np.arange(128, 132, 1),
         'min_samples_split':np.arange(2, 3), 'min_samples_leaf':np.arange(6, 8),
          'max_depth':np.arange(1, 3), 'max_features':np.arange(1, 8)}

grid_search_gbc = GridSearchCV(GradientBoostingClassifier(), params, cv=5, scoring='neg_mean_squared_error')
grid_search_gbc.fit(X, y)
grid_search_gbc.best_params_


# In[ ]:


gbc_model = GradientBoostingClassifier(**grid_search_gbc.best_params_)
gbc_score = cross_val_predict(gbc_model, X, y, cv=5)
f1_score(y, gbc_score)


# ### 3.3 KNeighborsClassifier

# In[ ]:


params = {'n_neighbors':np.arange(10, 25), 'p':np.arange(1, 6)}

grid_search_knn = GridSearchCV(KNeighborsClassifier(), params, cv=5, scoring='neg_mean_squared_error')
grid_search_knn.fit(X, y)
grid_search_knn.best_params_


# In[ ]:


knn_model = KNeighborsClassifier(**grid_search_knn.best_params_)
knn_score = cross_val_predict(knn_model, X, y, cv=5)
f1_score(y, knn_score)


# ### 3.4 RandomForestClassifier

# In[ ]:


params = {'n_estimators':np.arange(50, 1000, 100), 'max_depth':np.arange(3, 8)}

grid_search_rfc = GridSearchCV(RandomForestClassifier(random_state=42), params, cv=5, scoring='neg_mean_squared_error')
grid_search_rfc.fit(X, y)
grid_search_rfc.best_params_


# In[ ]:


rfc_model = RandomForestClassifier(random_state=42,**grid_search_rfc.best_params_)
rfc_score = cross_val_predict(rfc_model, X, y, cv=5)
f1_score(y, rfc_score)


# # 4. Launching

# Although after tuning all the models GradientBoostringClassifier showed the best result SVC works better on test set(0.73205 vs 0.79425
# ). So we will use SVC for predicting on the test set. 
# We created pipeline for data preparation and now we can just use it.

# In[ ]:


X_test = prepare_pipeline.fit_transform(test_data)


# And also we already created a model with the parameters we considered to be the best. So now we need only to fit model on the training set and predict target value on the test set.

# In[ ]:


svc_model.fit(X, y)
predictions = svc_model.predict(X_test)


# After we got predictions, we need to write it to .csv file.

# In[ ]:


my_submission = pd.DataFrame({'PassengerId': test_data.index.values, 'Survived': predictions})
my_submission.to_csv('submission.csv', index=False)


# # 5. References
# * [Parameter tuning for SVC](https://medium.com/all-things-ai/in-depth-parameter-tuning-for-svc-758215394769)
# * [Parameter tuning for Gradient Boosting](https://medium.com/all-things-ai/in-depth-parameter-tuning-for-gradient-boosting-3363992e9bae)
# * [Parameter tuning for KNN](https://medium.com/@mohtedibf/in-depth-parameter-tuning-for-knn-4c0de485baf6)
# * [Parameter tuning for Random Forest](https://medium.com/all-things-ai/in-depth-parameter-tuning-for-random-forest-d67bb7e920d)
# * Hands-On Machine Learning with Scikit-Learn and TensorFlow
