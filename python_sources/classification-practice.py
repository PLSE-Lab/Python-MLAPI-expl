#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('classic')
sns.set_style('white')

train_data = pd.read_csv("../input/titanic/train.csv")
test_data = pd.read_csv("../input/titanic/test.csv")


# In[ ]:


display(train_data.sample(3))
display(test_data.sample(3))


# In[ ]:


drop_cols = ['PassengerId', 'Name',  'Ticket', 'Cabin']

train_data = train_data.drop(drop_cols, axis = 1)
test_data = test_data.drop(drop_cols, axis = 1)


# In[ ]:


display(train_data.sample(3))
display(test_data.sample(3))


# In[ ]:


train_data.info()


# In[ ]:


test_data.info()


# In[ ]:


from sklearn.preprocessing import LabelEncoder

enc = LabelEncoder()

train_data['Sex'] = enc.fit_transform(train_data['Sex'])
test_data['Sex'] = enc.transform(test_data['Sex'])


# In[ ]:


from sklearn.impute import SimpleImputer

imp = SimpleImputer(strategy = 'most_frequent')
train_data['Embarked'] = imp.fit_transform(train_data[['Embarked']])


# In[ ]:


sns.heatmap(train_data.corr(), cmap = 'Blues', annot = True, fmt='.2f')

plt.title('Correlation between variables', fontsize = 15, pad = 15 )
plt.gcf().set_size_inches(8,6)
plt.show()


# In[ ]:


train_data = pd.get_dummies(train_data, columns=['Embarked'])
test_data = pd.get_dummies(test_data, columns=['Embarked'])


# In[ ]:


train_data['Age'] = imp.fit_transform(train_data[['Age']])
test_data['Age'] = imp.transform(test_data[['Age']])

imp.fit(train_data[['Fare']])
test_data['Fare'] = imp.transform(test_data[['Fare']])


# In[ ]:


display(train_data.sample(3))
display(test_data.sample(3))


# In[ ]:


from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(max_features = 'sqrt', max_leaf_nodes = 5)

train_X, train_y = train_data.drop(['Survived'], axis = 1), train_data['Survived']
clf.fit(train_X, train_y)


# In[ ]:


importances = clf.feature_importances_

columns = test_data.columns
indices = np.argsort(importances)[::-1]
names = [columns[i] for i in indices]
sns.barplot(names, importances[indices])

plt.gcf().set_size_inches(6, 4)
plt.xticks(rotation=90)

plt.title('Feature Importance of Random Forest Classifier', pad=10)
plt.show()


# In[ ]:


from sklearn.model_selection import cross_val_score

score = cross_val_score(clf, train_X, train_y ,cv=5)
print('CV score : {}'.format(score.mean()))


# In[ ]:


from sklearn.ensemble import AdaBoostClassifier

clf = AdaBoostClassifier()

score = cross_val_score(clf, train_X, train_y ,cv=5)
print('CV score : {}'.format(score.mean()))


# In[ ]:


from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': list(range(50,151))
}

gcv = GridSearchCV(clf, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)


# In[ ]:


get_ipython().run_cell_magic('time', '', '\ngcv.fit(train_X,train_y)')


# In[ ]:


gcv.best_params_


# In[ ]:


clf = gcv.best_estimator_

predictions = clf.predict(test_data)
print(predictions[0:3])


# In[ ]:


submission = pd.read_csv('../input/titanic/gender_submission.csv')

submission['Survived'] = predictions
display(submission.head(3))


# In[ ]:


submission.to_csv('result_2.csv',index=False)

