#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# Some libraries:

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from numpy import savetxt


# Here, I'm opening the datasets and move the column *Survived* to the end of the dataset, just for preference. 
# I delete some columns too, for some reasons: *Name* and *Ticket* are unique values, equivalent to the *PassengerId* column. And the *Cabin* has so many null spaces, approximaly 78%, making this column useless. 

# In[ ]:


base_train = pd.read_csv("/kaggle/input/titanic/train.csv")
base_train = base_train[['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked', 'Survived']]

del base_train['Name']
del base_train['Ticket']
del base_train['Cabin']

base_test = pd.read_csv("/kaggle/input/titanic/test.csv")
del base_test['Name']
del base_test['Ticket']
del base_test['Cabin']


# Well, here I pre-processed the data. First, I filled the null cells in the *Age* column with the average of the other ages.

# In[ ]:


imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imputer = imputer.fit(base_train.iloc[:, 3:4])
base_train.iloc[:, 3:4] = imputer.transform(base_train.iloc[:, 3:4])
base_test.iloc[:, 3:4] = imputer.transform(base_test.iloc[:, 3:4])


# And here, applied the label encoder to the *Embarked* and the *Sex* columns, to transform categorical values into numerical values.

# In[ ]:


labelencoder = LabelEncoder()
base_train['Embarked'] = labelencoder.fit_transform(base_train['Embarked'].astype(str))
base_test['Embarked'] = labelencoder.fit_transform(base_test['Embarked'].astype(str))
base_train.iloc[:, 2] = labelencoder.fit_transform(base_train.iloc[:, 2])
base_test.iloc[:, 2] = labelencoder.fit_transform(base_test.iloc[:, 2])


# So I also scaled the values so that some attributes are not valued more than others because of the scale difference.

# In[ ]:


scaler = StandardScaler()
base_train.iloc[:, 1:8] = scaler.fit_transform(base_train.iloc[:, 1:8])
base_test.iloc[:, 1:8] = scaler.fit_transform(base_test.iloc[:, 1:8])


# 
# Infinite values have been replaced by finite values as follows:

# In[ ]:


base_test[base_test==np.inf]=np.nan
base_test.fillna(base_test.mean(), inplace=True)


# And then I separated the dataset *train* to train the machine:

# In[ ]:


predictors_train = base_train.iloc[:, 0:8].values
class_train = base_train.iloc[:, 8].values


# I used Random Forest to train the machine and then made the predictions:

# In[ ]:


classificador = RandomForestClassifier(n_estimators = 40, criterion = 'entropy', random_state = 0)
classificador.fit(predictors_train, class_train)
predictions = classificador.predict(base_test)


# In[ ]:


output = pd.DataFrame({'PassengerId': base_test.PassengerId, 'Survived': predictions})
output.to_csv('submission.csv', index=False)


# That was my code. Very simple, with the little knowledge I acquired in the first classes of a course I am taking.
