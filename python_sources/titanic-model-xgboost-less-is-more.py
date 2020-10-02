#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # Less is more. Keep it simple
# 
# - All of this analysis was done straight to the point to achieve the goal quickly and efficiently without the unneccasy 'candy wrapping' of hundred uncessary histograms, charts and comparisons to look 'smart'. Time is money. So here it is, clean and simple.

# In[ ]:


import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[ ]:


df_train = pd.read_csv('/kaggle/input/titanic/train.csv')
df_test = pd.read_csv('/kaggle/input/titanic/test.csv')



# In[ ]:





# In[ ]:


df_train.head()


# - Quick correlation observation to determine the most fit variables for our feature or 'predictor' set:

# In[ ]:


df_train.corr()


# In[ ]:


df_train['Survived'] = df_train['Survived'].astype(int)
df_train['Survived']


# - define our target - Survived, and convert categorical features values into dummies aka numerical for xgboost processing:

# In[ ]:



y = df_train['Survived'].astype('int')
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin']

X = pd.get_dummies(df_train[features])
X_test = pd.get_dummies(df_test[features])




# - split the training data with train test split:

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)



# - define xgboost hyperparameters
# - training the model

# In[ ]:


from xgboost import XGBClassifier

model = XGBClassifier(learning_rate = 0.05,
                     n_estimators=300,
                     max_depth = 4)
model.fit(X_train, y_train)


# - Predictions with a round value:

# In[ ]:


y_pred = model.predict(X_test).astype(int)


# - Model evaluation of accuracy:

# In[ ]:


from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, predictions)
print('My model accuracy is: %.2f%%' % (accuracy *100.0))


# In[ ]:


import matplotlib.pyplot as plt
from xgboost import plot_importance

plot_importance(model)
plt.show()


# ### As i suspected, the most important feature in survival rate was the Fare. Obviously the VIP's had the priority boarding upon embarking and also upon emergency. From those VIP's first came the youngest ones, women and children in the family, hence the 'Age' being selected as the second most import feature in survival rate. Conclusion of the story - Money buys priority in life. 

# - making a kaggle export:

# In[ ]:


predictions = model.predict(df_test[features])


# In[ ]:


# make predictions for test data:

y_pred = model.predict(X_test).astype(int)
predictions = [round(value) for value in y_pred]

#create a df of predictions of Test data:
pred_df_file = pd.DataFrame({'Survived': predictions}).astype(int)

# combine the Passenger Id and their associated prediction:
submission = pd.concat([df_test['PassengerId'], pred_df_file], axis ='columns')

#output file for submission:

submission = pd.DataFrame({
    "PassengerId": df_test["PassengerId"],
    "Survived": y_pred })
submission.to_csv('titanic.csv', index=False)


# In[ ]:


print(submission)


# In[ ]:





# In[ ]:





# In[ ]:




