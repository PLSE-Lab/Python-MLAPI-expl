#!/usr/bin/env python
# coding: utf-8

# # Import Libs and Get Dataset

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


# In[ ]:


df = pd.read_csv('../input/datathon19/train.csv')
df.head()


# # Preprocessing: Using Label Encoder

# In[ ]:


x_train = df.iloc[:, 0:9]
y_train = df['class']
x_train.head()


# In[ ]:


x_test = pd.read_csv('../input/datathon19/test.csv')
x_test.head()


# In[ ]:


from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()

x_train['top_left_square'] = encoder.fit_transform(x_train['top_left_square'])
x_train['top_middle_square'] = encoder.fit_transform(x_train['top_middle_square'])
x_train['top_right_square'] = encoder.fit_transform(x_train['top_right_square'])
x_train['middle_left_square'] = encoder.fit_transform(x_train['middle_left_square'])
x_train['middle_middle_square'] = encoder.fit_transform(x_train['middle_middle_square'])
x_train['middle_right_square'] = encoder.fit_transform(x_train['middle_right_square'])
x_train['bottom_left_square'] = encoder.fit_transform(x_train['bottom_left_square'])
x_train['bottom_middle_square'] = encoder.fit_transform(x_train['bottom_middle_square'])
x_train['bottom_right_square'] = encoder.fit_transform(x_train['bottom_right_square'])

x_test['top_left_square'] = encoder.fit_transform(x_test['top_left_square'])
x_test['top_middle_square'] = encoder.fit_transform(x_test['top_middle_square'])
x_test['top_right_square'] = encoder.fit_transform(x_test['top_right_square'])
x_test['middle_left_square'] = encoder.fit_transform(x_test['middle_left_square'])
x_test['middle_middle_square'] = encoder.fit_transform(x_test['middle_middle_square'])
x_test['middle_right_square'] = encoder.fit_transform(x_test['middle_right_square'])
x_test['bottom_left_square'] = encoder.fit_transform(x_test['bottom_left_square'])
x_test['bottom_middle_square'] = encoder.fit_transform(x_test['bottom_middle_square'])
x_test['bottom_right_square'] = encoder.fit_transform(x_test['bottom_right_square'])


# In[ ]:


x_train.head()


# # Building the Model

# Previously I had used DecisionTreeClassifier which gave just about 89% accuracy in the below test I did with the Original dataset. Now trying RandomForestClassifier with n_estimators=20000

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier(n_estimators=20000)
model.fit(x_train,y_train)


# In[ ]:


x_test_pred = x_test.iloc[:, 0:9]
x_test_pred.head()


# # Make predictions using the trained model

# In[ ]:


y_pred = model.predict(x_test_pred)


# In[ ]:


y_pred


# In[ ]:


y_pred.shape


# In[ ]:


y_test_id = x_test['Id']
y_test_id.shape


# In[ ]:


combined = np.vstack((y_test_id, y_pred)).T
combined.shape


# In[ ]:


np.savetxt('datathon19.csv', combined, delimiter=',', fmt="%i")


# In[ ]:


asdf =  pd.read_csv('datathon19.csv')
asdf.head()


# In[ ]:


import csv
with open('datathon19.csv',newline='') as f:
    r = csv.reader(f)
    data = [line for line in r]
with open('datathon19.csv','w',newline='') as f:
    w = csv.writer(f)
    w.writerow(['Id','class'])
    w.writerows(data)


# In[ ]:


asdf =  pd.read_csv('datathon19.csv')
asdf.head()


# # Calculating Accuracy Using Oirignal Dataset

# This is just to show the accuracy score of the trained model. Previously done with DecisionTreeClassifier which yielded about 89% accuracy.

# In[ ]:


dfOrigin = pd.read_csv('../input/datathon19/tic_tac_toe_dataset.csv')


# In[ ]:


x = dfOrigin.iloc[:, 0:9]
y = dfOrigin['class']
x.head()


# 
# # Preprocessing: Using LabelEncoder

# In[ ]:


from sklearn.model_selection import train_test_split

x_train_origin, x_test_origin, y_train_origin, y_test_origin = train_test_split(x, y, random_state = 47, test_size = 0.25)


# In[ ]:


x_train_origin.head()
y_train_origin.head()


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=20000)
# model.fit(x_train_origin, y_train_origin)


# In[ ]:


x_train_origin['top_left_square'] = encoder.fit_transform(x_train_origin['top_left_square'])
x_train_origin['top_middle_square'] = encoder.fit_transform(x_train_origin['top_middle_square'])
x_train_origin['top_right_square'] = encoder.fit_transform(x_train_origin['top_right_square'])
x_train_origin['middle_left_square'] = encoder.fit_transform(x_train_origin['middle_left_square'])
x_train_origin['middle_middle_square'] = encoder.fit_transform(x_train_origin['middle_middle_square'])
x_train_origin['middle_right_square'] = encoder.fit_transform(x_train_origin['middle_right_square'])
x_train_origin['bottom_left_square'] = encoder.fit_transform(x_train_origin['bottom_left_square'])
x_train_origin['bottom_middle_square'] = encoder.fit_transform(x_train_origin['bottom_middle_square'])
x_train_origin['bottom_right_square'] = encoder.fit_transform(x_train_origin['bottom_right_square'])

x_test_origin['top_left_square'] = encoder.fit_transform(x_test_origin['top_left_square'])
x_test_origin['top_middle_square'] = encoder.fit_transform(x_test_origin['top_middle_square'])
x_test_origin['top_right_square'] = encoder.fit_transform(x_test_origin['top_right_square'])
x_test_origin['middle_left_square'] = encoder.fit_transform(x_test_origin['middle_left_square'])
x_test_origin['middle_middle_square'] = encoder.fit_transform(x_test_origin['middle_middle_square'])
x_test_origin['middle_right_square'] = encoder.fit_transform(x_test_origin['middle_right_square'])
x_test_origin['bottom_left_square'] = encoder.fit_transform(x_test_origin['bottom_left_square'])
x_test_origin['bottom_middle_square'] = encoder.fit_transform(x_test_origin['bottom_middle_square'])
x_test_origin['bottom_right_square'] = encoder.fit_transform(x_test_origin['bottom_right_square'])


# In[ ]:


x_test_origin.head()


# In[ ]:


x_test_origin.head()


# In[ ]:


y_train_origin =  encoder.fit_transform(y_train_origin)
y_test_origin =  encoder.fit_transform(y_test_origin)


# In[ ]:


y_test_origin[0:5]


# # Training the Model and making predictions

# In[ ]:


model.fit(x_train_origin, y_train_origin)


# In[ ]:


y_pred_origin = model.predict(x_test_origin)


# In[ ]:


y_pred_origin


# # Accuracy score

# Previously achieved 89% using DecisionTreeClassifier. Now, after using RandomForestClassifier there is a significant improvement, as the accuracy has gone up to almost 93%. Pretty decent.

# In[ ]:


from sklearn.metrics import accuracy_score
print('Accuracy Score on train data: ', accuracy_score(y_true=y_train_origin, y_pred=model.predict(x_train_origin)))
print('Accuracy Score on test data: ', accuracy_score(y_true=y_test_origin, y_pred=y_pred_origin))


# In[ ]:





# In[ ]:




