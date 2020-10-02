#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


train = pd.read_csv('../input/train.csv', encoding='ISO-8859-1')
train.head()


# In[3]:


test = pd.read_csv('../input/test.csv').drop(['name', 'display_name', 'production_budget', 'language', 'board_rating_reason'], axis=1)
test.head()


# In[4]:


train.shape, test.shape


# In[5]:


Concatenated = pd.concat([train, test], axis=0, sort=True)
Concatenated.head()


# In[6]:


Concatenated.info()


# In[7]:


Concatenated.groupby(['Category']).count()['id']


# In[8]:


sns.heatmap(Concatenated.isnull())


# In[9]:


sns.countplot(data=Concatenated, x='Category', hue='production_year')
Concatenated.groupby(['production_year']).count()['id']


# In[10]:


sns.countplot(data=Concatenated, x='Category', hue='movie_sequel')
Concatenated.groupby(['movie_sequel']).count()['id']


# In[11]:


sns.countplot(data=Concatenated, x='Category', hue='creative_type')


# In[12]:


sns.countplot(data=Concatenated, x='Category', hue='movie_board_rating_display_name')


# In[13]:


Concatenated.groupby(['source']).count()['id']


# In[14]:


Concatenated.groupby(['production_method']).count()['id']


# In[15]:


Concatenated.groupby(['genre']).count()['id']


# In[16]:


Concatenated.groupby(['language']).count()['id']


# In[17]:


Concatenated.groupby(['movie_board_rating_display_name']).count()['id']


# In[18]:


Concatenated.groupby(['movie_release_pattern_display_name']).count()['id']


# In[19]:


Concatenated = Concatenated.drop(['name', 'display_name', 'board_rating_reason', 'language','total'], axis=1)
Concatenated.head()


# In[20]:


Concatenated.corr()


# In[21]:


#handling categorical variables
creative_type = pd.get_dummies(Concatenated['creative_type'], drop_first = True)
source = pd.get_dummies(Concatenated['source'], drop_first = True)
production_method = pd.get_dummies(Concatenated['production_method'], drop_first = True)
genre = pd.get_dummies(Concatenated['genre'], drop_first = True)
movie_board_rating_display_name = pd.get_dummies(Concatenated['movie_board_rating_display_name'], drop_first = True)
movie_release_pattern_display_name = pd.get_dummies(Concatenated['movie_release_pattern_display_name'], drop_first = True)
#removed production_year as well
concatenated = pd.concat([Concatenated['id'], Concatenated['movie_sequel'], creative_type, source, production_method, genre, movie_board_rating_display_name, movie_release_pattern_display_name, Concatenated['Category']], axis=1)
concatenated.head()


# In[22]:


sns.heatmap(concatenated.isnull())


# In[23]:


concatenated.corr()


# In[24]:


Test = concatenated[concatenated['Category'].isnull()]
Train = concatenated[-concatenated['Category'].isnull()]
Test.drop(['Category'], inplace=True, axis=1)
Test.head()


# In[25]:


Train['Category'] = Train['Category'].astype(int)
Train = Train.drop(['id'], axis=1)
Train.head()


# In[26]:


#splitting Train for cross validation
from sklearn.model_selection import train_test_split
X = Train.drop(['Category'], axis=1)
y = Train['Category']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# In[27]:


from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score


# In[28]:


#Using Logistic Regression
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression(solver='lbfgs')
logmodel.fit(X_train, y_train)
pred = logmodel.predict(X_test)
c = (y_test != pred).sum()
print('Misclassified Samples: {}'.format(c))

accuracy = accuracy_score(pred, y_test)
print('Accuracy: {}'.format(accuracy))
print("Accuracy Mean through CV: {}".format(cross_val_score(logmodel, X, y, scoring='accuracy', cv=10).mean()))


# In[29]:


#Using Random Forests
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators = 1500)
rfc.fit(X_train, y_train)
pred_RFC = rfc.predict(X_test)
print("Accuracy: {}".format(cross_val_score(rfc, X, y, scoring='accuracy', cv=10).mean()))


# In[30]:


#trying knn
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=100)
knn.fit(X_train, y_train)
pred_knn = knn.predict(X_test)
print("Accuracy: {}".format(cross_val_score(knn, X, y, scoring='accuracy', cv=10).mean()))


# In[31]:


#trying xgb
from xgboost import XGBClassifier
xgb = XGBClassifier(eta=0.01, max_depth=2, n_estimaors=1000)
xgb.fit(X_train, y_train)
pred_xgb = xgb.predict(X_test)
print("Accuracy: {}".format(cross_val_score(xgb, X, y, scoring='accuracy', cv=10).mean()))


# In[32]:


#trying lgbm
from lightgbm import LGBMClassifier
lgbm = LGBMClassifier(objective='multiclass', random_state=101, max_bin=63, learning_rate=0.1, num_iterations=700)
lgbm.fit(X_train, y_train)
pred_lgbm = lgbm.predict(X_test)
print("Accuracy: {}".format(cross_val_score(lgbm, X, y, scoring='accuracy', cv=10).mean()))


# In[ ]:


#trying ann
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from keras.utils import np_utils
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
encoder = LabelEncoder()
encoder.fit(y)
encoded_y = encoder.transform(y)
dummy_y = np_utils.to_categorical(encoded_y)
X_train, X_test, y_train, y_test = train_test_split(X, dummy_y, test_size=0.3, random_state=101)
def ann_model():
    ann = Sequential()
    ann.add(Dense(activation='relu', init='uniform', output_dim=26, input_dim=52))
    ann.add(Dense(activation='relu', init='uniform', output_dim=13))
    ann.add(Dense(activation='softmax', init='uniform', output_dim=9))
    ann.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return ann
ann = KerasClassifier(build_fn=ann_model, epochs=500, batch_size=15)
print('Accuracy: {}'.format(cross_val_score(ann, X, dummy_y, cv=10).mean()))


# In[34]:


#using xgb for prediction
final_prediction = xgb.predict(Test.drop(['id'], axis=1))
final_prediction


# In[38]:


test['Category'] = final_prediction


# In[39]:


test.head()


# In[42]:


test.to_excel("Prediction.xlsx")


# In[ ]:




