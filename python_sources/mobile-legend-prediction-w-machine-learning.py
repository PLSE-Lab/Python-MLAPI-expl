#!/usr/bin/env python
# coding: utf-8

# In[ ]:


Hi, this post will be about pre-processing Mobile Legend dataset and applying it into Machine Learning Classifier e.g. Logistic Regression or Decision Tree.
I did my code mostly on Google Colab.


# In[ ]:


#import the essential to deal with data processing
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[ ]:


#mount your Google Drive and read the excel file
df = pd.read_excel('/content/drive/My Drive/ml_dataset3.xlsx')


# In[ ]:


heroArray = ['Akai', 'Aldous', 'Alice', 'Alpha', 'Alucard', 'Angela', 'Argus', 'Atlas', 'Aurora', 'Badang', 'Balmond', 'Bane', 'Baxia', 'Belerick', 'Bruno', 'Carmilla', 'Cecilion', 'Change', 'Chou', 'Claude', 'Clint', 'Cyclops', 'Diggie', 'Dyrroth', 'Esmeralda', 'Estes', 'Eudora', 'Fanny', 'Faramis', 'Franco', 'Freya', 'Gatotkaca', 'Gord', 'Granger', 'Grock', 'Guinevere', 'Gusion', 'Hanabi', 'Hanzo', 'Harith', 'Harley', 'Hayabusa', 'Helcurt', 'Hilda', 'Hylos', 'Irithel', 'Jawhead', 'Johnson', 'Kadita', 'Kagura', 'Kaja', 'Karina', 'Karrie', 'Khufra', 'Kimmy', 'Lancelot', 'Lapu-Lapu', 'Layla', 'Leomord', 'Lesley', 'Ling', 'Lolita', 'Lunox', 'Luo Yi', 'Lylia', 'Martis', 'Masha', 'Minotaur', 'Minsitthar', 'Miya', 'Moskov', 'Nana', 'Natalia', 'Odette', 'Pharsa', 'Popol And Kupa', 'Rafaela', 'Roger', 'Ruby', 'Saber', 'Selena', 'Silvanna', 'Sun', 'Terizla', 'Thamuz', 'Tigreal', 'Uranus', 'Vale', 'Valir', 'Vexana', 'Wanwan', 'XBorg', 'Yi Sun-Shin', 'Yu Zhong', 'Zhask', 'Zilong']


# In[ ]:


#The concept is similar to LabelEncoder where categorical variable (radiant/dire heroes) are changed to 1,2,3...
#This may not be ideal but further improvement suggestion is encouraged.

#Defined a function to LabelEncode the heroes column in excel file

def convertHero (var, arr):
  for count, i in enumerate(arr, start=1):
    if i == var:
      return count


# In[ ]:


#Then we apply the function on the 10 columns of heroes picked: radiant_hero_1...radiant_hero_5...dire_hero_1...dire_hero_5
df['rh_1'] = df['rh_1'].apply(lambda x: convertHero(x, heroArray))
df['rh_2'] = df['rh_2'].apply(lambda x: convertHero(x, heroArray))
df['rh_3'] = df['rh_3'].apply(lambda x: convertHero(x, heroArray))
df['rh_4'] = df['rh_4'].apply(lambda x: convertHero(x, heroArray))
df['rh_5'] = df['rh_5'].apply(lambda x: convertHero(x, heroArray))

df['dh_1'] = df['dh_1'].apply(lambda x: convertHero(x, heroArray))
df['dh_2'] = df['dh_2'].apply(lambda x: convertHero(x, heroArray))
df['dh_3'] = df['dh_3'].apply(lambda x: convertHero(x, heroArray))
df['dh_4'] = df['dh_4'].apply(lambda x: convertHero(x, heroArray))
df['dh_5'] = df['dh_5'].apply(lambda x: convertHero(x, heroArray))

df.head(10)


# In[ ]:


#similarly, we do the same for the roles.
roleArray = ['Tank', 'Assassin', 'Marksman', 'Fighter', 'Support', 'Mage']

def convertRole (var, arr):
  for count, role in enumerate(arr, start = 1):
    if role == var:
      return count
    else:
      pass


# In[ ]:


df['rh_1_role'] = df['rh_1_role'].apply(lambda x: convertRole(x, roleArray))
df['rh_2_role'] = df['rh_2_role'].apply(lambda x: convertRole(x, roleArray))
df['rh_3_role'] = df['rh_3_role'].apply(lambda x: convertRole(x, roleArray))
df['rh_4_role'] = df['rh_4_role'].apply(lambda x: convertRole(x, roleArray))
df['rh_5_role'] = df['rh_5_role'].apply(lambda x: convertRole(x, roleArray))

df['dh_1_role'] = df['dh_1_role'].apply(lambda x: convertRole(x, roleArray))
df['dh_2_role'] = df['dh_2_role'].apply(lambda x: convertRole(x, roleArray))
df['dh_3_role'] = df['dh_3_role'].apply(lambda x: convertRole(x, roleArray))
df['dh_4_role'] = df['dh_4_role'].apply(lambda x: convertRole(x, roleArray))
df['dh_5_role'] = df['dh_5_role'].apply(lambda x: convertRole(x, roleArray))


# In[ ]:


#we could visualize the 'outcome' column on how many win vs lose. 
#Currently the dataset is small as it's downloaded from Kaggle source: https://www.kaggle.com/iskelets/mobile-legends-data-set
import seaborn as sns
sns.countplot(df['outcome'], label = 'Count')


# In[ ]:


#Preparing training and test set
y = df.outcome
X = df.drop(['match_id', 'outcome'], axis=1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=0)


# In[ ]:


#Decision Tree
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()
classifier = classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn import metrics
print('Accuracy Score:', metrics.accuracy_score(y_test, y_pred))


# In[ ]:


#RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

forest_model = RandomForestRegressor(random_state=1)
forest_model.fit(X_train, y_train)
forest_pred = forest_model.predict(X_test)
print(mean_absolute_error(y_test, forest_pred))


# The accuracy achieved by Decision Tree is 0.6 while RandomForest is 0.46.
# Train test is split at 25% (standard).
# The dataset consist of only 10 column (input) which is very small. 
# Improvement suggestion encouraged please.
# Thank you.
