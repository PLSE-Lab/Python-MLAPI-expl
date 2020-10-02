#!/usr/bin/env python
# coding: utf-8

# In this notebook, I practice a few different techniques, creating a decision tree and neural network model to predict sport given physical features (Sex, Age, Weight, Height) -> what sport should you do, given your physical features?

# In[64]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[65]:


df = pd.read_csv('../input/athlete_events.csv')
df.info()


# Note that there are null entries in age, height, weight, and medal, but all other columns are complete. This is expected for medal, since only a small proportion of those entered in each event will win a medal. For age, height, and weight, this is simply missing data.

# Import packages for visualizations and modeling.

# In[66]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
import tensorflow as tf


# In[67]:


df.head()


# Things to note from the head:
# - Can see some missing height and weight is likely because this was not recorded for competitors from early modern olympics (1900, 1920).
# - Some sports and events from early modern olympics are no longer included in the current olympics (Tug-Of-War in 1900).
# - Sport is general, there may be many events to a sport.
# - Team and NOC may not directly match e.g. Denmark/Sweden -> DEN

# In[68]:


df[(df['Season']=='Summer')&(df['Year']==2016)]['Sport'].value_counts()


# These are the 34 sports that were in the most recent summer games (2016), noting that athletics and Swimming are very large (>1000 competitors), and Trampolining, Modern Pentathlon, Beach Volleyball, and Rythmic Gymnastics are small (<100)
# 
# Also, regarding number of competitors, note that there are duplicate rows with respect to weight, height, sex, and age information for competitors in sports like Athletics, Swimming, Gymnastics, as individuals compete in multiple events in these sports

# In[69]:


a = df[(df['Season']=='Summer')]['Sport'].value_counts()
allSports = list(a.keys())
len(allSports)


# In[70]:


a = df[(df['Season']=='Summer')&(df['Year']>=2000)]['Sport'].value_counts()
modernSports = list(a.keys())
len(modernSports)


# In[71]:


# To find a count of the number of times a sport has been included in the summer games
a = df[df['Season']=='Summer'].groupby(['Year'])['Sport'].value_counts()
yearIndex = 0
c = np.zeros(((int)((2020-1896) / 4),len(allSports)))
for year in range(1896,2020,4):
    if year in a:
        sportIndex = 0
        for sport in allSports:
            if (sport in a[year]):
                c[yearIndex][sportIndex] = a[year][sport]
            sportIndex = sportIndex + 1
    yearIndex = yearIndex + 1


# In[72]:


aa = df[(df['Season']=='Summer')]['Sport'].value_counts()
plt.figure(figsize=(16,8))
plt.subplot(121)
plt.barh(allSports,aa.values)
plt.subplot(122)
plt.barh(allSports,sum(c>0))


# The figure on the left shows the number of competitors for each sport over the whole dataset. The figure on the right shows the number of years in which a sport has been held.
# 
# Things to note:
# - Modern Pentathlon, which has been held at 24 of the games, but has a small number of competitors each year.
# - Several events in the modern list of events (Taekwondo, Baseball, Beach Volleyball) have only been held in a small number of games, and have small number of total competitors in the dataset
# - Not every sport is held at every games, for example Baseball and Softball were not held in 2016, but are back on the program for 2020.
# - There are 52 sports that have been held in the Summer games, of which 36 have been at games since 2000

# ### Model of the data
# 
# I propose to predict what event a competitor is in, given their height, weight, and age data.
# 
# It is unlikely that this will be able to be predicted with high accuracy, because there will be large overlap between the types of competitor in different sports. However, the top X sports for a given height, weight, age, may be fairly good indicator.
# 
# And this could be interesting information for someone: what sport would someone who is their height, weight, sex, age, likely to participate in at the olympics?
# 
# I propose to restrict data in the following ways:
# - remove competitors for which there is no height, weight, age data (all other data is complete)
# - restrict season to summer
# - remove duplicate competitors that competed in multiple events in the same sport in the same year
# - restrict the set of sports (see below)
# 
# I initially thought to restrict the year to after 1996, however, this reduces the number of data points significantly. Restricting the sports somehow, but including the earlier data should result in a better classifier.
# 
# To select a good set of sports to select over:
# - Has been held in the last X games
# - Has been held in at least Y games
# 
# Good values for X and Y are probably X=5 (since 2000), Y=5
# 
# It would also be possible to select for number of competitors, but the restriction to have been held in at least 5 years should already remove sports with too few competitors.
# 
# Also process data so that it can be used in a Decision Tree and a Neural Network

# In[73]:


df1 = df.dropna(subset=['Age', 'Height', 'Weight'])
df1 = df1[df1['Season'] == 'Summer']
df1 = df1.drop(columns=['ID', 'Team', 'Games', 'Season', 'City', 'NOC', 'Event', 'Medal'])
df1 = df1.drop_duplicates()
df1 = df1.drop(columns=['Name'])
le1 = LabelEncoder()
df1['Sex'] = le1.fit_transform(df1['Sex'])
aa = df1['Sport'].value_counts()
allSports1 = list(aa.keys())
a = df1.groupby(['Year'])['Sport'].value_counts()
yearIndex = 0
c = np.zeros(((int)((2020-1896) / 4),len(allSports1)))
for year in range(1896,2020,4):
    if year in a:
        sportIndex = 0
        for sport in allSports1:
            if (sport in a[year]):
                c[yearIndex][sportIndex] = a[year][sport]
            sportIndex = sportIndex + 1
    yearIndex = yearIndex + 1
aa = df1['Sport'].value_counts()
plt.figure(figsize=(16,8))
plt.subplot(121)
plt.barh(allSports1,aa.values)
plt.subplot(122)
plt.barh(allSports1,sum(c>0))


# After taking out competitors with no age, weight, or height entered, there are now 43 sports.

# In[74]:


chosenSports = modernSports
sportIndex = 0
for sport in allSports1:
    if sum(c>0)[sportIndex] < 5:
        if sport in chosenSports:
            chosenSports.remove(sport)
    sportIndex = sportIndex + 1
len(chosenSports)


# After choosing sports that have been included in the games since 2000, and are in at least 5 games, there are now 33 different sports.

# In[75]:


df1 = df1[df1['Sport'].isin(chosenSports)]
df1 = df1.drop(columns=['Year'])
sports = df1['Sport'].unique()
le2 = LabelEncoder()
df1['Sport'] = le2.fit_transform(df1['Sport'])
df1.info()


# In[76]:


sns.pairplot(df1)


# This pairplot shows that there are some trends between different sports, but there is also a very large overlap, particularly in the average age, height, and weight of competitors overall.
# 
# So, although it is unlikely that a classifier will do very well on this task (predict sport from sex, age, height, weight), here I will compare some standard techniques for predicting:
# - decision tree
# - neural networks

# In[77]:


X = df1.drop(columns=['Sport'])
y = df1['Sport']
scaler = StandardScaler()
scaler.fit(X.values)
scaled_features = scaler.fit_transform(X.values)
df_feat = pd.DataFrame(scaled_features, columns=X.columns)
X1 = df_feat
y1 = pd.DataFrame(y)
X_train, X_test, y_train, y_test = train_test_split(X1.values, y1.values, test_size=0.3)


# In[78]:


#Decision tree:
dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)
predictions_dt = dtree.predict(X_train)
print(classification_report(y_train,predictions_dt))
predictions_dt = dtree.predict(X_test)
print(classification_report(y_test,predictions_dt))


# Decision Tree has overfit the training set, reaching 60% precision, 56% recall, f1-score of 0.55, but on the test set reaching average 21% precision, 23% recall, with a f1-score of 0.21. It was very quick to train.

# In[79]:


# Neural network, several relu layers, finish with a softmax layer
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(4),
    tf.keras.layers.Dense(10, activation=tf.nn.relu),
    tf.keras.layers.Dense(20, activation=tf.nn.relu),
    tf.keras.layers.Dense(40, activation=tf.nn.relu),
    tf.keras.layers.Dense(len(chosenSports), activation=tf.nn.softmax)
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# In[80]:


model.fit(X_train, y_train, epochs=10)


# In[81]:


train_loss, train_acc = model.evaluate(X_train, y_train)
print('Train accuracy: {:5.2f}'.format(100*train_acc))
test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test accuracy: {:5.2f}'.format(100*test_acc))


# In[82]:


predictions_nn = model.predict(X_train)
predictions_nn1 = np.argmax(predictions_nn, axis=1)
print(classification_report(y_train,predictions_nn1))
predictions_nn = model.predict(X_test)
predictions_nn1 = np.argmax(predictions_nn, axis=1)
print(classification_report(y_test,predictions_nn1))


# The neural network performs worse than the decision tree on the training data, but generalises better to the test set. However, there are about one third of the classes that are not being predicted correctly at all.

# In[83]:


counts = np.zeros((len(chosenSports),), dtype=int)
i = 0
for prediction in predictions_nn:
    a = np.argsort(prediction)
    for j in range((len(chosenSports))):
        if y_test[i] == a[-(j+1)]:
            for c in range((len(chosenSports)-j)):
                counts[j+c] = counts[j+c]+1
    i = i+1
counts = counts / i
# accuracy for if target is included in top X suggested sports
print(counts)


# In[84]:


plt.plot(counts)


# For the neural network, it is also straightforward to convert a single prediction to choosing the top X sports. With 5 sports, the correct sport is included 67% of the time, with 10 sports, the sport is included 85% of the time.

# For the two models, here is a query for what sport is predicted for a given set of features.

# In[85]:


def sport_for_person(sex, age, height, weight, model_type):
    test = np.array([sex, age, height, weight], np.float64)
    test = test.reshape(1, -1)
    test1 = scaler.transform(test)
    toprint = ""
    if model_type == 'svc':
        prediction = svc_model.predict(test1)
        toprint = "[SVC] Suggested sport is: "
        toprint = toprint + sports[prediction]
        print(toprint[0])
    elif model_type == 'dt':
        prediction = dtree.predict(test1)
        toprint = "[DT] Suggested sport is: "
        toprint = toprint + sports[prediction]
        print(toprint[0])
    elif model_type == 'nn':
        prediction = model.predict(test1)
        a = np.argsort(prediction)
        toprint = "[NN] Suggested sports are: "
        for j in range(10):
            toprint = toprint + sports[a[0][-(j+1)]] + ' (' + (str)((int)(100*prediction[0][a[0][-(j+1)]])) + '%)'
            if (j!=9):
                toprint = toprint + ', '
        print(toprint)


# In[86]:


#sex: for male enter 1, for female enter 0
#age: in years
#height: in cm
#weight: in kg

sexes = [0, 1]
ages = [18, 24, 32]
heights = [150, 175, 200]
weights = [50, 80, 110]
for sex in sexes:
    for age in ages:
        for height in heights:
            for weight in weights:
                
                print('For a', age, 'year old', 'male' if sex else 'female', height, 'cm and', weight, 'kg:')
                sport_for_person(sex, age, height, weight,'dt')
                sport_for_person(sex, age, height, weight,'nn')


# Predictions are made, but they are not necessarily sensible ones. There is probably some interesting dividing up of sex-age-height-weight space for the different sports that could be looked into.

# ### Conclusions
# The models constructed do ok, considering the overlapping nature of the categories. More could be done to tune these, but doing so would not guarantee a model that can accurately predict a sport or event given a competitors details, in part because for a given set of characteristics, there may be a wide range of appropriate sports.
