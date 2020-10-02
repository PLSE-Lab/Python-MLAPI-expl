#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# This notebook is built to help students determine if they are eligible for an admission in their college preferred.
# 
# P.S. Some of the graphs are other people's ideas. I have put together things that I think is intuitive

# ## Initial Analysis
# 
# We would do the following in the notebook
# 
# 1. Load and prepare the data 
# 2. Checking out the data with correlation table and plot tables to see which features add more value
# 3. Analyze various features with graphs to understand which are important and which can be neglected
# 4. Normalizing the data and getting training and test data
# 5. Coming up with a tuned model model that would predict the next input

# ### Libaries that are used

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 


# ## Load and prepare the data
# 
# A critical step in working with neural networks is preparing the data correctly. Variables on different scales make it difficult for the network to efficiently learn the correct weights. Below, we've written the code to load and prepare the data.

# In[ ]:


data_path = '../input/Admission_Predict_Ver1.1.csv'

admissions = pd.read_csv(data_path)


# In[ ]:


admissions.head()


# In[ ]:


admissions.describe()


# ## Checking out the data
# This dataset has the admission percentage for a university based on various factors like GRE Score, TOEFL Score, University Rankig, SOP, LOR, CGPA and Research. 
# Below is a plot showing the number of students getting admitted just basedon the university ranking

# ### Correlation Table
# 
# Let us draw a correlation table to find out which are the important parameters to consider.
# Here we can see that the chance of admit is highly correlated with CGPA, GRE and TOEFEL scores are also correlated.

# In[ ]:


fig,ax = plt.subplots(figsize=(10, 10))
sns.heatmap(admissions.corr(), ax=ax, annot=True, linewidths=0.05, fmt= '.2f',cmap="magma")
plt.show()


# ### Pair plot  
# 
# The below pair plot indicates that GRE score TOEFL score and CGPA all are linearly related to each other Also it can be inferred that Research Students tend to Score higher by all means

# In[ ]:


cols=admissions.drop(labels='Serial No.',axis=1)
sns.pairplot(data=cols,hue='Research')


# ### Analyzing TOEFL Scores

# Analyzing the lowest, highest and mean TOEFL scores scored by the students

# In[ ]:


y = np.array([admissions["TOEFL Score"].min(),admissions["TOEFL Score"].mean(),admissions["TOEFL Score"].max()])
x = ["Lowest","Mean","Highest"]
plt.bar(x,y)
plt.title("TOEFL Scores")
plt.xlabel("Level")
plt.ylabel("TOEFL Score")
plt.show()


# ### Analyzing GRE Scores
# 
# Analyzing the spread of GRE scores of students

# In[ ]:


admissions["GRE Score"].plot(kind = 'hist',bins = 200,figsize = (6,6))
plt.title("GRE Scores")
plt.xlabel("GRE Score")
plt.ylabel("Frequency")
plt.show()


# ### Analyzing CGPA
# 
# From the below chart, we could understand that the students coming from higher ranked universities have better GPAs

# In[ ]:


admissions.plot(kind='scatter', x='University Rating', y='CGPA')


# ### Candidates who graduate from good universities have higher percentage of admissions

# In[ ]:


s = admissions[admissions["Chance of Admit "] >= 0.75]["University Rating"].value_counts()
plt.title("University Ratings of Candidates with a 75% acceptance chance")
s.plot(kind='bar',figsize=(20, 10))
plt.xlabel("University Rating")
plt.ylabel("Candidates")
plt.show()


# In[ ]:


fig = plt.figure(figsize = (20, 25))
j = 0
for i in admissions.columns:
    plt.subplot(6, 4, j+1)
    j += 1
    sns.distplot(admissions[i][admissions['Chance of Admit ']<0.72], color='r', label = 'Not Got Admission')
    sns.distplot(admissions[i][admissions['Chance of Admit ']>0.72], color='g', label = 'Got Admission')
    plt.legend(loc='best')
fig.suptitle('Admission Chance In University ')
fig.tight_layout()
fig.subplots_adjust(top=0.95)
plt.show()


# ### One hot encoding
# Here we have some categorical variables like University Rankig and Research. To include these in our model, we'll need to make binary dummy variables (or do one-hot encoding). This is simple to do with Pandas thanks to `get_dummies()`.

# In[ ]:


dummy_fields = ['University Rating', 'Research']
one_hot_admissions = admissions[:]
for each in dummy_fields:
    dummies = pd.get_dummies(one_hot_admissions[each], prefix=each, drop_first=False)
    one_hot_admissions = pd.concat([one_hot_admissions, dummies], axis=1)

to_be_dropped = ['University Rating', 'Research', 'Serial No.']
one_hot_admissions = one_hot_admissions.drop(to_be_dropped, axis=1)
one_hot_admissions.head()


# ### Normalizing the variables
# We could normalize variables like GRE, TOEFL, SOP, LOR and CGPA

# In[ ]:



processed_data = one_hot_admissions[:]

processed_data = processed_data/processed_data.max()
#processed_data = (processed_data - np.min(processed_data)) / (np.max(processed_data) - np.min(processed_data))


# ### Splitting the data into training, testing, and validation sets
# 
# We'll save the data for the last approximately 10% to use as a test set after we've trained the network. We'll use this set to make predictions and compare them with the actual percentage of admissions.

# In[ ]:


train_features = processed_data.drop('Chance of Admit ', axis=1)
train_targets = processed_data['Chance of Admit '].values

###This is another option####
from sklearn.model_selection import train_test_split
train_features,test_features,train_targets,test_targets = train_test_split(train_features,train_targets,test_size = 0.20,random_state = 42)


# ## Finally we have prepared our data. Now it's time to train it with neural nets !!! 

# In[ ]:


# Imports
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.utils import np_utils

# Building the model
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(train_features.shape[1],)))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compiling the model
model.compile(loss = 'mean_squared_error', optimizer='adam', metrics=['mse', 'mae'])
model.summary()


# ## Training the model

# In[ ]:


# Training the model
history = model.fit(train_features, train_targets, validation_split=0.2, epochs=100, batch_size=8, verbose=0)


# ## Evaluating the model
# 

# In[ ]:


#print(vars(history))
plt.plot(history.history['loss'])

plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


# In[ ]:


# Evaluating the model on the training and testing set
score = model.evaluate(train_features, train_targets)
print("score: ", score)
print("\n Training Accuracy:", score)
score = model.evaluate(test_features, test_targets)
print("score: ", score)
print("\n Testing Accuracy:", score)


# ## Prediction vs original labels

# In[ ]:


y_pred = model.predict(test_features)
plt.plot(test_targets)
plt.plot(y_pred)
plt.title('Prediction')

