#!/usr/bin/env python
# coding: utf-8

# # A Journey Through Titanic
# 
# Let's get started:

# In[ ]:


# Necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import math
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_palette('colorblind')
sns.set_style('whitegrid')
plt.rc('font', size = 15)


# In[ ]:


# Filter warning - use this with caution!
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


# Let's take a look at the training data
training_df = pd.read_csv('../input/train.csv')


# In[ ]:


# Here is some information
training_df.info()


# As we can see some passangers are missing age information, and a large portion of the passengers are missing cabin information.

# In[ ]:


# Here is a quick summary
training_df.describe()


# ## Data Visualization

# In[ ]:


# Let's make a count plot where we see the survival for each sex
fig, ax = plt.subplots(1, 1, figsize = (15,5))
sns.countplot(x = 'Sex', hue = 'Survived', data = training_df, ax = ax)


# Well obviously we see that the survival rate for females are drastically higher than the males.

# In[ ]:


# Now let's make a cluster map
sns.clustermap(training_df.corr(), annot = True, fmt = '.2f', cmap = 'coolwarm')


# As for the numerical variables, survival seems to be mostly correlated to fare (positively) and ticket class (inversely). This means those that paied more, hence in the higher class, are more likely to have survived.
# 
# At this point we haven't looked at name, ticket, cabin, and embarked variables. Let's look at the embarking, too:

# In[ ]:


# Let's make a count plot where we see the survival for each sex
fig, ax = plt.subplots(1, 1, figsize = (15,5))
sns.countplot(x = 'Embarked', hue = 'Survived', data = training_df, ax = ax)


# It's rather interesting to see those who embarked from Cherbourg have a larger survival fraction compared to Queenstown and Southampton. We can see if this is statistically significant:

# In[ ]:


val1 = training_df[(training_df['Embarked']=='C')&(training_df['Survived']==0)]['PassengerId'].count()
std1 = math.sqrt(val1) # Assuming Poisson - see below
val2 = training_df[(training_df['Embarked']=='C')&(training_df['Survived']==1)]['PassengerId'].count()
std2 = math.sqrt(val2) # Assuming Poisson - see below
print('A-B = %.2f +/- %.2f'%(val1-val2,math.sqrt(std1**2+std2**2))) # Simple uncorrelated error propagation for A-B


# Here we're assuming *A* (embarked from C and not survived) and *B* (embarked from C and survived) are following a Poisson distribution (*a discrete probability distribution that expresses the probability of a given number of events occurring in a fixed interval of time or space if these events occur with a known constant rate and independently of the time since the last event*). Then we look at the difference *A-B* and see if it's consistent w/ zero (i.e. comparible). If we further assume *A-B* is normally distributed, the difference between *A* and *B* is about 1.4$\sigma$ away from zero. **This means the effect is not statistically significant.**
# 
# We can further compare Cherbourg to Queenstown and Southampton but let's not get into that.

# ## Training a basic Deep Neural Network
# 

# Now let's train a simple neural network to predict the survival. For this we need to do two things, first convert some of the categorical data into numeric ones, and then standardize these. For this analysis we're going to leave out the names, passenger id, ticket (irrelevant - maybe not the last one but the information is probably encapsulated in other variables such as class), age (not strongly correlated and missing data) and cabin (missing data) information. The latter two is to avoid data imputation, which we can deal w/ in another time.

# In[ ]:


# Now refine the dataframe
training_df_refined = training_df.drop(['PassengerId','Name','Ticket','Age','Cabin'], axis = 1)


# In[ ]:


# Deal w/ categorical data
features = ['Sex','Embarked']
training_df_final = pd.get_dummies(training_df_refined, columns = features, drop_first = True)
training_df_final.head()


# Now let's standardize our data:

# In[ ]:


# Let's load the necessary libraries
from sklearn.preprocessing import StandardScaler


# In[ ]:


# Define a scaler, fit, and transform
sc = StandardScaler() # use the default configuration
sc.fit(X = training_df_final.drop('Survived', axis = 1))
scaled_data = sc.transform(X = training_df_final.drop('Survived', axis = 1))


# In[ ]:


# Put the scaled data into a new dataframe
training_df_final_scaled = pd.DataFrame(data = scaled_data, columns = training_df_final.columns[1:] ) 
training_df_final_scaled.head()


# Now split the data into training and testing (actual testing data doesn't have the labels so we cannot quantify how well we're doing simply):

# In[ ]:


# Split our datasets
from sklearn.model_selection import train_test_split


# In[ ]:


# Now prepare the data
X = training_df_final.drop('Survived', axis = 1).values
y = training_df_final['Survived'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3 , random_state = 101)


# In[ ]:


# Let's load Keras and build a sequential dense model
from keras.models import Sequential
from keras.layers import Dense


# In[ ]:


# A sequential model where we stack layers on top of each other
model = Sequential()
model.add(Dense(units = 10, activation='relu', input_dim = 7))
model.add(Dense(units = 10, activation='relu'))
model.add(Dense(units = 1, activation='sigmoid'))


# In[ ]:


# Now compile the method.
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


# In[ ]:


# Now fit the model
model.fit(X_train, y_train, epochs = 30, batch_size = 20, verbose = 0)


# ## Model Evaluation

# In[ ]:


# Load some useful functions
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


# In[ ]:


y_predict = model.predict_classes(X_test) # Here we predict the classes


# In[ ]:


# Let's look at the accuracy score
print(accuracy_score(y_test,y_predict))


# In[ ]:


# Let's look at the confusion matrix
print(confusion_matrix(y_test,y_predict))


# In[ ]:


# Let's look at the classification report
print(classification_report(y_test,y_predict))


# So we get an accuracy of 70-80% w/ this super simple example. We could spend more time in cleaning the data, perhaps imputing the missing information, do a bit of feature engineering to come up w/ derived variables, train different models, such as *Random Forests* etc., scan various values for the hyper-parameters, and more. All these to come soon.
