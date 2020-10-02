#!/usr/bin/env python
# coding: utf-8

# **IPL Game Prediction Kernel for SigTuple**<br>
# This kernel will aim to find which team will win the match and the probability score for each team.

# In[ ]:


# IMPORTING REQUIRED LIBRARIES
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# LOADING THE GIVEN DATA
train = pd.read_csv("../input/ipledited/train.csv")
test = pd.read_csv("../input/ipl-game-prediction/test.csv")
# VISUALIZING THE DATASET
test.head()


# In[ ]:


# TO CHECK THE DATASET FOR IMBALANCE - SO WE CAN EITHER UPSAMPLE OR DOWNSAMPLE
train['Winner (team 1=1, team 2=0)'].value_counts()


# In[ ]:


# BASIC ANALYSIS OF COLUMNS
print(train.columns)
print(train.dtypes)
train.describe(include="all")


# From the above basic analysis, we can conclude that:<br>
# * There are 3 types of data available - Integer, String and Float
# * The data is complete with no missing values - saving us the trouble of dropping columns or filling in extra values using correlation

# Let us do a test run with some basic algorithms to see how the algorithms perform.

# In[ ]:


# DROPPING THE CATEGORICAL VARIABLES AND SPLITTING THE DATA INTO TRAINING AND TESTING SETS
from sklearn.model_selection import train_test_split
p =  train.drop(['Game ID','Team 1','Team 2','City','DayOfWeek','DateOfGame','TimeOfGame','AvgWindSpeed','AvgHumidity','Winner (team 1=1, team 2=0)'],axis=1)
target = train['Winner (team 1=1, team 2=0)']
x_train,x_val,y_train,y_val = train_test_split(p,target,test_size=0.25,random_state=0)
test_target = test['Winner (team 1=1, team 2=0)']
q = test.drop(['Game ID','Team 1','Team 2','CityOfGame','Day','DateOfGame','TimeOfGame','AvgWindSpeed','AvgHumidity','Winner (team 1=1, team 2=0)'],axis=1)


# In[ ]:


from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
l = LogisticRegression()
l.fit(x_train,y_train)
y_pred = l.predict(x_val)
print(classification_report(y_val,y_pred))
print()
print(confusion_matrix(y_val,y_pred))
print()
print(accuracy_score(y_val,y_pred)*100)
y_lol = l.predict(q)
print()
print(classification_report(test_target,y_lol))
print()
print(confusion_matrix(test_target,y_lol))
print()
print(accuracy_score(test_target,y_lol)*100)


# Here, I am only printing out the algorithm with the best accuracy - with all 20 parameters. However, I will list the algorithm accuracies here:<br>
# * SVM = Training: 51.42% and Testing: 55.26%
# * Decision Tree = Training: 87.14% and Testing: 84.21%
# * Random Forest = Training: 84.28% and Testing: 85.52%
# * K Nearest Neighbor = Training: 47.14% and Testing: 61.84%
# * **Logistic Regression = Training: 95.16% and Testing: 92.10%**
# * Perceptron = Training: 48.57% and Testing: 47.36%
# * GaussianNB = Training: 94.28% and Testing: 85.52%
# * Stochastic Gradient Descent = Training: 50% and Testing: 47.36%

# **Visualizations of the Dataset** <br>
# This visualization of the parameters will allow us to see if there is any correlation between the parameters and winning the game for the team. This way we can choose which parameters to use in our model.
# Looking at the dataset description, I feel the following parameters must be explored in further depth

# In[ ]:


# NORMAL CORRELATION CHECKER
sns.barplot(x="Inn 1 Team 2 wickets taken_catches_runout",y="Winner (team 1=1, team 2=0)",color="yellow",data=train)


# Above, I am showing how I checked the dataset for correlation - by visualization of how the 20 parameters affect the WIN variable.<br>
# I found that all variables have some correlation with the WIN variables except the following:<br>
# * Inn 1 Team 2 wickets taken_catches_runout
# * Inn 1 Team 2 Extras conceded in_wides_No Balls
# * Inn 2 Team 2 NOP R>25,SR>125
# * Inn 2 Team 2 Total 6s
# * Inn 2 Team 1 Extras conceded in_wides_No Balls 

# Now I will resplit the dataset so we can test to see if removing some of the unnecessary parameters can improve the classifier.

# In[ ]:


# DROPPING THE CATEGORICAL VARIABLES AND SPLITTING THE DATA INTO TRAINING AND TESTING SETS
from sklearn.model_selection import train_test_split
p = train.drop(['Game ID','Team 1','Team 2','City','DayOfWeek','DateOfGame','TimeOfGame','AvgWindSpeed','AvgHumidity','Inn 1 Team 2 wickets taken_catches_runout','Inn 1 Team 2 Extras conceded in_wides_No Balls','Inn 2 Team 2 NOP R>25,SR>125','Inn 2 Team 2 Total 6s','Inn 2 Team 1 Extras conceded in_wides_No Balls','Winner (team 1=1, team 2=0)'],axis=1)
target = train['Winner (team 1=1, team 2=0)']
x_train,x_val,y_train,y_val = train_test_split(p,target,test_size=0.28,random_state=0)
test_target = test['Winner (team 1=1, team 2=0)']
q = test.drop(['Game ID','Team 1','Team 2','CityOfGame','Day','DateOfGame','TimeOfGame','AvgWindSpeed','AvgHumidity','Inn 1 Team 2 wickets taken_catches_runout','Inn 1 Team 2 Extras conceded in_wides_No Balls','Inn 2 Team 2 NOP R>25,SR>125','Inn 2 Team 2 Total 6s','Inn 2 Team 1 Extras conceded in_wides_No Balls','Winner (team 1=1, team 2=0)'],axis=1)


# In[ ]:


# RETESTING THE DATA TO SEE IF WE CAN GET ANY IMPROVEMENT IN PRECISION, RECALL and ACCURACY
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
l = LogisticRegression()
l.fit(x_train,y_train)
y_pred = l.predict(x_val)
print(classification_report(y_val,y_pred))
print()
print(confusion_matrix(y_val,y_pred))
print()
print(accuracy_score(y_val,y_pred)*100)
y_lol = l.predict(q)
print()
print(classification_report(test_target,y_lol))
print()
print(confusion_matrix(test_target,y_lol))
print()
print(accuracy_score(test_target,y_lol)*100)
y_prob = l.predict_proba(q)


# The accuracies after changing the parameters for each of the algorithms are as follows:<br>
# * SVM = Training: 51.42% and Testing: 55.26%
# * Decision Tree = Training: 82.85% and Testing: 85.52%
# * Random Forest = Training: 81.42% and Testing: 81.57%
# * K Nearest Neighbor = Training: 51.42% and Testing: 63.15%
# * **Logistic Regression = Training: 95.71% and Testing: 93.42%**
# * Perceptron = Training: 87.14% and Testing: 78.94%
# * GaussianNB = Training: 91.42% and Testing: 84.21%
# * Stochastic Gradient Descent = Training: 61.42% and Testing: 59.21%
# 

# Thus, we can see that even though the best algorithm has remained Logistic Regression but there have been improvements caused by the feature engineering:
# * **Training Accuray: 95.16% to 95.71%**
# * **Testing Accuracy: 92.10% to 93.42%**

# In[ ]:


# PRINTING THE TEST DATASET AND RESULTS
print("Team 1", "|", "Team 2","|", "Winner","|", "Probability")
for i in range(1,len(test)):
    print(test["Team 1"][i],"|",test['Team 2'][i],"|", y_lol[i],"|", y_prob[i])

output = pd.DataFrame({'Team 1': test['Team 1'], 'Team 2': test['Team 2'], 'Winner': y_lol, 'Probability' : list(y_prob)})
output.to_csv("Prediction.csv",index=False)


# Now that we have our predictions and our data into a CSV file - let me explain my thought process:
# * Inspite of knowing there are other models like Neural Networks and their types - I didn't use them because I felt that with training data having only 247 entries, the neural nets won't be able to bring out their best.
# * Still to test my curiosity I created a small ANN to see how the neural networks perform

# In[ ]:


import keras 
from keras.models import Sequential 
from keras.layers import Dense,Dropout

# MODEL
model = Sequential()

# layers
model.add(Dense(units = 12, kernel_initializer = 'uniform', activation = 'relu', input_dim = 15))
model.add(Dense(units = 9, kernel_initializer = 'uniform', activation = 'relu'))
model.add(Dropout(0.3))
model.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
model.compile(optimizer = 'Adagrad', loss = 'binary_crossentropy', metrics = ['accuracy'])

model.fit(x_train,y_train,batch_size=32,epochs=200)
y_pred = model.predict(x_val)
for i in range(len(y_pred)):
    if(y_pred[i]<0.5):
        y_pred[i] = 0
    else:
        y_pred[i] = 1
print(classification_report(y_val,y_pred))
print()
print(confusion_matrix(y_val,y_pred))
print()
print(accuracy_score(y_val,y_pred)*100)
y_lol = model.predict(q)
for i in range(len(y_lol)):
    if(y_lol[i]<0.5):
        y_lol[i] = 0
    else:
        y_lol[i] = 1
print()
print(classification_report(test_target,y_lol))
print()
print(confusion_matrix(test_target,y_lol))
print()
print(accuracy_score(test_target,y_lol)*100)


# Thus, we can see that even though we achieved the same training accuracy as our logistic regression model, our neural network is overfitting as our testing accuracy is 85.52% <br>
# Some of the things I would like to do:<br>
# * I would like to apply some regularization and early stopping to the neural model to see if the model will stop overfitting.
# * I would like to explore upsampling the minority class so we can increase the training data size - even though the difference is small.
# * I would like to run some more complex models like Gradient Boosting and GridSearchCV to try and do some better feature selection.
