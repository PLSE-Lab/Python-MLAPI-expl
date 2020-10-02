#!/usr/bin/env python
# coding: utf-8

# This is my code for a prime number predictor using SKLearn's Random Forest Regressor. My goal was to input a number n, and have the model return the nth prime number. I think it works pretty well, the predicted output is always within 5 of the true value.

# In[ ]:


#Charles Averill, 2019


# In[ ]:


#Imports
import pandas as pd

import sklearn

from itertools import count, islice
from math import sqrt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


# Here we create our training method, using the sklearn's train_test_split to split features and labels into test and train data.

# In[ ]:


#Training method
def train(features, labels):
    #Separate features and labels into training and testing sets
    ftrain, ftest, ltrain, ltest = train_test_split(features, labels, test_size=0.1)
    #Create Random Forest Regressor with 100 trees
    regressor = RandomForestRegressor(n_estimators=100)
    #Fit model and get accuracy
    model = regressor.fit(ftrain, ltrain)
    accuracy = regressor.score(ftest, ltest) * 100

    return model, accuracy


# Here we read in the first 100k primes from my dataset

# In[ ]:


df = pd.read_csv("../input/first-100000-prime-numbers/output.csv").drop(['Interval'], axis=1)


# Separating our data into features and labels

# In[ ]:


features = df.drop('Num', axis=1).values
labels = df['Num'].values


# Call the train function and print the returned accuracy

# In[ ]:


model, accuracy = train(features, labels)
print("Accuracy:", accuracy)


# Prompting for integer n to return the nth prime. Kaggle frontend doesn't support Python's input() function, so let's assume you asked for the 50th prime number.

# In[ ]:


labels = ['Rank']

csv = df.values

#num = int(input("What index prime do you want? "))
num = 50


# A method that checks primality

# In[ ]:


def isPrime(n):
    return n > 1 and all(n%i for i in islice(count(2), int(sqrt(n)-1)))


# Get the model's prediction for n. By adding the while loop, I change the prediction until it's prime.

# In[ ]:


#Predicting
inp = pd.DataFrame([[num]], columns = labels)

prediction1 = int(round(model.predict(inp)[0] - .5))
prediction2 = prediction1
#Makes predictions more accurate
while(not isPrime(prediction1) and not isPrime(prediction2)):
    prediction1 += 1
    prediction2 -= 1

#Print values
if(isPrime(prediction1)):
    print("Prediction:", prediction1)
else:
    print("Prediction:", prediction2)
#Only prints actual value if the csv has it
if(num < len(csv)):
    print("Actual Value:", csv[num - 1][1])


# This is one of the first models I've built without a tutorial at hand. It's not too complicated, but needed improvement in the last cell with my primality while loop. However, given that the prediction is usually less than 5 away from the true value, it shouldn't impact performance time that much. This was a lot of fun to write, and I look forward to expanding the complexity of my future projects.
# 
# Unfortunately, the models output by this code are about 500x larger than their training files (e.g. the 100k dataset is 1.6MB, and the model that trained on it is about 730MB), so there's no reason to use an AI for this instead of a simple program that loads primes from a CSV. It was an interesting experiment though!
