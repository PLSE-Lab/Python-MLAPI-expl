#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import time


# In[ ]:


np.random.seed(1988)


# In[ ]:


get_ipython().system(' dir')


# ### Data exploration

# In[ ]:


data = pd.read_json("../input/train.json")
test = pd.read_json("../input/test.json")
data.head()


# In[ ]:


#there is no NaNs
data[pd.isna(data['ingredients'])]


# In[ ]:


data.shape


# In[ ]:


def extract_ingredients(serie):
    list_ingredients=[]
    for lista in serie:
        for element in lista:
            if element in list_ingredients:
                pass
            elif element not in list_ingredients:
                list_ingredients.append(element)
            else:
                pass
        
    return list_ingredients      


# In[ ]:


ingredients = extract_ingredients(data['ingredients'])


# In[ ]:


len (ingredients)


# In[ ]:


#Types of differents cuisines:
data['cuisine'].unique().shape


# In[ ]:


cuisines = data['cuisine'].unique()
cuisines


# We will try two feature engineering methods, one hot encoding and feature hashing to see what is more effective.

# ## One hot encoding method

# In[ ]:


#Create columns
t = time.time()
for ingredient in ingredients:
    data[ingredient]=np.zeros(len(data["ingredients"]))

print("It took %i seg" %(time.time()-t))


# In[ ]:


def ohe(serie, dtset):    
    ind=0
    for lista in serie:
        for ingredient in lista:
            if ingredient in ingredients:
                dtset.loc[ind,ingredient]=1
            else:
                pass
        ind +=1


# In[ ]:


t = time.time()
ohe(data['ingredients'], data)
print('it took %i segs' % (time.time()-t))


# In[ ]:


predictors = ingredients
response = 'cuisine'


# In[ ]:


X = data[predictors]
Y = data[response]


# ### Neural Networks

# In[ ]:


from keras.models import Sequential
from keras.layers import Dense
import numpy as np


# In[ ]:


np.random.seed(1988)


# In[ ]:


len(cuisines)


# In[ ]:


from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

encoder = LabelEncoder()
encoder.fit(Y)
encoded_y_train = encoder.transform(Y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y_train = np_utils.to_categorical(encoded_y_train)


# In[ ]:


encoded_y_train


# In[ ]:


dummy_y_train


# Now we have and output vector, with the categories Ohe

# In[ ]:


# Create model:
def baseline_model():
    model = Sequential()
    model.add(Dense(100, input_dim=6714, activation='relu'))
    model.add(Dense(80, activation='relu'))
    model.add(Dense(20, activation='softmax')) # The output layer must create 20 output values,
    # one for each class. The output value with the largest value will be taken as the class predicted by the model.

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# In[ ]:


from keras.wrappers.scikit_learn import KerasClassifier
estimator = KerasClassifier(build_fn=baseline_model, epochs=5, batch_size=100)


# In[ ]:


estimator.fit(X, dummy_y_train)


# #### Make predictions:

# In[ ]:


# Create columns on test
t = time.time()
for ingredient in ingredients:
    test[ingredient]=np.zeros(len(test["ingredients"]))

print("It took %i seg" %(time.time()-t))


# In[ ]:


t = time.time()
ohe(test['ingredients'], test)
print('it took %i segs' % (time.time()-t))


# In[ ]:


y_pred = estimator.predict(test[predictors])
y_pred = encoder.inverse_transform(y_pred)


# In[ ]:


test['cuisine'] = y_pred


# In[ ]:


output = pd.DataFrame(test['id'])
output['cuisine'] = pd.Series(y_pred, name='cuisine')


# In[ ]:


output.to_csv('output.csv', index=False)

