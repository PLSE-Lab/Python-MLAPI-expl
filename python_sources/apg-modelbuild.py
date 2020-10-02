#!/usr/bin/env python
# coding: utf-8

# Import all the required libraries

# In[ ]:


import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import keras


# In[ ]:


train = pd.read_csv('../input/preprocesseddata/preprocessedData.csv')
test = pd.read_csv('../input/testfinaldata/testdata.csv')
train.head()


# Extract specific columns you wish to use as featues to train your model

# In[ ]:


features = ['Team','Opposition','TeamA Rating','TeamB Rating','Match Country','TeamA Home','TeamB Home','Ground']


# In[ ]:


X = train[features]
y = train.winbool
X.tail()


# Combine both train and test data, so as to encode all the categorical data to integer data.
# 
# * X : train data
# * X1: combined data
# * X2: test data

# In[ ]:


X1 = X.append(test,ignore_index=True)
X2 = X1[1042:]
X2.head()


# Create a backup of original train and test data, before encoding it.

# In[ ]:


trainBkp = X
testBkp = X2


# Apply encoding to select categorical columns of combined dataframe, to convert them to numerical values to be fed into the model.

# In[ ]:


number = LabelEncoder()
X1['Ground'] = number.fit_transform(X1['Ground'].astype('str'))
X1['Team'] = number.fit_transform(X1['Team'].astype('str'))
X1['Opposition'] = number.fit_transform(X1['Opposition'].astype('str'))
X1['Match Country'] = number.fit_transform(X1['Match Country'].astype('str'))


# In[ ]:


X1.head()


# Separate out encoded train and test data.

# In[ ]:


X = X1[:1042]
test = X1[1042:]
X.tail()


# In[ ]:


test.head()


# Take out actual train data from 'X' dataframe and keep the rest of data for cross-validation

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=12)


# In[ ]:


y.shape[0]


# In[ ]:


num_categories = 2
y_train = keras.utils.to_categorical(y_train, num_categories)
y_test= keras.utils.to_categorical(y_test, num_categories)
y_train


# Build the model

# In[ ]:


# Model Building
model = keras.models.Sequential()
model.add(keras.layers.Dense(50, activation="relu", input_dim = 8))
model.add(keras.layers.Dropout(0.1))
model.add(keras.layers.Dense(80, activation="relu"))
model.add(keras.layers.Dropout(0.1))
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dropout(0.1))
model.add(keras.layers.Dense(2, activation="softmax"))


# In[ ]:


# Compiling the model - adaDelta - Adaptive learning
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])


# In[ ]:


batch_size = 50
num_epoch = 5000
model_log = model.fit(X_train, y_train, batch_size = batch_size, epochs=num_epoch, verbose=1, validation_data=(X_test, y_test))


# Check the accuracy on the validation data.

# In[ ]:


train_score = model.evaluate(X_train, y_train, verbose=0)
test_score = model.evaluate(X_test, y_test, verbose=0)
print('Train accuracy:', train_score[1])
print('Test accuracy:', test_score[1])


# In[ ]:


model.summary()


# Now that the model is ready, we run the model on Test data.

# In[ ]:


prediction = model.predict_classes(test)
testBkp["Result"] = prediction
testBkp.head()


# Print the score of each team in the group matches

# In[ ]:


def winner(x):
    if x.Result == 1:
        x["Winning_Team"] = x.Team
    else:
        x["Winning_Team"] = x.Opposition
    return x

data_2019_final = testBkp.apply(winner, axis= 1)
results_2019 = data_2019_final.groupby("Winning_Team").size()
results_2019 = results_2019.sort_values(ascending=False)
print(results_2019)


# In[ ]:


p = []
p.append(results_2019.keys()[0])
p.append(results_2019.keys()[1])
p.append(results_2019.keys()[2])
p.append(results_2019.keys()[3])
p[0]


# In[ ]:


df = pd.DataFrame(columns=['Team','Opposition','TeamA Rating','TeamB Rating','Match Country','TeamA Home','TeamB Home','Ground'])


# In[ ]:


# train[train['Team']=='Sri Lanka']
# X1[134:]


# We have got the top 4 teams in the group stages. These 4 teams will proceed to the semi-finals.
# 
# The semi-finals are played according to the following rule.
# 
# * First - Fourth
# * Second - Third

# In[ ]:


df.loc[-1] = [p[0],p[3],p[0],p[3],'England',p[0],p[3],'Manchester']
df.loc[0] = [p[3],p[0],p[3],p[0],'England',p[3],p[0],'Manchester']
df.loc[1] = [p[1],p[2],p[1],p[2],'England',p[1],p[2],'Birmingham']
df.loc[2] = [p[2],p[1],p[2],p[1],'England',p[2],p[1],'Birmingham']
# adding a row
df.index = df.index + 1  # shifting index
df = df.sort_index() 
cleanup = {"TeamA Home":{"England":1, "India":0, "South Africa":0, "New Zealand":0, "Australia":0, "Pakistan":0,"Sri Lanka":0, "Bangladesh":0},
           "TeamB Home":{"England":1, "India":0, "South Africa":0, "New Zealand":0, "Australia":0, "Pakistan":0,"Sri Lanka":0, "Bangladesh":0},
           "TeamA Rating":{"England":1, "India":2, "South Africa":3, "New Zealand":4, "Australia":5, "Pakistan":6,"Sri Lanka":9,"Bangladesh":7},
           "TeamB Rating":{"England":1, "India":2, "South Africa":3, "New Zealand":4, "Australia":5, "Pakistan":6,"Sri Lanka":9,"Bangladesh":7},
}
df = df.replace(cleanup)
df


# Encode all the categorical data to integer values

# In[ ]:


cleanup = {"Team":{"Australia":1,"New Zealand":11, "Pakistan":13, "England":4, "South Africa": 15, "India": 6,"Sri Lanka":16,"Bangladesh":2},
            "Opposition": {"Australia":1,"New Zealand":11, "Pakistan":13, "England":4, "South Africa": 15, "India": 6,"Sri Lanka":16,"Bangladesh":2},
            "Match Country":{"England":3 },
            "Ground":{"Manchester": 68,"Birmingham":11},
         }

df1 = df.replace(cleanup)
df1.head()


# In[ ]:


predictionsemi = model.predict_classes(df1)
df['Result'] = -1
df["Result"] = predictionsemi
df
#df["Result"].head()


# In[ ]:


finalists = []
if df['Result'][0] == 1:
    finalists.append(df['Team'][0])
else:
    finalists.append(df['Opposition'][0])
if df['Result'][2] == 1:
    finalists.append(df['Team'][2])
else:
    finalists.append(df['Opposition'][2])
    


# And the finalists are: 

# In[ ]:


finalists


# In[ ]:


df = pd.DataFrame(columns=['Team','Opposition','TeamA Rating','TeamB Rating','Match Country','TeamA Home','TeamB Home','Ground'])
df   


# In[ ]:


df.loc[-1] = [finalists[0],finalists[1],finalists[0],finalists[1],'England',finalists[0],finalists[1],'Lord\'s']
df.loc[0] = [finalists[1],finalists[0],finalists[1],finalists[0],'England',finalists[1],finalists[0],'Lord\'s']
# adding a row
df.index = df.index + 1  # shifting index
df = df.sort_index()
cleanup = {"TeamA Home":{"England":1, "India":0, "South Africa":0, "New Zealand":0, "Australia":0, "Pakistan":0,"Sri Lanka":0, "Bangladesh":0},
           "TeamB Home":{"England":1, "India":0, "South Africa":0, "New Zealand":0, "Australia":0, "Pakistan":0,"Sri Lanka":0, "Bangladesh":0},
           "TeamA Rating": {"England":1, "India":2, "South Africa":3, "New Zealand":4, "Australia":5, "Pakistan":6,"Sri Lanka":9, "Bangladesh":7},
           "TeamB Rating":{"England":1, "India":2, "South Africa":3, "New Zealand":4, "Australia":5, "Pakistan":6,"Sri Lanka":9, "Bangladesh":7},

}
df = df.replace(cleanup)
df


# In[ ]:


cleanup = {"Team":{"Australia":1,"New Zealand":11, "Pakistan":13, "England":4, "South Africa": 15, "India": 6,"Sri Lanka":16,"Bangladesh":2},
            "Opposition": {"Australia":1,"New Zealand":11, "Pakistan":13, "England":4, "South Africa": 15, "India": 6,"Sri Lanka":16,"Bangladesh":2},
            "Match Country":{"England":3 },
            "Ground":{"Lord\'s": 67},
        }
df1 = df.replace(cleanup)
df1.head()


# In[ ]:


predictionsemi = model.predict_classes(df1)
df['Result'] = -1
df["Result"] = predictionsemi
df

