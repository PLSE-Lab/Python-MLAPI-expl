#!/usr/bin/env python
# coding: utf-8

# This kernel is a short and simple example of a neural network that achieves top 10% on "Titanic: Machine Learning from Disaster" challenge. This kernel does not present any statistical insights about the dataset and the viewer should already be familiar with the challenge. 
# 
# We load the dependencies.

# In[ ]:


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from keras.layers import Dense, Dropout, BatchNormalization
from keras.models import Sequential


# We load the train and test data using pandas. 

# In[ ]:


df_train = pd.read_csv("../input/titanic/train.csv")
df_test = pd.read_csv("../input/titanic/test.csv")


# We create a numerial and categorial transformer using the sklearn's pipeline. The numerical transformer fills the missing values with the mean and then standardizes the data. The categorical transformer fills the missing values with the most frequent and transforms the categories in one-hot vectors. 

# In[ ]:


numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent',)),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])


# We need to reate a custom transformer for the 'Name' column. It must replace the title of a person with a corresponding one-hot vector.

# In[ ]:


class TitleSelector(BaseEstimator, TransformerMixin):
    def __init__( self):
        self.dict_title = {
            "Capt":       0,
            "Col":        0,
            "Major":      0,
            "Jonkheer":   1,
            "Don":        1,
            "Sir" :       1,
            "Dr":         0,
            "Rev":        0,
            "the Countess":1,
            "Dona":       1,
            "Mme":        2,
            "Mlle":       3,
            "Ms":         2,
            "Mr" :        4,
            "Mrs" :       2,
            "Miss" :      3,
            "Master" :    5,
            "Lady" :      1
        }
   
    def fit(self, X, y=None):
        return self 
    
    def transform( self, X, y=None):
        for i, name in enumerate(X["Name"]):
            for title in self.dict_title.keys():
                if title in name:
                    X["Name"][i] = self.dict_title[title]
                    break
        
            assert X["Name"][i] in self.dict_title.values()
        
        return X
    
name_transformer = Pipeline(steps=[
    ('name', TitleSelector()),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])


# Now we create the ColumnTransformer object to fit and transform the data. 

# In[ ]:


num_cols = ["Age", "Fare", ]
cat_cols = ["Pclass", "Sex", "SibSp", "Parch", "Ticket", "Cabin", "Embarked"]
cols = num_cols + cat_cols + ["Name"]


preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_transformer, num_cols),
    ('name', name_transformer, ["Name"]),
    ('cat', categorical_transformer, cat_cols),
])

X_train = preprocessor.fit_transform(df_train[cols])
y_train = df_train["Survived"].values


# We create a neural network with two hidden layers. We use a very high dropout to strongly regularize the model. Also, batch normalization is used to stabilize the training.

# In[ ]:


model = Sequential()
model.add(Dense(32, input_dim=858, activation='relu'))
model.add(Dropout(0.9))
model.add(Dense(32, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.9))
model.add(Dense(1, activation='sigmoid'))


# Now its time to train! We use Adam optimizer and Binary Cross-entropy as the loss function. It should not take more than a few minutes on the CPU.

# In[ ]:


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=1000, batch_size=8)


# And voila! We trained the model! Now its time to save the predictions.

# In[ ]:


X_test = preprocessor.transform(df_test[cols])
y_pred = model.predict_classes(X_test)

df_pred = pd.DataFrame(df_test["PassengerId"])
df_pred["Survived"] = y_pred
df_pred.to_csv("submission.csv", index=False)

