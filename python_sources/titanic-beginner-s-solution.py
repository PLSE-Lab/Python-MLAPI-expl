#!/usr/bin/env python
# coding: utf-8

# # Importing necessary libraries

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# # Importing the TRAINING DATA and also finding the columns with Nan values in our dataframe i.e. x_train ,, also droping the unnecassary columns = ["PassengerId" , "Survived" , "Name" , "Ticket" , "Cabin"]

# In[ ]:


x_train = pd.read_csv('../input/titanic/train.csv')

null_columns=x_train.columns[x_train.isnull().any()]
print("NULL COLUMNS OF YOUR DATAFRAME : " , null_columns , '\n\n')


y_train = x_train.iloc[: , 1].values
print(y_train)


x_train = x_train.drop(["PassengerId" , "Survived" , "Name" , "Ticket" , "Cabin"] , axis = 1)
print(x_train)


# # Preprocessing of "Pclass" Column :
# ### Here the data is categorical but label encoding is not required  because data is already in numerical form.
# ### 1. One Hot Encoding as no category will get mathematical advantage
# 

# In[ ]:


from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
x_train = np.array(ct.fit_transform(x_train))

print("Column Number = 0 : " , x_train[: , 0])
print("Column Number = 1 : " , x_train[: , 1])
print("Column Number = 2 : " , x_train[: , 2])


# # Preprocessing of "Sex" column :
# ### 1. Filling the Nan values using IMPUTER using mode strategy
# ### 2. Label Encoding as it has Categorical data
# ### 3. One Hot Encoding after Label Encoding as no category will get mathematical advantage
# 
# ---
# 
# 

# In[ ]:



# FILLING MISSING VALUES USING MODE STRATEGY
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
x_train[: , 3] = imputer.fit_transform(x_train[: , 3].reshape(- 1 , 1)).reshape(-1)


# LABEL ENCODING
from sklearn.preprocessing import LabelEncoder , OneHotEncoder
label_encoder_x = LabelEncoder()
x_train[: , 3] = label_encoder_x.fit_transform(x_train[: , 3])

# ONE HOT ENCODING
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer(transformers = [ ('encoder' , OneHotEncoder() , [3]) ] , remainder = 'passthrough')
x_train = np.array(  ct.fit_transform(x_train)  )


print("Column Number = 3 : " , x_train[: , 3])
print("Column Number = 4 : " , x_train[: , 4])


# # Preprocessing of "Age" Column :
# ### 1. Filling missing values using mean strategy of imputer

# In[ ]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan , strategy = 'mean')
x_train[: , 5] = imputer.fit_transform(x_train[: , 5].reshape(-1 , 1)).reshape(-1)

print(x_train[: , 5])



# # Preprocessing of "Embarked" Column :
# ### 1. Filling missing values using MODE strategy of IMPUTER
# ### 2. LABEL ENCODING 
# ### 3. ONE HOT ENODING 

# In[ ]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan , strategy = 'most_frequent')
x_train[: , -1] = imputer.fit_transform(x_train[: , -1].reshape(-1 , 1)).reshape(891)

from sklearn.preprocessing import LabelEncoder , OneHotEncoder
label_encoder_x = LabelEncoder()
x_train[: , -1] = label_encoder_x.fit_transform(x_train[: , -1])



from sklearn.compose import ColumnTransformer
ct = ColumnTransformer(transformers = [ ('encoder' , OneHotEncoder() , [-1] ) ] , remainder = 'passthrough')
x_train = np.array(ct.fit_transform(x_train))



# # Applying Feature Scaling to x_train :

# In[ ]:


from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)


# # Model Training :

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 15, metric = 'minkowski', p = 2)
classifier.fit(x_train, y_train)


# # Importing the TEST dataset , checking for Nan value Columns , making passengerId column form test file and removing unnecessary columns = ["PassengerId"  , "Name" , "Ticket" , "Cabin"]:

# In[ ]:


x_test = pd.read_csv('../input/titanic/test.csv')

# PassengerId columns
passenger_id_col = np.array(x_test['PassengerId']).reshape(-1 , 1)




null_columns=x_test.columns[x_test.isnull().any()]
print("NULL COLUMNS OF YOUR DATAFRAME : " , null_columns , '\n\n')

x_test = x_test.drop(["PassengerId" , "Name" , "Ticket" , "Cabin"] , axis = 1)


# PassengerId columns



print(type(x_test))


# # Preprocessing of "Pclass" Column :
# ### Here the data is categorical but label encoding is not required  because data is already in numerical form.
# ### 1. One Hot Encoding as no category will get mathematical advantage

# In[ ]:


from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


ct = ColumnTransformer(transformers = [ ('encoder' , OneHotEncoder() , [0]) ] , remainder = "passthrough")
x_test = np.array(ct.fit_transform(x_test))
print(x_test[: , 0])


# # Preprocessing of "Sex" column :
# ### 1. Label Encoding as it has Categorical data
# ### 2. One Hot Encoding after Label Encoding as no category will get mathematical advantage

# In[ ]:


from sklearn.preprocessing import LabelEncoder , OneHotEncoder
label_encoder_x = LabelEncoder()
x_test[: , 3] = label_encoder_x.fit_transform(x_test[: , 3])


from sklearn.compose import ColumnTransformer
ct = ColumnTransformer(transformers = [ ('encoder' , OneHotEncoder() , [3]) ] , remainder = 'passthrough')
x_test = np.array(ct.fit_transform(x_test))

print(x_test[: , 3])


# # Preprocessing of "Age" Column :
# ### 1. Filling missing values using mean strategy of imputer

# In[ ]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer()
x_test[: , 5] = imputer.fit_transform(x_test[: , 5].reshape(-1 , 1)).reshape(-1)


# # Preprocessing of "Fare" Column :
# ### 1. Filling the missing values using mean strategy of imputer
# 

# In[ ]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan , strategy = 'mean')
x_test[: , -2] = imputer.fit_transform(x_test[: , -2].reshape(-1 , 1)).reshape(-1)

print(x_test[ : , -2])


# # Preprocessing of "Embarked" Column :
# ### 1. LABEL ENCODING 
# ### 2. ONE HOT ENODING

# In[ ]:


from sklearn.preprocessing import LabelEncoder , OneHotEncoder

label_encoder_x = LabelEncoder()
x_test[: , -1] = label_encoder_x.fit_transform(x_test[: , -1])

from sklearn.compose import ColumnTransformer
ct = ColumnTransformer(transformers = [ ('encoder' , OneHotEncoder() , [-1]) ] , remainder = 'passthrough')
x_test = np.array(ct.fit_transform(x_test))



# # Applying Feature Scaling to x_test :

# In[ ]:


from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_test = sc_x.fit_transform(x_test)


# # Predicting the test set values

# In[ ]:


y_pred = classifier.predict(x_test)
y_pred = y_pred.reshape(-1 , 1).astype(int)
print(y_pred)

final = np.concatenate( (passenger_id_col,y_pred), axis=1)
print(final)


# In[ ]:


print(final)


# In[ ]:


print(final.shape)


# In[ ]:


print(type(final))


# In[ ]:


data_to_be_submitted = pd.DataFrame(final , columns=['PassengerId' , 'Survived'])
print(data_to_be_submitted)


# In[ ]:


data_to_be_submitted = data_to_be_submitted.to_csv('data_to_be_submitted_csv_file.csv' , index = False)


# In[ ]:




