import pandas as pd
import numpy as np
import os 
import pickle as pkl
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


with open('../input/no_nans.pickle', 'rb') as f:
    df_0 = pkl.load(f) #7824791 observations, 28 columns (27 predictors, 1 response)
    
X = df_0.drop('HasDetections', axis=1)
y = df_0.loc[:, 'HasDetections'].values

X = X.astype('category')
print('dummying')
dummy_x = pd.get_dummies(data=X, drop_first=True, sparse=True)

data_with_response = dummy_x.assign(HasDetections=y)
print(data_with_response.shape)

print('pickling')
with open('dummy_x_y_df.pickle','wb') as f:
    pkl.dump(data_with_response, f)

X = dummy_x

X_train,X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=5)

print(X_train.shape)
print(y_train.shape)


with open('X_train.pickle', 'wb') as f:
    pkl.dump(X_train, f)
with open('X_test.pickle', 'wb') as f:
    pkl.dump(X_test, f)
with open('y_train.pickle', 'wb') as f:
    pkl.dump(y_train, f)
with open('y_test.pickle', 'wb') as f:
    pkl.dump(y_test, f)
