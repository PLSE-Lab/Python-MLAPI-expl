import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

training_set = pd.read_csv("../input/train.csv")
test_set = pd.read_csv("../input/test.csv")

from sklearn.preprocessing import StandardScaler

df = pd.concat([training_set,test_set],axis=0,sort='False', ignore_index = True)

df = df[df.columns.difference(['Id'])]

test_ids = test_set["Id"]

df = df.fillna(0)

df = pd.get_dummies(df)

from keras.models import Sequential
from keras.layers import Dense

df_train = df.iloc[:1460,:]
df_test = df.iloc[1460:,:]

X_train = df_train[df_train.columns.difference(['SalePrice'])].values
y_train = df_train[['SalePrice']].values
X_test = df_test[df_test.columns.difference(['SalePrice'])].values

sc_x = StandardScaler()
sc_y = StandardScaler()

X_train = sc_x.fit_transform(X_train)
X_test = sc_x.transform(X_test)
y_train = sc_y.fit_transform(y_train)

model = Sequential()
model.add(Dense(units=512, kernel_initializer="random_uniform",activation="tanh"))
model.add(Dense(units=256, kernel_initializer="random_uniform",activation="tanh"))
model.add(Dense(units=32, kernel_initializer="random_uniform",activation="relu"))
model.add(Dense(units=1, kernel_initializer="random_uniform",activation="selu"))
model.compile(optimizer="adam", loss='mean_squared_logarithmic_error', metrics=['mse'])
model.fit(X_train,y_train, validation_split=0.07, batch_size=32, nb_epoch=1000)

y_pred = model.predict(X_test)

y_pred = sc_y.inverse_transform(y_pred)

y_pred = pd.DataFrame(y_pred)
y_pred["Id"] = test_ids
y_pred = y_pred.rename(columns={0: "SalePrice"})
y_pred = y_pred[["Id","SalePrice"]]
y_pred.to_csv("Submission.csv", index=False)