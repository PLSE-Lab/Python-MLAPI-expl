import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense

dataset = pd.read_csv("../input/Admission_Predict.csv")

#Descision: if Chance of Admission is >= 50%, chance of admission = 1; else chance of admission = 0

for index, value in enumerate(dataset.iloc[:,-1]):
    if value >= 0.5:
        dataset.iloc[index,-1] = 1
    else:
        dataset.iloc[index,-1] = 0

dataset.head()

X = dataset.iloc[:,1:-1]
y = dataset.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2)

st_sc = StandardScaler()

X_train = st_sc.fit_transform(X_train)
X_test = st_sc.transform(X_test)

model = Sequential()
model.add(Dense(units = 32, input_dim = 7, kernel_initializer = "random_uniform", activation = "relu"))
model.add(Dense(units = 64, kernel_initializer = "random_uniform", activation = "relu"))
model.add(Dense(units = 128, kernel_initializer = "random_uniform", activation = "relu"))
model.add(Dense(units = 256, kernel_initializer = "random_uniform", activation = "relu"))
model.add(Dense(units = 128, kernel_initializer = "random_uniform", activation = "relu"))
model.add(Dense(units = 64, kernel_initializer = "random_uniform", activation = "relu"))
model.add(Dense(units = 32, kernel_initializer = "random_uniform", activation = "relu"))
model.add(Dense(units = 16, kernel_initializer = "random_uniform", activation = "relu"))
model.add(Dense(units = 1, kernel_initializer = 'random_uniform', activation = "sigmoid"))
model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])

model.summary()

model.fit(X_train, y_train, batch_size = 64, epochs = 1000, verbose = 0)

score = model.evaluate(X_test, y_test)
print(score)