from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
# Load data ignoring Rownumber, Customer ID and Surname
dataframe = pd.read_csv('../input/Churn_Modelling.csv')
# Transform Geography and Gender to numerical values
le = preprocessing.LabelEncoder()
encoded = dataframe.apply(le.fit_transform)
dataset = encoded.values
# X and Y values
X = dataset[:,3:13]
Y = dataset[:,13]
# Rescale min and max for X
scaler = MinMaxScaler(feature_range=(0, 1))
rescaledX = scaler.fit_transform(X)
# Using Relu activation, and Adam optimizer, 50 epochs
model = Sequential()
model.add(Dense(12, input_dim=10, init='uniform', activation='relu'))
model.add(Dense(10, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='relu'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Train model
history = model.fit(rescaledX, Y, nb_epoch=100, batch_size=50,  verbose=1)
# Print Accuracy
scores = model.evaluate(rescaledX, Y) 
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))