import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout

dataset = pd.read_csv('../input/PS_20174392719_1491204439457_log.csv')

X = dataset.iloc[:,:-2].values
y = dataset.iloc[:, -2].values

label_encoder_x = LabelEncoder()
X[:, 3] = label_encoder_x.fit_transform(X[:,3])
X[:, 6] = label_encoder_x.fit_transform(X[:,6])

o_h_enc = OneHotEncoder()

o_h_enc.fit(X[:,1].reshape(-1,1))
o_h_enc.categories_
X_2 = X[:,0].shape
X = np.concatenate((X[:,0].reshape(-1,1),o_h_enc.transform(X[:,1].reshape(-1,1)).toarray(), X[:,2:]), axis=1)

st_sc = StandardScaler()
X = st_sc.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2)

model = Sequential()
model.add(Dense(units = 64, input_dim = 13, activation = "relu"))
model.add(Dense(units = 128, activation = "relu"))
model.add(Dense(units = 256, activation = "relu"))
model.add(Dense(units = 512, activation = "relu"))
model.add(Dense(units = 256, activation = "relu"))
model.add(Dense(units = 128, activation = "relu"))
model.add(Dense(units = 64, activation = "relu"))
model.add(Dropout(0.4))
model.add(Dense(units = 32, activation = "relu"))
model.add(Dense(units = 1, activation = "sigmoid"))
model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])
model.summary()

model.fit(X_train, y_train, batch_size = 128, epochs = 7)

score = model.evaluate(X_test, y_test)
print(score)
