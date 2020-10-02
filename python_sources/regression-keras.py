import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")

train = train.set_index("Id")
test = test.set_index("Id")

test_index = test.index

#Clean the train and test data
combined = pd.concat([train, test], axis=0, sort=False)
Y = combined[combined["SalePrice"].notnull()]["SalePrice"].sort_index().values

#Get the log(y) to minimize values
log_Y = np.log(Y)

numeric_val_list = ["OverallQual", "GrLivArea", "YearBuilt", "MSSubClass", "OverallCond",
                    "GarageCars", "LotArea", "Fireplaces", "LotFrontage", "TotRmsAbvGrd",
                    "KitchenAbvGr", "FullBath"]
categorical_val_list = ["BsmtExposure", "BsmtFinType1", "Neighborhood", "BsmtQual", "MSZoning", "BsmtCond",
                        "Exterior1st", "KitchenQual", "Exterior2nd", "SaleCondition", "HouseStyle",
                        "LotConfig", "GarageFinish", "MasVnrType", "RoofStyle"]
numeric_df = combined[numeric_val_list]
categorical_df = combined[categorical_val_list]

#Cleaning the data
for key in numeric_val_list:
    numeric_df[key] = numeric_df[key].fillna(numeric_df[key].median())
for key in categorical_val_list:
    categorical_df[key] = categorical_df[key].fillna(categorical_df[key].value_counts().median())

categorical_df = pd.get_dummies(categorical_df)

#Split Data to train and test
train_c = categorical_df[categorical_df.index <= 1460] 
test_c = categorical_df[categorical_df.index > 1460]

train_n = numeric_df[numeric_df.index <= 1460]
test_n = numeric_df[numeric_df.index > 1460]

scale = StandardScaler()

train_n = scale.fit_transform(train_n)
test_n = scale.fit_transform(test_n)

train = np.concatenate((train_n, train_c.values), axis=1)
test = np.concatenate((test_n, test_c.values), axis=1)

X_train, X_val, Y_train, Y_val = train_test_split(train, log_Y, test_size=0.2)

#Design the Model
model = Sequential()
model.add(Dense(16, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(1))

optimizer = Adam(lr=0.002, beta_1=0.92, beta_2=0.99, amsgrad=False)
model.compile(optimizer=optimizer,
             loss='mean_squared_logarithmic_error',
             metrics=['mae'])

history=model.fit(X_train, Y_train, validation_data=(X_val, Y_val),
                  epochs=250, batch_size=256, verbose=0)

# summarize history for loss
import matplotlib.pyplot as plt
plt.plot(history.history['mae'])
plt.plot(history.history['val_mae'])
plt.title('model loss')
plt.ylabel('mae')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

print(history.history['val_mae'][-1])
print("Predicting the Test data...")
prediction = model.predict(test)
prediction = np.exp(prediction)
submission = pd.DataFrame()
submission['Id'] = test_index
submission['SalePrice'] = prediction
print("Saving prediction to output...")
submission.to_csv("prediction_regression.csv", index=False)
print("Done.")
print(submission)