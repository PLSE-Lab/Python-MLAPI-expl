#!/usr/bin/env python
# coding: utf-8

# # Importing and Seeding
# Seeding is important if you want to make a reproducible code. Though there will still be some variations, seeding here will minimize the randomness of each rerun of the kernel.

# In[ ]:


import numpy as np
import pandas as pd
import random
import time
import gc

from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

seed = random.randint(10, 10000)
seed = 2546 #From V21 of commit
print("This run's seed is:", seed)

np.random.seed(seed)
random.seed(seed)


# # Model Creation
# The model to be used is a simple regressor model using a Sequential model from Keras. It has three hidden Dense layer of sizes 128, 64 and 16 with activators sigmoid, relu and relu respectively. I also applied Batch Normalization and Dropout to avoid overfitting.
# 
# The loss function to be used is decided to be Mean Square Logarithmic Error or MSLE so that the Y_hat or predicted_value can be minimized. The metrics to be monitored is set to be the Mean Squared Error or MSE of the function. The two are similar but one is on the Logarithmic side while the other is not.

# In[ ]:


def create_model(input_shape):
    model = Sequential()
    model.add(Dense(128, input_dim=input_shape, activation='sigmoid'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(64, input_dim=input_shape, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(16, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(1))
    
    optimizer = Adam(lr=0.005, decay=0.)
    model.compile(optimizer=optimizer,
             loss='msle',
             metrics=['mse'])
    return model


# # Feature Expansion
# 
# To further simplify learning from the data, we would need to do some tweaks to it. One is to fill the N/A parts of the dataset. This can be done mostly by fitting Zero, Mean or Median of the column in its place. For this one, I chose the median for it's versatility in minimizing the amount of extremities.
# 
# To expand the features, I also did a simple boolean columns of form key>N where N is a number between the min and the max of the column.

# In[ ]:


def data_correction_numeric(keys):
    #We will access the global num_df for this
    for key in keys:
        mn, mx = abs(int(num_df[key].min())), abs(int(num_df[key].max()))
        if mx < mn:
            mn, mx = mx, mn
        print("Min:", mn, "Max:", mx)
        try:
            for suf in range(mn, mx, int((mx-mn)/3)):
                num_df[key+'>'+str(suf)] = num_df[key].map(lambda x: x>suf)
                num_df[key+'<'+str(suf)] = num_df[key].map(lambda x: x<suf)
        except:
            print("ERROR for %s" %key)
        x_val = num_df[key].median()
        num_df[key] = num_df[key].fillna(x_val)
        num_df[key] = num_df[key]-x_val

def data_correction_category(df, keys):
    for key in keys:
        x_val = 0#df[key].value_counts().median()
        df[key] = df[key].fillna(x_val)
    return df


# # Feature Engineering
# 
# Here, we will be fitting our data correction functions to the full train and test data.

# In[ ]:


#Read the input data
train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")
print(train.shape, test.shape)
train = train.set_index("Id")
test = test.set_index("Id")

test_index = test.index
#Clean the train and test data
combined = pd.concat([train, test], axis=0, sort=False)
print(combined.columns)

#Free some memory
del train, test

#Get the log(y) to minimize values
Y = combined[combined["SalePrice"].notnull()]["SalePrice"].sort_index().values
log_Y = np.log(Y)

del Y
gc.collect()


# In[ ]:


numeric_val_list = ["OverallQual", "GrLivArea", "YearBuilt", "MSSubClass", "OverallCond",
                    "GarageCars", "LotArea", "Fireplaces", "LotFrontage", "TotRmsAbvGrd",
                    "KitchenAbvGr", "FullBath"]
categorical_val_list = ["BsmtExposure", "BsmtFinType1", "Neighborhood", "BsmtQual", "MSZoning", "BsmtCond",
                        "Exterior1st", "KitchenQual", "Exterior2nd", "SaleCondition", "HouseStyle",
                        "LotConfig", "GarageFinish", "MasVnrType", "RoofStyle"]
num_df = combined[numeric_val_list]
cat_df = combined[categorical_val_list]

#Cleaning the data
data_correction_numeric(numeric_val_list)
cat_df = data_correction_category(cat_df, categorical_val_list)

cat_df = pd.get_dummies(cat_df)


# In[ ]:


num_df.columns


# In[ ]:


cat_df.columns


# # Final Adjustments
# 
# Here, we will be adjusting the values of the train and test data once more by fitting them to a scaler.

# In[ ]:


#Split Data to train and test
train_c = cat_df[cat_df.index <= 1460] 
test_c = cat_df[cat_df.index > 1460]
train_n = num_df[num_df.index <= 1460]
test_n = num_df[num_df.index > 1460]

del num_df, cat_df

scale = StandardScaler()

train_n = scale.fit_transform(train_n)
test_n = scale.fit_transform(test_n)

train = np.concatenate((train_n, train_c.values), axis=1)
test = np.concatenate((test_n, test_c.values), axis=1)

del train_c, train_n, test_c, test_n
gc.collect()


# We will also define a plotter so that we can see the train vs validation learning values.

# In[ ]:


# summarize history for loss
import matplotlib.pyplot as plt
def plotter(history, n):
    plt.plot(history.history['mse'])
    plt.plot(history.history['val_mse'])
    plt.title('MODEL MSE #%i' %n)
    plt.ylabel('MSE')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper right')
    plt.ylim(top=.1, bottom=0.01)
    plt.savefig('history_mse_{}.png'.format(n))
    plt.show()


# # Training
# Now, we fit the training model. I also use some callbacks: ReduceLROnPlateau for slow cooking, and EarlyStopping for, well, early stopping of the training. The patience values are decided after much trial and error, to ensure that there's enough room for adjustment.
# 
# After that, we train and predict the data over a ten-fold repetition of computation. This will minimize the overfitting of the data and will, hopefully, increase the accuracy of the prediction.

# In[ ]:


#Callbacks
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
lrr = ReduceLROnPlateau(monitor = 'val_mse',
                         patience = 200,
                         verbose = 1,
                         factor = 0.75,
                         min_lr = 1e-6)

es = EarlyStopping(monitor='val_loss',
                   mode='min',
                   verbose=1,
                   patience=1000,
                   restore_best_weights=True)

print("Shape of Train:", train.shape)
predictions = []

last_mse = []
folds = 10
for x in range(1, folds+1):
    #Separate train data into train and validation data
    X_train, X_val, Y_train, Y_val = train_test_split(train, log_Y, test_size=0.2, shuffle=True, random_state=seed)
    print("#"*72)
    print("Current RERUN: #%i" %(x))
    #Design the Model
    model = create_model(X_train.shape[1])
    #Start the training
    history=model.fit(X_train, Y_train, validation_data=(X_val, Y_val),
                  epochs=10000, batch_size=128, verbose=0,
                 callbacks=[es, lrr])
    #Predicting
    predict=model.predict(test)
    try:
        predictions = np.concatenate([predictions, predict], axis=1)
    except:
        predictions = predict
    #Show the MSE Plot
    plotter(history, x)
    loss, mse = model.evaluate(X_val, Y_val)
    print("Loss:", loss, "\tMSE:", mse)
    last_mse.append(mse)
    #Clear some Memory
    del X_train, X_val, Y_train, Y_val, model, history, predict, loss, mse
    gc.collect()


# # Prediction
# Finally, we will be saving our prediction. As we did a Ten-Fold prediction, we will be doing a weighted combination based on the MSE evaluation of the validation set for each fold of the training, which was save to the variable `last_mse`. Since the metrics we used is the `error` or the predictions, then, the larger the error, the smaller the effect, thus, I used the equation `(total - x)/((n-1)*total)` so as to reverse the relationship.

# In[ ]:


def ensemble(preds, metrics):
    over = sum(metrics)
    n = len(metrics)
    return [sum((over - metrics[x])*preds[i,x]/((n-1)*over) for x in range(n)) for i in range(len(preds))]


# In[ ]:


print("Predicting the Test data...")
prediction = ensemble(predictions, last_mse)
prediction = np.exp(prediction)
submission = pd.DataFrame()
submission['Id'] = test_index
submission['SalePrice'] = prediction
print("Saving prediction to output...")
submission.to_csv("prediction_regression.csv", index=False)
print("Done.")

print(submission)

x = np.mean(last_mse)
print(x, x**.5)

