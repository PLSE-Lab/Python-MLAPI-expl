#!/usr/bin/env python
# coding: utf-8

# # Boston House Price Predictions
# 
# Rather than applying traditional machine learning techniques on this dataset, as is the norm, in this notebook we'll look at forming a Deep Neural Network (DNN) for regression. This is normally difficult with such a small dataset and a DNN, since the complexity of this type of model tends to significantly overfit the data at hand.
# 
# Within this notebook, I'll exemplify some good practices to counteract overfitting during training, with a particular emphasis on DNNs.

# ---
# 
# ## 1. Import dependencies and dataset

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[ ]:


PATH =  "/kaggle/input/boston-house-prices/"


# In[ ]:


column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 
                'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

housing_df = pd.read_csv(PATH + 'housing.csv', header=None, delimiter=r"\s+", names=column_names)

print("Shape of housing dataset: {0}".format(housing_df.shape))

housing_df.head(5)


# ---
# ## 2. Split our data into training and testing partitions.

# In[ ]:


train_data = housing_df.iloc[:404, :].copy()
test_data = housing_df.iloc[404:, :].copy()

X_train = train_data.iloc[:, :-1].copy()
y_train = train_data.iloc[:, -1:].copy()

X_test = test_data.iloc[:, :-1].copy()
y_test = test_data.iloc[:, -1:].copy()


# In[ ]:


X_train.describe()


# ---
# ## 3. Normalization of our features
# 
# With many features that are heterogeneous, we should definitely consider standardising our data through normalisation. For this we should calculate the mean and standard deviation of the features within the training data, and use this to normalise both the training and test sets.
# 
# In the following code, we'll normalize our data with zero mean and unit standard deviation. We'll obtain the mean and standard deviation using the training partition, which we will then use to normalize both the training and test splits.

# In[ ]:


def feature_normalisation(train_data, test_data):
    """ Normalize our dataframe features with zero mean and unit
        standard deviation """
    
    std_data = train_data.copy()
    
    mean = train_data.mean(axis=0)
    std_dev = train_data.std(axis=0)
    
    # centre data around zero mean and give unit std dev
    std_data -= mean
    std_data /= std_dev
    
    # if test data passed to func, convert test data using train mean / std dev
    test_data -= mean
    test_data /= std_dev
        
    return std_data, test_data


# In[ ]:


X_train, X_test = feature_normalisation(X_train, X_test)


# Prior to producing a neural network model and making subsequent predictions on an evaluated model, lets visualise the relative importance of features using an off the shelf random forrest regressor.

# In[ ]:


ranf = RandomForestRegressor(random_state=1)
ranf.fit(X_train, y_train.values[:, 0])


# In[ ]:


columns = list(X_train.columns)

importances = ranf.feature_importances_
indices = np.argsort(importances)[::-1]
cols_ordered = []

for feat in range(X_train.shape[1]):
    print("{0:<5} {1:<25} {2:.5f}".format(feat + 1, columns[indices[feat]], importances[indices[feat]]))
    cols_ordered.append(columns[indices[feat]])
    
plt.figure(figsize=(6,4))
plt.bar(range(X_train.shape[1]), importances[indices], align='center')
plt.xticks(range(X_train.shape[1]), cols_ordered, rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.tight_layout()
plt.title("Random Forrest Feature Importances")
plt.show()


# It's clear that the only two really important features in our dataset appears to be LSTAT and RM. In comparison to these, the others are almost negigible in terms of their correlation to changing the output house price. However, since we are using a neural network in the following lines of code, we will retain all columns and train our model accordingly, regardless of their importance. If we were using more traditional methods, such as random forests, support vector machine or simple linear regression, we would likely benefit from reduction of some of these less important features.

# ---
# ## 4. Formation of Deep Neural Network for regression
# 
# Forming a deep neural network for regression is relatively simple, especially when armed with high-level libraries like Keras or PyTorch. For regression tasks, we simply need to ensure our final output layer has no activation, unlike in classification tasks where we employ sigmoid or softmax output activations.

# In[ ]:


from keras import models
from keras import layers


# In[ ]:


def nn_model(dropout=False):
    """ Create a basic Deep NN for regression """
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
    if dropout:
        model.add(layers.Dropout(0.5))
    model.add(layers.Dense(64, activation='relu'))
    if dropout:
        model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model


# With such a small dataset, we're best employing K-means cross-validation, which makes more from our data than using just one dedicated partition of samples for the validation set.

# In[ ]:


k = 4

num_val_samples = len(X_train) // k

epochs = 100

scores = []

# prepare validation and training partitions
for i in range(k):
    print('Cross-validation fold number {0}'.format(i))
    val_samples_x = X_train[i * num_val_samples: (i + 1) * num_val_samples]
    val_samples_y = y_train[i * num_val_samples: (i + 1) * num_val_samples]
    
    print("X Val: {0}, y Val: {1}".format(val_samples_x.shape, val_samples_y.shape))
    
    train_samples_x = np.concatenate([X_train[:i * num_val_samples],
                                      X_train[(i + 1) * num_val_samples:]], axis=0)
    
    train_samples_y = np.concatenate([y_train[:i * num_val_samples], 
                                      y_train[(i + 1) * num_val_samples:]], axis=0)
    
    print("X Train: {0}, y Train: {1}".format(train_samples_x.shape, train_samples_y.shape))
    
    # instantiate model and fit training samples, then evaluate on val partition
    model = nn_model()
    model.fit(train_samples_x, train_samples_y, epochs=epochs, batch_size=1, verbose=0)
    val_mse, val_mae = model.evaluate(val_samples_x, val_samples_y, verbose=0)
    scores.append(val_mae)


# In[ ]:


print(np.mean(scores))


# Repeat again but obtain a record of our validation performance at each epoch across all folds.

# In[ ]:


k = 4

num_val_samples = len(X_train) // k

epochs = 100

mae_histories = []

# prepare validation and training partitions
for i in range(k):
    print('Cross-validation fold number {0}'.format(i))
    val_samples_x = X_train[i * num_val_samples: (i + 1) * num_val_samples]
    val_samples_y = y_train[i * num_val_samples: (i + 1) * num_val_samples]
    
    print("X Val: {0}, y Val: {1}".format(val_samples_x.shape, val_samples_y.shape))
    
    train_samples_x = np.concatenate([X_train[:i * num_val_samples],
                                      X_train[(i + 1) * num_val_samples:]], axis=0)
    
    train_samples_y = np.concatenate([y_train[:i * num_val_samples], 
                                      y_train[(i + 1) * num_val_samples:]], axis=0)
    
    print("X Train: {0}, y Train: {1}".format(train_samples_x.shape, train_samples_y.shape))
    
    # instantiate model and fit training samples, then evaluate on val partition
    model = nn_model()
    history = model.fit(train_samples_x, train_samples_y, 
                        epochs=epochs, batch_size=1, 
                        verbose=0, validation_data=(val_samples_x, val_samples_y))
    
    val_mae_hist = history.history['val_mae']
    
    mae_histories.append(val_mae_hist)


# In[ ]:


average_mae_hist = [np.mean([x[i] for x in mae_histories]) for i in range(epochs)]


# In[ ]:


plt.figure(figsize=(10,6))
plt.plot(range(1, len(average_mae_hist) + 1), average_mae_hist)
plt.xlabel("Epochs")
plt.ylabel("Validation MAE")
plt.show()


# In[ ]:


plt.figure(figsize=(10,6))
plt.plot(range(1, len(average_mae_hist) + 1), average_mae_hist)
plt.xlabel("Epochs")
plt.ylabel("Validation MAE")
plt.xlim(0.0, 100.0)
plt.show()


# As we can see, even after a very low number of iterations we begin to over-fit on our data. This is expected with such a small dataset of only 500 samples. To counteract this, we have made use of cross-folds validation. Something that we can apply further to this is regularisation.
# 
# Below we'll apply dropout regularisation in an effort to reduce overfitting.

# In[ ]:


k = 4

num_val_samples = len(X_train) // k

epochs = 100

reg_mae_histories = []

# prepare validation and training partitions
for i in range(k):
    print('Cross-validation fold number {0}'.format(i))
    val_samples_x = X_train[i * num_val_samples: (i + 1) * num_val_samples]
    val_samples_y = y_train[i * num_val_samples: (i + 1) * num_val_samples]
    
    print("X Val: {0}, y Val: {1}".format(val_samples_x.shape, val_samples_y.shape))
    
    train_samples_x = np.concatenate([X_train[:i * num_val_samples],
                                      X_train[(i + 1) * num_val_samples:]], axis=0)
    
    train_samples_y = np.concatenate([y_train[:i * num_val_samples], 
                                      y_train[(i + 1) * num_val_samples:]], axis=0)
    
    print("X Train: {0}, y Train: {1}".format(train_samples_x.shape, train_samples_y.shape))
    
    # instantiate dropout regularised model and fit training samples with val data for eval
    model = nn_model(dropout=True)
    history = model.fit(train_samples_x, train_samples_y, 
                        epochs=epochs, batch_size=1, 
                        verbose=0, validation_data=(val_samples_x, val_samples_y))
    
    val_mae_hist = history.history['val_mae']
    
    reg_mae_histories.append(val_mae_hist)

average_reg_mae_hist = [np.mean([x[i] for x in reg_mae_histories]) for i in range(epochs)]


# In[ ]:


plt.figure(figsize=(10,6))
plt.plot(range(1, len(average_mae_hist) + 1), average_mae_hist, label='Original Model')
plt.plot(range(1, len(average_reg_mae_hist) + 1), average_reg_mae_hist, label='Regularised Model')
plt.xlabel("Epochs")
plt.ylabel("Validation MAE")
plt.xlim(1.0, 100.0)
plt.legend(loc='best')
plt.show()


# Our dropout regularised model is much better in terms of reducing overfitting.

# ---
# ## 5. Formation of our evaluated model into a final model
# 
# #### Finally, lets make a final model with the entire training set, followed by predictions for our test set using the trained model.

# In[ ]:


# produce our deep NN model using dropout regularisation, trained on all training data
final_model = nn_model(dropout=True)
history = final_model.fit(X_train, y_train, epochs=200, batch_size=1, verbose=0)


# In[ ]:


hist_dict = history.history

trg_loss = history.history['loss']
trg_acc = history.history['mae']

epochs = range(1, len(trg_acc) + 1)

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(epochs, trg_loss, label='Training Loss')
ax[1].plot(epochs, trg_acc, label='Training MAE')
ax[0].set_ylabel('Training Loss')
ax[1].set_ylabel('Training MAE')

plt.show()


# In[ ]:


test_preds = final_model.predict(X_test)


# In[ ]:


test_mse, test_mae = final_model.evaluate(X_test, y_test, verbose=0)


# In[ ]:


print("Test set performance: \n- Test MSE: {0} \n- Test MAE: {1}".format(test_mse, test_mae))


# Our final test set performance is not bad - only 3.21 Mean Absolute Error (MAE)! Basically the average amount that our predictions of house prices deviated from the actual values was $3210. When we consider how varying houses can be, coupled with their relatively high prices, this is not a bad average error to have in our set of predictions, especially with the extremely low amount of data-preprocessing and feature engineering conducted in this notebook to obtain this model.
