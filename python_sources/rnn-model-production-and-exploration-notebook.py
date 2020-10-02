#!/usr/bin/env python
# coding: utf-8

# # RNN Model Production and Exploration Notebook

# In[ ]:


# import modules
import kagglegym
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, LSTM
from keras.regularizers import l2
from keras.optimizers import SGD
from keras.preprocessing.sequence import pad_sequences
from sklearn import preprocessing as pp
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from time import time

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# Start environment
env = kagglegym.make()
observation = env.reset()
train = observation.train


# Lets print our dataset head to see how it looks like:

# In[ ]:


observation.train.head()


# In[ ]:


# Data preprocessing

# https://www.kaggle.com/bguberfain/two-sigma-financial-modeling/univariate-model-with-clip/run/482189/code
# Clipped target value range to use
low_y_cut = -0.086093
high_y_cut = 0.093497

y_is_above_cut = (train.y > high_y_cut)
y_is_below_cut = (train.y < low_y_cut)
y_is_within_cut = (~y_is_above_cut & ~y_is_below_cut)

# Select the features to use
excl = ['id', 'sample', 'y', 'timestamp']
#feature_vars = [c for c in train.columns if c not in excl]
features_to_use = ['technical_30', 'technical_20', 'technical_19', 'fundamental_11']
target_var = ['y']

features = train.loc[y_is_within_cut, features_to_use]
X_train = features.values

targets = train.loc[y_is_within_cut, target_var]
y_train = targets.values

im = pp.Imputer(strategy='median')
X_train = im.fit_transform(X_train)
X_scaler = pp.RobustScaler()
X_train = X_scaler.fit_transform(X_train)
y_scaler = pp.RobustScaler()
y_train = y_scaler.fit_transform(y_train.reshape([-1,1]))

X_train = pd.DataFrame(X_train, columns=features_to_use)
y_train = pd.DataFrame(y_train, columns=target_var)
preprocess_dict = {'fillna': im, 'X_scaler': X_scaler, 'y_scaler': y_scaler}

del y_is_above_cut, y_is_below_cut, excl, targets, features


# Lets take a peek in our dataset head again.

# In[ ]:


X_train.head()


# Right! Now we have scaled values and without NaN values. Better this way!
# Now we can start to build our models. This time we gonna try some deep neural network arquitetures and see how it performs. Let's get started!

# In[ ]:


def dnn(shape,timesteps,l2_coef,drop_coef):
    model = Sequential()
    model.add(LSTM(shape[1], input_shape=(timesteps, shape[0])))
    model.add(Dense(shape[2], input_dim=shape[0], init='he_uniform', W_regularizer=l2(l2_coef)))
    model.add(Activation('relu'))
    model.add(Dropout(drop_coef))
    model.add(Dense(shape[3], init='he_uniform', W_regularizer=l2(l2_coef)))

    optm = SGD(lr=0.01, momentum=0.9, decay=0.0, nesterov=True)
    model.compile(loss='mse',
                  optimizer=optm,
                  metrics=None,
                  verbose=2)
    return model


# In[ ]:


print("Training model")
t0 = time()
timesteps = 1
X_train_ts = pad_sequences(np.reshape(X_train.values, (X_train.values.shape[0], timesteps, X_train.values.shape[1])))
print(X_train_ts.shape)
model1 = dnn(shape=[4,8,64,1],timesteps=timesteps,l2_coef=0.0001,drop_coef=0.7)
model1.fit(X_train_ts, y_train.values,
          nb_epoch=33,
          batch_size=X_train.values.shape[0],
          verbose=2
          );
print("Done! Training time:", time() - t0)


# In[ ]:


print("Evaluating model on training set")
t0 = time()
m1_loss = model1.evaluate(X_train_ts, y_train.values, batch_size=X_train.values.shape[0], verbose=0)
print("Done! Eval time:",time() - t0)
print("Mean squared error for train dataset:",m1_loss)


# In[ ]:


print("Predicting target on training dataset")
t0 = time()
m1_preds = model1.predict(X_train_ts, batch_size=X_train.values.shape[0], verbose=0)
score = r2_score(y_train, m1_preds)
print("Done! Prediction time:",time() - t0)
print("R2 score for train dataset",score)


# In[ ]:


# Predict-step-predict routine ################################################################################
def gen_preds(model, preprocess_dict, features_to_use, print_info=True):
    env = kagglegym.make()
    # We get our initial observation by calling "reset"
    observation = env.reset()

    im = preprocess_dict['fillna']
    X_scaler = preprocess_dict['X_scaler']
    y_scaler = preprocess_dict['y_scaler']
    
    reward = 0.0
    reward_log = []
    timestamps_log = []
    pos_count = 0
    neg_count = 0

    total_pos = []
    total_neg = []

    print("Predicting")
    t0= time()
    while True:
    #    observation.features.fillna(mean_values, inplace=True)

        # Predict with model
        features_dnn = im.transform(observation.features.loc[:,features_to_use].values)
        features_dnn = X_scaler.transform(features_dnn)
        
        features_dnn_ts = pad_sequences(np.reshape(features_dnn,
                                                   (features_dnn.shape[0], timesteps, features_dnn.shape[1])))

        y_dnn = model.predict(features_dnn_ts,batch_size=features_dnn.shape[0],
                              verbose=0).clip(low_y_cut, high_y_cut)

        # Fill target df with predictions 
        observation.target.y = y_scaler.inverse_transform(y_dnn)

        observation.target.fillna(0, inplace=True)
        target = observation.target
        timestamp = observation.features["timestamp"][0]
        
        observation, reward, done, info = env.step(target)

        timestamps_log.append(timestamp)
        reward_log.append(reward)

        if (reward < 0):
            neg_count += 1
        else:
            pos_count += 1

        total_pos.append(pos_count)
        total_neg.append(neg_count)
        
        if timestamp % 100 == 0:
            if print_info:
                print("Timestamp #{}".format(timestamp))
                print("Step reward:", reward)
                print("Mean reward:", np.mean(reward_log[-timestamp:]))
                print("Positive rewards count: {0}, Negative rewards count: {1}".format(pos_count, neg_count))
                print("Positive reward %:", pos_count / (pos_count + neg_count) * 100)

            pos_count = 0
            neg_count = 0

        if done:
            break
    print("Done: %.1fs" % (time() - t0))
    print("Total reward sum:", np.sum(reward_log))
    print("Final reward mean:", np.mean(reward_log))
    print("Total positive rewards count: {0}, Total negative rewards count: {1}".format(np.sum(total_pos),
                                                                                        np.sum(total_neg)))
    print("Final positive reward %:", np.sum(total_pos) / (np.sum(total_pos) + np.sum(total_neg)) * 100)
    print(info)


# In[ ]:


gen_preds(model1, preprocess_dict, features_to_use)


# ### Analysing Training Results

# In[ ]:


market_df = observation.train[['timestamp', 'y']].groupby('timestamp').agg([np.mean, np.std]).reset_index()
y_mean = np.array(market_df['y']['mean'])
t = market_df['timestamp']

print("Predicting target on training dataset")
t0 = time()
m1_preds = model1.predict(X_train_ts, batch_size=X_train_ts.shape[0], verbose=0)
score = r2_score(y_train, m1_preds)
print("Done! Prediction time:",time() - t0)
print("R2 score for train dataset",score)
cum_ret = np.log(1+y_mean).cumsum()
pred_ret = pd.DataFrame(np.vstack((observation.train.timestamp.loc[y_is_within_cut], m1_preds[:,0])).T,
                        columns=['timestamp','y']).groupby('timestamp').agg([np.mean, np.std]).reset_index()
cum_pred1 = np.log(1+pred_ret['y']['mean']).cumsum()

fig, ax = plt.subplots(figsize=(12,7))
ax.set_xlabel("Timestamp");
ax.set_title("Cumulative target signal and predictions over time");
sns.tsplot(cum_ret,t,ax=ax,color='b');
sns.tsplot(cum_pred1,t,ax=ax,color='r');
ax.set_ylabel('Target / Prediction');

fig, ax = plt.subplots(figsize=(12,7))
ax.set_title("Target Variable Distribution. (True vs Prediction)");
#plt.ylim([0, 100000])
sns.distplot(observation.train.y ,ax=ax, color='b', kde=False, bins=100);
sns.distplot(m1_preds ,ax=ax, color='r', kde=False, bins=100);
ax.set_ylabel('Target / Prediction');

