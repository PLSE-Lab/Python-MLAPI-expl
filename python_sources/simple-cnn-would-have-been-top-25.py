#!/usr/bin/env python
# coding: utf-8

# 
# This is a simple kernel which demonstrates that it would have been **possible to get a top 25 result (private LB 2.40) using a simple CNN without any feature engineering and without utilizing the p4677 information about the test data**. The approach is different from all other ideas I've seen here: Everything is based on the *signal power density*, which is simply the acoustic signal amplitude squared (after subtracting the mean). I am making this kernel public for two reasons:
# * I haven't seen any other kernel using the power density
# * This seems to be the only way to use neural nets directly on the data: the power density is always > 0 and can therefore be averaged over many data points (unlike the original signal, where positive and negative amplitudes would cancel, leading to a loss of information). A 150,000 points window can therefore be efficiently downsampled to several 100 data points.
# * I achieved a (unfortunately late :) ) top 25 result without much effort. The classic story: I didn't like my public LB for this method and didn't choose it... Maybe someone else is interested in improving this result? Using the p4677 info and optimizing the CNN via CV might lead to a (late) top result...
# 
# All that was done to achieve this result is the following:
# 1. split the train data into non-overlapping 150k data point windows, analogous to the test data
# 2. remove the global mean from train/test, square the result, reshape the squared data into (300,500) windows and take the average over each 300 point window, resulting in downsampled 300 point power density windows.
# 3. train a CNN (3 convolution layers, 2 dense layers)
# 4. multiply the predicted ttf by a factor which accounts for the fact that the mean power density in test is ~5% higher than in train (more on that below).
# 
# I probably didn't even use the best possible CNN topology / parameters.
# Hope that this solution is of interest! Thanks to the organizers for this great and frustrating :) competition, and congratulation to the winners!
# 

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from keras.regularizers import l1


# Read the train data and prepare train_X and train_y:

# In[ ]:


df_train = pd.read_csv('../input/train.csv', dtype = {'acoustic_data': np.int16, 'time_to_failure': np.float32} ) # float32 is enough :)

train_X = np.resize(df_train['acoustic_data'].values, (len(df_train['acoustic_data']) // 150000, 150000)).astype(np.float32) # rearrange into 150k windows
train_X = (train_X - np.mean(train_X)) ** 2 # calculate power density
train_X = np.mean(train_X.reshape(-1,300,500), axis=2) # downsample 500x

train_y = np.resize(df_train['time_to_failure'].values, (len(df_train['time_to_failure']) // 150000, 150000))[:,-1] # train_y is ttf on right window edge

del df_train # free some memory

print(train_X.shape, train_y.shape)


# Read the submission file and the test data and prepare test_X:

# In[ ]:


df_subm = pd.read_csv('../input/sample_submission.csv')

test_X = []

for fname in df_subm['seg_id'].values:
    test_X.append(pd.read_csv('../input/test/' + fname + '.csv').acoustic_data.values.astype(np.int16))
test_X = np.array(test_X).astype(np.float32)

test_X = (test_X - np.mean(test_X)) ** 2  # calculate power density
test_X = np.mean(test_X.reshape(-1,300,500), axis=2) # downsample 500x

print(test_X.shape)


# This is what the individual windows look like: some examples from train(left) and test(right):

# In[ ]:


fig, axes = plt.subplots(1,2)
fig.set_size_inches(16,5)
axes[0].grid(True)
axes[1].grid(True)

axes[0].plot(train_X[28]);
axes[0].plot(train_X[38]);
axes[1].plot(test_X[28]);
axes[1].plot(test_X[38]);


# Now something important: as we will see below, the model (like all other models in this competition, due to the underlying physics) is actually unable to truly predict ttf for individual quake cycles. For ~ the first half of the cycle, the acoustic data is always the same in every cycle. There is no information in the data which tells whether there will be a minor quake in 3 sec or a major one in 5 sec. So the model *has* to predict a kind of average ttf over all cycles (see plot of ttf below).
# 
# Now the test data has a higher mean power density than the train data. This means that the cycles in test will on average be longer. It is therefore reasonable to multiply the predicted ttfs by this ratio, which is ~1.05:

# In[ ]:


ratio_mean_train_test = np.mean(test_X) / np.mean(train_X)
print('Ratio of mean power in train/test : ', np.mean(test_X) / np.mean(train_X))


# Now prepare the X data for the CNN:

# In[ ]:


def prepare_X_for_cnn(data):
    data = np.log10(data) # take log10 to handle the huge peaks
    data -= np.mean(data) # remove mean
    data /= np.std(data) # set std to 1.0
    data = np.expand_dims(data, axis=-1) # reshaping for CNN/RNN
    return data

train_X = prepare_X_for_cnn(train_X)
test_X  = prepare_X_for_cnn(test_X)


# Another look at the same windows as above after the CNN preparation. I've added another window to train which contains a major quake. The log10 scales everything such that a CNN can handle it:

# In[ ]:


fig, axes = plt.subplots(1,2)
fig.set_size_inches(16,5)
axes[0].grid(True)
axes[1].grid(True)

axes[0].plot(train_X[28]);
axes[0].plot(train_X[38]);
axes[0].plot(train_X[29]);
axes[1].plot(test_X[28]);
axes[1].plot(test_X[38]);


# Now the CNN: The parameters have been set according to the results of a quick RandomGridCV. I'm sure that there are better topologies:

# In[ ]:


def model_cnn():
    model = Sequential([
        Conv1D(filters=16, kernel_size=3, activation='relu', input_shape = train_X.shape[1:]),
        MaxPooling1D(2),
        Conv1D(filters=128, kernel_size=3, activation='relu'),
        MaxPooling1D(2),
        Conv1D(filters=16, kernel_size=3, activation='relu'),
        MaxPooling1D(2),
        Flatten(),
        Dropout(0.1),
        Dense(16, activation='relu', kernel_regularizer=l1(0.01)),
        Dense(16, activation='relu', kernel_regularizer=l1(0.01)),
        Dense(1, activation='linear') # regression
    ])
    model.compile(
        loss='mse',
        optimizer='adam',
        metrics=['mae']
    )
    return model

model = model_cnn()
model.summary()
model.save_weights('/tmp/model_weights_init.h5')


# Same CV approach here: 16 epochs, average of 64 runs as the net isn't really stable after 16 epochs. batch_size etc. could still be tuned...

# In[ ]:


test_y_pred = []
num_iter = 64

for i in range(num_iter):
    model.load_weights('/tmp/model_weights_init.h5')
    model.fit(train_X, train_y, epochs=16,  verbose=0)
    test_y_pred.append(model.predict(test_X))
    
test_y_pred = np.array(test_y_pred).reshape(num_iter,-1)


# Average the ttfs: 

# In[ ]:


test_y_pred_avg = np.mean(test_y_pred, axis=0)
test_y_pred_avg


# This are the predicted train ttfs. Note that the predictions are an average over all cycles. Training the CNN for more epochs would lead to overfitting, as it would simply memorize all train data points:

# In[ ]:


fig, axes = plt.subplots(1,1)
fig.set_size_inches(16,5)
axes.grid(True)

axes.plot(model.predict(train_X));
axes.plot(train_y);


# Multiply by the corrective factor (see above) and submit:

# In[ ]:


df_subm['time_to_failure'] = test_y_pred_avg * ratio_mean_train_test
df_subm.to_csv('submission.csv', index=False)


# That's all. Hope this was helpful, maybe even for the organizers :). Remember: the CNN didn't get any info on mean, std, etc. directly but learned all that *from the shape of the peaks*.
