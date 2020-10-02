#!/usr/bin/env python
# coding: utf-8

# # Titanic: Machine Learning From Disaster
# In this notebook I'm going to expand on my previous attempt that used scikit-learn random forests and try to use TensorFlow as the learning framework this time.
# 
# Step 1: Load the modules and see what versions we have installed.

# In[ ]:


import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
print(f'matplotlib: {matplotlib.__version__}')
print(f'tensorflow: {tf.__version__}')
print(f'pandas    : {pd.__version__}')
print(f'numpy     : {np.__version__}')


# ## Format the data
# Next step is to format the data so that we can use it to actually train and test our data.

# In[ ]:


# Load the data
df = pd.read_csv("../input/titanic/train.csv")
df.describe()


# In[ ]:


df.head(10)


# The formatting that we will apply includes the following:
# * **One-hot encode**: 'Sex', 'Embarked'
# * **Remove**: 'Name', 'Ticket', 'Cabin'
# * **Fill *null* values** with the mean of the associated column.

# In[ ]:


from sklearn import preprocessing

def format_feats(in_feats):
    x = in_feats.values #returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    return pd.DataFrame(x_scaled, columns=in_feats.columns)

# Apply some data formatting
def format_data(data):
    # One-hot encode 'Embarked' column
    data = pd.get_dummies(data, columns=['Sex','Embarked'])
    # Drop columns that require additional processing
    data = data.drop(['Name','Ticket','Cabin'], axis=1)
    # Fill null values with the mean of the column
    data.fillna(data.mean(), inplace=True)
    # Return the results
    return data

# This should split the data into our features and our labels
data = format_data(df)
data.describe()


# Let's see what the breakdown of our labels is between our two categories. Ideally it will be 50/50, or really close to it.

# In[ ]:


# Plot histogram of labels
data.hist(column=['Survived'],bins=2);


# Ok, so that's not great, there's about 36% more labels in the first classification. If we leave our data this way, it has the potential to skew the classification algorithm towards the '0' label, since that seems to be the safest bet when the class is uncertain. so we need to remove entries from our training until we get to a more 50/50 split.

# In[ ]:


# Get the number of labels that are in the second class 'Survived==1'
def tally_survivors(dat):
    num_survived = len(dat[dat['Survived']==1])
    num_died     = len(dat[dat['Survived']==0])
    print(f'   survivors    : {num_survived}')
    print(f'   non-survivors: {num_died}')
    return num_died-num_survived

print('BEFORE TRIM:')
diff = tally_survivors(data)

# Reduce the number of non-survivors in the training set
# to match the number of survivors
# =============

# Get the list of indices for non-survivors
indices = data[data['Survived']==0].index
# Get a list of indices to remove and remove them
removed = np.random.choice(np.array(indices), size=diff, replace=False)
train   = data.drop(index=removed)

# Plot the remaining distribution
print('AFTER TRIM:')
new_diff = tally_survivors(train)
train['Survived'].hist(bins=2);


# Now get the features and labels for training.

# In[ ]:


# Features
feats  = train.drop(['Survived'], axis=1)
# Classification labels
labels = train['Survived']


# ## Building the model
# Now we need to build a model that is capable of being trained and generating predictions. For this attempt I will be using TensorFlow.
# 
# Note that we will create a function for generating our model from a list of nodes per layer. This will help us to more easily tune these parameters as we search for the best model.

# In[ ]:


# Generate the model
from tensorflow import nn
from tensorflow.keras import layers, Sequential

# Set Dropout rate
drpout = 0.2

# Create a function for model construction
# This will help for testing different model architectures.
def model_construct(inputs, n=[16], outputs=2,
                    activ=nn.relu):
    # Add the outputs to the list of nodes
    n.append(outputs)
    
    # Input layer
    layer_list = []
    layer_list.append(layers.Dense(units=n[0],
                                   activation=activ,
                                   input_shape=[inputs,]))
    layer_list.append(layers.Dropout(rate=drpout))
    
    # Loop over the hidden layers
    for i in range(len(n)-1):
        layer_list.append(layers.Dense(units=n[i+1], activation=activ))
        layer_list.append(layers.Dropout(rate=drpout))
        
    # Remove the last dropout layer
    layer_list.pop()
    # Change final activation function
    layer_list[-1] = layers.Dense(units=2, activation='softmax')
    
    # Put it all together
    return Sequential(layers=layer_list)


# And for training/testing the model...

# In[ ]:


from tensorflow.keras.callbacks import EarlyStopping

def train_model(model, epochs=5, verbose=False, valsplit=0.2):
    # Define a callback for early stopping
    callbacks = []
    if (valsplit > 0.0):
        early_stop = EarlyStopping(monitor='val_loss', patience=50)
        callbacks.append(early_stop)
        
    # Compile the model with the appropriate loss function and optimizer
    # Make sure to track accuracy
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(0.001),
                  metrics=['accuracy'])
    history = model.fit(feats, labels, 
                        validation_split=valsplit,
                        epochs=epochs, verbose=verbose, callbacks=callbacks)
    
    # ================================================
    # Everything below here is just for visualization
    # ================================================
    
    hist = history.history
    
    # Print the final information
    print(f"   Train Acc.: {hist['accuracy'][-1]:0.2f}%")
    print(f"   Train loss: {hist['loss'][-1]}")
    if (valsplit > 0.0):
        print(f"   Test Acc. : {hist['val_accuracy'][-1]:0.2f}%")
        print(f"   Test loss : {hist['val_loss'][-1]}")
    
    # Now plot the accuracy and loss over time
    plt.subplot(211)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.plot(hist['accuracy'], label='train acc.')
    if (valsplit > 0.0):
        plt.plot(hist['val_accuracy'], label='train acc.')
        plt.legend();
    
    # Plot the loss
    plt.subplot(212)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.plot(history.history['loss'], label='train loss')
    plt.title('Loss')
    if (valsplit > 0.0):
        plt.plot(history.history['val_loss'], label='test loss')
        plt.legend();
    
    return hist['val_accuracy'][-1]


# In[ ]:


# Some tracking stats for the best model
model    = None
accuracy = 0.0


# In[ ]:


def update_best(test_model, best_model, best_accuracy):
    # Summary of model
    print(test_model.summary())
    
    # Train and get model accuracy
    acc = train_model(test_model, epochs=1000)
    
    # Update best model
    if (acc > best_accuracy):
        print("Model is better!")
        best_model = test_model
        best_accuracy = acc
    else:
        print("Model not better :(")
        
    return


# In[ ]:


inputs = len(feats.columns)
print(f'Num inputs: {inputs}')


# In[ ]:


# Give it a try
print("Test 1:")
model1 = model_construct(inputs, n=[256])
update_best(model1, model, accuracy)


# In[ ]:


print("Test 2:")
update_best(model_construct(inputs, n=[256, 64]), model, accuracy)


# In[ ]:


print("Test 3:")
model3 = model_construct(inputs, n=[8], activ='relu')
update_best(model3, model, accuracy)


# Now we keep the model with the highest accuracy, but since they all have pretty much the same accuracy we will just take the simplest model.

# In[ ]:


model = model1
print(model.summary())
accuracy = train_model(model, epochs=1000, valsplit=0.0)


# ## Submit the best result
# Now that we have the best result, we will get the predictions and submit them.

# In[ ]:


# Load and process the testing data
test_df    = pd.read_csv("../input/titanic/test.csv")
test_feats = format_data(test_df)

# Compute the results
results = model.predict(test_feats)
results = [np.argmax(res) for res in results]
print(results[:10])
plt.hist(results, bins=2, range=(-0.5,1.5), density=True)

# Load it all into a dataframe
submission_df = pd.DataFrame({'PassengerId': test_df['PassengerId'], 
                              'Survived'   : results})
submission_df.describe()


# In[ ]:


submission_df.to_csv('submission.csv', index=False)


# In[ ]:




