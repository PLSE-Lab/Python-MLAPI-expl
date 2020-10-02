#!/usr/bin/env python
# coding: utf-8

# # Deep Learning Grid Search
# 
# In this project, we will learn how to use the scikit-learn grid search capability.
# 
# We are going to learn the following topics:
# 
# * How to use Keras models in scikit-learn.
# * How to use grid search in scikit-learn.
# * How to tune batch size and training epochs.
# * How to tune learning rate
# * How to tune network weight initialization.
# * How to tune activation functions.
# * How to tune dropout regularization.
# * How to tune the number of neurons in the hidden layer.

# In[ ]:


import sys
import pandas
import numpy
import sklearn
import keras

print('Python: {}'.format(sys.version))
print('Pandas: {}'.format(pandas.__version__))
print('Numpy: {}'.format(numpy.__version__))
print('Sklearn: {}'.format(sklearn.__version__))
print('Keras: {}'.format(keras.__version__))


# In[ ]:


import pandas as pd
import numpy as np

names = ['n_pregnant', 'glucose_concentration', 'blood_pressuer (mm Hg)', 'skin_thickness (mm)', 'serum_insulin (mu U/ml)',
        'BMI', 'pedigree_function', 'age', 'class']
#df = pd.read_csv('../input/diabetes.csv', names = names)
df = pd.read_csv('../input/diabetes.csv')
df.head()


# In[ ]:


# Describe the dataset
df.describe()


# In[ ]:


df[df['Glucose'] == 0]


# In[ ]:


# Preprocess the data, mark zero values as NaN and drop
columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

for col in columns:
    df[col].replace(0, np.NaN, inplace=True)
    
df.describe()


# In[ ]:


# Drop rows with missing values
df.dropna(inplace=True)

# summarize the number of rows and columns in df
df.describe()


# In[ ]:


# Convert dataframe to numpy array
dataset = df.values
print(dataset.shape)


# In[ ]:


# split into input (X) and an output (Y)
X = dataset[:,0:8]
Y = dataset[:, 8].astype(int)


# In[ ]:


print(X.shape)
print(Y.shape)
print(Y[:5])


# In[ ]:


# Normalize the data using sklearn StandardScaler
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().fit(X)


# In[ ]:


print(scaler)


# In[ ]:


# Transform and display the training data
X_standardized = scaler.transform(X)

data = pd.DataFrame(X_standardized)
data.describe()


# In[ ]:


# import necessary sklearn and keras packages
from sklearn.model_selection import GridSearchCV, KFold
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import Adam

# Do a grid search for the optimal batch size and number of epochs
# import necessary packages
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint


# In[ ]:



# Define a random seed
seed = 6
np.random.seed(seed)

# Start defining the model
def create_model():
    # create model
    model = Sequential()
    model.add(Dense(8, input_dim = 8, kernel_initializer='normal', activation='relu'))
    model.add(Dense(16, input_dim = 8, kernel_initializer='normal', activation='relu'))
    model.add(Dense(8, input_dim = 16, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # compile the model
    adam = Adam(lr = 0.01)
    model.compile(loss = 'binary_crossentropy', optimizer = adam, metrics = ['accuracy'])
    return model

# create the model
model = KerasClassifier(build_fn = create_model, verbose = 0)

# define the grid search parameters
batch_size = [16, 32, 64,128]
epochs = [2, 5, 10]

# make a dictionary of the grid search parameters
param_grid = dict(batch_size=batch_size, epochs=epochs)

# build and fit the GridSearchCV
grid = GridSearchCV(estimator = model, param_grid = param_grid, cv = KFold(random_state=seed), verbose = 10)
grid_results = grid.fit(X_standardized, Y)

# summarize the results
print("Best: {0}, using {1}".format(grid_results.best_score_, grid_results.best_params_))
means = grid_results.cv_results_['mean_test_score']
stds = grid_results.cv_results_['std_test_score']
params = grid_results.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print('{0} ({1}) with: {2}'.format(mean, stdev, param))


# In[ ]:


best_batch_size = 64
best_epochs = 10 # 100


# In[ ]:



# Do a grid search for learning rate and dropout rate
# import necessary packages
from keras.layers import Dropout

# Define a random seed
seed = 6
np.random.seed(seed)

# Start defining the model
def create_model(learn_rate, dropout_rate):
    # create model
    model = Sequential()
    model.add(Dense(8, input_dim = 8, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(16, input_dim = 8, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(8, input_dim = 16, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='sigmoid'))

    # compile the model
    adam = Adam(lr = learn_rate)
    model.compile(loss = 'binary_crossentropy', optimizer = adam, metrics = ['accuracy'])
    return model

# create the model
model = KerasClassifier(build_fn = create_model, epochs = best_epochs, batch_size = best_batch_size, verbose = 0)

# define the grid search parameters
learn_rate = [0.001, 0.01, 0.1]
dropout_rate = [0.0, 0.2, 0.4,0.6]

# make a dictionary of the grid search parameters
param_grid = dict(learn_rate=learn_rate, dropout_rate=dropout_rate)

# build and fit the GridSearchCV
grid = GridSearchCV(estimator = model, param_grid = param_grid, cv = KFold(random_state=seed), verbose = 10)
grid_results = grid.fit(X_standardized, Y)

# summarize the results
print("Best: {0}, using {1}".format(grid_results.best_score_, grid_results.best_params_))
means = grid_results.cv_results_['mean_test_score']
stds = grid_results.cv_results_['std_test_score']
params = grid_results.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print('{0} ({1}) with: {2}'.format(mean, stdev, param))


# In[ ]:


best_dropout_rate = 0.0
best_learn_rate = 0.01


# In[ ]:



# Do a grid search to optimize kernel initialization and activation functions
# import necessary packages

# Define a random seed
seed = 6
np.random.seed(seed)

# Start defining the model
def create_model(activation, init):
    # create model
    model = Sequential()
    model.add(Dense(8, input_dim = 8, kernel_initializer= init, activation= activation))
    model.add(Dense(16, input_dim = 8, kernel_initializer= init, activation= activation))
    model.add(Dense(8, input_dim = 16, kernel_initializer= init, activation= activation))
    model.add(Dense(1, activation='sigmoid'))

    # compile the model
    adam = Adam(lr = best_learn_rate)
    model.compile(loss = 'binary_crossentropy', optimizer = adam, metrics = ['accuracy'])
    return model

# create the model
model = KerasClassifier(build_fn = create_model, epochs = best_epochs, batch_size = best_batch_size, verbose = 0)

# define the grid search parameters
activation = ['softmax', 'relu', 'tanh', 'linear']
init = ['uniform', 'normal', 'zero']

# make a dictionary of the grid search parameters
param_grid = dict(activation = activation, init = init)

# build and fit the GridSearchCV
grid = GridSearchCV(estimator = model, param_grid = param_grid, cv = KFold(random_state=seed), verbose = 10)
grid_results = grid.fit(X_standardized, Y)

# summarize the results
print("Best: {0}, using {1}".format(grid_results.best_score_, grid_results.best_params_))
means = grid_results.cv_results_['mean_test_score']
stds = grid_results.cv_results_['std_test_score']
params = grid_results.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print('{0} ({1}) with: {2}'.format(mean, stdev, param))


# In[ ]:


best_activation = 'relu'
best_init = 'normal'


# In[ ]:



# Do a grid search to find the optimal number of neurons in each hidden layer
# import necessary packages

# Define a random seed
seed = 6
np.random.seed(seed)

# Start defining the model
def create_model(neuron1, neuron2, neuron3):
    # create model
    model = Sequential()
    model.add(Dense(neuron1, input_dim = 8, kernel_initializer= best_init, activation= best_activation))
    model.add(Dense(neuron2, input_dim = neuron1, kernel_initializer= best_init, activation= best_activation))
    model.add(Dense(neuron3, input_dim = neuron2, kernel_initializer= best_init, activation= best_activation))
    model.add(Dense(1, activation='sigmoid'))

    # compile the model
    adam = Adam(lr = best_learn_rate)
    model.compile(loss = 'binary_crossentropy', optimizer = adam, metrics = ['accuracy'])
    return model

# create the model
model = KerasClassifier(build_fn = create_model, epochs = best_epochs, batch_size = best_batch_size, verbose = 0)

# define the grid search parameters
neuron1 = [8, 16, 32]
neuron2 = [16, 32, 64]
neuron3 = [8, 16, 32]

# make a dictionary of the grid search parameters
param_grid = dict(neuron1 = neuron1, neuron2 = neuron2, neuron3 = neuron3)

# build and fit the GridSearchCV
grid = GridSearchCV(estimator = model, param_grid = param_grid, cv = KFold(random_state=seed), refit = True, verbose = 10)
grid_results = grid.fit(X_standardized, Y)

# summarize the results
print("Best: {0}, using {1}".format(grid_results.best_score_, grid_results.best_params_))
means = grid_results.cv_results_['mean_test_score']
stds = grid_results.cv_results_['std_test_score']
params = grid_results.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print('{0} ({1}) with: {2}'.format(mean, stdev, param))


# In[ ]:


best_neuron1 = 16
best_neuron2 = 16
best_neuron3 = 8


# In[ ]:


from sklearn.model_selection import train_test_split
#best model
model = Sequential()
model.add(Dense(best_neuron1, input_dim = 8, kernel_initializer= best_init, activation= best_activation))
model.add(Dense(best_neuron2, input_dim = best_neuron1, kernel_initializer= best_init, activation= best_activation))
model.add(Dense(best_neuron3, input_dim = best_neuron2, kernel_initializer= best_init, activation= best_activation))
model.add(Dense(1, activation='sigmoid'))


# In[ ]:


# compile the model
adam = Adam(lr = best_learn_rate)
model.compile(loss = 'binary_crossentropy', optimizer = adam, metrics = ['accuracy'])
ckpt_model = 'pima-weights_best_t.hdf5'
checkpoint = ModelCheckpoint(ckpt_model, 
                            monitor='val_acc', 
                            verbose=1,
                            save_best_only=True,
                            mode='max')
callbacks_list = [checkpoint]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=0)

print(X_train.shape)
print(X_test.shape)

history = model.fit(X_train,
                    y_train,
                    validation_data=(X_test, y_test),
                    nb_epoch=best_epochs,
                    batch_size=best_batch_size,
                    callbacks=callbacks_list,
                    verbose=1)


# In[ ]:


model.load_weights("pima-weights_best_t.hdf5")


# In[ ]:


scores = model.evaluate(X_test, y_test, verbose=0)
print("%s: %.3f%%" % (model.metrics_names[1], scores[1]*100))


# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
# Model accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'])
plt.show()


# In[ ]:


# Model Losss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'])
plt.show()


# In[ ]:


# confusion matrix
import seaborn as sns
from sklearn.metrics import confusion_matrix
# Predict the values from the validation dataset
y_pred = model.predict(X_test)
y_final = (y_pred > 0.5).astype(int).reshape(X_test.shape[0])
# compute the confusion matrix
confusion_mtx = confusion_matrix(y_test, y_final) 
# plot the confusion matrix
f,ax = plt.subplots(figsize=(8, 8))
sns.heatmap(confusion_mtx, annot=True, linewidths=0.01,cmap="Greens",linecolor="gray", fmt= '.1f',ax=ax)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()


# In[ ]:


from sklearn.metrics import classification_report

# Generate a classification report
report = classification_report(y_test, y_final, target_names=['0','1'])

print(report)

