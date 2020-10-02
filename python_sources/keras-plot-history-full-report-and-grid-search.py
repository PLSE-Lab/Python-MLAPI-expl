#!/usr/bin/env python
# coding: utf-8

# # Keras - Plot History, Full Report and Grid Search
# 
# This notebook provides examples implementing the following
# 
# - Plot History : plot loss and accuracy from the history
# - Full Report : print a full report and plot a confusion matrix
# - Grid Search : uses the GridSearchCV and show how to resolve the issue relative to the multiclass models when using custom scoring

# In[39]:


import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn import datasets
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras.wrappers.scikit_learn import KerasClassifier

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")

seed = 1000


# ## Functions definition
# 
# ### Plot Keras History
# 
# Plot loss and accuracy for the training and validation set.

# In[40]:


def plot_history(history):
    loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' not in s]
    val_loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' in s]
    acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' not in s]
    val_acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' in s]
    
    if len(loss_list) == 0:
        print('Loss is missing in history')
        return 
    
    ## As loss always exists
    epochs = range(1,len(history.history[loss_list[0]]) + 1)
    
    ## Loss
    plt.figure(1)
    for l in loss_list:
        plt.plot(epochs, history.history[l], 'b', label='Training loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))
    for l in val_loss_list:
        plt.plot(epochs, history.history[l], 'g', label='Validation loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))
    
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    ## Accuracy
    plt.figure(2)
    for l in acc_list:
        plt.plot(epochs, history.history[l], 'b', label='Training accuracy (' + str(format(history.history[l][-1],'.5f'))+')')
    for l in val_acc_list:    
        plt.plot(epochs, history.history[l], 'g', label='Validation accuracy (' + str(format(history.history[l][-1],'.5f'))+')')

    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


# ## Create a Full Multiclass Report

# In[41]:


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        title='Normalized confusion matrix'
    else:
        title='Confusion matrix'

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    
## multiclass or binary report
## If binary (sigmoid output), set binary parameter to True
def full_multiclass_report(model,
                           x,
                           y_true,
                           classes,
                           batch_size=32,
                           binary=False):

    # 1. Transform one-hot encoded y_true into their class number
    if not binary:
        y_true = np.argmax(y_true,axis=1)
    
    # 2. Predict classes and stores in y_pred
    y_pred = model.predict_classes(x, batch_size=batch_size)
    
    # 3. Print accuracy score
    print("Accuracy : "+ str(accuracy_score(y_true,y_pred)))
    
    print("")
    
    # 4. Print classification report
    print("Classification Report")
    print(classification_report(y_true,y_pred,digits=5))    
    
    # 5. Plot confusion matrix
    cnf_matrix = confusion_matrix(y_true,y_pred)
    print(cnf_matrix)
    plot_confusion_matrix(cnf_matrix,classes=classes)


# ## Load Data

# In[42]:


iris = datasets.load_iris()
x = iris.data
y = to_categorical(iris.target)
labels_names = iris.target_names
xid, yid = 0, 1

le = LabelEncoder()
encoded_labels = le.fit_transform(iris.target_names)

plt.scatter(x[:,xid],x[:,yid],c=y,cmap=plt.cm.Set1,edgecolor='k')
plt.xlabel(iris.feature_names[xid])
plt.ylabel(iris.feature_names[yid])


# ## Split Train/Val/Test

# In[44]:


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=seed)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.8, random_state=seed)


# ## Basic Keras Model
# 
# Create a very basic MLNN with a single Dense layer.

# In[21]:


model = Sequential()
model.add(Dense(8,activation='relu',input_shape = (4,)))
model.add(Dense(3,activation='softmax'))
model.compile(optimizer = 'rmsprop',
             loss='categorical_crossentropy',
             metrics=['accuracy'])

history = model.fit(x_train, 
                    y_train,
                    epochs = 200,
                    batch_size = 16,
                    verbose=0,
                    validation_data=(x_val,y_val))


# In[22]:


plot_history(history)


# ### Full report on the Validation Set

# In[32]:


full_multiclass_report(model,
                       x_val,
                       y_val,
                       le.inverse_transform(np.arange(3)))


# ### Full report on the test set

# In[34]:


full_multiclass_report(model,
                       x_test,
                       y_test,
                       le.inverse_transform(np.arange(3)))


# ## Grid Search
# 
# Using grid search in keras can lead to an issue when trying to use custom scoring with multiclass models. 
# 
# Assume you creates a multiclass model as above with Iris. 
# 
# With keras, you usually encode `y` as categorical data like this: `[[0,1,0],[1,0,0], ...]`
# 
# But when you try to use a custom scoring such as below:  
# 
# ```python
# grid = GridSearchCV(model,
#                     param_grid=param_grid,
#                     return_train_score=True,
#                    scoring=['precision_macro','recall_macro','f1_macro'],
#                     refit='precision_macro')
# grid_results = grid.fit(x_train,y_train)
# ```
# 
# You get this error:
# 
# ```
# ValueError: Classification metrics can't handle a mix of multilabel-indicator and multiclass targets
# ```
# 
# After searching in keras code ans sklearn details I've finally found how to resolve it. I wrote the details here :  https://github.com/keras-team/keras/issues/9331
# 
# Basically, you simply have to **not encode** your output as categorical when using GridSearchCV, or cross_validation with custom scoring. Keras Wrapper already does it for you and so is able to make it compatible with sklearn multiclass model (in sklearn the one-hot is interpreted as multilabel)..
# 
# That's why the code below reload the iris labels to remove the one-hot encoding before using GridSearchCV.

# In[36]:


y = iris.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=seed)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.8, random_state=seed)

def create_model(dense_layers=[8],
                 activation='relu',
                 optimizer='rmsprop'):
    model = Sequential()

    for index, lsize in enumerate(dense_layers):
        # Input Layer - includes the input_shape
        if index == 0:
            model.add(Dense(lsize,
                            activation=activation,
                            input_shape=(4,)))
        else:
            model.add(Dense(lsize,
                            activation=activation))
            
    model.add(Dense(3,activation='softmax'))
    model.compile(optimizer = optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

model = KerasClassifier(build_fn=create_model,
                        epochs=10, 
                        batch_size=5,
                        verbose=0)

param_grid = {'dense_layers': [[4],[8],[8,8]],
              'activation':['relu','tanh'],
              'optimizer':('rmsprop','adam'),
              'epochs':[10,50],
              'batch_size':[5,16]}

grid = GridSearchCV(model,
                    param_grid=param_grid,
                    return_train_score=True,
                    scoring=['precision_macro','recall_macro','f1_macro'],
                    refit='precision_macro')

grid_results = grid.fit(x_train,y_train)

print('Parameters of the best model: ')
print(grid_results.best_params_)


# Here we found the following as the best parameters:
# 
# ```
# Parameters of the best model: 
# {'activation': 'tanh', 'batch_size': 5, 'dense_layers': [8, 8], 'epochs': 50, 'optimizer': 'rmsprop'}
# ```
# 
# So lets apply them on a candidate model and check the results.
# 

# In[38]:


## First redefine y as categorical variable
y = to_categorical(iris.target,3)

## Rebuild the training and test set with the categorical y
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=seed)

## Capture the best params
params = grid_results.best_params_

## create the model with the best params found
model = create_model(dense_layers=params['dense_layers'],
                     activation=params['activation'],
                     optimizer=params['optimizer'])

## Then train it and display the results
history = model.fit(x_train,
                    y_train,
                    epochs=params['epochs'],
                    batch_size=params['batch_size'],
                    verbose = 0)

model.summary()
plot_history(history)
full_multiclass_report(model,
                       x_test,
                       y_test,
                       classes=le.inverse_transform(np.arange(3)))


# This is summarizing plot function, full report and usage of grid search with Keras.
# 
# Thank you

# In[ ]:




