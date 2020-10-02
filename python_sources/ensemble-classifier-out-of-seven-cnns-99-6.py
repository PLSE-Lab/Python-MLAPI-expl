#!/usr/bin/env python
# coding: utf-8

# # Playing with ensemble classifiers based on seven different CNN models (~99.5 % accuracy)

# Hellouu peoples.
# 
# This notebook presents my results of playing around with CNNs in Keras. I train seven different architectures and use them two create an ensemble classifier in order to compare accuracies. Why seven? I am so glad you asked. The number is motivated by a mixture of random coincidence and [this epic movie](https://en.wikipedia.org/wiki/Seven_Samurai). 
# 
# Before I start I want to give credit to [Yassine Ghouzam's notebook](https://www.kaggle.com/yassineghouzam/introduction-to-cnn-keras-0-997-top-6), which I used as inspiration and which is probably the better notbook if you are working with Keras for the first time. You might also want to have a look at [this](http://cs231n.github.io/convolutional-networks/#architectures), which helped me a lot.
# 
# This is what you're going to find here:
# 
# __1. Import of libraries and import of the data__
# 
# __2. First overview of data__
#   1. Plot images
#   2. Check distribution
#   3. Check for missing values
#   
# __3. Create train, validation and test datasets__
# 
# __4. Define and train the convolutional neural networks__
#   1. Model 1
#   2. Model 2
#   3. Model 3
#   4. Model 4
#   5. Model 5
#   6. Model 6
#   7. Model 7
#   
# __5. Ensemble classifiers and confusion matrices__
#   1. Overview of model performance so far:
#   2. Ensemble classifier based on summing the probabilities
#   3. Ensemble classifier based on a majority vote
#   
# __6. Conlusion__
# 
# __7. Output routines__

# # Import of libraries and import of the data

# In[ ]:


#Data
import pandas as pd
import numpy as np
from scipy import stats
import sys

#Plotting
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.offline as py
import plotly.figure_factory as ff
import plotly.graph_objs as go
from plotly import tools
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import colorlover as cl
from IPython.display import HTML, SVG
import random

random.seed(42)
init_notebook_mode(connected=True)
#%matplotlib inline


# In[ ]:


df_train = pd.read_csv('../input/train.csv')
df_comp = pd.read_csv('../input/test.csv')


# In[ ]:


iplot(ff.create_table(df_train.iloc[0:10,0:10]), filename='jupyter-table1')


# # First overview of data

# ### Plot images

# Let's create an overview of the data so that we get a feeling of what we're dealing with. You can change the displayed number by changing *sel_int*.

# In[ ]:


n_size = 10
sel_int = 5
num_array = df_train[df_train.label == sel_int]

fig = tools.make_subplots(rows=10, cols=10, print_grid=False)
for row in range(1,n_size+1):
    for column in range(1,n_size+1):
        trace = go.Heatmap(z=num_array.iloc[row*10-10+column-1, 1:].values.reshape((28,28))[::-1], colorscale=[[0,'rgb(0,0,0)'],[1,'rgb(255,255,255)']], showscale=False)
        fig.append_trace(trace, row, column)
        fig['layout']['xaxis'+str(((row-1)*10 + column))].update(showticklabels=False, ticks='')
        fig['layout']['yaxis'+str(((row-1)*10 + column))].update(showticklabels=False, ticks='')
        
fig['layout'].update(height=500, width=500)
fig['layout']['margin'].update(l=10, r=10, b=10, t=10)
iplot(fig, filename='number_plot')


# ### Check distribution

# Let's check whether the labels are evenly distributed:

# In[ ]:


#df.label.value_counts().values
trace = go.Bar(x=df_train.label.value_counts().index,y=df_train.label.value_counts().values)
layout = go.Layout(xaxis=dict(title='Number', nticks=10),
                  yaxis=dict(title='# Occurance'),
                  width = 600,
                  height = 400
                  )
figure = go.Figure(data = [trace],
                  layout = layout)
figure['layout']['margin'].update(l=50, r=50, b=50, t=50)
iplot(figure)


# ### Check for missing values

# In[ ]:


df_train.describe()


# In[ ]:


df_train.isnull().sum().sum()


# No missing values - perfect.

# # Create train, validation and test datasets

# As usual we need to split our data into a training, validation and test data set.

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


Y = df_train.label
X = df_train.drop('label', axis=1)

X = X / 255
X_comp = df_comp / 255

X_train, X_cross, Y_train, Y_cross = train_test_split(X, Y,test_size=0.1, random_state=42)
X_valid, X_test, Y_valid, Y_test = train_test_split(X_cross, Y_cross, test_size=0.5, random_state=42)


# Let's make sure there is nothing going wrong with the distribution of the labels in the three sets:

# In[ ]:


trace1 = go.Bar(x=Y_train.value_counts().index,y=Y_train.value_counts().values/Y_train.value_counts().values.sum(), name='Training set')
trace2 = go.Bar(x=Y_valid.value_counts().index,y=Y_valid.value_counts().values/Y_valid.value_counts().values.sum(), name='Validation set')
trace3 = go.Bar(x=Y_test.value_counts().index,y=Y_test.value_counts().values/Y_test.value_counts().values.sum(), name='Test set')
fig = go.Figure(data=[trace1, trace2, trace3])
fig['layout'].update(xaxis=dict(title='Number', nticks=10), yaxis=dict(title='# Occurance'), width = 600, height = 400)
fig['layout']['margin'].update(l=50, r=50, b=50, t=50)
iplot(fig)


# # Define and train the convolutional neural networks

# Here's where the meat comes on the table. In the next sections the six CNNs are defined and trained using a GPU and the ImageDataGenerator from Keras to make the networks more robust.

# In[ ]:


from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, MaxPool2D, Dropout, Flatten
from keras.utils import plot_model, to_categorical
from keras.utils.vis_utils import model_to_dot
from keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import confusion_matrix, accuracy_score


# In[ ]:


X_train = X_train.values.reshape(X_train.shape[0],28,28,1)
X_valid = X_valid.values.reshape(X_valid.shape[0],28,28,1)
X_test = X_test.values.reshape(X_test.shape[0],28,28,1)
X_comp = X_comp.values.reshape(X_comp.shape[0],28,28,1)

Y_train = to_categorical(Y_train)
Y_valid = to_categorical(Y_valid)
Y_test = to_categorical(Y_test)


# In[ ]:


datagen = ImageDataGenerator(height_shift_range=0.1,
                             width_shift_range=0.1,
                             #brightness_range=(0,0.1),
                             rotation_range=10,
                             zoom_range=0.1,
                             fill_mode='constant',
                             cval=0
                            )

datagen.fit(X_train)


# ## Model 1

# In[ ]:


model = Sequential()
droprate = 0.175
model.add(Conv2D(kernel_size=(2,2), filters=128, strides=(1,1), padding='same',activation='relu', input_shape=(28,28,1)))
model.add(Conv2D(kernel_size=(2,2), filters=128, strides=(1,1), padding='same',activation='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(droprate))
model.add(Conv2D(kernel_size=(2,2), filters=64, strides=(1,1), padding='same',activation='relu'))
model.add(Conv2D(kernel_size=(2,2), filters=64, strides=(1,1), padding='same',activation='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(droprate))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(droprate))
model.add(Dense(10, activation='softmax'))

model1 = model
model1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

plot_model(model1, show_shapes=True, to_file='Network_9.png')
SVG(model_to_dot(model1, show_shapes=True).create(prog='dot', format='svg'))
# In[ ]:


epochsN = 25
batch_sizeN = 63
history1 = model1.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_sizeN), validation_data=(X_valid, Y_valid), steps_per_epoch=len(X_train)/batch_sizeN, epochs=epochsN, verbose=2)


# In[ ]:


model1.evaluate(X_test, Y_test, verbose=0)


# In[ ]:


model1.save('model_1.h5')


# In[ ]:


history = history1
fig = tools.make_subplots(rows=1, cols=2, print_grid=False)
trace1 = go.Scatter(x=list(range(1,epochsN+1)), y=history.history['loss'], name='Training Loss')
trace2 = go.Scatter(x=list(range(1,epochsN+1)), y=history.history['val_loss'], name='Validation Loss')
trace3 = go.Scatter(x=list(range(1,epochsN+1)), y=history.history['acc'], name='Training Accuracy')
trace4 = go.Scatter(x=list(range(1,epochsN+1)), y=history.history['val_acc'], name='Validation Accuracy')
fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 1, 1)
fig.append_trace(trace3, 1, 2)
fig.append_trace(trace4, 1, 2)
fig['layout'].update(xaxis1=dict(title='Epoch', nticks=10), yaxis1=dict(title='Loss', type='log'), width = 1200, height = 400)
fig['layout'].update(xaxis2=dict(title='Epoch', nticks=10), yaxis2=dict(title='Accuracy', type = 'log'))
fig['layout']['margin'].update(l=50, r=50, b=50, t=50)

#fig['layout'].update(xaxis=dict(title='Epoch', nticks=10), yaxis=dict(title='Loss'), width = 600, height = 400)
#fig['layout']['margin'].update(l=50, r=50, b=50, t=50)
iplot(fig)


# ## Model 2

# In[ ]:


del model
model = Sequential()
droprate = 0.15
model.add(Conv2D(kernel_size=(2,2), filters=128, strides=(1,1), padding='same',activation='relu', input_shape=(28,28,1)))
#model.add(Conv2D(kernel_size=(2,2), filters=128, strides=(1,1), padding='same',activation='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(droprate))
model.add(Conv2D(kernel_size=(2,2), filters=64, strides=(1,1), padding='same',activation='relu'))
#model.add(Conv2D(kernel_size=(2,2), filters=64, strides=(1,1), padding='same',activation='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(droprate))
model.add(Conv2D(kernel_size=(2,2), filters=32, strides=(1,1), padding='same',activation='relu'))
#model.add(Conv2D(kernel_size=(2,2), filters=64, strides=(1,1), padding='same',activation='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(droprate))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(droprate))
model.add(Dense(10, activation='softmax'))

model2 = model
model2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

plot_model(model1, show_shapes=True, to_file='Network_9.png')
SVG(model_to_dot(model1, show_shapes=True).create(prog='dot', format='svg'))
# In[ ]:


epochsN = 35
batch_sizeN = 63
history2 = model2.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_sizeN), validation_data=(X_valid, Y_valid), steps_per_epoch=len(X_train)/batch_sizeN, epochs=epochsN, verbose=2)


# In[ ]:


model2.evaluate(X_test, Y_test, verbose=0)


# In[ ]:


model2.save('model_2.h5')


# In[ ]:


history = history2
fig = tools.make_subplots(rows=1, cols=2, print_grid=False)
trace1 = go.Scatter(x=list(range(1,epochsN+1)), y=history.history['loss'], name='Training Loss')
trace2 = go.Scatter(x=list(range(1,epochsN+1)), y=history.history['val_loss'], name='Validation Loss')
trace3 = go.Scatter(x=list(range(1,epochsN+1)), y=history.history['acc'], name='Training Accuracy')
trace4 = go.Scatter(x=list(range(1,epochsN+1)), y=history.history['val_acc'], name='Validation Accuracy')
fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 1, 1)
fig.append_trace(trace3, 1, 2)
fig.append_trace(trace4, 1, 2)
fig['layout'].update(xaxis1=dict(title='Epoch', nticks=10), yaxis1=dict(title='Loss', type='log'), width = 1200, height = 400)
fig['layout'].update(xaxis2=dict(title='Epoch', nticks=10), yaxis2=dict(title='Accuracy', type = 'log'))
fig['layout']['margin'].update(l=50, r=50, b=50, t=50)

#fig['layout'].update(xaxis=dict(title='Epoch', nticks=10), yaxis=dict(title='Loss'), width = 600, height = 400)
#fig['layout']['margin'].update(l=50, r=50, b=50, t=50)
iplot(fig)


# ## Model 3

# In[ ]:


del model
model = Sequential()
droprate = 0.2
model.add(Conv2D(kernel_size=(2,2), filters=128, strides=(1,1), padding='same',activation='relu', input_shape=(28,28,1)))
model.add(Conv2D(kernel_size=(2,2), filters=128, strides=(1,1), padding='same',activation='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(droprate))
model.add(Conv2D(kernel_size=(2,2), filters=64, strides=(1,1), padding='same',activation='relu'))
model.add(Conv2D(kernel_size=(2,2), filters=64, strides=(1,1), padding='same',activation='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(droprate))
model.add(Conv2D(kernel_size=(2,2), filters=32, strides=(1,1), padding='same',activation='relu'))
model.add(Conv2D(kernel_size=(2,2), filters=32, strides=(1,1), padding='same',activation='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(droprate))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(droprate))
model.add(Dense(10, activation='softmax'))

model3 = model
model3.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

plot_model(model1, show_shapes=True, to_file='Network_9.png')
SVG(model_to_dot(model1, show_shapes=True).create(prog='dot', format='svg'))
# In[ ]:


epochsN = 40
batch_sizeN = 63
history3 = model3.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_sizeN), validation_data=(X_valid, Y_valid), steps_per_epoch=len(X_train)/batch_sizeN, epochs=epochsN, verbose=2)


# In[ ]:


model3.evaluate(X_test, Y_test, verbose=0)


# In[ ]:


model3.save('model_3.h5')


# In[ ]:


history = history3
fig = tools.make_subplots(rows=1, cols=2, print_grid=False)
trace1 = go.Scatter(x=list(range(1,epochsN+1)), y=history.history['loss'], name='Training Loss')
trace2 = go.Scatter(x=list(range(1,epochsN+1)), y=history.history['val_loss'], name='Validation Loss')
trace3 = go.Scatter(x=list(range(1,epochsN+1)), y=history.history['acc'], name='Training Accuracy')
trace4 = go.Scatter(x=list(range(1,epochsN+1)), y=history.history['val_acc'], name='Validation Accuracy')
fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 1, 1)
fig.append_trace(trace3, 1, 2)
fig.append_trace(trace4, 1, 2)
fig['layout'].update(xaxis1=dict(title='Epoch', nticks=10), yaxis1=dict(title='Loss', type='log'), width = 1200, height = 400)
fig['layout'].update(xaxis2=dict(title='Epoch', nticks=10), yaxis2=dict(title='Accuracy', type = 'log'))
fig['layout']['margin'].update(l=50, r=50, b=50, t=50)

#fig['layout'].update(xaxis=dict(title='Epoch', nticks=10), yaxis=dict(title='Loss'), width = 600, height = 400)
#fig['layout']['margin'].update(l=50, r=50, b=50, t=50)
iplot(fig)


# ## Model 4

# In[ ]:


del model
model = Sequential()
droprate = 0.20
model.add(Conv2D(kernel_size=(2,2), filters=128, strides=(1,1), padding='same',activation='relu', input_shape=(28,28,1)))
model.add(Conv2D(kernel_size=(2,2), filters=128, strides=(1,1), padding='same',activation='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(droprate))
model.add(Conv2D(kernel_size=(2,2), filters=64, strides=(1,1), padding='same',activation='relu'))
model.add(Conv2D(kernel_size=(2,2), filters=64, strides=(1,1), padding='same',activation='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(droprate))
model.add(Conv2D(kernel_size=(2,2), filters=32, strides=(1,1), padding='same',activation='relu'))
model.add(Conv2D(kernel_size=(2,2), filters=32, strides=(1,1), padding='same',activation='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(droprate))
model.add(Conv2D(kernel_size=(2,2), filters=16, strides=(1,1), padding='same',activation='relu'))
model.add(Conv2D(kernel_size=(2,2), filters=16, strides=(1,1), padding='same',activation='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(droprate))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(droprate))
model.add(Dense(10, activation='softmax'))

model4 = model
model4.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

plot_model(model1, show_shapes=True, to_file='Network_9.png')
SVG(model_to_dot(model1, show_shapes=True).create(prog='dot', format='svg'))
# In[ ]:


epochsN = 90
batch_sizeN = 63
history4 = model4.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_sizeN), validation_data=(X_valid, Y_valid), steps_per_epoch=len(X_train)/batch_sizeN, epochs=epochsN, verbose=2)


# In[ ]:


model4.evaluate(X_test, Y_test, verbose=0)


# In[ ]:


model4.save('model_4.h5')


# In[ ]:


history = history4
fig = tools.make_subplots(rows=1, cols=2, print_grid=False)
trace1 = go.Scatter(x=list(range(1,epochsN+1)), y=history.history['loss'], name='Training Loss')
trace2 = go.Scatter(x=list(range(1,epochsN+1)), y=history.history['val_loss'], name='Validation Loss')
trace3 = go.Scatter(x=list(range(1,epochsN+1)), y=history.history['acc'], name='Training Accuracy')
trace4 = go.Scatter(x=list(range(1,epochsN+1)), y=history.history['val_acc'], name='Validation Accuracy')
fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 1, 1)
fig.append_trace(trace3, 1, 2)
fig.append_trace(trace4, 1, 2)
fig['layout'].update(xaxis1=dict(title='Epoch', nticks=10), yaxis1=dict(title='Loss', type='log'), width = 1200, height = 400)
fig['layout'].update(xaxis2=dict(title='Epoch', nticks=10), yaxis2=dict(title='Accuracy', type = 'log'))
fig['layout']['margin'].update(l=50, r=50, b=50, t=50)

#fig['layout'].update(xaxis=dict(title='Epoch', nticks=10), yaxis=dict(title='Loss'), width = 600, height = 400)
#fig['layout']['margin'].update(l=50, r=50, b=50, t=50)
iplot(fig)


# ## Model 5

# In[ ]:


del model
model = Sequential()
droprate = 0.1
model.add(Conv2D(kernel_size=(2,2), filters=64, strides=(1,1), padding='same',activation='relu', input_shape=(28,28,1)))
model.add(Conv2D(kernel_size=(2,2), filters=64, strides=(2,2), padding='valid',activation='relu'))
model.add(Dropout(droprate))
model.add(Conv2D(kernel_size=(2,2), filters=32, strides=(1,1), padding='same',activation='relu'))
model.add(Conv2D(kernel_size=(2,2), filters=32, strides=(2,2), padding='valid',activation='relu'))
model.add(Dropout(droprate))
model.add(Conv2D(kernel_size=(2,2), filters=16, strides=(1,1), padding='same',activation='relu'))
model.add(Conv2D(kernel_size=(3,3), filters=16, strides=(2,2), padding='valid',activation='relu'))
model.add(Flatten())
model.add(Dropout(droprate))
model.add(Dense(128, activation='relu'))
model.add(Dropout(droprate))
model.add(Dense(10, activation='softmax'))

model5 = model
model5.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

plot_model(model1, show_shapes=True, to_file='Network_9.png')
SVG(model_to_dot(model1, show_shapes=True).create(prog='dot', format='svg'))
# In[ ]:


epochsN = 90
batch_sizeN = 63
history5 = model5.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_sizeN), validation_data=(X_valid, Y_valid), steps_per_epoch=len(X_train)/batch_sizeN, epochs=epochsN, verbose=2)


# In[ ]:


model5.evaluate(X_test, Y_test, verbose=0)


# In[ ]:


model5.save('model_5.h5')


# In[ ]:


history = history5
fig = tools.make_subplots(rows=1, cols=2, print_grid=False)
trace1 = go.Scatter(x=list(range(1,epochsN+1)), y=history.history['loss'], name='Training Loss')
trace2 = go.Scatter(x=list(range(1,epochsN+1)), y=history.history['val_loss'], name='Validation Loss')
trace3 = go.Scatter(x=list(range(1,epochsN+1)), y=history.history['acc'], name='Training Accuracy')
trace4 = go.Scatter(x=list(range(1,epochsN+1)), y=history.history['val_acc'], name='Validation Accuracy')
fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 1, 1)
fig.append_trace(trace3, 1, 2)
fig.append_trace(trace4, 1, 2)
fig['layout'].update(xaxis1=dict(title='Epoch', nticks=10), yaxis1=dict(title='Loss', type='log'), width = 1200, height = 400)
fig['layout'].update(xaxis2=dict(title='Epoch', nticks=10), yaxis2=dict(title='Accuracy', type = 'log'))
fig['layout']['margin'].update(l=50, r=50, b=50, t=50)

#fig['layout'].update(xaxis=dict(title='Epoch', nticks=10), yaxis=dict(title='Loss'), width = 600, height = 400)
#fig['layout']['margin'].update(l=50, r=50, b=50, t=50)
iplot(fig)


# ## Model 6

# In[ ]:


del model
model = Sequential()
droprate = 0.15
model.add(Conv2D(kernel_size=(2,2), filters=64, strides=(1,1), padding='same',activation='relu', input_shape=(28,28,1)))
model.add(Conv2D(kernel_size=(2,2), filters=64, strides=(1,1), padding='same',activation='relu'))
model.add(Conv2D(kernel_size=(2,2), filters=64, strides=(1,1), padding='same',activation='relu'))
model.add(Conv2D(kernel_size=(2,2), filters=64, strides=(2,2), padding='valid',activation='relu'))
model.add(Dropout(droprate))
model.add(Conv2D(kernel_size=(2,2), filters=32, strides=(1,1), padding='same',activation='relu'))
model.add(Conv2D(kernel_size=(2,2), filters=32, strides=(1,1), padding='same',activation='relu'))
model.add(Conv2D(kernel_size=(2,2), filters=32, strides=(1,1), padding='same',activation='relu'))
model.add(Conv2D(kernel_size=(2,2), filters=16, strides=(2,2), padding='valid',activation='relu'))
model.add(Flatten())
model.add(Dropout(droprate))
model.add(Dense(256, activation='relu'))
model.add(Dropout(droprate))
model.add(Dense(10, activation='softmax'))

model6 = model
model6.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

plot_model(model1, show_shapes=True, to_file='Network_9.png')
SVG(model_to_dot(model1, show_shapes=True).create(prog='dot', format='svg'))
# In[ ]:


epochsN = 45
batch_sizeN = 63
history6 = model6.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_sizeN), validation_data=(X_valid, Y_valid), steps_per_epoch=len(X_train)/batch_sizeN, epochs=epochsN, verbose=2)


# In[ ]:


model6.evaluate(X_test, Y_test, verbose=0)


# In[ ]:


model6.save('model_6.h5')


# In[ ]:


history = history6
fig = tools.make_subplots(rows=1, cols=2, print_grid=False)
trace1 = go.Scatter(x=list(range(1,epochsN+1)), y=history.history['loss'], name='Training Loss')
trace2 = go.Scatter(x=list(range(1,epochsN+1)), y=history.history['val_loss'], name='Validation Loss')
trace3 = go.Scatter(x=list(range(1,epochsN+1)), y=history.history['acc'], name='Training Accuracy')
trace4 = go.Scatter(x=list(range(1,epochsN+1)), y=history.history['val_acc'], name='Validation Accuracy')
fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 1, 1)
fig.append_trace(trace3, 1, 2)
fig.append_trace(trace4, 1, 2)
fig['layout'].update(xaxis1=dict(title='Epoch', nticks=10), yaxis1=dict(title='Loss', type='log'), width = 1200, height = 400)
fig['layout'].update(xaxis2=dict(title='Epoch', nticks=10), yaxis2=dict(title='Accuracy', type = 'log'))
fig['layout']['margin'].update(l=50, r=50, b=50, t=50)

#fig['layout'].update(xaxis=dict(title='Epoch', nticks=10), yaxis=dict(title='Loss'), width = 600, height = 400)
#fig['layout']['margin'].update(l=50, r=50, b=50, t=50)
iplot(fig)


# ## Model 7

# In[ ]:


del model
model = Sequential()
droprate = 0.35
model.add(Conv2D(kernel_size=(2,2), filters=64, strides=(1,1), padding='same',activation='relu', input_shape=(28,28,1)))
model.add(Conv2D(kernel_size=(2,2), filters=64, strides=(1,1), padding='same',activation='relu'))
#model.add(Conv2D(kernel_size=(2,2), filters=64, strides=(1,1), padding='same',activation='relu'))
model.add(Conv2D(kernel_size=(2,2), filters=64, strides=(2,2), padding='valid',activation='relu'))
model.add(Dropout(droprate))
model.add(Conv2D(kernel_size=(3,3), filters=128, strides=(1,1), padding='same',activation='relu'))
model.add(Conv2D(kernel_size=(3,3), filters=128, strides=(1,1), padding='same',activation='relu'))
#model.add(Conv2D(kernel_size=(3,3), filters=128, strides=(1,1), padding='same',activation='relu'))
model.add(Conv2D(kernel_size=(2,2), filters=128, strides=(2,2), padding='valid',activation='relu'))
model.add(Dropout(droprate))
model.add(Conv2D(kernel_size=(3,3), filters=256, strides=(1,1), padding='valid',activation='relu'))
model.add(Conv2D(kernel_size=(3,3), filters=256, strides=(1,1), padding='valid',activation='relu'))
#model.add(Conv2D(kernel_size=(3,3), filters=256, strides=(1,1), padding='same',activation='relu'))
model.add(Conv2D(kernel_size=(3,3), filters=256, strides=(2,2), padding='valid',activation='relu'))
model.add(Dropout(droprate))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))

model7 = model
model7.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

plot_model(model1, show_shapes=True, to_file='Network_9.png')
SVG(model_to_dot(model1, show_shapes=True).create(prog='dot', format='svg'))
# In[ ]:


epochsN = 60
batch_sizeN = 63
history7 = model7.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_sizeN), validation_data=(X_valid, Y_valid), steps_per_epoch=len(X_train)/batch_sizeN, epochs=epochsN, verbose=2)


# In[ ]:


model7.evaluate(X_test, Y_test, verbose=0)


# In[ ]:


model7.save('model_7.h5')


# In[ ]:


history = history7
fig = tools.make_subplots(rows=1, cols=2, print_grid=False)
trace1 = go.Scatter(x=list(range(1,epochsN+1)), y=history.history['loss'], name='Training Loss')
trace2 = go.Scatter(x=list(range(1,epochsN+1)), y=history.history['val_loss'], name='Validation Loss')
trace3 = go.Scatter(x=list(range(1,epochsN+1)), y=history.history['acc'], name='Training Accuracy')
trace4 = go.Scatter(x=list(range(1,epochsN+1)), y=history.history['val_acc'], name='Validation Accuracy')
fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 1, 1)
fig.append_trace(trace3, 1, 2)
fig.append_trace(trace4, 1, 2)
fig['layout'].update(xaxis1=dict(title='Epoch', nticks=10), yaxis1=dict(title='Loss', type='log'), width = 1200, height = 400)
fig['layout'].update(xaxis2=dict(title='Epoch', nticks=10), yaxis2=dict(title='Accuracy', type = 'log'))
fig['layout']['margin'].update(l=50, r=50, b=50, t=50)

#fig['layout'].update(xaxis=dict(title='Epoch', nticks=10), yaxis=dict(title='Loss'), width = 600, height = 400)
#fig['layout']['margin'].update(l=50, r=50, b=50, t=50)
iplot(fig)


# # Ensemble classifiers and confusion matrices

# #Optionally load the models from disk:
# model1 = load_model('model_1.h5')
# model2 = load_model('model_2.h5')
# model3 = load_model('model_3.h5')
# model4 = load_model('model_4.h5')
# model5 = load_model('model_5.h5')
# model6 = load_model('model_6.h5')
# model7 = load_model('model_7.h5')

# ## Overview of model performance so far:

# Let's make a list and plot the performance of the created models:

# In[ ]:


trained_models = [model1, model2, model3, model4, model5, model6, model7]


# In[ ]:


acc_scores = pd.Series()
for num, model in enumerate(trained_models):
    acc_scores.loc['Model ' + str(num + 1)] = accuracy_score(np.argmax(Y_test, axis=1), np.argmax(model.predict(X_test), axis=1))


# In[ ]:


trace = go.Bar(x=acc_scores.values ,y=acc_scores.index, orientation='h')
layout = go.Layout(xaxis=dict(title='Accuracy', nticks=10, range=[0.985, 1]),
                  #yaxis=dict(title='Model'),
                  width = 600,
                  height = 400
                  )
figure = go.Figure(data = [trace],
                  layout = layout)
figure['layout']['margin'].update(l=50, r=50, b=50, t=50)
iplot(figure)


# Let's get the score of the best performing model:

# In[ ]:


print(acc_scores.idxmax(), ': ', acc_scores[acc_scores.idxmax()])


# As a reference, let's plot the confusion matrix for this model:

# In[ ]:


ind_best_model = acc_scores.reset_index().loc[:, 0].idxmax(axis=0)
Y_test_pred = trained_models[ind_best_model].predict(X_test)
confM = confusion_matrix(np.argmax(Y_test, axis=1), np.argmax(Y_test_pred, axis=1))

fig = ff.create_annotated_heatmap(x=list(map(str, range(0,10))), y=list(map(str, range(0,10))), z=np.log(confM+1), annotation_text=confM, colorscale='Jet')
fig['layout'].update(xaxis=dict(title='Predictated Label'), yaxis=dict(title='Actual Label', autorange='reversed'), width = 600, height = 600)

iplot(fig)


# ## Ensemble classifier based on summing the probabilities:

# Now lets see whether we can get a better classification by summing up the probabilities of the six different models:

# In[ ]:


def summing_classifier(data, model_list):
    total_pred_prob = model_list[0].predict(data)
    for model in model_list[1:]:
        total_pred_prob += model.predict(data)
        
    return np.argmax(total_pred_prob, axis=1)


# This gives a score of:

# In[ ]:


acc_scores.loc['Summing Classifier'] = accuracy_score(np.argmax(Y_test, axis=1), summing_classifier(X_test, trained_models))
acc_scores.loc['Summing Classifier']


# Which we can compare to the average of the six models:

# In[ ]:


acc_scores.iloc[0:6].mean()


# The confusion matrix for the summing classifier looks like this:

# In[ ]:


confM = confusion_matrix(np.argmax(Y_test, axis=1), summing_classifier(X_test, trained_models))

fig = ff.create_annotated_heatmap(x=list(map(str, range(0,10))), y=list(map(str, range(0,10))), z=np.log(confM+1), annotation_text=confM, colorscale='Jet')
fig['layout'].update(xaxis=dict(title='Predictated Label'), yaxis=dict(title='Actual Label', autorange='reversed'), width = 600, height = 600)

iplot(fig)


# ## Ensemble classifier based on majority vote:

# Let's try if we get a different result by determining the label with a majority vote of the six different models:

# In[ ]:


def voting_classifier(data, model_list):
    pred_list = np.argmax(model_list[0].predict(data), axis=1).reshape((1,len(data)))
    for model in model_list[1:]:
        pred_list = np.append(pred_list, [np.argmax(model.predict(data), axis=1)], axis=0)
    return np.array(list(map(lambda x: np.bincount(x).argmax(), pred_list.T)))


# This results in a accuracy of:

# In[ ]:


acc_scores.loc['Voting Classifier'] = accuracy_score(np.argmax(Y_test, axis=1), voting_classifier(X_test, trained_models))
acc_scores.loc['Voting Classifier']


# And here is the confusing matrix we see that we make different mistakes (but not more or less):

# In[ ]:


confM = confusion_matrix(np.argmax(Y_test, axis=1), voting_classifier(X_test, trained_models))

fig = ff.create_annotated_heatmap(x=list(map(str, range(0,10))), y=list(map(str, range(0,10))), z=np.log(confM+1), annotation_text=confM, colorscale='Jet')
fig['layout'].update(xaxis=dict(title='Predictated Label'), yaxis=dict(title='Actual Label', autorange='reversed'), width = 600, height = 600)

iplot(fig)


# # Conclusion

# Let's plot the accuracies again:

# In[ ]:


trace = go.Bar(x=acc_scores.sort_values(ascending=True).values ,y=acc_scores.sort_values(ascending=True).index, orientation='h')
layout = go.Layout(xaxis=dict(title='Accuracy', nticks=10, range=[0.985, 1]),
                  #yaxis=dict(title='Model'),
                  width = 600,
                  height = 400
                  )
figure = go.Figure(data = [trace],
                  layout = layout)
figure['layout']['margin'].update(l=130, r=50, b=50, t=50)
iplot(figure)


# So that's it. While I wouldn't trust the accuracies determined from a small test set of ~2000 images too much, it seems that the ensemble enhances the accurance beyond what the best single model can do.

# # Output routines

# ### Competition predictions of the best single model:

# In[ ]:


best_model_results = pd.DataFrame({'Label' : np.argmax(trained_models[ind_best_model].predict(X_comp), axis=1)})
best_model_results = best_model_results.reset_index().rename(columns={'index' : 'ImageId'})
best_model_results['ImageId'] = best_model_results['ImageId'] + 1
best_model_results.to_csv('best_model_result_kaggle.csv', index=False)


# ### Competition predictions of the ensemble based on the sum:

# In[ ]:


esmbl_sum_results = pd.DataFrame({'Label' : summing_classifier(X_comp, trained_models)})
esmbl_sum_results = esmbl_sum_results.reset_index().rename(columns={'index' : 'ImageId'})
esmbl_sum_results['ImageId'] = esmbl_sum_results['ImageId'] + 1
esmbl_sum_results.to_csv('esmbl_sum_result_kaggle.csv', index=False)


# ### Competition predictions of the ensemble based on the vote:

# In[ ]:


esmbl_vote_results = pd.DataFrame({'Label' : voting_classifier(X_comp, trained_models)})
esmbl_vote_results = esmbl_vote_results.reset_index().rename(columns={'index' : 'ImageId'})
esmbl_vote_results['ImageId'] = esmbl_vote_results['ImageId'] + 1
esmbl_vote_results.to_csv('esmbl_vote_result_kaggle.csv', index=False)

