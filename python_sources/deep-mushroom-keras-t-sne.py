#!/usr/bin/env python
# coding: utf-8

# ## Summary
# 
# There are several "classic" models which fit well with this dataset and achieve a great accuracy. We are going to use a neural network to experiment its potential to transform raw input data into useful features to difference the two possible classes. We have implemented a neural network with Keras and obtained the values of the hidden layer for each input. We have used t-SNE to project this data in a two dimension plot where we can see the points of each class are grouped.

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import math


# In[ ]:


data = pd.read_csv("../input/mushrooms.csv")


# In[ ]:


data.head()


# In[ ]:


data.info()


# In[ ]:


data['stalk-root'].value_counts()


# More than 30% of the values of **stalk-root** are missing values

# In[ ]:


100*len(data.loc[data['stalk-root']=='?']) / sum(data['stalk-root'].value_counts())


# In[ ]:


data = data.drop('stalk-root', 1)


# We prepare the data to be used in the neural network model:

# In[ ]:


Y = pd.get_dummies(data.iloc[:,0],  drop_first=False)
X = pd.DataFrame()
for each in data.iloc[:,1:].columns:
    dummies = pd.get_dummies(data[each], prefix=each, drop_first=False)
    X = pd.concat([X, dummies], axis=1)
    


# We build the neural network with Keras:

# In[ ]:



from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from sklearn.model_selection import cross_val_score
from keras import backend as K

seed = 123456 

def create_model():
    model = Sequential()
    model.add(Dense(20, input_dim=X.shape[1], kernel_initializer='uniform', activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(2, activation='softmax'))
    sgd = SGD(lr=0.01, momentum=0.7, decay=0, nesterov=False)
    model.compile(loss='binary_crossentropy' , optimizer='sgd', metrics=['accuracy'])
    return model


# We train the model and get the associated training graphs:

# In[ ]:


model = create_model()
history = model.fit(X.values, Y.values, validation_split=0.33, epochs=200, batch_size=100, verbose=0)


# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()


# We have good accuracy although this model tens to overfit:

# In[ ]:


print("Training accuracy: %.2f%% / Validation accuracy: %.2f%%" % 
      (100*history.history['acc'][-1], 100*history.history['val_acc'][-1]))


# We are going to obtain the values of the layer previous to the output layer:

# In[ ]:


from keras import backend as K
import numpy as np

layer_of_interest=0
intermediate_tensor_function = K.function([model.layers[0].input],[model.layers[layer_of_interest].output])
intermediate_tensor = intermediate_tensor_function([X.iloc[0,:].values.reshape(1,-1)])[0]


# In[ ]:


intermediates = []
color_intermediates = []
for i in range(len(X)):
    output_class = np.argmax(Y.iloc[i,:].values)
    intermediate_tensor = intermediate_tensor_function([X.iloc[i,:].values.reshape(1,-1)])[0]
    intermediates.append(intermediate_tensor[0])
    if(output_class == 0):
        color_intermediates.append("#0000ff")
    else:
        color_intermediates.append("#ff0000")


# The penultimate layer has 10 neurons. We are going to build a t-SNE projection:

# In[ ]:


from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, random_state=0)
intermediates_tsne = tsne.fit_transform(intermediates)


# In[ ]:


plt.figure(figsize=(8, 8))
plt.scatter(x = intermediates_tsne[:,0], y=intermediates_tsne[:,1], color=color_intermediates)
plt.show()


# ## Conclusion
# 
# We have obtained a clear image where the different classes are very identificable (poison and edible mushrooms)
