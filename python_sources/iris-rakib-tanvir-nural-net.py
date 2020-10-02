#!/usr/bin/env python
# coding: utf-8

# # IRIS WITH NURAL NETWORK (TENSORFLOW & KERAS)

# In[ ]:


## import library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
try:
    get_ipython().system('pip install tensorflow-gpu')
    import tensorflow as tf
except:
    get_ipython().system('pip install tensorflow')
    import tensorflow as tf
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


## import iris data from seaborn data
iris = sns.load_dataset('iris')


# In[ ]:


iris.head()


# ## label encoding

# In[ ]:


from sklearn.preprocessing import LabelEncoder
def encode(df):
    encoder = LabelEncoder()
    target=encoder.fit_transform(df)
    return np.array(target)


# In[ ]:


target = encode(iris['species'])


# In[ ]:


target


# ## adding to the main dataframe

# In[ ]:


iris['target'] = np.array(target)


# In[ ]:


iris_working = iris.drop('species',axis=1)


# In[ ]:


iris_working.head()


# In[ ]:


corr = iris_working.corr()


# In[ ]:


sns.heatmap(corr,cmap='coolwarm',annot=True)


# ## single plot for relation with the target

# In[ ]:


corr2 = iris_working.corr()['target']


# In[ ]:


corr2.plot()


# ## train_test_split

# In[ ]:


X = iris_working.drop('target',axis=1)
y = iris_working[['target']]


# In[ ]:


X.head()


# In[ ]:


y.head()


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.25)


# ## Shape of the data

# In[ ]:


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# # Normalize (i am doing it manually. you can do it with built in function)

# In[ ]:


def normalize(df):
    result = df.copy()
    for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result


# In[ ]:


X_train = normalize(X_train)
X_test = normalize(X_test)


# In[ ]:


X_train.head()


# In[ ]:


X_test.head()


# ## Nural Net Model

# In[ ]:


def NNmodel():
    
    model = tf.keras.models.Sequential() ## making th sequental model
    ## add layer we have to change the shape with flatten
    model.add(tf.keras.layers.Flatten())
    ## perceptron 128 per layer actication function rectified linear
    model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))
    model.add(tf.keras.layers.Dropout(.25))
    model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))
    model.add(tf.keras.layers.Dropout(.25))
    model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))
    model.add(tf.keras.layers.Dropout(.25))
    model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))
    ## final activation function softmax and 3 cause data will be 3 catagory
    model.add(tf.keras.layers.Dense(3,activation=tf.nn.softmax))
    
    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    return model


# In[ ]:


model = NNmodel()


# # Train the model with 30 epoch

# In[ ]:


model.fit(np.array(X_train),np.array(y_train),epochs=30)


# ## find loss and accuracy

# In[ ]:


loss,acc=model.evaluate(X_test,y_test)


# In[ ]:


print ("LOSS : "+str(loss))
print ("ACCURACY : "+str(acc))


# ## find accuracy and loss for different epochs (This is gonna take time) (Select GPU if you have any)

# In[ ]:


loss_array=[]
accuracy_array=[]
for epoch in range(1,200):
    tmpmodel = NNmodel()
    tmpmodel.fit(np.array(X_train),np.array(y_train),epochs=epoch)
    loss,acc=tmpmodel.evaluate(X_test,y_test)
    loss_array.append(loss)
    accuracy_array.append(acc)
    
    


# In[ ]:


x = list(range(1,200))
plt.grid()
sns.lineplot(x, accuracy_array, color='green',label='accuracy', linestyle='-', markersize=12)
sns.lineplot(x, loss_array, color='red', linestyle='--',label='loss', markersize=12)

#lt.legend()


# # Predict value

# In[ ]:


model.fit(np.array(X_train),np.array(y_train),epochs=30)
predicted=model.predict(X_test)


# In[ ]:


predicted


# In[ ]:


y_pred=[]
for item in range(len(predicted)):
    y_pred.append(np.argmax(predicted[item]))    


# In[ ]:


y_pred


# In[ ]:


from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test,y_pred)


# # MEAN SQUARED ERROR (you can use in the loop too)

# In[ ]:


print ("MEAN SQUARED LOSS "+str(mse))


# In[ ]:




