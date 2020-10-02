#!/usr/bin/env python
# coding: utf-8

# * **1. Data preparation**
#     * 1.1 Load train data
#     * 1.2 Check for null and missing values
#     * 1.3 Normalization
#     * 1.4 Reshape
#     * 1.5 Label encoding
#     * 1.6 Split training and valdiation set
# * **2. CNN**
#     * 2.1 Define the model
#     * 2.2 Set the optimizer and annealer
# * **3. Evaluate the model**
#     * 3.1 Training and validation curves
#     * 3.2 Confusion matrix
# * **4. Prediction and submition**
#     * 4.1 Load test data
#     * 4.1 Predict and Submit results

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[ ]:


#Load training data
df= pd.read_csv("../input/train.csv")
Y= df["label"]
X= df.drop("label",axis=1)


# In[ ]:


# One Hot Encoding for Y
from keras.utils.np_utils import to_categorical
Y_cat =to_categorical(Y)
Y_cat.shape


# In[ ]:


# No. of digits
sns.countplot(Y)
print(Y.value_counts())


# In[ ]:


#>> Check for null value
X.isnull().any().any()

#>> Normaliztion
# X values are ranging from 0 to 255    
X.describe()
X =X /255.0


# In[ ]:


#>> Reshaping Data
# keras require a shape 4D tensor
#[Batch Size, Height of Image, Width of Image, No. of Color Channels]

X=X.values.reshape(-1,28,28,1)
X.shape


# In[ ]:


# Change Value of num to plot Different Images
num =4
plt.imshow(X[num][:,:,0],cmap="gray")


# In[ ]:


# Split the train and the validation set for the fitting
from sklearn.model_selection import train_test_split

X_train, X_val, Y_train, Y_val = train_test_split(X, Y_cat, test_size = 0.1)


# In[ ]:


# Importing model config
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten
from keras.optimizers import RMSprop


# In[ ]:


model= Sequential()

model.add(Conv2D(32,(4,4),padding="Same",activation="relu", input_shape=(28,28,1)  ))
model.add(Conv2D(32, (4,4), padding="Same", activation="relu"))
model.add(MaxPool2D())

model.add(Dropout(0.2))
model.add(Conv2D(32,(3,3),padding="Same",activation="relu" ))
model.add(Conv2D(32, (3,3), padding="Same", activation="relu"))
model.add(MaxPool2D(strides=(2,2)))

model.add(Flatten())
model.add(Dense(256, activation="relu"))
model.add(Dropout(0.4))
model.add(Dense(10, activation="softmax"))

model.compile(RMSprop(lr=0.001),"categorical_crossentropy",metrics=["accuracy"]   )
model.summary()


# 1) **1 \* 32 \* 16 (Weights) + 32(bias) = 544 learning parameters**                                 
#     Convolution layer => with 32 nodes/no. of filters; having size of (4\*4) each; Input channel/nodes from previous is 1
# 
#                                     
# 2) **32 \* 32 \* 16 (Weights) + 32(bias) = 16416 learning parameters**                   
#     Convolution layer; with 32 nodes/no. of filters; having size of (4\*4) each; Input channel/nodes from previous layer are 32
#              
# 3) MaxPooling layer of size (2,2) it decreases height and width by 2. and it does not have any parameters for learning.
# 
# 4) **32 \* 32 \* 9 (Weights) + 32(bias) = 9248 learning parameters**                                 
#     Convolution layer => with 32 nodes/no. of filters; having size of (3\*3) each; Input channel/nodes from previous is 32
# 
#                                     
# 5) **32 \* 32 \* 9 (Weights) + 32(bias) = 9248 learning parameters**                   
#     Convolution layer; with 32 nodes/no. of filters; having size of (3\*3) each; Input channel/nodes from previous layer are 32
#              
# 6) MaxPooling layer of size (2,2) it decreases height and width by 2. and it does not have any parameters for learning.
# 
# 7) Flatten, It reshape(-1) in a 1D , ** 7 \* 7 \* 32 **
# 
# 8) Dense, Fully Connected layer with 256 nodes, **1568 * 256 (Weights) + 256 bias =401664 learning parameters**
# 
# 9) Dropout layer with 40% units to prevent overfitting.
# 
# 10) Dense, Fully Connected layer with 10 nodes [0,1...,9] for output.
# 
# 

# In[ ]:


# Set a learning rate annealer
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
lr_reduction= ReduceLROnPlateau(monitor='val_acc',factor=0.4, min_lr=0.00001,patience=1,verbose=1)


# In[ ]:


# Training model with lr reduction and Early stopping
history =model.fit(X_train,Y_train,batch_size=64, epochs=15, validation_split=0.1,verbose=1,
         callbacks=[lr_reduction])


# In[ ]:


historydf=pd.DataFrame(history.history, index=history.epoch)
historydf.plot()


# In[ ]:


# Training model with 5 more epochs reduction and Early stopping
history =model.fit(X_train,Y_train,batch_size=64, epochs=5, validation_split=0.1,verbose=1,
         callbacks=[lr_reduction])


# In[ ]:


## Saving model
from keras.models import load_model
model.save("mnist_5.h5")
#model= load_model("mnist_5.h5")


# In[ ]:


# Predict from model on Validation data
Y_pred = model.predict(X_val)

# argmax for predicted value 
y_predicted= np.argmax(Y_pred,axis = 1) 

# argmax for true values
y_true = np.argmax(Y_val,axis = 1)


# In[ ]:


# Validation data Score
from sklearn.metrics import accuracy_score
accuracy_score(y_true,y_predicted)


# In[ ]:


# Plot confusion matrix
from sklearn.metrics import confusion_matrix
import itertools

def plot_confusion_matrix(cm, rangee):

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xticks(rangee, rangee, rotation=40)
    plt.yticks(rangee, rangee)

   
    
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center")
                

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# confusion matrix
conf_matrix = confusion_matrix(y_true, y_predicted) 
#print(conf_matrix)

# plot the confusion matrix
plot_confusion_matrix(conf_matrix, rangee = range(10)) 


# In[ ]:


# Finding wrong predicted images
error = y_true - y_predicted !=0

Y_true_err = y_true[error]
Y_pred_err = y_predicted[error]
X_val_err =  X_val[error]
print("Wrong preicted Y true values",Y_true_err)


# ## Try different values of num, for Wrong Predicted images

# In[ ]:


# 0 for first wrong predicted image
num = 0

plt.imshow(X_val_err[num][:,:,0],cmap="gray")
plt.show()
print("True Value: ",Y_true_err[num] )
print("Predicted Value: ",Y_pred_err[num])


# In[ ]:


# Retraining model on Validation data 

history2 =model.fit(X_val,Y_val,batch_size=32, epochs=2, validation_split=0.1,verbose=1,
         callbacks=[lr_reduction,EarlyStopping(monitor='loss', patience=2)])


# In[ ]:


# Again, Checking Accuracy score on Validation data

y_pred2=model.predict(X_train)
y_predd2=np.argmax(y_pred2,axis=1)
y_truee2 = np.argmax(Y_train,axis=1)

accuracy_score(y_truee2,y_predd2)


# In[ ]:


# Importing test.csv
df_tst = pd.read_csv("../input/test.csv")

# Reshaping test data
df_tst=df_tst.values.reshape(-1,28,28,1)

# Predicting with model
df_tst_result=model.predict(df_tst)
y_result=np.argmax(df_tst_result,axis=1)


# In[ ]:


range_n=np.arange(1,len(y_result) + 1 )

final_results=pd.concat([pd.DataFrame(range_n) , pd.DataFrame(y_result)],axis=1)
final_results.columns=(["ImageId","Label"])

final_results.to_csv("result.csv")
# Upload "result.csv"

