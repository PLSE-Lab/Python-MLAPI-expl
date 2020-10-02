#!/usr/bin/env python
# coding: utf-8

# # 1. Data preprocessing
# 
# ## Importing the relevant libraries and datasets

# In[ ]:


import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt


# In[ ]:


train_df=pd.read_csv('../input/digit-recognizer/train.csv')
train_df.head()


# In[ ]:


test_df=pd.read_csv('../input/digit-recognizer/test.csv')
test_df.head()


# As we can see, the above data is divided into training and testing datasets. We will also create a separate validation dataset. The label section in the training dataset tells us the actual digit that the model must accurately determine.
# 
# 
# Since we are dealing with image data in the form of pixel values, we will need to preprocess our data for it to be fed into the neural network.
# 
# 
# ## Visualising the data

# Let us check how the data actually can be visualised. For this purpose, we need to reshape our 1D data into 2D data of matrix of 28X28. This will give us an idea how actually the data looks like.

# In[ ]:


temp_df=train_df.copy()
temp_df.drop('label',axis=1,inplace=True)
temp_df=np.array(temp_df).reshape(-1,28,28,1)
temp_df.shape


# In[ ]:


labels=train_df['label']
plt.imshow(temp_df[9][:,:,0])
print('The number is:{}'.format(labels[9]))


# ## Extracting data from the csv files

# After glancing through the training dataframe, we can realise that the first column named as label is the target column while everything else are inputs for the nueral net.
# 
# Let us try to separate this data.

# In[ ]:


unscaled_inputs=train_df.iloc[:,1:].values


# In[ ]:


targets=train_df.iloc[:,0].values


# ## Standardize the inputs

# In[ ]:


scaled_inputs=preprocessing.scale(unscaled_inputs)


# ## Shuffling the data

# In case the data was arranged in some particular order, we would want to remove any bias by shuffling the data completely. This will make the dataset more homogeneous in nature and prevent any undue bias in the model.

# In[ ]:


total_indices=scaled_inputs.shape[0]


# In[ ]:


print('Total amount of data in the training dataset: {}'.format(total_indices))


# Let us now shuffle all these 42000 indices to make the data homogeneous in nature.

# In[ ]:


shuffled_indices=np.arange(total_indices)


# In[ ]:


np.random.shuffle(shuffled_indices)


# In[ ]:


shuffled_indices


# As we can see, the indices have now been all shuffled.

# In[ ]:


shuffled_inputs=scaled_inputs[shuffled_indices]
shuffled_targets=targets[shuffled_indices]


# ## Splitting the dataset into train,validation and test sets

# In[ ]:


samples_count=total_indices

train_samples_count=int(0.8*samples_count)
validation_samples_count=int(0.1*samples_count)
test_samples_count=samples_count-train_samples_count-validation_samples_count


# As we can see from above few codes, we have allocated **80%** of the dataset for **training** , **10%** for **cross validation** and the remaining **10%** for **testing purpose**.

# In[ ]:


train_inputs=shuffled_inputs[:train_samples_count]
train_targets=shuffled_targets[:train_samples_count]

validation_inputs=shuffled_inputs[train_samples_count:train_samples_count+validation_samples_count]
validation_targets=shuffled_targets[train_samples_count:train_samples_count+validation_samples_count]

test_inputs=shuffled_inputs[train_samples_count+validation_samples_count:]
test_targets=shuffled_targets[train_samples_count+validation_samples_count:]


# From the above code, we have separated all the train, validation and test data and separated the inputs from the targets aswell.

# ## Saving the three datasets into .npz form to be used in further neural network

# In[ ]:


np.savez('MNIST_train',inputs=train_inputs,target=train_targets)
np.savez('MNIST_validation',inputs=validation_inputs,target=validation_targets)
np.savez('MNIST_test',inputs=test_inputs,target=test_targets)


# ## Loading the NPZ files

# In[ ]:


npz=np.load('MNIST_train.npz')
train_inputs=npz['inputs'].astype(np.float)
train_targets=npz['target'].astype(np.int)


# In[ ]:


npz=np.load('MNIST_test.npz')
test_inputs=npz['inputs'].astype(np.float)
test_targets=npz['target'].astype(np.int)


# In[ ]:


npz=np.load('MNIST_validation.npz')
validation_inputs=npz['inputs'].astype(np.float)
validation_targets=npz['target'].astype(np.int)


# # 2. Deep learning model

# ## Creating the neural network model

# From the .CSV files, it is clear that we have the values for a total of **784 pixels** for each digit. This means,it is in the form of a rank 3 tensor as **28 X 28 X 1** . 
# 
# The above situation is a problem because it is not possible to feed these values as input in simple neural networks. For convolutional neural networks, there is no issue with such a tensor input. In this case however, we need to apply the layer flattening option provided by Keras.
# 
# 
# As we have 784 pixels for each digit, so, out input nodes (or values) will be 784.
# 
# Let us take the number of hidden layers as 50
# 
# The digits may range from 0-9. Hence, the number of output values is taken as 10
# 
# 
# We are implementing **three sets of hidden layers** initially.
# The activation function we plan to use for the hidden layer is **'Relu'**
# 
# 
# For backpropogation of the output layer, the activation function used is **Softmax** 

# In[ ]:


input_size=784
output_size=10
hidden_layer_size=50

model=tf.keras.Sequential([
    #Input layer
    tf.keras.layers.Dense(input_size),
    
    #Hidden layer 1
    tf.keras.layers.Dense(hidden_layer_size,activation='relu'),
    #Hidden layer 2
    tf.keras.layers.Dense(hidden_layer_size,activation='relu'),
    #Hidden layer 3
    tf.keras.layers.Dense(hidden_layer_size,activation='relu'),
    
    #Output layer
    tf.keras.layers.Dense(output_size,activation='softmax')
])


# ## Choosing the optimizer and the loss function

# In[ ]:


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# ## Training the dataset
# 
# 
# We have an option of setting an early stopping criteria which checks the steps where the validation loss increases in the subsequent steps. We can set it to any value less than the numer of epochs. This helps to control the overfit issue. However, since we have already used the validation datasets, we will comment out the code in the model.
# 
# 
# 

# In[ ]:


NUM_EPOCHS=50
BATCH_SIZE=100

early_stopping=tf.keras.callbacks.EarlyStopping(patience=20)

model.fit(train_inputs,train_targets,
          batch_size=BATCH_SIZE,
          epochs=NUM_EPOCHS,
          callbacks=[early_stopping],
          validation_data=(validation_inputs,validation_targets),
          verbose=2,validation_steps=10)


# ## Testing the model
# 
# Initial testing on a part of the training data will be first done to check how the neural net performs.

# In[ ]:


test_loss,test_accuracy=model.evaluate(test_inputs,test_targets)


# In[ ]:


print('\n Test loss:{0:.2f} Test accuracy: {1:.2f} %'.format(test_loss,test_accuracy*100))


# In[ ]:


values=model.predict(test_inputs)


# In[ ]:


pd.DataFrame(values).head()


# As we can see, the entries which have 1 are corresponding to the digits of their column name. 

# ## Final testing on new dataset

# Once the model has been completely trained, we import the test dataset provided to us.

# In[ ]:


test_df.head()


# In[ ]:


unscaled_inputs_test=test_df.values
scaled_inputs=preprocessing.scale(unscaled_inputs_test)


# Unlike the training dataset, we shall not shuffle the testing  dataset since we need to preserve the order for submission purpose.

# In[ ]:


test_inputs=scaled_inputs


# In[ ]:


test_inputs.shape


# As we can see, we have 28000 different images with their pixel intensities.

# In[ ]:


np.savez('Final_test',inputs=test_inputs)


# In[ ]:


npz_test=np.load('Final_test.npz')
test_inputs=npz_test['inputs'].astype(float)


# In[ ]:


values_df=pd.DataFrame(model.predict(test_inputs))


# In[ ]:


values_df.head()


# In[ ]:


values_df=values_df[values_df>0.5]


# We need to sort out the labels for each entry now. We can convert every element in the dataframe into int datatype such that we have only 1s and 0s to make it more readable.

# In[ ]:


values_df


# In[ ]:


values_df[values_df>0.5]=1
values_df


# In[ ]:


values_df.fillna(0,inplace=True)
values_df.head()


# In[ ]:


values_df.size


# In[ ]:


predictions_df=pd.DataFrame(values_df[values_df==1].stack())
predictions_df


# In[ ]:


predictions_df[0].isna().value_counts()


# In[ ]:


predictions_df.shape[0]


# In[ ]:


predictions_df.drop(0,axis=1,inplace=True)


# In[ ]:


predictions_df


# The neural net could predict for the above number of cases. Rest could not be identified.
# 
# Let us organise the dataframe properly.

# In[ ]:


predictions_df.reset_index(inplace=True)


# In[ ]:


predictions_df.rename(columns={'level_1':'Label'},inplace=True)
predictions_df.head()


# In[ ]:


image_id=pd.DataFrame(np.arange(0,28000),columns=['ImageId'])
image_id['ID']=image_id['ImageId']
image_id.head()


# In[ ]:



predictions_df.rename(columns={'level_0':'ImageId'},inplace=True)


# In[ ]:


predictions_df.head()


# In[ ]:


final_preds=predictions_df.copy()


# In[ ]:


final_preds=pd.merge_ordered(final_preds,image_id,on='ImageId',fill_method='None')


# In[ ]:


final_preds['Label'].isna().value_counts()


# Sadly, we could not recognize about few test cases. Let us see the various results amongst the recognized numbers.

# In[ ]:


final_preds['Label'].value_counts()


# In a brute and inaccurate manner, we will fill the null values with mode of the Label column.

# In[ ]:


final_preds['Label'].fillna(final_preds['Label'].mode()[0],inplace=True)


# In[ ]:


final_preds=final_preds.astype(int)


# In[ ]:


final_preds['ImageId']=final_preds['ImageId']+1


# In[ ]:


final_preds.drop('ID',axis=1,inplace=True)
final_preds.head()


# In[ ]:


final_preds['Label'].unique()


# In[ ]:


sample_sub=pd.read_csv('../input/digit-recognizer/sample_submission.csv')
sample_sub.dtypes


# In[ ]:


final_preds['ImageId']=sample_sub['ImageId']


# In[ ]:


final_preds.to_csv('Final_submission.csv',index=False)


# In[ ]:




