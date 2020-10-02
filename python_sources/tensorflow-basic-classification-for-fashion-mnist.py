#!/usr/bin/env python
# coding: utf-8

# In[ ]:



import numpy as np
import os
import tensorflow as tf
import gzip
import matplotlib.pyplot as plt
print(os.listdir("../input"))


# In[ ]:


def load_data():
    with open("../input/train-labels-idx1-ubyte",'rb') as lbpath:
        y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)
    
    with open("../input/train-images-idx3-ubyte",'rb') as imgpath:
        x_train = np.frombuffer(imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28)
    
    with open("../input/t10k-labels-idx1-ubyte",'rb') as lbpath:
        y_test = np.frombuffer(lbpath.read(), np.uint8, offset=8)
        
    with open("../input/t10k-images-idx3-ubyte",'rb') as imgpath:
        x_test = np.frombuffer(imgpath.read(), np.uint8, offset=16).reshape(len(y_test), 28, 28)

    return (x_train,y_train),(x_test,y_test)


# In[ ]:


(train_images, train_labels), (test_images, test_labels) = load_data()


# In[ ]:


print(len(train_images),len(train_labels))


# In[ ]:


print(len(test_images),len(test_labels))


# In[ ]:


class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


# In[ ]:


plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)


# In[ ]:


train_images = train_images/255.0
test_images = test_images/255.0


# In[ ]:


plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.imshow(train_images[i])
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.xlabel(class_names[train_labels[i]])


# In[ ]:


model = tf.keras.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256,activation=tf.nn.relu),
    tf.keras.layers.Dense(10,activation=tf.nn.softmax)
])


# In[ ]:


model.compile(optimizer=tf.train.AdamOptimizer(),loss="sparse_categorical_crossentropy",metrics=["accuracy"])


# In[ ]:


model.fit(x=train_images,y=train_labels,epochs=5)


# In[ ]:


model.evaluate(x=test_images,y=test_labels)


# In[ ]:


predictions = model.predict(test_images)


# In[ ]:


predictions[0]


# In[ ]:


print(np.argmax(predictions[0]))
print(test_labels[0])


# In[ ]:


def plot_image(i, predictions_array, true_label, img):
    true_label = true_label[i]
    img = img[i]
    predicted_label = np.argmax(predictions_array[i])
    plt.imshow(img)
    
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'
    
    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)
    


# In[ ]:


def plot_value_array(i, predictions_array, true_label):
      predictions_array, true_label = predictions_array[i], true_label[i]
      plt.grid(False)
      plt.xticks(range(10), class_names, rotation=45)
      plt.yticks([])
      thisplot = plt.bar(range(10), predictions_array, color="#777777")
      plt.ylim([0, 1]) 
      predicted_label = np.argmax(predictions_array)

      thisplot[predicted_label].set_color('red')
      thisplot[true_label].set_color('blue')
        


# In[ ]:


i = 0
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions,  test_labels)


# In[ ]:


i = 12
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions,  test_labels)


# In[ ]:





# In[ ]:





# In[ ]:




