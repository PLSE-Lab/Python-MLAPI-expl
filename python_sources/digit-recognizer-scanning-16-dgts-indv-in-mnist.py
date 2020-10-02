#!/usr/bin/env python
# coding: utf-8

# >1) FIRST OF ALL MAKING THE CONVOLUTIONAL NEURAL NETWORK FOR THIS DATASET AND PREDICTING THE TEST SET RESULT.
# 
# **SECOND AND MOST IMPORTANT**
# 
# >2) MNIST ORIGINAL DATASET "SINGLE IMAGE" CONSISTS OF 16 DIGITS. I CAME ACROSS THE SITUATION WHERE I HAD TO PREDICT THOSE 16 DIGITS IN SINGLE IMAGE AS A REQUIREMENT. SO AT THE END OF KERNEL ,IS SOLUTION FOR THAT. I HAVE USED PAINT FOR GETTING INFORMATION OF PIXEL VALUES FOR CROPPING 16 DIGITS INDIVIDUALLY IN THAT IMAGE.

# # IMPLEMENTATION USING CNN

# ### Importing the libraries

# In[ ]:


import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator


# In[ ]:


import numpy as np


# In[ ]:


tf.__version__


# ## Part 1 - Data Preprocessing

# ### Preprocessing the Training set

# In[ ]:


train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
training_set = train_datagen.flow_from_directory('../input/mnistasjpg/trainingSet/trainingSet',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')


# ### Preprocessing the Test set

# In[ ]:


test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory('../input/mnistasjpg/testSet',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'input')


# ## Part 2 - Building the CNN

# ### Initialising the CNN

# In[ ]:


cnn = tf.keras.models.Sequential()


# ### Step 1 - Convolution

# In[ ]:


cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))


# ### Step 2 - Pooling

# In[ ]:


cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))


# ### Adding a second convolutional layer

# In[ ]:


cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))


# ### Step 3 - Flattening

# In[ ]:


cnn.add(tf.keras.layers.Flatten())


# ### Step 4 - Full Connection

# In[ ]:


cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))


# ### Step 5 - Output Layer

# In[ ]:


cnn.add(tf.keras.layers.Dense(units=10, activation='softmax'))


# ## Part 3 - Training the CNN

# ### Compiling the CNN

# In[ ]:


cnn.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])


# ### Training the CNN on the Training set and evaluating it on the Test set

# In[ ]:


cnn.fit(x = training_set, epochs = 10)


# ## Part 4 - Making a single prediction from test set

# In[ ]:


import numpy as np
from PIL import Image
from keras.preprocessing import image
test_image = image.load_img('../input/mnistasjpg/testSet/testSet/img_10.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = cnn.predict(test_image)


# In[ ]:


#test_image
#test_image.show()


# In[ ]:


training_set.class_indices


# In[ ]:


if result[0][0] == 1:
    prediction = 0
elif result[0][1] == 1:
    prediction = 1
elif result[0][2] == 1:
    prediction = 2
elif result[0][3] == 1:
    prediction = 3
elif result[0][4] == 1:
    prediction = 4
elif result[0][5] == 1:
    prediction = 5
elif result[0][6] == 1:
    prediction = 6
elif result[0][7] == 1:
    prediction = 7
elif result[0][8] == 1:
    prediction = 8
elif result[0][9] == 1:
    prediction = 9

print(prediction)


# 

# #This is Single Digit prediction out of 16 digits in mnist original dataset image. One sample of MNIST Original dataset is shown below. And i am predicting **1st digit** of below image.

# ![samples_0000.png](attachment:samples_0000.png)

# In[ ]:


from PIL import Image
test_image = Image.open('../input/mnist-digits/mnist_digits/samples_0000.png').crop((0, 0, 95, 95))
coordinate = x, y = 0, 0
print(test_image.getpixel(coordinate))
test_image = test_image.convert('RGB')
test_image.save('/kaggle/working/img_new1.png')
test_image = image.load_img('/kaggle/working/img_new1.png', target_size = (64, 64))
test_image.save('/kaggle/working/img_new1.png')
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = cnn.predict(test_image)


# In[ ]:


if result[0][0] == 1:
    prediction = 0
elif result[0][1] == 1:
    prediction = 1
elif result[0][2] == 1:
    prediction = 2
elif result[0][3] == 1:
    prediction = 3
elif result[0][4] == 1:
    prediction = 4
elif result[0][5] == 1:
    prediction = 5
elif result[0][6] == 1:
    prediction = 6
elif result[0][7] == 1:
    prediction = 7
elif result[0][8] == 1:
    prediction = 8
elif result[0][9] == 1:
    prediction = 9

print(prediction)


# In[ ]:





# **MNIST ORIGINAL DATASET  "SINGLE IMAGE"  CONSISTS OF 16 DIGITS. I CAME ACROSS THE SITUATION WHERE I WANT TO PREDICT THOSE 16 DIGITS IN SINGLE IMAGE AS A REQUIREMENT. SO HERE IS BELOW SOLUTION FOR THAT. I HAVE USED PAINT FOR GETTING INFORMATION OF PIXEL VALUES FOR CROPPING SINGLE DIGITS IN THAT IMAGE.**
# 
# -**PREDICTING 16 DIGITS INDIVIDUALLY IN SINGLE IMAGE OF MNIST ORIGINAL DATASET. ONE SAMPLE IS SHOWN BELOW. I HAVE ATTACHED THE MNIST SAMPLE IMAGES FOR THIS EXERCISE. HOPE THAT HELPS SOME WAY.**
# 
# 

# ![samples_0000.png](attachment:samples_0000.png)

# In[ ]:


lst = []
rs = []
x_cursor = 73
y_cursor = 69
for i in range(0,4):
    for j in range(0, 4):
        test_image = Image.open('../input/gan-generated/samples/samples_0000.png').crop((x_cursor, y_cursor, x_cursor + 95, y_cursor + 95))
        test_image = test_image.convert('RGB')
        #test_image.show()
        test_image.save('/kaggle/working/img_new1.jpg')
        test_image = image.load_img('/kaggle/working/img_new1.jpg', target_size = (64, 64))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = cnn.predict(test_image)
        rs = cnn.predict(test_image/255)
        if result[0][0] == 1:
            prediction = 0
        elif result[0][1] == 1:
            prediction = 1
        elif result[0][2] == 1:
            prediction = 2
        elif result[0][3] == 1:
            prediction = 3
        elif result[0][4] == 1:
            prediction = 4
        elif result[0][5] == 1:
            prediction = 5
        elif result[0][6] == 1:
            prediction = 6
        elif result[0][7] == 1:
            prediction = 7
        elif result[0][8] == 1:
            prediction = 8
        elif result[0][9] == 1:
            prediction = 9    
       # print(rs)    
        if rs.all() < 0.7:
            lst.append(10)
        else:    
            lst.append(prediction)
        x_cursor += 115

    y_cursor += 115
    x_cursor = 72
        


# # Below is output of Sample image passed above for prediction

# In[ ]:


np.reshape(lst, (-1, 4))


# #-PREDICTING 16 DIGITS INDIVIDUALLY IN SINGLE IMAGE OF MNIST ORIGINAL DATASET **TRAINING SET SAMPLE**. ONE SAMPLE IS SHOWN BELOW. I HAVE ATTACHED THE MNIST SAMPLE IMAGES FOR THIS EXERCISE. HOPE THAT HELPS SOME WAY

# ![img_0.png](attachment:img_0.png)

# In[ ]:


lst1 = []
x_cursor = 37
y_cursor = 35
for i in range(0,4):
    for j in range(0, 4):
        test_image = Image.open('../input/kaggle-mnist-digits/mnist_digits_kaggle/img_0.png').crop((x_cursor, y_cursor, x_cursor + 48, y_cursor + 48))
        test_image = test_image.convert('RGB')
        test_image.show()
        test_image.save('/kaggle/working/img_new1.jpg')
        test_image = image.load_img('/kaggle/working/img_new1.jpg', target_size = (64, 64))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = cnn.predict(test_image)
        if result[0][0] == 1:
            prediction = 0
        elif result[0][1] == 1:
            prediction = 1
        elif result[0][2] == 1:
            prediction = 2
        elif result[0][3] == 1:
            prediction = 3
        elif result[0][4] == 1:
            prediction = 4
        elif result[0][5] == 1:
            prediction = 5
        elif result[0][6] == 1:
            prediction = 6
        elif result[0][7] == 1:
            prediction = 7
        elif result[0][8] == 1:
            prediction = 8
        elif result[0][9] == 1:
            prediction = 9
        
        lst1.append(prediction)
        
        x_cursor += 58
    y_cursor += 56
    x_cursor = 37
        


# # Below is output of Sample image passed above for prediction

# In[ ]:


np.reshape(lst1, (-1, 4))


# # Prediction for all testSet Images

# In[ ]:


'''lst_test = []
import os
for filename in os.listdir('../input/mnistasjpg/testSet/testSet'):
    img_path = os.path.join('../input/mnistasjpg/testSet/testSet', filename)
    
    
    test_image = image.load_img(img_path, target_size = (64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = cnn.predict(test_image)
    if result[0][0] == 1:
        prediction = 0
    elif result[0][1] == 1:
        prediction = 1
    elif result[0][2] == 1:
        prediction = 2
    elif result[0][3] == 1:
        prediction = 3
    elif result[0][4] == 1:
        prediction = 4
    elif result[0][5] == 1:
        prediction = 5
    elif result[0][6] == 1:
        prediction = 6
    elif result[0][7] == 1:
        prediction = 7
    elif result[0][8] == 1:
        prediction = 8
    elif result[0][9] == 1:
        prediction = 9
        
    lst_test.append(prediction)'''
        
    
        


    
    


# In[ ]:


#lst_test


# # Prediction of all images in mnist original dataset folder attached below.

# ![img_2.png](attachment:img_2.png)![img_3.png](attachment:img_3.png)![img_0.png](attachment:img_0.png)

# In[ ]:


test_list = []
import os
for filename in os.listdir('../input/kaggle-mnist-digits/mnist_digits_kaggle'):
    img_path = os.path.join('../input/kaggle-mnist-digits/mnist_digits_kaggle', filename)
    
    lst1 = []
    x_cursor = 37
    y_cursor = 35
    for i in range(0,4):
        for j in range(0, 4):
            test_image = Image.open(img_path).crop((x_cursor, y_cursor, x_cursor + 48, y_cursor + 48))
            test_image = test_image.convert('RGB')
            test_image.show()
            test_image.save('/kaggle/working/img_new1.jpg')
            test_image = image.load_img('/kaggle/working/img_new1.jpg', target_size = (64, 64))
            test_image = image.img_to_array(test_image)
            test_image = np.expand_dims(test_image, axis = 0)
            result = cnn.predict(test_image)
            if result[0][0] == 1:
                prediction = 0
            elif result[0][1] == 1:
                prediction = 1
            elif result[0][2] == 1:
                prediction = 2
            elif result[0][3] == 1:
                prediction = 3
            elif result[0][4] == 1:
                prediction = 4
            elif result[0][5] == 1:
                prediction = 5
            elif result[0][6] == 1:
                prediction = 6
            elif result[0][7] == 1:
                prediction = 7
            elif result[0][8] == 1:
                prediction = 8
            elif result[0][9] == 1:
                prediction = 9
        
            lst1.append(prediction)
        
            x_cursor += 58
    
        y_cursor += 56
        x_cursor = 37
    lst1 = np.reshape(lst1, (-1, 4))
    test_list.append(lst1)
    lst1 = []
    
        


    
    


# # Prediction of images in order attached above

# In[ ]:


#print(test_list)
np.reshape(test_list, (-1, 4))


# #I had GAN Generated output which were produced in onw of my project. Predicting the digits in these gan generated images.
# 

# In[ ]:


os.listdir('../input/gan-generated/samples')
#os.path.join('../input/gan-generated/samples', 'samples_0057.png')


# # Prediction of digits in images are in order shown in above output

# In[ ]:


test_list_gan = []
import os
for filename in os.listdir('../input/gan-generated/samples'):
    img_path = os.path.join('../input/gan-generated/samples', filename)
    
    
    lst_gan = []
    x_cursor = 72
    y_cursor = 70
    for i in range(0,4):
        for j in range(0, 4):
            test_image = Image.open(img_path).crop((x_cursor, y_cursor, x_cursor + 95, y_cursor + 95))
            test_image = test_image.convert('RGB')
            #test_image.show()
            test_image.save('/kaggle/working/img_new1.jpg')
            test_image = image.load_img('/kaggle/working/img_new1.jpg', target_size = (64, 64))
            test_image = image.img_to_array(test_image)
            test_image = np.expand_dims(test_image, axis = 0)
            result = cnn.predict(test_image)
            if result[0][0] == 1:
                prediction = 0
            elif result[0][1] == 1:
                prediction = 1
            elif result[0][2] == 1:
                prediction = 2
            elif result[0][3] == 1:
                prediction = 3
            elif result[0][4] == 1:
                prediction = 4
            elif result[0][5] == 1:
                prediction = 5
            elif result[0][6] == 1:
                prediction = 6
            elif result[0][7] == 1:
                prediction = 7
            elif result[0][8] == 1:
                prediction = 8
            elif result[0][9] == 1:
                prediction = 9
        
            lst_gan.append(prediction)
        
            x_cursor += 115

        y_cursor += 115
        x_cursor = 72
    lst_gan = np.reshape(lst_gan, (-1, 4))
    test_list_gan.append(lst_gan)
        
    


# In[ ]:


print(test_list_gan)


# # READ A NUMBER PLATE

# In[ ]:


from PIL import Image
test_image = Image.open('../input/read-plate/plateread.png')
x_len, y_len = test_image.size


# # Calculating x_cor_values

# In[ ]:


test_image.getpixel((0, 0))!=(0, 0, 0)


# In[ ]:


x_before = []
x_after = [0, ]
k = 0
i = 0 
i_plus = 0
while i < x_len:
    #print(i)
    j=0
    flag = False
    i_plus = i + 1
    while flag==False and j < y_len:
        if test_image.getpixel((i, j))!=(0, 0, 0):
            x_before.append(i)
            flag = True
            
            flag1 = False
            k = i
            while k < x_len:
                #print(k)
                j = 0
                flag1 = True
                while flag1 == True and j < y_len:
                    #
                    #print(j)
                    if test_image.getpixel((k, j)) != (0, 0, 0):
                        flag1 = False
                    
                    j += 1
                
                if flag1 == True:
                    x_after.append(k)
                    i_plus = k
                    #print(i_plus)
                    k = x_len
                    
                k += 1
            
            
        j += 1
        #print(i, j)
    
    i = i_plus
    

    
x_before.append(x_len)    


# In[ ]:


x_before


# In[ ]:


x_after


# In[ ]:


x_cor_values = []
for i in range(len(x_before)):
    x_cor_values.append(int((x_before[i] + x_after[i])/2))


# In[ ]:


x_cor_values


# # calculating y_cor_values

# In[ ]:


y_before = []
y_after = []
k = 0
i = 0 
i_plus = 0

for i in range(len(x_before)-1):
    flag = True
    l=0
    while flag and l < y_len:
        j = x_cor_values[i]
        flag = True
        while flag and j < x_cor_values[i+1]:
            if test_image.getpixel((j, l)) != (0, 0, 0):
                flag=False
                
                y_before.append(l/2)
                
                k = l
                while k < y_len:
                    flag1 = True
                    j = x_cor_values[i]
                    while flag1 and j < x_cor_values[i+1]:
                        if test_image.getpixel((j, k)) != (0, 0, 0):
                            flag1 = False
                            
                        j += 1    
                    
                    if flag1 == True:
                        y_after.append((k + y_len)/2)
                        k = y_len
                
                    k += 1
                    
                
            j += 1
            
        l += 1
            
                


# In[ ]:


y_before


# In[ ]:


y_after


# # Final Evaluation of Image

# In[ ]:


lst = []
import numpy as np
from keras.preprocessing import image
for i in range(len(x_before)-1):
    test_image = Image.open('../input/read-plate/plateread.png').crop((x_cor_values[i], y_before[i],x_cor_values[i+1], y_after[i]))
    test_image = test_image.convert('RGB')
    test_image.save('/kaggle/working/img_new1.png')
    test_image = image.load_img('/kaggle/working/img_new1.png', target_size = (64, 64))
    test_image.save('/kaggle/working/img_new1.png')
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = cnn.predict(test_image)
    
    if result[0][0] == 1:
        prediction = 0
    elif result[0][1] == 1:
        prediction = 1
    elif result[0][2] == 1:
        prediction = 2
    elif result[0][3] == 1:
        prediction = 3
    elif result[0][4] == 1:
        prediction = 4
    elif result[0][5] == 1:
        prediction = 5
    elif result[0][6] == 1:
        prediction = 6
    elif result[0][7] == 1:
        prediction = 7
    elif result[0][8] == 1:
        prediction = 8
    elif result[0][9] == 1:
        prediction = 9

    print(prediction)
    lst.append(prediction)
    


# In[ ]:


lst

