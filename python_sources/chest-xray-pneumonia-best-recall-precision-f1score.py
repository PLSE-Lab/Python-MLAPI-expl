#!/usr/bin/env python
# coding: utf-8

# Hello everyone. I developed a medical software before so I am a little familiar with DICOM images and xrays etc. Even tho this particular project doesn't require much knowledge about DICOM since the input is coming in JPEG format, I was still interested in it. Simply because there is a lot of use for ML models in healthcare moving forward. 
# 
# Since it's already been awhile this dataset is up. I first reviewed what others accomplished before me. Most notable was a research paper published on Cell Magazine - [Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning](https://www.cell.com/cell/pdf/S0092-8674%2818%2930154-5.pdf) In the second part of the article, the researchers said using InceptionV3 and Transfer Learning they achieved an **Accuracy of 92.8%, with a Sensitivity/Recall of 93.2% and a Specificity/Precision of 90.1% using the same Pneumonia dataset** we have here. I wanted to see if I can beat that since I had a few ideas to try, and I did surpass all these metrics on the 624 sample test images, not the 8 images in validation set - which would be trivial to attain 100%.  
# 
# My current metrics are :   
# 
# Base Training : **Accuracy: 93.59%    Recall: 96.92%    Precision: 93%    F1: 0.95**  
# Fine Tuning : **Accuracy: 93.75%      Recall: 98.97%    Precision: 92%    F1: 0.95**
# 
# ![v_best_model_28x12.png](attachment:v_best_model_28x12.png) 
# 
# On an attempt to finetune the network, I was able to achive a similar performance with a more smooth distribution between Recall / Precision since there is a trade-off between the two. Nevertheless, the accuracy and F1-score on average were the same for both models. So Let me explain how I got these... 
# 
# ![v_best_model_35x4.png](attachment:v_best_model_35x4.png)

# When we first look at the dataset, we understand that we are working with images labeled either having or not having a pneumonia. Having said that, out of 5,216 training images, we have 1,341 Normal, and 2,875 Pneumonia. There is an imbalance on these two sets of samples but we will discuss about that later.

# In[ ]:


import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import preprocess_input
from keras.applications.inception_v3 import preprocess_input as incep_preprocess_input

batch_size = 128
target_size = (299, 299)
train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
                                   shear_range=0.2, zoom_range=0.2,
                                   horizontal_flip=True, fill_mode='nearest')

train_generator = train_datagen.flow_from_directory('/kaggle/input/chest-xray-pneumonia/chest_xray/chest_xray/train/',
                                                    target_size=target_size, color_mode='rgb',
                                                    batch_size=batch_size, class_mode='binary',
                                                    shuffle=True, seed=42)

val_datagen = ImageDataGenerator(preprocessing_function=incep_preprocess_input)

val_generator = val_datagen.flow_from_directory('/kaggle/input/chest-xray-pneumonia/chest_xray/chest_xray/test/',
                                                target_size=target_size, color_mode="rgb",
                                                batch_size=batch_size, shuffle=False, class_mode="binary")


step_size_train = train_generator.n // train_generator.batch_size
step_size_valid = val_generator.n // val_generator.batch_size

df = pd.DataFrame({'data':train_generator.classes})
no_pne = int(df[df.data==train_generator.class_indices['NORMAL']].count())
yes_pne = int(df[df.data==train_generator.class_indices['PNEUMONIA']].count())
print("Normal:{}  Pneumonia:{}".format(no_pne, yes_pne));


# Many people already pointed out that using a model trained on **ImageNet** dataset such as **Inception**, **VGG** or **ResNet** would be a good starting point than training your own neural network from scratch. You can do that, but 5,216 images are not that much to train an entire network off off to a point where it generalize enough. The paper I mentioned at the beginning also trained on InceptionV3. 
# So I first set up a model on top of InceptionV3, simply excluding the top layer, and adding my own fully connected layer on top of that to see what kind of performance we are getting... After a few experimentation, i settled for a simple 2 dense layers, and a **BatchNormalization** and a **DropOut** in between the two. As you see I am setting all the layer of Inception (base_model) trainable property to false, we do not want to train and mess their weights. We only want to train our final layers that we added. However, more on this later...

# In[ ]:


from keras.applications import InceptionV3,VGG16
from keras.models import Model
from keras.layers import Flatten, Dense, BatchNormalization, Dropout
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint

print("Using InceptionV3")
base_model = InceptionV3(weights=None, input_shape=(299,299, 3), include_top=False)
base_model.load_weights('/kaggle/input/weights/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5')

x = base_model.output
x = Flatten()(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.33)(x)
x = BatchNormalization()(x)
output = Dense(1, activation='sigmoid')(x)

for layer in base_model.layers:
    layer.trainable = False
    
model = Model(inputs=base_model.input, outputs=output)

model.compile(loss='binary_crossentropy', optimizer=RMSprop(learning_rate=0.0001), metrics=['accuracy'])

model.summary()


# After we set up the model, I set up 3 checkpoints , based on trainin accuracy, validation accuracy and validation loss. I wanted to save the best of each and compare the 3 against each other. I commented out to kick off the training so it won't run on the screen now.

# In[ ]:


chkpt1 = ModelCheckpoint(filepath="best_model_acc.hd5", monitor='accuracy', verbose=1, save_best_only=True, save_weights_only=False)
chkpt2 = ModelCheckpoint(filepath="best_model_val_acc.hd5", monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=False)
chkpt3 = ModelCheckpoint(filepath="best_model_val_loss.hd5", monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False)

#history = model.fit_generator(generator=train_generator,
#                    steps_per_epoch=step_size_train,
#                    validation_data=val_generator,
#                    validation_steps=step_size_valid,
#                    callbacks=[chkpt1,chkpt2,chkpt3],
#                    epochs=10, verbose=1)


# We also need to calculate other metrics than accuracy. Why? Because let's think about it. We do not ONLY want to know how much we guessed correct, we also need to know which patients were normal and we predicted sick (which is not as bad, it may mean another doctor check), and which patients were sick but we predicted them to be normal (which is much dangerous than first case, as patient may be discharged and the sickness can cause death). In these binary cases a simple confusion matrix gives us a better insight. With InceptionV3 I was able to get a good Recall but the Precision was not good. There were a lot of false positives (normal patients predicted to be sick). I think that's partly because the network was biased towards marking everything as sick therefore getting high precision on pneumonia cases, but not in normal cases. 
# 
# ![best_inc.png](attachment:best_inc.png)
# 
# 
# 
# I would like to point a discussion here though. Even though InceptionV3 is a great model for ImageNet dataset and overperformed VGG16. There is an architectural difference that makes using one or the other very different for our case. Remember the reason we said we want to start with a base model is because we do not have enough training data (5,216 images). But what does that mean? How much data we need? The data we need depends on the complexity of our network. In other words, the more number of weights(parameters) our network has to adjust/learn, the more data we need for them to distinguish without overfitting (like almost memorizing the training data, but not drawing any general conclusions). 
# 
# VGG16 is 16 layers deep where as InceptionV3 is 46 layers deep. Not only that, but the number of neurons and other parameters are also different between the two. When we print the summary of our model using InceptionV3, we see that it has about 8.3 million parameters compared to VGG16 based model having a mere 524,000 trainable parameters. 
# 
# ![inc-trainable.png](attachment:inc-trainable.png)
# 
# ![vgg-trainable.png](attachment:vgg-trainable.png)
# 

# Now considering we have 5,216 images, I believe vgg based model would do better here so I set up another network, the final layers being same, just replacing the base model to VGG16

# In[ ]:


base_model = VGG16(weights=None, input_shape=(150, 150, 3), include_top=False)
base_model.load_weights('/kaggle/input/weights/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')

x = base_model.output
x = Flatten()(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.33)(x)
x = BatchNormalization()(x)
output = Dense(1, activation='sigmoid')(x)

for layer in base_model.layers:
    layer.trainable = False
    
model = Model(inputs=base_model.input, outputs=output)

model.compile(loss='binary_crossentropy', optimizer=RMSprop(learning_rate=0.0001), metrics=['accuracy'])

model.summary()


# After training on VGG16 I immediately see some improvements, especially on false positives (normal patients predicted to be sick) which you saw at the beginning of the kernel.
# 
# ![Screen%20Shot%202019-10-22%20at%209.16.16%20PM.png](attachment:Screen%20Shot%202019-10-22%20at%209.16.16%20PM.png)
# 
# That's 93.5% Accuracy, 96.9% Recall and 93% Precision which is the best on the Test data set of 624 images as of now as far as I know.
# 
# 

# # Fine-Tuning
# So is that it? Are we done? Can we fine tune this model? The article mentioned about a similar attemt. The authors have trained a based network off off InceptionV3. Then in second stage, took that model as base (including inceptionv3 layers), made ALL layers trainable (including the ones from InceptionV3) and found out the performance was worsened. Couple of other Kagglers also mentioned that in their kernels if i am not mistaken. Here I would like to remind you again the discussion about the number of parameters of our model and the size of training data...
# 
# With InceptionV3 initial layers non-trainable, we were already looking at 8.3 million parameters (we only added 64 neurons, but it has to connect to each output fo the final layer of the InceptionV3) If you scroll back up a little, the inceptionv3 model had another 21 million non-trainable parameters. So if were to unfreeze all the layers, we are looking at training now around 30 million parameters. If we didn't have enough data for 8 million parameters, we surely don't have enough data for 30 million parameters. That would be 15 million paramaters if we were to unfreeze and train all layer of VGG16 based model. (see the screenshot above)
# 
# How about, maybe, unfreezing not ALL the layers, but just the last layer of the VGG16(since that performed better anyways) that our fully dense layer is connected to ??? Basically we are keeping all the lower layers which are known for learning more basic features/shapes and colors and letting the last layer learn from our dataset. In that case, using VGG we are looking at 2.8 million parameters
# 

# In[ ]:


for layer in base_model.layers:
    if layer.name != 'block5_conv3':
        layer.trainable = False
    else:
        layer.trainable = True
        print("Setting 'block5_conv3' trainable")

for layer in model.layers:
    print("{} {}".format(layer.name, layer.trainable))

model.summary()


# ![Screen%20Shot%202019-10-22%20at%209.29.51%20PM.png](attachment:Screen%20Shot%202019-10-22%20at%209.29.51%20PM.png)

# That's a lot more than 500K params we trained initially, but hey, its still less than 8 million of basic InceptionV3 based model. Now setting a small learning rate on top of this model.

# In[ ]:


model.compile(loss='binary_crossentropy', optimizer=RMSprop(learning_rate=0.0001), metrics=['accuracy'])
#Load first stage weights to fine tune off off
#model.load_weights("vgg_best_model.hd5")


# This ran for awhile as well, the overall occuracy and F1 score (combination of Recall & Precision) didn't improve really. But at the same accuracy and F1 score, it was a different distribution between how many false negatives and false positives. But the total was almost the same 39 vs 40 wrong out of 624 test images.

# ![v_best_model_35x4.png](attachment:v_best_model_35x4.png)

# # Data Imbalance
# Let's talk about the imbalance in the data itself due to having 3,875 pneumonia images, but only 1,341 normal cases and what it means for us.
# 
# I read in a few other kernels that some researchers tried to address this by augmentation, meaning transforming normal images a little bit (rotation of a few degrees, horizontal flip etc) and saving them as well, and bringing the training image split to a more equal distribution. However that didn't seem to quite improve the predictions. And there are two reasons for that in my opinion. 
# 
# 1. You still have same number of patients/studies at hand, you really didn't increase number of unique training samples by rotating same data set. 
# 2. But the second reason is more worthy to talk. Bear with me here...
# 
# In order to address the 1st issue, there is actually an easier way in Keras. [fit_generator](https://keras.io/models/model/)     has a parameter **class_weights** : By providing class weights for each of the 2 classes in your output, you can give more importance to data in one class over the other. It's a better way to overcome class imbalance issue than fake augmentation in my opinion. So I gave this a try too... 
# 
# 

# In[ ]:


import pandas as pd
import math

df = pd.DataFrame({'data':train_generator.classes})
no_pne = int(df[df.data==train_generator.class_indices['NORMAL']].count())
yes_pne = int(df[df.data==train_generator.class_indices['PNEUMONIA']].count())

imb_rat = round(yes_pne / no_pne, 2)

no_weight = imb_rat
yes_weight = 1.0

cweights = {
    train_generator.class_indices['NORMAL']:no_weight,
    train_generator.class_indices['PNEUMONIA']:yes_weight
}

text = "Normal:{:.0f}\nPneumonia:{:.0f}\nImbalance Ratio: {:.2f}\n".format(no_pne, yes_pne, imb_rat)
print(text)
text = "Using class_weights as:\nNormal:{:.2f}\nPneumonia:{:.2f}\n".format(no_weight, yes_weight)
print(text)

#history = model.fit_generator(generator=train_generator,
#                    steps_per_epoch=step_size_train,
#                    validation_data=val_generator,
#                    validation_steps=step_size_valid,
#                    callbacks=[chkpt1,chkpt2,chkpt3],
#                    class_weight=cweights,
#                    epochs=20, verbose=1)


# Great so it should do better right? No actually it didn't, it actually started doing worse. If you notice the model now giving more important to Normal cases because they were the minority in the training sample. So The model started penalizing 2-3 (avg 2.89) false negatives (pneumonias predicted to be normal) compared to each false positive (a Normal predicted to be Pneumonia)... Let's think about this for a second...
# 
# When we talked about the confusion matrix, we also mentioned that in this particular health application, we would be more concerned about pnuemonia patients going undetected by the model (predicted to be normal) - simply because it can cause death; whereas a normal patient predicting to have pneuomonia might not have as much terrifying affect. It may mean the repeat of the study taking a better image, another doctor visit maybe to find out it was a false positive. 
# 
# Sooo with the current class_weights assigned we did the exact opposite of what was needed. Instead of giving more important to false negatives, we made false positives more important. Then one might think, doing the exact opposite right? but that won't work neither. Why? Because we alrady have 3 times as much Pneumonia images, that the system is already 3 times biased towards that under regular conditions. If it marked everything as Pneumonia, it would have done around 75% accuracy on the test set. So to even increase that ratio wouldn't be really beneficial. It will improve the number of false positives and more normal patients would required to do secondary imaging or doctor visits causing more issues than helping.
# 
# ***So what do we do about it? Nothing. Absolute nothing. ***
# 
# Long story short: For us detecting a Pneumonia correctly is more important that detecting a normal patient's lack of pneumonia. The data set gives us 3 times as much more Pneumonia images than normal, and therefore our model is already biasing towards detecting Pneumonia cases correctly closer to that ratio. And I think that itself takes care of everything in this particular case. Say if we had 3,000 images each class for a total of 6,000 training samples; the model would treat both cases equally, in that case, I would actually use **class_weights**, and assign more importance to samples from Pneumonia group just to make sure detection of that is more prioritized and the loss there is penalized more as well. 
# 
# 

# # **Final Thoughts**
# 
# * As others also found out using a ImageNet based trained model might be good for a different Image Recognition project you have.
#         This could change in future, should much more medical images (in the size of ImageNet training set of 22 million images) be available for training data sets, then an entire vgg or inceptionv3 (or maybe another architectural model) could be trained entirely on that image set and do a better performace. I truely believe that day will come sooner than later.
# * Depending on your own training data size, you may have to consider the number of your models parameters whether you base it on which ImageNet trained model on be it InceptionV3,VGG16,ResNet,Xception etc.
# * Fine-Tuning is not only ALL OR NOTHING type of thing. You may have to experience with unfreezing all layers, just 1 final layers, maybe coule layers etc... Again taking your models required parameters into consideration.
# * If you have imbalance in your training data and want to address it, you can use Keras' built-in **class_weights** parameter to experience with difference ratios there.
# 
# Thank you for taking the time to read my Kernel. All the code in this article can be found on my [GitHub](https://github.com/judopro/Chest-Xray-Pneumonia) page as well. 
