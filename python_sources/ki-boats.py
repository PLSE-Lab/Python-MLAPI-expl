#!/usr/bin/env python
# coding: utf-8

# # Dit is de kaggle opdracht van KI 
# In deze opdracht gaan we een CONV model maken.
# De dataset die we gaan gebruiken is een dataset gemaakt door de docenten zelf en bevatten 5 categorieen.

# In[ ]:


# het eerste wat gedaan wordt is de libraries inladen
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob

from keras.preprocessing import image
# voor het model
from keras.applications import VGG16
from keras.models import Sequential, load_model, Model
from keras.layers import Flatten, Dense, Dropout
from keras.layers import Conv2D, MaxPool2D, GlobalMaxPooling2D

from keras import optimizers
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

from keras import regularizers

seed=123
random.seed(seed)


# In[ ]:


from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

import tensorflow as tf 

if tf.test.gpu_device_name(): 
    print('\n\nDefault GPU Device:{}'.format(tf.test.gpu_device_name()))
else:
   print("Please install GPU version of TF")


# In[ ]:


# hier krijg je 2 lijsten met de locaties van de foto's en een locatie met de 5 categorie namen
files = glob.glob('../input/kunstmatigeintelligentie20192020/Archive/Category*/*.jpg') # creerd een lijst met alle locatie's van de foto's

path="../input/kunstmatigeintelligentie20192020/Archive"
names = (os.listdir(path))
names.sort()
print(names)
print(files[0])


# In[ ]:


# hier zijn 5 functies voor wat visualisatie

#kijken naar de dimensie van de verschillende foto's
# Dit is zelf al onderzorgd en er is in ons opzichte de beste verhouding gekozen width, height, channels
def dimensie():
    array = pd.DataFrame(columns=('width', 'height'))
    for fname in files:
        img = image.load_img(fname)
        x = image.img_to_array(img)
        width = len(x[0])
        height = len(x)
        df = pd.DataFrame([[width, height]], columns=('width', 'height'))
        array = array.append(df)
    
    width = int(round(np.mean(array['width']))+1)
    height = 139
    
    return (height+6), width, len(x[0,0]), array
    

# lijst met 5 samples van elke categorie voor visualisatie
def plotlijst():
    plot_lijst=[]
    aantal=[]
    for i in range(1, 6):
        lenght = len(glob.glob('../input/kunstmatigeintelligentie20192020/Archive/Category%s/*.jpg' % i))
        aantal.append(lenght)
        files = random.sample(glob.glob('../input/kunstmatigeintelligentie20192020/Archive/Category%s/*.jpg' % i), 5)
        plot_lijst.append(files)    
    
    return plot_lijst, aantal



# functie voor visualisatie
def visueel(plot_lijst, height, width):        
    fig = plt.figure(figsize=(17, 30))
    plt.subplots_adjust(left=.2, bottom=.5)
    for row, lijst in enumerate(plot_lijst):        
        for col, index in enumerate(lijst):
            img = image.load_img(index, target_size=(height, width))   
            subplot = fig.add_subplot(5, len(lijst), row*len(lijst)+col+1)
            subplot.set_title('Category %s' % (row+1))
            for label in (subplot.get_xticklabels() + subplot.get_yticklabels()):
                    label.set_fontsize(10)
            plt.imshow(img)

def acc_loss_plot(history):
    
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs = range(1, len(acc)+1)
    
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    
    plt.figure()
    
    
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    
    
    plt.show()
    
    
def layer_outputs(model, img):    
    
    img_tensor = image.img_to_array(img)
    img_tensor = img_tensor.reshape((1,) + img_tensor.shape)
    #length = len(model.layers)
    layer_outputs = [layer.output for layer in model.layers[:4]]
    activation_model = Model(inputs=model.input, outputs=layer_outputs)
    
    activations = activation_model.predict(img_tensor)    
    
    layer_names = []
    for layer in model.layers[:4]:
        layer_names.append(layer.name)
    
    images_per_row = 8
    
    # Now let's display our feature maps
    for layer_name, layer_activation in zip(layer_names, activations):
        # This is the number of features in the feature map
        n_features = layer_activation.shape[-1]
    
        # The feature map has shape (1, size, size, n_features)
        height = layer_activation.shape[1]
        width = layer_activation.shape[2]
        
    
        # We will tile the activation channels in this matrix
        n_cols = n_features // images_per_row
        display_grid = np.zeros((height * n_cols, images_per_row * width))
    
        # We'll tile each filter into this big horizontal grid
        for col in range(n_cols):
            for row in range(images_per_row):
                channel_image = layer_activation[0,
                                                 :, :,
                                                 col * images_per_row + row]
                # Post-process the feature to make it visually palatable
                channel_image -= channel_image.mean()
                channel_image /= channel_image.std()
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                display_grid[col * height : (col + 1) * height,
                             row * width : (row + 1) * width] = channel_image
    
        # Display the grid
        scale = 1. / height
        plt.figure(figsize=(scale * display_grid.shape[1],
                            scale * display_grid.shape[0]))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')
        
    plt.show()  


# # Hier zien we visueel enkele dinge
# 
# Zoals te zien is in de plaatjes hieronder zijn er 5 categorieen aan boten<br>
# categorie 1: olieschepen<br>
# categorie 2: militaire schepen<br>
# categorie 3: grote vrachtschepen<br>
# categorie 4: cruiseschepen<br>
# categorie 5: iets<br>
# 
# de verdeling van de categorienen kan je ook goed zijn in het bovenste plaatje.<br>
# 
# Ook is goed te zien dat er zwart wit, grayscale foto's tussen zitten. Dit is belangrijk om mee rekening te houden, want stel dat er in een categorie veel zwartwit foto's zijn vergeleken met de andere categorieen kan het gebeuren dat ons model onterecht de verkeerde categorie kiest vanwege grayscale. Hier zal nog verder naar gekeken worden.
# 
# 
# 
# 

# In[ ]:


plot_lijst, aantal = plotlijst()
sns.barplot(x=names, y=aantal) # plot van de verdeling

max(aantal) / sum(aantal) # als er alleen gekozen wordt op category 1 heb je een acc van 34%

height=139
width=210
channel=3
visueel(plot_lijst, height, width)


# Wat ook goed is om te doen is een random baseline uit te rekenen om er achter te komen hoe goed het model minimaal moet zijn om beter te zijn dan random. Als we dan naar het antwoor dheironder kijken dan zien we dat we minimaal een model moeten hebben die 22.8% van de afbeeldingen goed voorspeld.
# 

# In[ ]:


probability = [i / sum(aantal) for i in aantal]
random_base = sum([(tal*prob)/sum(aantal) for tal, prob in zip(aantal, probability)])
print(round(random_base*100,2))


# Zoals jullie weten zijn de handelingen hieronder om de data te prepareren. Door gebruik te maken van meerder augmentaties krijg je heel makelijk meer data. Dit kan handig zijn als je een kleine dataset hebt.

# In[ ]:


batch_size=32
train_datagen = image.ImageDataGenerator(
      rescale=1/255,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest',
      validation_split=.2)

train_generator = train_datagen.flow_from_directory(
        path,
        target_size=(height, width),
        color_mode='rgb',
        class_mode='categorical',
        classes=names,
        batch_size=batch_size,
        subset='training'
        )

val_generator = train_datagen.flow_from_directory(
        path,
        target_size=(height, width),
        color_mode='rgb',
        class_mode='categorical',
        classes=names,
        batch_size=batch_size,
        subset='validation'
        )


# ## Augmentatie
# Hier onder is te zien wat de augmentatie kan doen. Zo is te zien dat de afbeelding horizontale gedraait is. Wat verschoven is naar links,rechts, boven en ook ingezoomd is.

# In[ ]:


img = image.load_img('../input/kunstmatigeintelligentie20192020/Archive/Category1/2780099.jpg')

x = image.img_to_array(img)
print(x.shape)
x = x.reshape((1,)+x.shape)

i = 0
for batch in train_datagen.flow(x, batch_size=1):
    
    plt.figure(i)
    imgplot = plt.imshow(image.array_to_img(batch[0]))
    i += 1
    if i % 4 == 0:
        break
    
plt.show()


# # Transfer-learning
# Er wordt gebruik gemaakt van transfer-learning. In dit geval wordt er gebruik gemaakt van VGG16.
# Zie: https://arxiv.org/pdf/1409.1556.pdf, voor meer uitleg.
# VGG16 is erg lang getrained op de imagenet dataset. Hierin zitten ook verschillende boten en is bruikbaar voor ons probleem.
# voor meer informatie over transfer-learning kan je dit boek gebruiken, <b>Deep Learning for Vision Systems</b> bij <b>Mohamed Elgendy</b>.<br>
# <br>
# <b>Het laden van de vgg16 lukte niet dus het is op een andere manier gedaan</b>

# In[ ]:


vgg16 =load_model('../input/modellen1/vgg16.h5')
"""
vgg16 = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(height, width, channel),
             pooling='max')
"""
vgg16.summary()


# Zoals je hierboven ziet heeft dit model pittig wat parameters. 14,714,688 om precies te zijn en dan hebben we de dense layers er nog niet aan toe gevoegd. Het is daarom verstandig om een gpu te gebruiken. Ook is het mogelijk om de layers te frezen zodat je alleen nog de dense layers hoeft te trainen. hiervoor hebben wij niet gekozen.

# In[ ]:


model = Sequential()
model.add(vgg16)
model.add(Dense(4096, activation='relu'))
model.add(Dropout(.5))
model.add(Dense(4096, activation='relu'))
model.add(Dropout(.5))
model.add(Dense(5, activation='softmax'))

model.compile(loss='categorical_crossentropy',
        optimizer=optimizers.Adam(learning_rate=1e-6),
        metrics=['acc'])

num_epochs = 60

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=np.sqrt(.1), patience=3, min_lr=2e-7, verbose=1)
callbacks = [EarlyStopping(patience=5), reduce_lr] #ModelCheckpoint('model_eind.h5', save_best_only=True),

history  = model.fit_generator(train_generator,
                               steps_per_epoch=int(5002*.8/batch_size),
                               epochs=num_epochs,
                               validation_data=val_generator,
                               validation_steps=int(5002*.2/batch_size),
                               callbacks=callbacks)


# # Het model
# Het model wat je hier ziet is het vgg16 model met ipv een Flatten() layer een globalmaxpooling2d layer. De reden waarom dit gedaan is is omdat de hiermee er meer informatie behouden wordt in de afbeelding dan met het gebruik van Flatten()
# 
# De Adam optimizer, zie heir voor meer uitleg https://towardsdatascience.com/adam-latest-trends-in-deep-learning-optimization-6be9a291375c, is hier ook gebruikt ipv RMSprop() deze keuze is gemaakt omdat Adam vaak beter resultaten behaald. Let op!!! Niet altijd. Het is dus altijd goed om ook te kijken hoe andere optimizers het doen
# 
# Ook wordt er gebruik gemaakt van callbacks. De reden is dat dmv callbacks je acties kan toevoegen in het model.
# zo wordt er veel gebruik gemaakt van EarlyStopping en ModelCheckpoint.
# EarlyStoppping word gebruikt om er voor te zorgen dat je model niet door blijft trainen als het niks leert. Zo zie je hierboven dat er 60 epochs zijn, maar niet 
# alle 60 epochs gerunt worden omdat het model niks extra's meer leerde. (geen lagere val_loss na 6 achtereenvolgende epochs) 
# 
# ModelCheckpoint wordt gebruikt om het model met het beste op te slaan. De laagste val_loss, in ons geval, wordt dus opgeslagen. Heeft het model dus geen lagere val_loss daarna wordt het niet gezien als een beter model en slaan we die dus niet op. In het geval hierboven zou je dus kunnen zeggen dat ModelCheckpoint het model van epoch 9 zou opslaan met een val_loss van 0.1130 en een val_acc van 0.8913. 
# 
# ReduceLROnPlateau wordt gebruikt om de learning rate van het model te verlagen als de val_loss over meerdere epochs, in ons geval 5 epcosh (patience=5), niet verbeterd. Het kan dan dus zijn dat het model niet meer beter leert.

# In[ ]:


model.evaluate_generator(val_generator)


# Zoals je hieronder ziet komt er het zelfde resultaat uit als bij epoch 9

# In[ ]:


acc_loss_plot(history)


# Hierboven zie je de plot die er uiteindelijk uitgekomen is.

# ## Als test hebben we ook gebruik gemaakt van VGG16 maxpooling en VGG16 globalpooling hieronder zullen we enkele resultaten daarvan geven met de bijborende veranderingen zoals lr, optimizer. <br> 
# <b>Al deze modelen zijn getrained op de voledige vgg16 model</b> <br>
# 1: <b>maxpooling, RMSprop</b> <br>
# Bij het model met een lr van 2e-6 kwamen we op een val_loss van <b>.1113</b> en een val_acc van <b>.907</b> <br>
# Bij het model met een lr van 2e-7 kwamen we op een val_loss van <b>.1346</b> en een val_acc van <b>.890</b> <br>
# 
# 1: <b>maxpooling, Adam</b> <br>
# Bij het model met een lr van 2e-5 kwamen we op een val_loss van <b>.2221</b> en een val_acc van <b>.879</b> <br>
# Bij het model met een lr van 2e-6 kwamen we op een val_loss van <b>.1286</b> en een val_acc van <b>.888</b> <br>
# 
# 1: <b>maxpooling, SGD</b> <br>
# Bij het model met een lr van 2e-5 kwamen we op een val_loss van <b>1.263</b> en een val_acc van <b>.3346</b> <br>
# Bij het model met een lr van 2e-6 kwamen we op een val_loss van <b>1.079</b> en een val_acc van <b>.4083</b> <br>
# 
# 2: <b>averagepooling, RMSprop</b> <br>
# Bij het model met een lr van 2e-6 kwamen we op een val_loss van <b>.2492</b> en een val_acc van <b>.8178</b> <br>
# Bij het model met een lr van 2e-7 kwamen we op een val_loss van <b>1.539</b> en een val_acc van <b>.348</b> <br>
# 
# 2: <b>averagepooling, Adam</b> <br>
# Bij het model met een lr van 2e-5 kwamen we op een val_loss van <b>.1691</b> en een val_acc van <b>.877</b> <br>
# Bij het model met een lr van 2e-6 kwamen we op een val_loss van <b>.0570</b> en een val_acc van <b>.904</b> <br>
# 
# 2: <b>averagepooling, SGD</b> <br>
# Bij het model met een lr van 2e-6 kwamen we op een val_loss van <b>.2492</b> en een val_acc van <b>.8178</b> <br>
# Bij het model met een lr van 2e-7 kwamen we op een val_loss van <b>1.539</b> en een val_acc van <b>.348</b> <br>
# 
# 3: <b>Dense, RMSprop</b> <br>
# Bij het model met een lr van 2e-5 kwamen we op een val_loss van <b>.1528</b> en een val_acc van <b>.866</b> <br>
# Bij het model met een lr van 2e-6 kwamen we op een val_loss van <b>.0964</b> en een val_acc van <b>.901</b> <br>
# Bij het model met een lr van 2e-7 kwamen we op een val_loss van <b>.2528</b> en een val_acc van <b>.871</b> <br>
# 
# 3: <b>Dense, Adam</b> <br>
# Bij het model met een lr van 2e-6 kwamen we op een val_loss van <b>.0229</b> en een val_acc van <b>.914</b> <br>
# Bij het model met een lr van 2e-7 kwamen we op een val_loss van <b>.7218</b> en een val_acc van <b>.751</b> <br>
# 
# 3: <b>Dense, SGD</b> <br>
# Bij het model met een lr van 2e-6 kwamen we op een val_loss van <b>.992</b> en een val_acc van <b>.55</b> <br>

# # Submission file
# En dan nu een submision file aanmaken die gebruikt kan worden om te kijken hoe goed het model is

# In[ ]:


model = load_model('../input/modellen1/best_model.h5')
test_files=glob.glob('../input/kunstmatigeintelligentie20192020/Images/Images/*.jpg')
test_images= [image.load_img(x, target_size=(height, width)) for x in test_files]
test_images = np.array([image.img_to_array(x) for x in test_images])

test_images = test_images.astype('float32') / 255

lijst = os.listdir('../input/kunstmatigeintelligentie20192020/Images/Images')

predict = model.predict_classes(test_images)+1

  
data = {'id':lijst,
        'Category':predict}

df = pd.DataFrame(data, columns = ['id', 'Category'])
df.head()


# # Bagging
# Door gebruik te maken van bagging kan je uiteindelijke voorspelling hoger uitkomen.
# Dit kan zo zijn omdat je meerdere modellen hebt gebruikt al deze modellen hebben verschillende gewichten in hun model en zullen daardoor ook allemaal andere voorspellingen hebben. Als je daarbij ervan uitgaat dat 1 model soms een fout en andere modellen deze fout niet maken dat als er gebruikt wordt gemaakt van meerdere modellen de kans op een foute voorspelling kleiner kan worden en er uiteindelijk meer juiste voorspellingen ontstaan. Door dit te gebruiken is de scoring met 2% gestegen vergeleken met het gebruiken van een model.

# De modellen zijn niet ingeladen op kaggle, maar hier onder is de code te zien hoe wij het zelf gedaan hebben.

# In[ ]:


model1 = load_model('../input/modellen/max_adam_4096_lr7_3.h5')
model2 = load_model('../input/modellen/model_best_acc_1.h5')
model3 = load_model('../input/modellen/model_best_acc_4.h5')
model4 = load_model('../input/modellen1/best_model.h5')
model5 = load_model('../input/modellen/dense_rmsprop_best.h5')


test_files=glob.glob('../input/kunstmatigeintelligentie20192020/Images/Images/*.jpg')  # data locaties
# laat afbeeldingen in ipv flow_from_directory
test_images= [image.load_img(x, target_size=(height, width)) for x in test_files] 
# Maakt er arrays van om te kunnen voorspelen
test_images = np.array([image.img_to_array(x) for x in test_images])
# rescale
test_images = test_images.astype('float32') / 255

model_list = [model1, model2, model3, model4, model5]
y = np.empty((1250), dtype=int)

# maakt lijst met voorspelingen
for model in model_list:
    predict = model.predict_classes(test_images)+1
    
    y = np.column_stack((y, predict))

# gebruikt de meest voorkomende voorspelling
y = y[:,1:]
voorspel = np.array([], dtype=int)
for i in y:
    count = np.bincount(i)
    voorspel = np.append(voorspel, np.argmax(count))
    
    


lijst = os.listdir('../input/kunstmatigeintelligentie20192020/Images/Images')
    
data = {'id':lijst,
        'Category':voorspel}
    
df = pd.DataFrame(data, columns = ['id', 'Category'])

print(y[0:5,])
print(df.head())


# # zelf gemaakte code vvg16

# In[ ]:


regu_rate = 1e-5

model = Sequential()
 
# block
model.add(Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same', input_shape=(height, width, channel),
                 kernel_regularizer=regularizers.l2(regu_rate)))
model.add(Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same',
                 kernel_regularizer=regularizers.l2(regu_rate)))
model.add(MaxPool2D((2,2), strides=(2,2)))
 
# block
model.add(Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same', 
                 kernel_regularizer=regularizers.l2(regu_rate)))
model.add(Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same', 
                 kernel_regularizer=regularizers.l2(regu_rate)))
model.add(MaxPool2D((2,2), strides=(2,2)))
 
# block
model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same', 
                 kernel_regularizer=regularizers.l2(regu_rate)))
model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same', 
                 kernel_regularizer=regularizers.l2(regu_rate)))
model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same', 
                 kernel_regularizer=regularizers.l2()))
model.add(MaxPool2D((2,2), strides=(2,2)))
 
# block
model.add(Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same', 
                 kernel_regularizer=regularizers.l2(regu_rate)))
model.add(Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same', 
                 kernel_regularizer=regularizers.l2(regu_rate)))
model.add(Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same', 
                 kernel_regularizer=regularizers.l2(regu_rate)))
model.add(MaxPool2D((2,2), strides=(2,2)))
 
# block
model.add(Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same', 
                 kernel_regularizer=regularizers.l2(regu_rate)))
model.add(Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same', 
                 kernel_regularizer=regularizers.l2(regu_rate)))
model.add(Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same', 
                 kernel_regularizer=regularizers.l2(regu_rate)))
model.add(MaxPool2D((2,2), strides=(2,2)))
 
# block #
model.add(GlobalMaxPooling2D())
model.add(Dense(2048, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2048, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(5, activation='softmax'))
 


model.compile(optimizer=optimizers.RMSprop(lr=1e-4),
              loss='categorical_crossentropy',
              metrics=['acc'])

print(model.summary())

