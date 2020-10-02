#!/usr/bin/env python
# coding: utf-8

# In[ ]:


cd /kaggle/input/pokemon-images-and-types/images/


# In[ ]:


dir = "images/"


# In[ ]:


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import imageio


# In[ ]:


def load_pokemon():
    data = []
    for i in os.listdir(dir):
        
        img = imageio.imread(dir + i)
        img = Image.fromarray(img)
        img.load()
        
        if(len(img.split()) == 4):
        
        # replace alpha channel with white color
            im = Image.new('RGB', img.size, (255, 255, 255))
            im.paste(img, mask=img.split()[3])
           
        
        else:
            im = img
        pixels = tf.keras.preprocessing.image.img_to_array(im)
        pixels = pixels.astype("float32")
        pixels /= 255.
        data.append(pixels)
    return np.stack(data)


# In[ ]:


dataset = load_pokemon()


# In[ ]:


dataset.shape


# In[ ]:


plt.figure(figsize = (15,15))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.axis("off")
    plt.imshow(dataset[i])
plt.show()


# In[ ]:


def discriminator2(inp_shape = (120,120,3)):
    tf.keras.backend.clear_session()
    base_model = tf.keras.applications.MobileNetV2(input_shape = inp_shape, include_top = False, weights="imagenet")
    base_model.trainable = True
    glob_avg_pool = tf.keras.layers.GlobalAveragePooling2D()
    pred_layer = tf.keras.layers.Dense(1, "sigmoid")
    model = tf.keras.models.Sequential([base_model,
            glob_avg_pool,
            pred_layer])
    model.compile(optimizer = tf.keras.optimizers.Adam(0.0002, beta_1=0.5), loss = "binary_crossentropy", metrics = ['acc'])
    return model


# In[ ]:


def discriminator(inp_shape = (120,120,3)):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(128, (3,3), strides = (2,2), padding="same", input_shape = inp_shape),
        tf.keras.layers.LeakyReLU(0.2),
        
        tf.keras.layers.Conv2D(128, (3,3), padding="same",  strides = (2,2)),
        tf.keras.layers.LeakyReLU(0.2),
        
        tf.keras.layers.Conv2D(64, (3,3), padding="same",  strides = (2,2)),
        tf.keras.layers.LeakyReLU(0.2),
        
        tf.keras.layers.Conv2D(64, (3,3), padding = "same", strides = (2,2)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Flatten(),
        
        tf.keras.layers.Dense(1, activation = "sigmoid")
    ])
    model.compile(optimizer = tf.keras.optimizers.Adam(0.0002, beta_1=0.5), loss = "binary_crossentropy", metrics = ['acc'])
    return model


# In[ ]:


d_model = discriminator()


# In[ ]:


d_model.summary()


# In[ ]:


cd /kaggle/working


# In[ ]:


tf.keras.utils.plot_model(d_model, show_shapes = True)


# In[ ]:


def generate_fake_samples(n_samples):
    rand_samp = np.random.randn(120 * 120 * 3 * n_samples)
    rand_samp = -1 + rand_samp * 2
    X = rand_samp.reshape(n_samples, 120, 120, 3)
    y = np.zeros(shape = (n_samples,1))
    return X,y


# In[ ]:


def train_discriminator(model, dataset, num_iterations = 20, n_batch = 128):
    half_batch = int(n_batch/2)
    for i in range(num_iterations):
        X_real, y_real = generate_real_samples(dataset, half_batch)
       
        _, real_acc = model.train_on_batch(X_real, y_real)
        
        X_fake, y_fake = generate_fake_samples(half_batch)
      
        _, fake_acc = model.train_on_batch(X_fake, y_fake)
        print("Batch {}: Real acc: {} and Fake acc: {}".format(i+1, real_acc*100, fake_acc* 100))


# In[ ]:


train_discriminator(d_model, dataset)


# In[ ]:


X_real, y_real = generate_real_samples(dataset,800)
X_fake, y_fake = generate_fake_samples(800)
X,y = np.vstack((X_real, X_fake)), np.vstack((y_real, y_fake))


# In[ ]:


history = d_model.fit(X,y, epochs = 5)


# In[ ]:


history


# In[ ]:


plt


# In[ ]:


def generator(latent_dim = 100):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(128 * 15 * 15, input_dim = latent_dim),
        tf.keras.layers.LeakyReLU(0.2),
        tf.keras.layers.Reshape((15,15,128)),
        
        # 30 * 30
        tf.keras.layers.Conv2DTranspose(128, (3,3), strides = (2,2), padding = "same"),
        tf.keras.layers.LeakyReLU(0.2),
        
        #60 * 60
        tf.keras.layers.Conv2DTranspose(128, (3,3), strides = (2,2), padding = "same"),
        tf.keras.layers.LeakyReLU(0.2),
        
        # 120 * 120
        tf.keras.layers.Conv2DTranspose(64, (3,3), strides = (2,2), padding = "same"),
        tf.keras.layers.LeakyReLU(0.2),
        
        tf.keras.layers.Conv2D(3, (3,3), padding = "same", activation = "sigmoid")
        
    ])
    return model


# In[ ]:





# In[ ]:


d_model.summary()


# In[ ]:


g_model = generator()


# In[ ]:


g_model.summary()


# In[ ]:


def generate_real_samples(dataset, n_size = 128):
    ind = np.random.randint(0, dataset.shape[0], n_size)
    data = dataset[ind]
    y = np.ones((n_size, 1))
    return data, y


# In[ ]:


# plt.imshow(generate_fake_examples(g_model)[0][3])


# In[ ]:


def generate_latent_space(n_size = 128, latent_dim = 100):
    points = np.random.randn(n_size * latent_dim)
    points = points.reshape((n_size, latent_dim))
    return points


# In[ ]:


def generate_fake_examples(g_model, n_size = 128, latent_dim = 100):
    latent_space = generate_latent_space(n_size, latent_dim)
    preds = g_model.predict(latent_space)
    y = np.zeros((n_size , 1))
    return preds, y


# In[ ]:


def gan(g_model, d_model):
    d_model.trainable = False
    model = tf.keras.models.Sequential([
        g_model,
        d_model
    ])
    model.compile(optimizer = tf.keras.optimizers.Adam(0.0002, beta_1=0.5), loss = "binary_crossentropy")
    return model


# In[ ]:


gan_model = gan(g_model, d_model)


# In[ ]:


gan_model.summary()


# In[ ]:


def plot_samples(data):
    plt.figure(figsize = (15,15))
    for i in range(7*7):
        plt.subplot(7,7,i+1)
        plt.axis("off")
        plt.imshow(data[i])
    plt.show()


# In[ ]:


def summarize_performance(g_model, dataset, n_size = 128):
    X_real, y_real = generate_real_samples(dataset)
    _,accr = d_model.evaluate(X_real, y_real)
    
    X_fake, y_fake = generate_fake_examples(g_model)
    _, accf = d_model.evaluate(X_fake, y_fake)
    
    print("Real samples Acc: {}".format(accr*100))
    print("Fake samples Acc: {}".format(accf*100))
    
    plot_samples(X_fake)
    


# In[ ]:


cd 


# In[ ]:


cd /kaggle/working


# In[ ]:


def plot_history(d1_hist, d2_hist, g_hist, a1_hist, a2_hist):
	# plot loss
	plt.subplot(2, 1, 1)
	plt.plot(d1_hist, label='d-real')
	plt.plot(d2_hist, label='d-fake')
	plt.plot(g_hist, label='gen')
	plt.legend()
	# plot discriminator accuracy
	plt.subplot(2, 1, 2)
	plt.plot(a1_hist, label='acc-real')
	plt.plot(a2_hist, label='acc-fake')
	plt.legend()
	# save plot to file
	plt.close()


# In[ ]:


def train(g_model, d_model, gan_model, dataset,epochs = 1500, latent_dim = 100, batch_size = 128):
    half_batch = int(batch_size/2)
    batch_per_epoch = int(len(dataset)/batch_size)
    d1_hist, d2_hist, g_hist, a1_hist, a2_hist = list(), list(), list(), list(), list()
    for i in range(epochs):
        for j in range(batch_per_epoch):
            X_real, y_real = generate_real_samples(dataset, half_batch)
            d_loss1, d_acc1 = d_model.train_on_batch(X_real, y_real)
            
            X_fake, y_fake = generate_fake_examples(g_model, half_batch)
            d_loss2, d_acc2 = d_model.train_on_batch(X_fake, y_fake)
            
            X_gan = generate_latent_space()
            y_gan = np.ones((batch_size, 1))
            
            g_loss = gan_model.train_on_batch(X_gan, y_gan)
            
            d1_hist.append(d_loss1)
            d2_hist.append(d_loss2)
            g_hist.append(g_loss)
            a1_hist.append(d_acc1)
            a2_hist.append(d_acc2)
            
            if((j+1) % 5 == 0):
                print('>%d, d1=%.3f, d2=%.3f g=%.3f, a1=%d, a2=%d' %
                    (i+1, d_loss1, d_loss2, g_loss, int(100*d_acc1), int(100*d_acc2)))
        
        if((i+1) % 50 == 0):
            summarize_performance(g_model, dataset)
    plot_history(d1_hist, d2_hist, g_hist, a1_hist, a2_hist)
            


# In[ ]:


# Trained for 1500 epochs
train(g_model, d_model, gan_model, dataset)


# In[ ]:


g_model.save("pokemon.h5")


# ## Vizualizaing Generated Pokemons 

# In[ ]:


X_input = generate_latent_space(n_size = 49)
preds = g_model.predict(X_input)
plot_samples(preds)


# In[ ]:





# In[ ]:




