#!/usr/bin/env python
# coding: utf-8

# # GENERATIVE ADVERSERIAL NETWORK (GANs)
# 
# WHAT ARE GANs?
# 
# GAN is an artificial neural network architecture. But we call it GAN's. Because, in fact, it is the combination of two separate network structures. The GANs generate new data that has the same statistics as the data given to it.
# 
# We said that GAN consists of two separate networks. One of these two networks receives real data and sends it to a controller structure. The second network structure creates another data which should represent the original data and sends this data to the same supervisory structure. In the supervisory network or structure part, these data are tested and as a result of the test, the similarity of the data from the generative structure to the original data is measured. If the counterfeit network is not successful enough, it tries to generate new data by updating their weights. As a result, the weights of the simulated network are updated after a certain location and learn to create a dataset that is very similar to the original data or well represents the original data.
# 
# In other words, one of the networks uses the original data and the other tries to emulate the same data. While both networks try to prove to the supervisory structure that their data is original, our copycat network becomes able to create the original data from scratch.
# 
# If we represent this situation with an image, it would look like this:
# 
# ![alt text](https://miro.medium.com/max/958/1*-gFsbymY9oJUQJ-A3GTfeg.png)
# 
# Picture Source: https://medium.com/@devnag/generative-adversarial-networks-gans-in-50-lines-of-code-pytorch-e81b79659e3f
# 
# **GAN SAMPLE STRUCTURE**
# ![alt text](https://miro.medium.com/max/1741/1*t78gwhhw-hn1CgXc1K89wA.png)
# 
# Picture Source: https://medium.com/datadriveninvestor/generative-adversarial-network-gan-using-keras-ce1c05cfdfd3
# 
# **Imroving of GAN Epoch by Epoch**
# ![alt text](https://miro.medium.com/max/1952/1*xm6_ZfvfKSHe2KS49DT8TQ.png)
# 
# 
# Picture Source: https://medium.com/datadriveninvestor/generative-adversarial-network-gan-using-keras-ce1c05cfdfd3
# 
# 
# As a result of this structure, our ultimate goal is to produce new data. The more similar it is to the new data produced, the more successful our network is. The new data we produce in a good structure is too high quality to be distinguished from the reality.
# 
# As an example, let's look at human faces that were actually produced with GANs (These people never lived in our world):
# 
# B
# 
# ![alt text](https://wp-assets.futurism.com/2018/12/ai1.jpg)
# 
# Picture Source: https://futurism.com/incredibly-realistic-faces-generated-neural-network

# # CODING OF GAN ARCHITECTURE
# If we understand how it works, we can start building an exemplary GAN architecture.
# 
# In this study, we will try to observe the results of the GAN network using mnist dataset.

# ## IMPORTING LIBRARIES

# In[ ]:


from keras.layers import Dense, Dropout, ReLU, LeakyReLU, BatchNormalization
from keras.models import Sequential, Model, Input
from keras.optimizers import Adam
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
import warnings
import pandas as pd
warnings.filterwarnings("ignore")


# ## IMPORTING DATA

# ### RESHAPE DATA
# 

# In[ ]:


x_train = pd.read_csv("../input/fashionmnist/fashion-mnist_train.csv")
x_test = pd.read_csv("../input/fashionmnist/fashion-mnist_test.csv")

x_train = np.array(x_train)
x_test = np.array(x_test)

x_train = x_train.astype("float32")/255.0
x_test = x_test.astype("float32")/255.0

x_train = x_train[:,:-1]
x_test = x_test[:,:-1]

print(x_train.shape)
print(x_test.shape)


# 
# ## MODEL BUILDING
# 
# The GANs model consists of two networks, Generator and Discriminator. We will set up these networks separately and then create our overall model.
# 
# Let's write our function to create a generator network.

# In[ ]:


def create_generator():
    generator = Sequential()
    generator.add(Dense(512, input_dim=100))
    generator.add(ReLU())

    generator.add(Dense(1024))
    generator.add(ReLU())

    generator.add(Dense(512))
    generator.add(ReLU())

    #set output sizes to 784 to match our data
    generator.add(Dense(784, activation="tanh"))


    #it will be fake and real two classes, we will build our model similar to classification.
    generator.compile(loss="binary_crossentropy", optimizer=Adam(lr=0.001, beta_1 = 0.5))
    return generator

g = create_generator()
g.summary()


# 
# Now let's write the part of our Discriminator network.
# 
# But the part that needs to be understood here is that the Discriminator network should match the images that the input shape will take. Because we will give the pictures directly to this network,  itself will not produce like the generator.
# 

# In[ ]:


def create_discriminator():
    discriminator = Sequential()
    discriminator.add(Dense(1024, input_dim=784))
    discriminator.add(ReLU())
    discriminator.add(Dropout(0.4))
    
    discriminator.add(Dense(512))
    discriminator.add(ReLU())
    discriminator.add(Dropout(0.4))

    discriminator.add(Dense(512))
    discriminator.add(ReLU())

    discriminator.add(Dense(1, activation="sigmoid"))

    discriminator.compile(loss="binary_crossentropy", optimizer=Adam(lr=0.001, beta_1=0.5))

    return discriminator

d = create_discriminator()
d.summary()


# # CREATING THE GAN MODEL
# 
# After creating our Generator and Discriminator networks, we can now combine them into a single structure and create our GAN model.
# 
# Before we get started, we need to know that we don't want to develop the discriminator part so that the generator part, that is, the producer part of our network, can update itself. Because if we train the discriminator part that will make the fake / real distinction, this part of the network will improve itself, so no matter how realistic the fake data generated, it will realize that the data is fake and the generator part will always fail. Therefore, the producer network will see every output unsuccessful and will have difficulty developing itself. However, when he approaches the truth, he needs to know that he is successful and update himself accordingly. For these reasons, our discriminator network will be closed to train process.
# 

# In[ ]:


def create_gan(generator, discriminator):
    discriminator.trainable = False

    #lets get started to specify an input and give it to generator
    gan_input = Input(shape=(100,))
    #generator will give us a value after it taked this part
    x = generator(gan_input)
    #the value we get will be entered into the discriminator and checked 
    # and returned to us a GAN output
    gan_output = discriminator(x)

    #now we can build a gun model
    gan = Model(inputs=gan_input, outputs=gan_output)
    gan.compile(loss="binary_crossentropy", optimizer="adam", metrics=["acc"])

    #as a result, let's turn our gan model
    return gan

gan = create_gan(g, d)
g.summary()


# ## GAN Model Training

# In[ ]:


epochs = 50
batch_size = 256
acc_list = []
hsitory = []

def gan_train(g, d, gan):
    for e in range(epochs):
        print("Epoch continues : ", e+1)
        for _ in range(batch_size):
            noise = np.random.normal(0,1,[batch_size, 100])
            generated_img = g.predict(noise)
            
            batch_img = x_train[np.random.randint(low=0, high=x_train.shape[0], size=batch_size)]
            
            x = np.concatenate([batch_img, generated_img])


            y_disc = np.zeros(batch_size*2)
            y_disc[:batch_size] = 1

            d.trainable = True
            d.train_on_batch(x, y_disc)

            noise = np.random.normal(0,1,[batch_size, 100])
            y_gen = np.ones(batch_size)

            d.trainable = False

            history = gan.train_on_batch(noise, y_gen)
        acc_list.append(history[0])
        history.append(history)
 
    print("Training Done...")

gan_train(g,d,gan)


# # Visualization of GAN Results
# 
# It would be unreasonable to expect a very good result because we only did 50 epochs. The higher the number of Epoch, the better we can get.

# In[ ]:


noise = np.random.normal(loc=0, scale=1, size=[100,100])
generated_images = g.predict(noise)
generated_images = generated_images.reshape(100,28,28)
plt.imshow(generated_images[66], interpolation="nearest")
plt.title("The Picture\nof GAN")
plt.show()
gan

