#!/usr/bin/env python
# coding: utf-8

# ### WorkFlow
# - Data Exploration
# - Data Cleaning
# - GAN variables setup
# - GAN network setup
# - GAN Cost and Optimizer
# - GAN iteration
# - View outputs

# In[ ]:


from PIL import Image
import os
root="../input/img_align_celeba/img_align_celeba/"
allim=os.listdir("../input/img_align_celeba/img_align_celeba")
allim[:10]
print(len(allim))
#get a small sample
allim=allim[:20000]
print(len(allim))


# #### Data Exploration

# In[ ]:


#visualize data
import matplotlib.pyplot as plt
plt.figure(figsize=(20,10))
for i in range (30):
    impath=root+allim[i]
    im=Image.open(impath)
    plt.subplot(3,10,i+1)
    plt.imshow(im)


# #### Data Cleaning 

# In[ ]:


# transfer image to numpy array, resize 3d to 1d
import numpy as np
def getImage(id,w=30,h=36):
    path=root+id
    im=Image.open(path)
    im=im.resize([w,h],Image.NEAREST)
    im=np.array(im)
    im=im.reshape(w*h*3)
    return im


for i in range(len(allim)):
    allim[i]=getImage(allim[i])


print(allim[3].shape)
print(len(allim))
    


# In[ ]:


#visualize resized images
plt.figure(figsize=(12,6))
for i in range (30):
    plt.subplot(3,10,i+1)
    plt.imshow(allim[i].reshape(36,30,3))


# In[ ]:


#normalize data for tanh(GAN generation function)
allim=np.array(allim)
allim=allim/255*2-1
allim.shape


# #### GAN variables setup

# In[ ]:


#input real image and noise
import tensorflow as tf
def inputs(dim_real,dim_noise):
    input_reals=tf.placeholder(tf.float32, [None, dim_real], name='input_reals')
    input_noises=tf.placeholder(tf.float32, [None, dim_noise], name='input_noises')
    return input_reals, input_noises


# In[ ]:


#generator
def generator(noises,nn_units,out_dimension,alpha=0.01,reuse=False):
    
    with tf.variable_scope("generator", reuse=reuse):
        hidden1=tf.layers.dense(input_noises,nn_units)
        #leaky relu
        hidden1=tf.maximum(alpha*hidden1, hidden1)
        hidden1=tf.layers.dropout(hidden1,rate=0.2)

        logits=tf.layers.dense(hidden1, out_dimension)
        outputs=tf.tanh(logits)

        return logits,outputs


# In[ ]:


#discriminator
def discriminator(image,nn_units,alpha=0.01,reuse=False):
    with tf.variable_scope("discriminator", reuse=reuse):
        hidden1=tf.layers.dense(image,nn_units)
        #leaky relu
        hidden1=tf.maximum(alpha*hidden1, hidden1)

        logits=tf.layers.dense(hidden1, 1)
        outputs=tf.sigmoid(logits)

        return logits,outputs


# In[ ]:


dim_real=allim[0].shape[0]
print(dim_real)
dim_noise=100
gen_units=128
dis_units=128
LR=0.001
alpha=0.01


# #### GAN network setup

# In[ ]:



tf.reset_default_graph()
input_reals, input_noises=inputs(dim_real,dim_noise)
gen_logits, gen_outputs=generator(input_noises,gen_units, dim_real)
dis_real_logits, dis_real_outputs=discriminator(input_reals,dis_units)
dis_fake_logits, dis_fake_outputs=discriminator(gen_outputs,dis_units, reuse=True)


# In[ ]:


dis_real_cost=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_real_logits,
                                                                     labels=tf.ones_like(dis_real_logits)))
dis_fake_cost=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_fake_logits,
                                                                     labels=tf.zeros_like(dis_fake_logits)))
dis_total_cost=tf.add(dis_real_cost, dis_fake_cost)

gen_cost=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_fake_logits,
                                                                labels=tf.ones_like(dis_fake_logits)))


# In[ ]:


train_vars=tf.trainable_variables()
gen_vars=[var for var in train_vars if var.name.startswith('generator')]
dis_vars=[var for var in train_vars if var.name.startswith('discriminator')]
gen_optimizer=tf.train.AdamOptimizer(LR).minimize(gen_cost,var_list=gen_vars)
dis_optimizer=tf.train.AdamOptimizer(LR).minimize(dis_total_cost,var_list=dis_vars)


# #### Gan iteration

# In[ ]:


init=tf.global_variables_initializer()
sess=tf.Session()
sess.run(init)

batch_size=64
count=0
g_cost=0
d_cost=0
samples=[]
for i in range(40000):
    startindex=(i*batch_size)%(len(allim)-batch_size)
    endindex=startindex+batch_size
    batch_real=allim[startindex:endindex] #shape is (64,3240)
    batch_noise=np.random.uniform(-1,1,size=(batch_size, dim_noise))
    sess.run(dis_optimizer,feed_dict={input_reals:batch_real,input_noises:batch_noise})
    sess.run(gen_optimizer, feed_dict={input_noises:batch_noise})
    g_cost+=(sess.run(gen_cost, feed_dict={input_noises:batch_noise}))
    d_cost+=(sess.run(dis_total_cost, feed_dict={input_reals:batch_real,input_noises:batch_noise}))
    
    if (i+1)%1000==0:
        count+=1
        print("ITER:",count,"| GEN COST:",g_cost/(1000),"DIS COST:",d_cost/(1000))
        g_cost=0
        d_cost=0
        gen_samples=sess.run(generator(input_noises,gen_units,dim_real, reuse=True),
                            feed_dict={input_noises:batch_noise})
        samples.append(gen_samples)


# In[ ]:


sess.close()


# #### View output

# In[ ]:


print(len(samples))
print(len(samples[1]))
print((samples[0][0].shape))


# In[ ]:


#generated logits
samples[0][0][0]


# In[ ]:


#generated outputs
samples[0][1][0]


# In[ ]:


plt.figure(figsize=(10,10))
for i in range(64):
    plt.subplot(8,8,i+1)
    img=((samples[0][1][i]+1)*255/2).astype(np.uint8)
    plt.imshow(img.reshape(36,30,3))


# In[ ]:


plt.figure(figsize=(10,10))
for i in range(64):
    plt.subplot(8,8,i+1)
    img=((samples[9][1][i]+1)*255/2).astype(np.uint8)
    plt.imshow(img.reshape(36,30,3))


# In[ ]:


plt.figure(figsize=(10,10))
for i in range(64):
    plt.subplot(8,8,i+1)
    img=((samples[39][1][i]+1)*255/2).astype(np.uint8)
    plt.imshow(img.reshape(36,30,3))


# In[ ]:


c=0
for i in range(64):
    path='img'+str(c)+'.jpg'
    img=((samples[39][1][i]+1)*255/2).astype(np.uint8)
    img=img.reshape(36,30,3)
    plt.imsave(path,img)
    c+=1

