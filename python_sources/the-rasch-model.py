#!/usr/bin/env python
# coding: utf-8

# # The Rasch Model #
# 
# This notebook demonstrates implementation of the Rasch model in TensorFlow. All theoretical parts are taken from an excellent textbook "Bayesian Reasoning and Machine Learning" by David Barber. The Rasch Model is covered in chapter 22 of the book.

# Consider an exam in which student $s$ answers question $q$ either correctly $x_{qs} = 1$ or incorrectly $x_{qs} = 0$.
# For a set of $N$ students and $Q$ questions, the performance of all students is given in the $Q \times N$ binary
# matrix $X$. Based on this data alone we wish to evaluate the ability of each student, and at the same time estimate difficulty of each question. To learn both, we assign the probability that a student $s$ gets a question $q$ correct based on the student's latent ability $\alpha_s$ and the latent difficulty of the question $\delta_q$:
# 
# $$p(x_{qs} = 1|\alpha, \delta) = \sigma(\alpha_s -\delta_q)$$
# Where $\sigma$ is sigmoid function.
# 
# Making the i.i.d. assumption, the likelihood of the data $X$ under this model is:
# 
# $$p(X|\alpha, \delta) = \prod_{s=1}^S\prod_{q=1}^Q \sigma(\alpha_s-\delta_q)^{x_{qs}} (1-\sigma(\alpha_s-\delta_q))^{1-x_{qs}}$$

# The log likelihood is then:
# 
# $$L \equiv log(X|\alpha, \beta) = \sum_{q,s} { x_{qs} log \sigma(\alpha_s - \delta_q) + 
# (1 - x_{qs}) log (1 - \sigma(\alpha_s - \delta_q))}$$
# 

# And the partial derivatives are:
# 
# $$\frac{\partial L}{\partial \alpha_s} = \sum_{q=1}^Q(x_{qs} - \sigma(\alpha_s - \delta_q))$$
# 
# $$\frac{\partial L}{\partial \delta_q} = - \sum_{s=1}^S(x_{qs} - \sigma(\alpha_s - \delta_q))$$
# 
# But since we are going to use TensorFlow, it will calculate the derivatives automatically, so these are just for the information

# In[ ]:


#Import
import numpy as np
import pandas as pd
import itertools
import tensorflow as tf
np.random.seed(1239)


# In[ ]:


def sigmoid(x):
    return 1/(1+np.exp(-x))


# In[ ]:


#First we generate the test data

#The synthetic question:
synthetic_questions = np.arange(-1.9, 3.1, 1)
synthetic_students = np.arange(0,2,0.1)
synthetic_logits = synthetic_students.reshape(-1,1) - synthetic_questions.reshape(1,-1)
synthetic_probs = sigmoid(synthetic_logits)
synthetic_data = (synthetic_probs > np.random.rand(synthetic_probs.shape[0],synthetic_probs.shape[1])).astype('float')


# In[ ]:



synthetic_data


# In[ ]:


data_shape = synthetic_data.shape
learning_rate = 0.1
tf.reset_default_graph()
X = tf.placeholder(dtype='float' ,shape=data_shape, name="X")
alpha = tf.Variable(initial_value=np.zeros((data_shape[0],1)), name="alpha", dtype='float')
delta = tf.Variable(initial_value=np.zeros((1,data_shape[1])), name="delta", dtype='float')
log_likelihood = tf.reduce_sum(X * tf.log(tf.sigmoid(alpha-delta)) + (1-X) * tf.log(1-tf.sigmoid(alpha-delta)))
cost = -log_likelihood
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
training_op = optimizer.minimize(cost)


# In[ ]:


init = tf.global_variables_initializer()
n_epochs = 4000


with tf.Session() as sess:
    sess.run(init)

    for epoch in range(n_epochs):
        if epoch % 1000 == 0:
            print("Epoch", epoch, "Cost =", cost.eval(feed_dict={X: synthetic_data}))
        sess.run(training_op, feed_dict={X: synthetic_data})
    
    best_alpha = alpha.eval()
    best_delta = delta.eval()


# In[ ]:


best_alpha


# In[ ]:


best_delta


# It got the questions in the right order, and the students are also roughly in the right order, but are affected by chance.
# 
# One of the improvements of this model would be to add priors for $\alpha$ and $\delta$, which will cause regularization and the smoothing of both student ability scores and the question difficulty score.
