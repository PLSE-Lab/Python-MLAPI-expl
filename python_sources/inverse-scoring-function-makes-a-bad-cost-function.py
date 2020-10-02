#!/usr/bin/env python
# coding: utf-8

# **Summary: ...because it produces overconfident and too optimistic models.**
# 
# An inexperienced data scientist like me might be tempted to use the inverse of the scoring function as a cost function to train models. Let's see how it would turn out.

# In[ ]:


import numpy as np
import pandas as pd
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import datetime
from tensorflow.python.framework import ops
from kaggle.competitions import twosigmanews


# In[ ]:


env = twosigmanews.make_env()
(market_train_df, news_train_df) = env.get_training_data()
del news_train_df


# ## Training and test set
# Let's take an array of all days that we observe in the training data set and split it into training days and test days. Based on that, we construct our training and test set. As hinted by [Maxwell's work](https://www.kaggle.com/maxwell110/naive-experiment-on-evaluation-metric), data from the turbulent days of the financial crisis might disturb our model. So, for better reproducibility, we ignore the days from 2007 and 2008 altogether.

# In[ ]:


np.random.seed(1010)
days_past = np.sort(market_train_df.time.unique())
days_past = days_past[days_past > datetime.datetime(2009, 1, 1, tzinfo=days_past[0].tzinfo)]
days_shuffled = np.random.permutation(days_past)
days_test = np.sort(days_shuffled[0:int(len(days_past) * 0.02)])
days_train = np.sort(days_shuffled[int(len(days_past) * 0.02):])
train_df = market_train_df[market_train_df['time'].isin(days_train)].dropna(subset=['returnsOpenPrevMktres10', 'returnsOpenNextMktres10'])
test_df = market_train_df[market_train_df['time'].isin(days_test)].dropna(subset=['returnsOpenPrevMktres10', 'returnsOpenNextMktres10'])


# ## The benchmark
# Let's start with the simplest possible approach: Take the returns of the previous 10 days (`returnsOpenPrevMktres10`) and use it as a confidence value for the next 10 days.

# In[ ]:


def daily_score(test_daily):
    return np.sum(test_daily.confidenceValue * test_daily.returnsOpenNextMktres10 * test_daily.universe)


# In[ ]:


test_df = test_df.assign(confidenceValue = test_df.returnsOpenPrevMktres10)
sc = [daily_score(test_df[test_df.time == day]) for day in days_test]
print("Test score: ", np.mean(sc) / np.std(sc))


# Not bad of a score! In fact, [this simple model](https://www.kaggle.com/poznyakovskiy/naive-prediction) produces a fairly decent score even in the evaluation data.
# 
# If we regard this as a linear model
# $$Z = W * X + b$$
# our current benchmark corresponds to a model with $W = 1$ and $b = 0$. Surely, we can improve it by tweaking the values for W and b.

# ## A linear model with TensorFlow
# Let's train a simple linear model. Here, we take just the same scoring function

# In[ ]:


def score(Z, Y):
    x = tf.multiply(Z, Y)
    mu, var = tf.nn.moments(x, axes=1)
    return mu / tf.sqrt(var)


# and negate it, so that our cost function is the inverse of the score.

# In[ ]:


def cost(Z, Y):
    return -score(Z, Y)


# Let's build and train a TensorFlow model.

# In[ ]:


ops.reset_default_graph()
X_train = [train_df['returnsOpenPrevMktres10']]
Y_train = [train_df['returnsOpenNextMktres10']]
X = tf.placeholder(tf.float32, [1, None], name="X")
Y = tf.placeholder(tf.float32, [1, None], name="Y")

# We initialize W with 1 and b with 0, so we are starting just with our benchmark
W = tf.get_variable('W', 1, initializer = tf.constant_initializer(1))
b = tf.get_variable('b', 1, initializer = tf.zeros_initializer())
Z = tf.add(tf.multiply(W, X), b)
cf = cost(Z, Y)

epoch_costs = []
optimizer = tf.train.AdamOptimizer(learning_rate = 0.005).minimize(cf)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(200):
        _ , epoch_cost = sess.run([optimizer, cf], feed_dict={X: X_train, Y: Y_train})
        epoch_costs.append(epoch_cost)
        if epoch % 50 == 0:
            print('Costs after epoch %i: %f' % (epoch, epoch_cost))
    W_final = sess.run(W)
    b_final = sess.run(b)
    
plt.plot(np.squeeze(epoch_costs))
plt.ylabel('cost')
plt.xlabel('iterations (per tens)')
plt.show()


# So far, everything looks fair. The training cost decreases; the graph looks a little bit bumpy near the end, but this is because the learning rate is tuned up here. Let's see which score we get on the test set:

# In[ ]:


test_df = test_df.assign(confidenceValue = [(v * W_final + b_final)[0] for v in test_df.returnsOpenPrevMktres10])
sc = [daily_score(test_df[test_df.time == day]) for day in days_test]
print("Test score: ", np.mean(sc) / np.std(sc))


# *Yikes!* This is not what we wanted to achieve. Maybe we've overfit it? Let's look at the parameters:

# In[ ]:


print("Weight: ", W_final)
print("Bias: ", b_final)


# This does not look like a typical overfit model. The weight is close to zero, but look at the bias: It is huge! With this settings, the model will always produce large positive values. Let's have look at the actual predictions:

# In[ ]:


test_df.confidenceValue[0:10]


# Basically, our model bets on the asset value always going up a lot, each day, for every asset. Why?
# 
# ## The reason why it fails
# Let's start with a question: When will our cost function produce a small cost? Looking at the definition, we can conclude that it happens in two cases:
# * The return is positive, and the confidence value is also positive
# * The return is negative, and so is the confidence value
# That also means that if we encounter even the smallest positive return, say, 0.0001, the model would attempt to squeeze the most of it by multiplying it with a large positive confidence. That means that **the model favors a large positive bias**.
# 
# "But what if the confidence value is close to 1, but the actual return is -0.0001? Wouldn't it deal a blow to the total cost?" -- one might ask. This is correct, but let's take a look at the summary statistics for the returns for next 10 days:

# In[ ]:


plt.hist(train_df.returnsOpenNextMktres10, bins=60, range=(-0.25, 0.25))
plt.show()


# The distribution shows a smooth bell curve with a mean slightly above zero:

# In[ ]:


print("Mean returns: ", np.mean(train_df.returnsOpenNextMktres10))


# That means that the absolute value of all returns below 0 is less than the absolute value of all returns above 0. So, **the model** basically ignores the possibility of a return going below 0 and instead **leverages the difference between positive and negative returns to maximize the total score**.
# 
# ### What is the wrong underlying assumption?
# Recall the description of the evaluation function:
# > If you expect a stock to have a large positive return--compared to the broad market--over the next ten days, you might assign it a large, positive confidenceValue (near 1.0). If you expect a stock to have a negative return, you might assign it a large, negative confidenceValue (near -1.0). If unsure, you might assign it a value near zero.
# 
# With the previously mentioned example of encountering a future return of 0.0001 in the training set, it would be actually more correct to assign a confidence value close to 0. But the cost computed with the inverse scoring function would actually be higher than if the model goes all-in. That means that **we are not rewarding the model for being unconfident** even in cases where it is appropriate (near-zero or volatile returns).
# 
# ## Conclusion
# * Models using the inverted scoring function as a cost function are in danger of training to an overwhelmingly high bias;
# * Such models tend to be overconfident and over-optimistic because the scoring function favors large confidence values;
# * A better cost function should be rewarded for uncertainty when the returns have small positive or small negative values.
