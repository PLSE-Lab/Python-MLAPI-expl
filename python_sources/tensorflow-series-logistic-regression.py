#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn import datasets
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


iris = datasets.load_iris()
x = iris.data

y = iris.target
index = np.where(y==1)
index = np.append(index, np.where(y==0))
x = x[index]
y = y[index].reshape(-1,1)
print(x.shape)
print(y.shape)

tf.reset_default_graph()
X = tf.placeholder(tf.float32, [None, x.shape[-1]], name='X')
Y = tf.placeholder(tf.float32, [None, 1], name='Y')
W = tf.Variable(tf.zeros([x.shape[-1],1]), name='Weights')
b = tf.Variable(tf.zeros([1]), name='bias')
logits = tf.matmul(X,W) + b

cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=Y))
init = tf.global_variables_initializer()

optimizer = tf.train.GradientDescentOptimizer(0.06).minimize(cross_entropy)
pred = tf.round(tf.sigmoid(logits))
accu = tf.reduce_mean(tf.cast(tf.equal(pred, Y), dtype=tf.float32))
loss = []
accuracy = []
with tf.Session() as sess:
    sess.run(init)
    file_writer = tf.summary.FileWriter("iris/train", sess.graph)
    for i in range(1000):
        sess.run(optimizer, feed_dict={X: x, Y:y})
        loss.append(sess.run(cross_entropy, feed_dict={X:x, Y:y}))
        accuracy.append(sess.run(accu, feed_dict={X:x, Y:y}))
        if i%100==0:
            print(sess.run(cross_entropy, feed_dict={X:x, Y:y}))
            #print(sess.run(W))
    print(sess.run(accu, feed_dict={X:x, Y:y}))       
plt.plot(accuracy)
plt.plot(loss)


# In[ ]:


from IPython.core.display import display, HTML

html_string = """
<blockquote class="imgur-embed-pub" lang="en" data-id="UEsX76x"><a href="//imgur.com/UEsX76x"></a></blockquote><script async src="//s.imgur.com/min/embed.js" charset="utf-8"></script>
"""
h = display(HTML(html_string))


# Notice there is a very interesting behavior from the figure above,  the blue line is accuracy while the brownish yellow is the loss, the accuracy already reached 1 even when the loss is still decreasing. Logstic regression decision boundaries is not fixed as in SVM (linearly separable dataset).  Cross-entropy loss, or log loss, measures the performance of a classification model whose output is a probability value between 0 and 1. In fact, cross-entropy loss can be used in regression if the regression output is in between 0 and 1.

# In[ ]:




