#!/usr/bin/env python
# coding: utf-8

# ## Factorization Machines 

# Factorization Machines model all interactions between variables using factorized parameters. The model equation for a factorization machine of degree $d=2$ is defined as:
# $$
# \hat{y}(x) = \omega_0 + \sum_{i=1}^n\omega_ix_i + \sum_{i=1}^n\sum_{j=i+1}^n v_i^Tv_jx_ix_j.
# $$

# The complexity of straight forward computation of the equation is in $\mathcal{O}(kn^2)$. However, we have:
# \begin{align}
# \sum_{i=1}^n\sum_{j=i+1}^n v_i^Tv_jx_ix_j &= \frac{1}{2}\sum_{i=1}^n\sum_{j=1}^n v_i^Tv_jx_ix_j - \frac{1}{2}\sum_{i=1}^n v_i^Tv_ix_ix_i\\
# &= \frac{1}{2}(\sum_{i=1}^n\sum_{j=1}^n\sum_{f=1}^k v_{if}v_{jf}x_ix_j - \sum_{i=1}^n\sum_{f=1}^k v_{if}v_{if}x_ix_i)\\
# &= \frac{1}{2}\sum_{f=1}^k((\sum_{i=1}^n v_{if}x_i)^2 - \sum_{i=1}^n v_{if}^2x_i^2)
# \end{align}
# This equation has only linear complexity in both $k$ and $n$, i.e., its computation is in $\mathcal{O}(kn)$.

# ## d-way Factorization Machines

# The 2-way FM described so far can easily be generalized to a d-way FM:
# $$
# \hat{y}(x) = \omega_0 + \sum_{i=1}^n\omega_ix_i + \sum_{l=2}^d\sum_{i_1=1}^n\cdot\cdot\cdot\sum_{i_l=i_{l-1}+1}^n(\prod_{j=1}^l x_{i_j})(\sum_{f=1}^{k_l}\prod_{j=1}^l v_{i_jf}^{(l)}).
# $$

# ## Implementation

# In[ ]:


from sklearn import metrics
import tensorflow as tf
import numpy as np

# evaluation
def get_auc(y, y_pre):
    fpr, tpr, thresholds = metrics.roc_curve(y.astype(int), y_pre, pos_label=1)
    return metrics.auc(fpr, tpr)

# hyper-parameters
vector_dim = 8
learning_rate = 1e-4
l2_factor = 1e-2
max_training_step = 400
train_rate = 0.8

# split data
data = np.loadtxt(fname='/kaggle/input/data', delimiter='\t')
thredhold = int(train_rate * len(data))
x_train = data[:thredhold, :-1]
y_train = data[:thredhold, -1]
x_test = data[thredhold:, :-1]
y_test = data[thredhold:, -1]
feature_num = len(data[0])-1

# construct graph
# model parameters
w_0 = tf.Variable(0.0)
w = tf.Variable(tf.zeros(shape=[feature_num]))
v = tf.Variable(tf.truncated_normal(shape=[feature_num, vector_dim], mean=0.0, stddev=0.01))

# construct loss
x = tf.placeholder(shape=[None, feature_num], dtype=tf.float32)
y = tf.placeholder(shape=[None], dtype=tf.float32)

linear_term = w_0 + tf.reduce_sum(tf.expand_dims(w, axis=0) * x, axis=1)
square_of_sum = tf.square(tf.reduce_sum(tf.expand_dims(x, axis=2) * tf.expand_dims(v, axis=0), axis=1))
sum_of_square = tf.reduce_sum(tf.square(tf.expand_dims(v, axis=0) * tf.expand_dims(x, axis=2)), axis=1)
y_pre = tf.sigmoid(linear_term + 0.5 * tf.reduce_sum(square_of_sum - sum_of_square, axis=1))

cross_entropy = - y * tf.log(y_pre) - (1 - y) * tf.log(1 - y_pre)
train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy + l2_factor * tf.add_n([tf.nn.l2_loss(item) for item in [w_0, w, v]]))

accuracy = tf.reduce_mean(tf.cast(tf.less(tf.abs(y - y_pre), 0.5), dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(max_training_step):
        sess.run(train_op, {x: x_train, y: y_train})
        acc, y_pre_value = sess.run([accuracy, y_pre], {x: x_test, y: y_test})
        print('----step%3d accuracy: %.3f, auc: %.3f' % (step, acc, get_auc(y_test, y_pre_value)))


# ## Some References

# 1. Rendle, Steffen. "Factorization machines." Data Mining (ICDM), 2010 IEEE 10th International Conference on. IEEE, 2010.
# 2. Juan, Yuchin, et al. "Field-aware factorization machines for CTR prediction." Proceedings of the 10th ACM Conference on Recommender Systems. ACM, 2016.
# 3. Guo, Huifeng, et al. "Deepfm: a factorization-machine based neural network for ctr prediction." arXiv preprint arXiv:1703.04247 (2017).
# 4. Qu, Yanru, et al. "Product-based neural networks for user response prediction." Data Mining (ICDM), 2016 IEEE 16th International Conference on. IEEE, 2016.
