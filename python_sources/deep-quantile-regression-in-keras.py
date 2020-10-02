#!/usr/bin/env python
# coding: utf-8

# # Deep Quantile Regression in Keras

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from scipy import stats


# We build a synthetic dataset

# In[ ]:


def mu(x):
    return np.sin(np.pi * x)


def sigma(x):
    return 0.5 + np.where(x > 0, 1, 2) * np.abs(x)


X = np.random.uniform(-3, 3, (5000, 1))
y = np.random.normal(mu(X), sigma(X))
X_out = np.concatenate([np.random.uniform(-5, -3, (1000, 1)), np.random.uniform(3, 5, (1000, 1))], 0)
y_out  = np.random.normal(mu(X_out), sigma(X_out))

plt.figure(figsize=(16, 8))
plt.scatter(X, y, alpha=0.5, color='tab:blue')
plt.scatter(X_out, y_out, alpha=0.5, color='tab:orange');


# Then we define the quantile-loss function. The huber-like loss is preferable to the absolute error since it is differentiable

# In[ ]:


def QuantileLoss(perc, delta=1e-4):
    perc = np.array(perc).reshape(-1)
    perc.sort()
    perc = perc.reshape(1, -1)
    def _qloss(y, pred):
        I = tf.cast(y <= pred, tf.float32)
        d = K.abs(y - pred)
        correction = I * (1 - perc) + (1 - I) * perc
        # huber loss
        huber_loss = K.sum(correction * tf.where(d <= delta, 0.5 * d ** 2 / delta, d - 0.5 * delta), -1)
        # order loss
        q_order_loss = K.sum(K.maximum(0.0, pred[:, :-1] - pred[:, 1:] + 1e-6), -1)
        return huber_loss + q_order_loss
    return _qloss


# In[ ]:


perc_points = [0.01, 0.25, 0.5, 0.75, 0.99]
model = keras.Sequential([
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(5)
])
model.compile(optimizer=keras.optimizers.Adam(2e-3), loss=QuantileLoss(perc_points))
model.fit(X, y, epochs=100, verbose=0)


# In[ ]:


xx = np.linspace(X_out.min(), X_out.max(), 500)
pred = model.predict(xx)

plt.figure(figsize=(16, 8))
plt.scatter(X, y, alpha=0.5, color='tab:blue')
plt.scatter(X_out, y_out, alpha=0.5, color='tab:orange')
plt.plot(xx, pred, color='tab:red', linestyle='--')
plt.xlabel('X')
plt.ylabel('y');


# Now we visualize the real cumulative distribution in various points VS the one estimated by the model

# In[ ]:


xs = np.array([-4, -2, 0, 2, 4]).reshape(-1, 1)
for i in range(xs.shape[0]):
    x0 = xs[i:(i + 1), :]
    mu0, sigma0 = mu(x0).squeeze(), sigma(x0).squeeze()
    z = np.linspace(mu0 - 4 * sigma0, mu0 + 4 * sigma0, 100)
    p = stats.norm(mu0, sigma0).cdf(z).reshape(-1)
    plt.figure(figsize=(10, 5))
    plt.plot(z, p, color='tab:blue', label='CDF')
    plt.scatter(model.predict(x0), perc_points, color='tab:red', label='NN')
    plt.title(f"x = {x0.squeeze()}")
    plt.legend()
    plt.ylim(0, 1)


# In[ ]:




