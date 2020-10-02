#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra


# ## linear fit

# ### sample data

# In[ ]:


num_data = 100
significant_figure = 3
data_x = np.round(np.sort(np.random.random_sample(num_data)) - 0.5, significant_figure)
W, B = 0.9, 0.3
sigma = (np.random.random_sample(num_data) - 0.5) / 10
data_y = W*data_x + B + sigma


# ### model

# In[ ]:


from keras.models import Model
from keras.layers import Input, Dense
from keras.optimizers import SGD

x = Input(shape=(1,))
y = Dense(units=1, activation='linear')(x)

# y = w*x + b
model = Model(inputs=x, output=y)
model.summary()


# ### compile, training
# iterate training N times, take average as fitted value

# In[ ]:


def train(data_x, data_y, model, lr=.5, verbose=0):
    model.compile(loss='mean_squared_error', optimizer=SGD(lr=lr), metrics=['mse'])
    model.fit(data_x, data_y, epochs=10, verbose=verbose)
    calc_w, calc_b = model.get_weights()
    return calc_w.reshape(1), calc_b

N = 5
array_w, array_b = np.zeros(N), np.zeros(N)
for i in range(N):
    tmp_w, tmp_b = train(data_x, data_y, model)
    array_w[i] = tmp_w
    array_b[i] = tmp_b

fit_w = np.average(array_w)
std_w = np.std(array_w)
fit_b = np.average(array_b)
std_b = np.std(array_b)


# In[ ]:


import matplotlib.pyplot as plt

print('w={0:.3f}({1:.0f}), b={2:.3f}({3:.0f})'
      .format(fit_w, std_w*np.power(10,significant_figure), fit_b, std_b*np.power(10,significant_figure)))
calc_y = fit_w*data_x + fit_b
plt.plot(data_x, data_y, 'bo')
plt.plot(data_x, calc_y, 'r-')
plt.show()


# ## quadratic function

# ### sample data
# use the same data_x above

# In[ ]:


data_y = W*np.power(data_x, 2) + B + sigma


# ### model

# In[ ]:


from keras.layers import Lambda

x = Input(shape=(1,))
x_squared = Lambda(lambda x:np.power(x,2))(x)
y = Dense(units=1, activation='linear')(x_squared)

# y = w*pow(x,2) + b
model = Model(inputs=x, output=y)
model.summary()


# ### train

# In[ ]:


N = 5
array_w, array_b = np.zeros(N), np.zeros(N)
for i in range(N):
    tmp_w, tmp_b = train(data_x, data_y, model, lr=.8)
    array_w[i] = tmp_w
    array_b[i] = tmp_b

fit_w = np.average(array_w)
std_w = np.std(array_w)
fit_b = np.average(array_b)
std_b = np.std(array_b)


# In[ ]:


print('w={0:.3f}({1:.0f}), b={2:.3f}({3:.0f})'
      .format(fit_w, std_w*np.power(10,significant_figure), fit_b, std_b*np.power(10,significant_figure)))
calc_y = fit_w*np.power(data_x,2) + fit_b
plt.plot(data_x, data_y, 'bo')
plt.plot(data_x, calc_y, 'r-')
plt.show()

