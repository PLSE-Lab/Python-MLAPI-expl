# calculate cross entropy loss for binary classification with keras
from numpy import asarray
from keras import backend
from keras.losses import binary_crossentropy

# define classification data
t = asarray([1, 1, 1, 0, 0, 0]) # true classification
p = asarray([0.8, 0.9, 0.6, 0.1, 0.4, 0.2]) # predicted classification 

# convert to keras variables
y_true = backend.variable(t)
y_pred = backend.variable(p)

# calculate the average cross-entropy
mean_ce = backend.eval(binary_crossentropy(y_true, y_pred))
print('Average Cross Entropy: %.3f nats' % mean_ce)