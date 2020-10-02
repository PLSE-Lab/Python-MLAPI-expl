import pandas as pd
from sklearn import metrics
from sklearn.decomposition import PCA
import numpy
import theano
import theano.tensor as T
rng = numpy.random

import time
start = time.time()
def elapsed(): return time.time() - start

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train_x = train.ix[:,1:].values.astype('uint8')
train_y = train.ix[:,0].values.astype('uint8')
test_x = test.ix[:,:].values.astype('uint8')

# Interesting to note: for SVM, PCA was used to increase performance
# for KNN, PCA does not affect performance but it reduces the running time of script by one third.
print('PCA reduction at %ds' % elapsed())
pca = PCA(n_components=36, whiten=True)
pca.fit(train_x)
feats_before_pca = len(train_x[0])
train_x = pca.transform(train_x)
test_x = pca.transform(test_x)

feats = len(train_x[0])
print('Num feats: %d->%d using PCA' % (feats_before_pca, feats))
training_steps = 10000

print('Declare Theano symbolic variables at %ds' % elapsed())
x = T.matrix("x")
y = T.vector("y")
w = theano.shared(rng.randn(feats), name="w")
b = theano.shared(0., name="b")


print('Construct Theano expression graph at %ds' % elapsed())
p_1 = 1 / (1 + T.exp(-T.dot(x, w) - b))   # Probability that target = 1
prediction = p_1 > 0.5                    # The prediction thresholded
xent = -y * T.log(p_1) - (1-y) * T.log(1-p_1) # Cross-entropy loss function
cost = xent.mean() + 0.01 * (w ** 2).sum()# The cost to minimize
gw, gb = T.grad(cost, [w, b])             # Compute the gradient of the cost
                                          # (we shall return to this in a
                                          # following section of this tutorial)

print('Compiling Theano function at %ds' % elapsed())
train_fn = theano.function(
              inputs=[x,y],
              outputs=[prediction, xent],
              updates=((w, w - 0.1 * gw), (b, b - 0.1 * gb)))

predict_fn = theano.function(inputs=[x], outputs=prediction)

print('Training theano linear regression model at %ds' % elapsed())
for i in range(training_steps):
        pred, err = train_fn(train_x, train_y)

print('Output at %ds' % elapsed())
test_y = predict_fn(test_x)
pd.DataFrame({"ImageId": range(1,len(test_y)+1), "Label": test_y}).to_csv('out.csv', index=False, header=True)

print('Exit at %ds' % elapsed())