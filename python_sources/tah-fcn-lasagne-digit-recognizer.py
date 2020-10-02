import numpy as np
import pandas as pd
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from theano.tensor.nnet import softmax
from nolearn.lasagne import BatchIterator

train = pd.read_csv("../input/train.csv")
test  = pd.read_csv("../input/test.csv")


train_labels = train.ix[:,0].values.astype(np.int32)
train_images = train.ix[:,1:].values.astype(np.float32)
train_images /= train_images.std(axis=None)
train_images -= train_images.mean()

class CropBatchIterator(BatchIterator):
    crop = 4
    def transform(self, Xb, yb):
        Xb, yb = super(CropBatchIterator, self).transform(Xb, yb)
        bs = Xb.shape[0]
        sz = 28 - self.crop
        new_Xb = np.zeros([bs, 1, sz, sz], dtype=np.float32)
        for i in range(bs):
            dx = np.random.randint(self.crop+1)
            dy = np.random.randint(self.crop+1)
            new_Xb[i] = Xb[i,:,dy:dy+sz,dx:dx+sz]
        return new_Xb, yb
    
net5 = NeuralNet(
    layers=[
        ('input', layers.InputLayer),
        ('hidden', layers.DenseLayer),
        ('output', layers.DenseLayer),
        ],
    input_shape=(None, 1, 28, 28),
    hidden_num_units=128,
    output_num_units=10,
    output_nonlinearity=softmax,


    update=nesterov_momentum,
    update_learning_rate=0.1,
    update_momentum=0.9,

    use_label_encoder=True,
    regression=False,  
    max_epochs=32,  
    verbose=1,
    )

X = train_images.reshape(-1,1,28,28)
y = train_labels
_ = net5.fit(X, y)