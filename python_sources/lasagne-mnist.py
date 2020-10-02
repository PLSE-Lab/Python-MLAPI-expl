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
    crop = 2
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
        ('conv1', layers.Conv2DLayer),
        ('pool1', layers.MaxPool2DLayer),
        ('conv2', layers.Conv2DLayer),
        ('pool2', layers.MaxPool2DLayer),
        ('dropout2', layers.DropoutLayer),
        ('conv3', layers.Conv2DLayer),
       # ('pool3', layers.MaxPool2DLayer),
        ('dropout3', layers.DropoutLayer),
        ('hidden4', layers.DenseLayer),
        ('dropout4', layers.DropoutLayer),
        ('hidden5', layers.DenseLayer),
        ('output', layers.DenseLayer),
        ],
    #MAde 26 instead of 24 , due to smaller crop used:
    input_shape=(None, 1, 26, 26),
    conv1_num_filters=32, conv1_filter_size=(3, 3), conv1_stride=(2,2), pool1_pool_size=(2, 2),
    conv2_num_filters=32, conv2_filter_size=(2, 2), pool2_pool_size=(2, 2), 
    conv3_num_filters=32, conv3_filter_size=(2, 2),# pool3_pool_size=(2, 2), 
    dropout3_p = 0.4,
    hidden4_num_units=32, dropout4_p=0.4,
    hidden5_num_units=32,
    output_num_units=10, 
    output_nonlinearity=softmax,

    batch_iterator_train=CropBatchIterator(batch_size=128),
    batch_iterator_test=CropBatchIterator(batch_size=128),

    update=nesterov_momentum,
    update_learning_rate=0.01,
    update_momentum=0.9,

    use_label_encoder=True,
    regression=False,  
    max_epochs=32,  
    verbose=1,
    )

X = train_images.reshape(-1,1,28,28)
y = train_labels
_ = net5.fit(X, y)