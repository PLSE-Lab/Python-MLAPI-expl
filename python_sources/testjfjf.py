import csv
import numpy as np
import scipy
import pandas
from numpy.random import randint
# Lasagne (& friends) imports
import theano
from nolearn.lasagne import BatchIterator, NeuralNet
from lasagne.objectives import aggregate, binary_crossentropy
from lasagne.layers import InputLayer, DropoutLayer, DenseLayer,Conv1DLayer,MaxPool1DLayer
from lasagne.updates import nesterov_momentum
from theano.tensor.nnet import sigmoid
import theano.tensor as T
import gc

# Silence some warnings from lasagne
import warnings
warnings.filterwarnings('ignore', '.*topo.*')
warnings.filterwarnings('ignore', module='.*lasagne.init.*')
warnings.filterwarnings('ignore', module='.*nolearn.lasagne.*')

# We are reusing the training and loading of the CSV files from many of
# the scripts of previous teams.  Our approach was to create a Naive Neural
# Network to classify and predict that would yield the best results based 
# on a mixture of parameters and many different trial runs.  If time wasn't
# an issue the more time steps you take and the larger the sample would
# yield better results accordingly.


SUBJECTS = list(range(1,13))
TRAIN_SERIES = list(range(1,9))
TEST_SERIES = [9,10]

N_ELECTRODES = 32
N_EVENTS = 6

TRAIN_SIZE = 5*1024 

class Source:

    mean = None
    std = None

    def load_raw_data(self, subject, series):
        raw_data = [self.read_csv(self.path(subject, i, "data")) for i in series]
        self.data = np.concatenate(raw_data, axis=0)
        raw_events = [self.read_csv(self.path(subject, i, "events")) for i in series]
        self.events = np.concatenate(raw_events, axis=0)
    
    def normalize(self):
        self.data = self.data - self.mean
        self.data /= self.std
        
    @staticmethod
    def path(subject, series, kind):
        prefix = "train" if (series in TRAIN_SERIES) else "test"
        return "../input/{0}/subj{1}_series{2}_{3}.csv".format(prefix, subject, series, kind)
    
    csv_cache = {}
    @classmethod
    def read_csv(klass, path):
        if path not in klass.csv_cache:
            if len(klass.csv_cache):
                klass.csv_cache.popitem()
            klass.csv_cache[path] = pandas.read_csv(path, index_col=0).values
        return klass.csv_cache[path]
        
class TrainSource(Source):

    def __init__(self, subject, series_list):
        self.load_raw_data(subject, series_list)
        self.mean = self.data.mean(axis=0)
        self.std = self.data.std(axis=0)
        self.normalize()
        self.principle_components = scipy.linalg.svd(self.data, full_matrices=False)
        self.std2 = self.data.std(axis=0)
        self.data /= self.std2

        
class TestSource(Source):

    def __init__(self, subject, series, train_source):
        self.load_raw_data(subject, series)
        self.mean = train_source.mean
        self.std = train_source.std
        self.principle_components = train_source.principle_components
        self.normalize()
        self.data /= train_source.std2
        

class SubmitSource(TestSource):

    def __init__(self, subject, a_series, train_source):
        TestSource.__init__(self, subject, [a_series], train_source)

    def load_raw_data(self, subject, series):
        [a_series] = series
        self.data = self.read_csv(self.path(subject, a_series, "data"))
        
        
# Lay out the Neural net.


SAMPLE_SIZE = 2048
DOWNSAMPLE = 8 
TIME_POINTS = SAMPLE_SIZE // DOWNSAMPLE
 
# This is where we classify our data.  the NeuralNet.fit and NeuralNet.predict_proba
# methods were not playing nicely with our data so followed format in example.

class IndexBatchIterator(BatchIterator):
    def __init__(self, source, *args, **kwargs):
        super(IndexBatchIterator, self).__init__(*args, **kwargs)
        self.source = source
        if source is not None:
            x = source.data
            self.augmented = np.zeros([len(x)+(SAMPLE_SIZE-1), N_ELECTRODES], dtype=np.float32)
            self.augmented[SAMPLE_SIZE-1:] = x
            self.augmented[:SAMPLE_SIZE-1] = x[0]
        self.Xbuf = np.zeros([self.batch_size, N_ELECTRODES, TIME_POINTS], np.float32) 
        self.Ybuf = np.zeros([self.batch_size, N_EVENTS], np.float32) 
    
    def transform(self, X_indices, y_indices):
        X_indices, y_indices = super(IndexBatchIterator, self).transform(X_indices, y_indices)
        [count] = X_indices.shape
        # Use preallocated space
        X = self.Xbuf[:count]
        Y = self.Ybuf[:count]
        for i, ndx in enumerate(X_indices):
            if ndx == -1:
                ndx = np.random.randint(len(self.source.events))
            sample = self.augmented[ndx:ndx+SAMPLE_SIZE]
            X[i] = sample[::-1][::DOWNSAMPLE].transpose()
            if y_indices is not None:
                Y[i] = self.source.events[ndx]
        Y = None if (y_indices is None) else Y
        return X, Y
    

# Creating a Neural Network with multiple layers for efficiency
# and accuracy
    
def create_net(train_source, test_source, batch_size=128, max_epochs=20): 

    def loss(x,t):
        return aggregate(binary_crossentropy(x, t))

    nnet =  NeuralNet(
        y_tensor_type = theano.tensor.matrix,
        layers = [
            ('input', InputLayer),
            ('conv1', Conv1DLayer),
            ('pool1', MaxPool1DLayer),
            ('dropout1', DropoutLayer),
            ('conv2', Conv1DLayer),
            ('pool2', MaxPool1DLayer),
            ('dropout2', DropoutLayer),
            ('conv3', Conv1DLayer),
            ('pool3', MaxPool1DLayer),
            ('dropout3', DropoutLayer),
            ('hidden4', DenseLayer),
            ('dropout4', DropoutLayer),
            ('hidden5', DenseLayer),
            ('output', DenseLayer)
        ],

        input_shape=(None, N_ELECTRODES, TIME_POINTS),
        conv1_num_filters=32, conv1_filter_size=3, pool1_pool_size=2,
        dropout1_p=0.1,
        conv2_num_filters=64, conv2_filter_size=2, pool2_pool_size=2,
        dropout2_p=0.2,
        conv3_num_filters=128, conv3_filter_size=2, pool3_pool_size=2,
        dropout3_p=0.3,
        hidden4_num_units = 1024,
        dropout4_p=0.5,
        hidden5_num_units = 1024,
        output_num_units = N_EVENTS, output_nonlinearity = None,
        batch_iterator_train = IndexBatchIterator(train_source, batch_size=batch_size),
        batch_iterator_test = IndexBatchIterator(test_source, batch_size=batch_size),
        max_epochs=max_epochs,
        verbose=1,
        update = nesterov_momentum, 
        update_learning_rate = 0.03,
        update_momentum = 0.9,
        objective_loss_function = loss,
        regression = True
        )

    return nnet


# Do the training.

train_indices = np.zeros([TRAIN_SIZE], dtype=int) - 1

def train(factory, subj, max_epochs=20, valid_series=[1,2], params=None):
    tseries = sorted(set(TRAIN_SERIES) - set(valid_series))
    train_source = TrainSource(subj, tseries)
    test_source = TestSource(subj, valid_series, train_source)
    net = factory(train_source, test_source, max_epochs=max_epochs)
    if params is not None:
        net.load_params_from(params)
    net.fit(train_indices, train_indices)
    return (net, train_source)
 

def train_all(factory, max_epochs=30, init_epochs=30, valid_series=[1,2]):
    info = {}
    params = None
    for subj in SUBJECTS:
        print("Subject:", subj)
        epochs = max_epochs + init_epochs
        net, train_source = train(factory, subj, epochs, valid_series, params)
        params = net.get_all_params_values()
        info[subj] = (params, train_source)
        init_epochs = 0
    return (factory, info)   
  
 
def make_submission(train_info, name):
    factory, info = train_info
    with open(name, 'w') as file:
        file.write("id,HandStart,FirstDigitTouch,BothStartLoadPhase,LiftOff,Replace,BothReleased\n")
        for subj in SUBJECTS:
            weights, train_source = info[subj]
            for series in [9,10]:
                print("Subject:", subj, ", series:", series)
                submit_source = SubmitSource(subj, series, train_source)  
                indices = np.arange(len(submit_source.data))
                net = factory(train_source=None, test_source=submit_source)
                net.load_weights_from(weights)
                probs = net.predict_proba(indices)
                for i, p in enumerate(probs):
                    id = "subj{0}_series{1}_{2},".format(subj, series, i)
                    file.write(id + ",".join(str(x) for x in p) + '\n')
        
        
if __name__ == "__main__":
    train_info = train_all(create_net, max_epochs=3)
    make_submission(train_info, "finch_sledge3.csv") 