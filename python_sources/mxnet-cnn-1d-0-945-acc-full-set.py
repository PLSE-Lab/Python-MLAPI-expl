
### mxnet 1.1.0 - cnn_1d - 0.945 ACC ON 20% TEST

import numpy as np
seed = 2018
np.random.seed(seed)

import os
from time import time

from tqdm import tqdm

import librosa

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import mxnet as mx
mx.random.seed(seed)
####################################################################

ctx = mx.gpu()

model_name = 'mxnet_cnn_1d'
saved_params = model_name + '.params'
saved_log = model_name + '.log'

input_shape = (5000, 1)

batch_size = 32
epochs = 100
es_patience = 8
rlr_patience = 5
rlr_factor = 0.1

SR = 8000
####################################################################

target_names = ['Ae. aegypti', 'Ae. albopictus', 'An. gambiae', 'An. arabiensis', 'C. pipiens', 'C. quinquefasciatus']

X_names = []
y = []
target_count = []

for i, target in enumerate(target_names):
    target_count.append(0)
    path = './Wingbeats/' + target + '/'
    for [root, dirs, files] in os.walk(path, topdown = False):
        for filename in files:
            name,ext = os.path.splitext(filename)
            if ext == '.wav':
                name = os.path.join(root, filename)
                y.append(i)
                X_names.append(name)
                target_count[i]+=1
                # if target_count[i] > 20000:
                #     break

    print (target, '#recs = ', target_count[i])

print ('total #recs = ', len(y))

X_names, y = shuffle(X_names, y, random_state = seed)
X_train, X_valid, y_train, y_valid = train_test_split(X_names, y, stratify = y, test_size = 0.20, random_state = seed)

print ('train #recs = ', len(X_train))
print ('test #recs = ', len(X_valid))
####################################################################

def random_data_shift(data, u):
    if np.random.random() < u:
        data = np.roll(data, int(round(np.random.uniform(-(len(data)*0.25), (len(data)*0.25)))))
    return data

def load_data(data_paths, data_labels, mode = None):
    data_batch = []
    
    for i in range(len(data_paths)):
        data, rate = librosa.load(data_paths[i], sr = SR)
        
        if mode == 'augm':
            data = random_data_shift(data, u = 1.0)

        data_batch.append(data)

    data_batch = np.array(data_batch, np.float32)
    data_labels = np.array(data_labels, np.float32)

    data_batch = np.expand_dims(data_batch, axis = -1)

    data_batch = np.transpose(data_batch, (0, 2, 1))

    data_batch = mx.nd.array(data_batch, ctx)
    data_labels = mx.nd.array(data_labels, ctx)

    train_iter = mx.io.NDArrayIter(data = data_batch, label = data_labels, batch_size = len(data_labels))

    return train_iter
####################################################################

def get_net(num_classes):
    data = mx.sym.var('data')

    bn0 = mx.sym.BatchNorm(data = data)

    conv1 = mx.sym.Convolution(data = bn0, num_filter = 16, kernel = (3,), pad = (1,))
    bn1 = mx.sym.BatchNorm(data = conv1)
    act1 = mx.sym.Activation(data = bn1, act_type = "relu")
    pool1 = mx.sym.Pooling(data = act1, pool_type = "max", kernel = (2,), stride = (2,))

    conv2 = mx.sym.Convolution(data = pool1, num_filter = 32, kernel = (3,), pad = (1,))
    bn2 = mx.sym.BatchNorm(data = conv2)
    act2 = mx.sym.Activation(data = bn2, act_type = "relu")
    pool2 = mx.sym.Pooling(data = act2, pool_type = "max", kernel = (2,), stride = (2,))

    conv3 = mx.sym.Convolution(data = pool2, num_filter = 64, kernel = (3,), pad = (1,))
    bn3 = mx.sym.BatchNorm(data = conv3)
    act3 = mx.sym.Activation(data = bn3, act_type = "relu")
    pool3 = mx.sym.Pooling(data = act3, pool_type = "max", kernel = (2,), stride = (2,))

    conv4 = mx.sym.Convolution(data = pool3, num_filter = 128, kernel = (3,), pad = (1,))
    bn4 = mx.sym.BatchNorm(data = conv4)
    act4 = mx.sym.Activation(data = bn4, act_type = "relu")
    pool4 = mx.sym.Pooling(data = act4, pool_type = "max", kernel = (2,), stride = (2,))

    conv5 = mx.sym.Convolution(data = pool4, num_filter = 256, kernel = (3,), pad = (1,))
    bn5 = mx.sym.BatchNorm(data = conv5)
    act5 = mx.sym.Activation(data = bn5, act_type = "relu")
    pool5 = mx.sym.Pooling(data = act5, pool_type = "max", kernel = (2,), stride = (2,))

    conv6 = mx.sym.Convolution(data = pool5, num_filter = 512, kernel = (3,), pad = (1,))
    bn6 = mx.sym.BatchNorm(data = conv6)
    act6 = mx.sym.Activation(data = bn6, act_type = "relu")
    pool6 = mx.sym.Pooling(data = act6, pool_type = "max", kernel = (2,), stride = (2,))

    globpool = mx.sym.Pooling(data = pool6, global_pool = True, pool_type = "avg", kernel = (1,))

    drop1 = mx.symbol.Dropout(data = globpool, p = 0.5)

    fc = mx.sym.FullyConnected(data = drop1, num_hidden = num_classes)
    net = mx.sym.SoftmaxOutput(data = fc, name = 'softmax')

    return net
####################################################################

net = get_net(len(target_names))

model = mx.mod.Module(symbol = net, 
    data_names = ('data', ), 
    label_names = ('softmax_label', ), 
    context = ctx)

model.bind(for_training = True,
    data_shapes = [('data', (batch_size, input_shape[1], input_shape[0]))], 
    label_shapes = [('softmax_label', (batch_size,))])

model.init_params()

nag = mx.optimizer.Optimizer.create_optimizer('nag', rescale_grad = 1./batch_size, wd = 0.000001)
model.init_optimizer(optimizer = nag, optimizer_params = (('learning_rate', 0.01), 
    ('momentum', 0.9), 
    ('lazy_update', True),
    ('')))

train_metric = mx.metric.create('acc')
valid_metric = mx.metric.create('acc')
####################################################################

es_counter = 0
rlr_counter = 0

best_acc = 0.0
total_time = 0.0

t_acc = 0.0
v_acc = 0.0

with open(saved_log, 'wb') as my_file:
    my_file.write("\nseed: " + str(seed) 
        + " \ninput_shape: " + str(input_shape)
        + " \nbatch_size: " + str(batch_size)
        + " \nepochs: " + str(epochs)
        + " \nes_patience: " + str(es_patience)
        + " \nrlr_patience: " + str(rlr_patience)
        + " \nrlr_factor: " + str(rlr_factor)
        + " \nSR: " + str(SR)
        + "\n\n")

for epoch in range(epochs):
    print '\033[93m' + "Epoch: " + str(epoch+1) + '\033[0m'

    with open(saved_log, 'a') as my_file:
        my_file.write("Epoch: " + str(epoch+1) + '\n')

    start_time = time()

    train_metric.reset()
    valid_metric.reset()
    ####################################################################

    train_tqdm = tqdm(range(0, len(X_train), batch_size), ascii = True)
    train_tqdm.set_description("Train_acc: " + str(("{0:.5f}".format(
                round(t_acc,5)))) + " ###")

    for start in train_tqdm:
        end = min(start + batch_size, len(X_train))

        train_iter = load_data(X_train[start:end], y_train[start:end], mode = 'augm')

        for train_batch in train_iter:
            model.forward(train_batch, is_train = True)
            model.update_metric(train_metric, train_batch.label)  
            model.backward()                          
            model.update()

            t_acc = train_metric.get()[1]
            
            train_tqdm.set_description("Train_acc: " + str(("{0:.5f}".format(
                round(t_acc,5)))) + " ###") 

    with open(saved_log, 'a') as my_file:
        my_file.write("Train_acc: %s" % ( 
            ("{0:.5f}".format(round(t_acc,5)))))
    ####################################################################
   
    valid_tqdm = tqdm(range(0, len(X_valid), batch_size), ascii = True)
    valid_tqdm.set_description("Valid_acc: " + str(("{0:.5f}".format(
                round(v_acc,5)))) + " ###")

    for start in valid_tqdm:
        end = min(start + batch_size, len(X_valid))

        valid_iter = load_data(X_valid[start:end], y_valid[start:end], mode = None)

        for valid_batch in valid_iter:
            model.forward(valid_batch, is_train = False)
            model.update_metric(valid_metric, valid_batch.label)

            v_acc = valid_metric.get()[1]

            valid_tqdm.set_description("Valid_acc: " + str(("{0:.5f}".format(
                round(v_acc,5)))) + " ###")

    with open(saved_log, 'a') as my_file:
        my_file.write("\nValid_acc: %s" % (
            ("{0:.5f}".format(round(v_acc,5)))
            + "\n"))

    end_time = time()
    ####################################################################

    es_counter+=1
    rlr_counter+=1

    if v_acc > best_acc:
        print '\033[91m' + '\tvalid_acc improved:', str(("{0:.5f}".format(
            round(best_acc,5)))), '--->', str(("{0:.5f}".format(
            round(v_acc,5)))) + '\033[0m'
        
        with open(saved_log, 'a') as my_file:
            my_file.write('\tvalid_acc improved: ' + str(("{0:.5f}".format(
                round(best_acc,5)))) + ' ---> ' + str(("{0:.5f}".format(
                round(v_acc,5))))
                + "\n")

        model.save_params(saved_params)
        best_acc = v_acc

        es_counter = 0
        rlr_counter = 0

    epoch_time = end_time - start_time
    total_time += epoch_time

    print "Epoch_sec: %s \nTotal_sec: %s" % ( 
        ("{0:.5f}".format(round(epoch_time,5))),
        ("{0:.5f}".format(round(total_time,5))))

    with open(saved_log, 'a') as my_file:
        my_file.write("Epoch_sec: %s \nTotal_sec: %s" % ( 
            ("{0:.5f}".format(round(epoch_time,5))),
            ("{0:.5f}".format(round(total_time,5)))
            + "\n"))
    ####################################################################

    if es_counter == es_patience:
        print '\033[93m' + "Early stopped!" + '\033[0m'
        
        with open(saved_log, 'a') as my_file:
            my_file.write("Early stopped!"
                + "\n")
        
        break

    if rlr_counter == rlr_patience:
        nag.set_learning_rate(nag.learning_rate * rlr_factor)

        print '\033[93m' + "Reduced learning rate: " + str(
            nag.learning_rate / rlr_factor) + ' ---> ' + str(
            nag.learning_rate) + '\033[0m'

        with open(saved_log, 'a') as my_file:
            my_file.write("Reduced learning rate: " + str(
                nag.learning_rate / rlr_factor) + ' ---> ' + str(
                nag.learning_rate) 
                + "\n")

        rlr_counter = 0
