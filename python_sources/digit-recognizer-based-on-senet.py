#!/usr/bin/env python
# coding: utf-8

# # Title: Digit Recognizer based on SENet.  
# The original source code is provided at the following link.  
# https://github.com/YeongHyeon/Kaggle-MNIST  
# 
# The performance of this deep neural network is 0.99871. This is the 122th of 2270 teams (top 6%) as of November 29, 2019.  
# ![score](https://raw.githubusercontent.com/YeongHyeon/Kaggle-MNIST/master/figures/performance.png)

# # 1. Calling python libraries.

# In[ ]:


import os, inspect, warnings, argparse
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ["CUDA_VISIBLE_DEVICES"]='0'
warnings.filterwarnings('ignore')

import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from sklearn.utils import shuffle

PACK_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))


# # 2. Definition of the Dataset class (with MNIST dataset).

# In[ ]:


class Dataset(object):

    def __init__(self, normalize=True):

        print("\nInitializing Dataset...")

        self.normalize = normalize

        self.x_tr, self.y_tr = None, None
        self.x_te, self.y_te = None, None

        ftr = open("../input/digit-recognizer/train.csv", "r")
        while(True):
            content = ftr.readline()
            if not content: break
            if("pixel" in content): continue
            tmp_label = content.replace("\n", "").split(",")[0]
            tmp_data = np.reshape(np.asarray(content.replace("\n", "").split(",")[1:]), (1, 28, 28))
            if(self.y_tr is None):
                self.y_tr = tmp_label
                self.x_tr = tmp_data
            else:
                self.y_tr = np.append(self.y_tr, tmp_label)
                self.x_tr = np.append(self.x_tr, tmp_data, axis=0)
        ftr.close()

        fte = open("../input/digit-recognizer/test.csv", "r")
        while(True):
            content = fte.readline()
            if not content: break
            if("pixel" in content): continue
            tmp_label = 0
            tmp_data = np.reshape(np.asarray(content.replace("\n", "").split(",")), (1, 28, 28))
            if(self.y_te is None):
                self.y_te = tmp_label
                self.x_te = tmp_data
            else:
                self.y_te = np.append(self.y_te, tmp_label)
                self.x_te = np.append(self.x_te, tmp_data, axis=0)
        fte.close()

        self.x_tr = np.ndarray.astype(self.x_tr, np.float32)
        self.x_te = np.ndarray.astype(self.x_te, np.float32)

        self.num_tr, self.num_te = self.x_tr.shape[0], self.x_te.shape[0]
        self.idx_tr, self.idx_te = 0, 0

        print("Number of data\nTraining: %d, Test: %d\n" %(self.num_tr, self.num_te))

        x_sample, y_sample = self.x_te[0], self.y_te[0]
        self.height = x_sample.shape[0]
        self.width = x_sample.shape[1]
        try: self.channel = x_sample.shape[2]
        except: self.channel = 1

        self.min_val, self.max_val = x_sample.min(), x_sample.max()
        self.num_class = 10

        print("Information of data")
        print("Shape  Height: %d, Width: %d, Channel: %d" %(self.height, self.width, self.channel))
        print("Value  Min: %.3f, Max: %.3f" %(self.min_val, self.max_val))
        print("Class  %d" %(self.num_class))
        print("Normalization: %r" %(self.normalize))
        if(self.normalize): print("(from %.3f-%.3f to %.3f-%.3f)" %(self.min_val, self.max_val, 0, 1))

    def reset_idx(self): self.idx_tr, self.idx_te = 0, 0

    def label2vector(self, labels):

        labels_v = None
        for idx_l, label in enumerate(labels):
            tmp_v = np.expand_dims(np.eye(self.num_class)[int(label)], axis=0)
            if(labels_v is None): labels_v = tmp_v
            else: labels_v = np.append(labels_v, tmp_v, axis=0)

        return labels_v

    def next_train(self, batch_size=1, fix=False):

        start, end = self.idx_tr, self.idx_tr+batch_size
        x_tr, y_tr = self.x_tr[start:end], self.y_tr[start:end]
        x_tr = np.expand_dims(x_tr, axis=3)

        terminator = False
        if(end >= self.num_tr):
            terminator = True
            self.idx_tr = 0
            self.x_tr, self.y_tr = shuffle(self.x_tr, self.y_tr)
        else: self.idx_tr = end

        if(fix): self.idx_tr = start

        if(x_tr.shape[0] != batch_size):
            x_tr, y_tr = self.x_tr[-1-batch_size:-1], self.y_tr[-1-batch_size:-1]
            x_tr = np.expand_dims(x_tr, axis=3)

        if(self.normalize):
            min_x, max_x = x_tr.min(), x_tr.max()
            x_tr = (x_tr - min_x) / (max_x - min_x)

        return x_tr, self.label2vector(labels=y_tr), terminator

    def next_test(self, batch_size=1):

        start, end = self.idx_te, self.idx_te+batch_size
        x_te, y_te = self.x_te[start:end], self.y_te[start:end]
        x_te = np.expand_dims(x_te, axis=3)

        terminator = False
        if(end >= self.num_te):
            terminator = True
            self.idx_te = 0
        else: self.idx_te = end

        if(self.normalize):
            min_x, max_x = x_te.min(), x_te.max()
            x_te = (x_te - min_x) / (max_x - min_x)

        return x_te, self.label2vector(labels=y_te), terminator


# # 3. Definition of the SENet.

# In[ ]:


class SENet(object):

    def __init__(self, height, width, channel, num_class, leaning_rate=1e-3):

        print("\nInitializing Neural Network...")
        self.height, self.width, self.channel = height, width, channel
        self.num_class, self.k_size = num_class, 3
        self.leaning_rate = leaning_rate

        self.x = tf.placeholder(tf.float32, [None, self.height, self.width, self.channel])
        self.y = tf.placeholder(tf.float32, [None, self.num_class])
        self.batch_size = tf.placeholder(tf.int32, shape=[])

        self.weights, self.biasis = [], []
        self.w_names, self.b_names = [], []

        self.y_hat = self.build_model(input=self.x)

        self.smce = tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.y_hat)
        self.loss = tf.reduce_mean(self.smce)

        self.optimizer = tf.train.AdamOptimizer(             self.leaning_rate, beta1=0.9, beta2=0.999).minimize(self.loss)

        self.score = tf.nn.softmax(self.y_hat)
        self.pred = tf.argmax(self.score, 1)
        self.correct_pred = tf.equal(self.pred, tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

        tf.summary.scalar('softmax_cross_entropy', self.loss)
        self.summaries = tf.summary.merge_all()

    def build_model(self, input):

        print("SE-1")
        conv1_1 = self.conv2d(input=input, stride=1, padding='SAME',             filter_size=[self.k_size, self.k_size, 1, 16], activation="relu", name="conv1_1")

        glo_avg_pool1 = tf.reduce_sum(conv1_1, axis=(1, 2))
        squeeze1 = self.fully_connected(input=glo_avg_pool1, num_inputs=16,             num_outputs=8, activation="relu", name="squeeze1")
        exitation1 = tf.reshape(self.fully_connected(input=squeeze1, num_inputs=8,             num_outputs=16, activation="sigmoid", name="exitation1"), [-1, 1, 1, 16])
        exitated1 = tf.multiply(conv1_1, exitation1)

        conv1_2 = self.conv2d(input=exitated1, stride=1, padding='SAME',             filter_size=[self.k_size, self.k_size, 16, 16], activation="relu", name="conv1_2")
        max_pool1 = self.maxpool(input=conv1_2, ksize=2, strides=2, padding='SAME', name="max_pool1")

        print("SE-2")
        conv2_1 = self.conv2d(input=max_pool1, stride=1, padding='SAME',             filter_size=[self.k_size, self.k_size, 16, 32], activation="relu", name="conv2_1")

        glo_avg_pool2 = tf.reduce_sum(conv2_1, axis=(1, 2))
        squeeze2 = self.fully_connected(input=glo_avg_pool2, num_inputs=32,             num_outputs=16, activation="relu", name="squeeze2")
        exitation2 = tf.reshape(self.fully_connected(input=squeeze2, num_inputs=16,             num_outputs=32, activation="sigmoid", name="exitation2"), [-1, 1, 1, 32])
        exitated2 = tf.multiply(conv2_1, exitation2)

        conv2_2 = self.conv2d(input=exitated2, stride=1, padding='SAME',             filter_size=[self.k_size, self.k_size, 32, 32], activation="relu", name="conv2_2")
        max_pool2 = self.maxpool(input=conv2_2, ksize=2, strides=2, padding='SAME', name="max_pool2")

        print("SE-3")
        conv3_1 = self.conv2d(input=max_pool2, stride=1, padding='SAME',             filter_size=[self.k_size, self.k_size, 32, 64], activation="relu", name="conv3_1")

        glo_avg_pool3 = tf.reduce_sum(conv3_1, axis=(1, 2))
        squeeze3 = self.fully_connected(input=glo_avg_pool3, num_inputs=64,             num_outputs=32, activation="relu", name="squeeze3")
        exitation3 = tf.reshape(self.fully_connected(input=squeeze3, num_inputs=32,             num_outputs=64, activation="sigmoid", name="exitation3"), [-1, 1, 1, 64])
        exitated3 = tf.multiply(conv3_1, exitation3)

        conv3_2 = self.conv2d(input=exitated3, stride=1, padding='SAME',             filter_size=[self.k_size, self.k_size, 64, 64], activation="relu", name="conv3_2")

        print("FullCon")
        [n, h, w, c] = conv3_2.shape
        fullcon_in = tf.reshape(conv3_2, shape=[self.batch_size, h*w*c], name="fullcon_in")
        fullcon1 = self.fully_connected(input=fullcon_in, num_inputs=int(h*w*c),             num_outputs=512, activation="relu", name="fullcon1")
        fullcon2 = self.fully_connected(input=fullcon1, num_inputs=512,             num_outputs=self.num_class, activation=None, name="fullcon2")

        return fullcon2

    def initializer(self):
        return tf.initializers.variance_scaling(distribution="untruncated_normal", dtype=tf.dtypes.float32)

    def maxpool(self, input, ksize, strides, padding, name=""):

        out_maxp = tf.nn.max_pool(value=input,             ksize=ksize, strides=strides, padding=padding, name=name)
        print("Max-Pool", input.shape, "->", out_maxp.shape)

        return out_maxp

    def activation_fn(self, input, activation="relu", name=""):

        if("sigmoid" == activation):
            out = tf.nn.sigmoid(input, name='%s_sigmoid' %(name))
        elif("tanh" == activation):
            out = tf.nn.tanh(input, name='%s_tanh' %(name))
        elif("relu" == activation):
            out = tf.nn.relu(input, name='%s_relu' %(name))
        elif("lrelu" == activation):
            out = tf.nn.leaky_relu(input, name='%s_lrelu' %(name))
        elif("elu" == activation):
            out = tf.nn.elu(input, name='%s_elu' %(name))
        else: out = input

        return out

    def variable_maker(self, var_bank, name_bank, shape, name=""):

        try:
            var_idx = name_bank.index(name)
        except:
            with tf.variable_scope("vars", reuse=tf.AUTO_REUSE):
                variable = tf.get_variable(name=name,                     shape=shape, initializer=self.initializer())

            var_bank.append(variable)
            name_bank.append(name)
        else:
            variable = var_bank[var_idx]

        return var_bank, name_bank, variable

    def conv2d(self, input, stride, padding,         filter_size=[3, 3, 16, 32], dilations=[1, 1, 1, 1], activation="relu", name=""):

        self.weights, self.w_names, weight = self.variable_maker(var_bank=self.weights, name_bank=self.w_names,             shape=filter_size, name='%s_w' %(name))
        self.biasis, self.b_names, bias = self.variable_maker(var_bank=self.biasis, name_bank=self.b_names,             shape=[filter_size[-1]], name='%s_b' %(name))

        out_conv = tf.nn.conv2d(
            input=input,
            filter=weight,
            strides=[1, stride, stride, 1],
            padding=padding,
            use_cudnn_on_gpu=True,
            data_format='NHWC',
            dilations=dilations,
            name='%s_conv' %(name),
        )
        out_bias = tf.math.add(out_conv, bias, name='%s_add' %(name))

        print("Conv", input.shape, "->", out_bias.shape)
        return self.activation_fn(input=out_bias, activation=activation, name=name)

    def fully_connected(self, input, num_inputs, num_outputs, activation="relu", name=""):

        self.weights, self.w_names, weight = self.variable_maker(var_bank=self.weights, name_bank=self.w_names,             shape=[num_inputs, num_outputs], name='%s_w' %(name))
        self.biasis, self.b_names, bias = self.variable_maker(var_bank=self.biasis, name_bank=self.b_names,             shape=[num_outputs], name='%s_b' %(name))

        out_mul = tf.matmul(input, weight, name='%s_mul' %(name))
        out_bias = tf.math.add(out_mul, bias, name='%s_add' %(name))

        print("Full-Con", input.shape, "->", out_bias.shape)
        return self.activation_fn(input=out_bias, activation=activation, name=name)


# # 4. The function for training and test.

# In[ ]:


def training(sess, saver, neuralnet, dataset, epochs, batch_size, normalize=True):

    print("\nTraining to %d epochs (%d of minibatch size)" %(epochs, batch_size))

    summary_writer = tf.summary.FileWriter(PACK_PATH+'/Checkpoint', sess.graph)

    iteration = 0

    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()

    test_sq = 20
    test_size = test_sq**2
    for epoch in range(epochs):

        while(True):
            x_tr, y_tr, terminator = dataset.next_train(batch_size) # y_tr does not used in this prj.

            _, summaries = sess.run([neuralnet.optimizer, neuralnet.summaries],                 feed_dict={neuralnet.x:x_tr, neuralnet.y:y_tr, neuralnet.batch_size:x_tr.shape[0]},                 options=run_options, run_metadata=run_metadata)
            loss, accuracy = sess.run([neuralnet.loss, neuralnet.accuracy],                 feed_dict={neuralnet.x:x_tr, neuralnet.y:y_tr, neuralnet.batch_size:x_tr.shape[0]})
            summary_writer.add_summary(summaries, iteration)

            iteration += 1
            if(terminator): break

        print("Epoch [%d / %d] (%d iteration)  Loss:%.3f, Acc:%.3f"             %(epoch, epochs, iteration, loss, accuracy))
        saver.save(sess, PACK_PATH+"/Checkpoint/model_checker")
        summary_writer.add_run_metadata(run_metadata, 'epoch-%d' % epoch)

def test(sess, saver, neuralnet, dataset, batch_size):

    if(os.path.exists(PACK_PATH+"/Checkpoint/model_checker.index")):
        print("\nRestoring parameters")
        saver.restore(sess, PACK_PATH+"/Checkpoint/model_checker")

    print("\nTest...")

    fsubmit = open("submission.csv", "w")
    fsubmit.write("ImageId,Label\n")
    print("ImageId,Label")
    cntid = 1

    confusion_matrix = np.zeros((dataset.num_class, dataset.num_class), np.int32)
    while(True):
        x_te, y_te, terminator = dataset.next_test(1) # y_te does not used in this prj.
        class_score = sess.run(neuralnet.score,             feed_dict={neuralnet.x:x_te, neuralnet.batch_size:x_te.shape[0]})

        label, logit = np.argmax(y_te[0]), np.argmax(class_score)
        confusion_matrix[label, logit] += 1
        print("%d,%d" %(cntid,logit))
        fsubmit.write("%d,%d\n" %(cntid,logit))
        cntid += 1

        if(terminator): break
    fsubmit.close()


# # 5. The main function for run the whole source code

# In[ ]:


def main():

    dataset = Dataset(normalize=FLAGS.datnorm)
    neuralnet = SENet(height=dataset.height, width=dataset.width, channel=dataset.channel,         num_class=dataset.num_class, leaning_rate=FLAGS.lr)
    
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess = tf.Session(config=sess_config)
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    training(sess=sess, neuralnet=neuralnet, saver=saver, dataset=dataset, epochs=FLAGS.epoch, batch_size=FLAGS.batch, normalize=True)
    test(sess=sess, neuralnet=neuralnet, saver=saver, dataset=dataset, batch_size=FLAGS.batch)
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--datnorm', type=bool, default=True, help='Data normalization')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for training')
    parser.add_argument('--epoch', type=int, default=100, help='Training epoch')
    parser.add_argument('--batch', type=int, default=32, help='Mini batch size')

    FLAGS, unparsed = parser.parse_known_args()

    main()


# In[ ]:




