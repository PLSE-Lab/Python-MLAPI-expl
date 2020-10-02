#!/usr/bin/env python
# coding: utf-8

# In[ ]:


cd  ../input/tensorflow-rnn-shakespeare-master/tensorflow-rnn-shakespeare-master/


# In[ ]:


import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib import rnn  
import os
import time
import math
import numpy as np
import my_txtutils as txt
tf.set_random_seed(0)


# In[ ]:


SEQLEN = 30
BATCHSIZE = 200
ALPHASIZE = txt.ALPHASIZE
INTERNALSIZE = 512
NLAYERS = 3
learning_rate = 0.001  
dropout_pkeep = 0.8    

# load data
shakedir = "shakespeare/*.txt"
codetext, valitext, bookranges = txt.read_data_files(shakedir, validation=True)

# display some stats on the data
epoch_size = len(codetext) // (BATCHSIZE * SEQLEN)
txt.print_data_stats(len(codetext), len(valitext), epoch_size)


lr = tf.placeholder(tf.float32, name='lr')  
pkeep = tf.placeholder(tf.float32, name='pkeep')  
batchsize = tf.placeholder(tf.int32, name='batchsize')

# inputs
X = tf.placeholder(tf.uint8, [None, None], name='X')    
Xo = tf.one_hot(X, ALPHASIZE, 1.0, 0.0)                 
# expected outputs = same sequence shifted by 1 since we are trying to predict the next character
Y_ = tf.placeholder(tf.uint8, [None, None], name='Y_')  
Yo_ = tf.one_hot(Y_, ALPHASIZE, 1.0, 0.0)               
# input state
Hin = tf.placeholder(tf.float32, [None, INTERNALSIZE*NLAYERS], name='Hin')  # [ BATCHSIZE, INTERNALSIZE * NLAYERS]

# using a NLAYERS=3 layers of GRU cells, unrolled SEQLEN=30 times
# dynamic_rnn infers SEQLEN from the size of the inputs Xo


cells = [rnn.GRUCell(INTERNALSIZE) for _ in range(NLAYERS)]

dropcells = [rnn.DropoutWrapper(cell,input_keep_prob=pkeep) for cell in cells]
multicell = rnn.MultiRNNCell(dropcells, state_is_tuple=False)
multicell = rnn.DropoutWrapper(multicell, output_keep_prob=pkeep)  # dropout for the softmax layer

Yr, H = tf.nn.dynamic_rnn(multicell, Xo, dtype=tf.float32, initial_state=Hin)


H = tf.identity(H, name='H')  # just to give it a name


# In[ ]:



Yflat = tf.reshape(Yr, [-1, INTERNALSIZE])    # [ BATCHSIZE x SEQLEN, INTERNALSIZE ]
Ylogits = layers.linear(Yflat, ALPHASIZE)     # [ BATCHSIZE x SEQLEN, ALPHASIZE ]
Yflat_ = tf.reshape(Yo_, [-1, ALPHASIZE])     # [ BATCHSIZE x SEQLEN, ALPHASIZE ]
loss = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Yflat_)  # [ BATCHSIZE x SEQLEN ]
loss = tf.reshape(loss, [batchsize, -1])      # [ BATCHSIZE, SEQLEN ]
Yo = tf.nn.softmax(Ylogits, name='Yo')        # [ BATCHSIZE x SEQLEN, ALPHASIZE ]
Y = tf.argmax(Yo, 1)                          # [ BATCHSIZE x SEQLEN ]
Y = tf.reshape(Y, [batchsize, -1], name="Y")  # [ BATCHSIZE, SEQLEN ]
train_step = tf.train.AdamOptimizer(lr).minimize(loss)

# stats for display
seqloss = tf.reduce_mean(loss, 1)
batchloss = tf.reduce_mean(seqloss)
accuracy = tf.reduce_mean(tf.cast(tf.equal(Y_, tf.cast(Y, tf.uint8)), tf.float32))
loss_summary = tf.summary.scalar("batch_loss", batchloss)
acc_summary = tf.summary.scalar("batch_accuracy", accuracy)
summaries = tf.summary.merge([loss_summary, acc_summary])


# In[ ]:


# Init Tensorboard stuff. This will save Tensorboard information into a different
# folder at each run named 'log/<timestamp>/'. Two sets of data are saved so that
# you can compare training and validation curves visually in Tensorboard.
timestamp = str(math.trunc(time.time()))
summary_writer = tf.summary.FileWriter("/kaggle/working/log/" + timestamp + "-training")
validation_writer = tf.summary.FileWriter("/kaggle/working/log/" + timestamp + "-validation")

# Init for saving models. They will be saved into a directory named 'checkpoints'.
# Only the last checkpoint is kept.
if not os.path.exists("/kaggle/working/checkpoints"):
    os.mkdir("/kaggle/working/checkpoints")
saver = tf.train.Saver(max_to_keep=1000)

# for display: init the progress bar
DISPLAY_FREQ = 50
_50_BATCHES = DISPLAY_FREQ * BATCHSIZE * SEQLEN
progress = txt.Progress(DISPLAY_FREQ, size=111+2, msg="Training on next "+str(DISPLAY_FREQ)+" batches")

# init
istate = np.zeros([BATCHSIZE, INTERNALSIZE*NLAYERS])  # initial zero input state
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
step = 0


# In[ ]:


# training loop
for x, y_, epoch in txt.rnn_minibatch_sequencer(codetext, BATCHSIZE, SEQLEN, nb_epochs=10):

    # train on one minibatch
    feed_dict = {X: x, Y_: y_, Hin: istate, lr: learning_rate, pkeep: dropout_pkeep, batchsize: BATCHSIZE}
    _, y, ostate = sess.run([train_step, Y, H], feed_dict=feed_dict)

    # log training data for Tensorboard display a mini-batch of sequences (every 50 batches)
    if step % _50_BATCHES == 0:
        feed_dict = {X: x, Y_: y_, Hin: istate, pkeep: 1.0, batchsize: BATCHSIZE}  # no dropout for validation
        y, l, bl, acc, smm = sess.run([Y, seqloss, batchloss, accuracy, summaries], feed_dict=feed_dict)
        txt.print_learning_learned_comparison(x, y, l, bookranges, bl, acc, epoch_size, step, epoch)
        summary_writer.add_summary(smm, step)

    
    if step % _50_BATCHES == 0 and len(valitext) > 0:
        VALI_SEQLEN = 1*1024  
        bsize = len(valitext) // VALI_SEQLEN
        txt.print_validation_header(len(codetext), bookranges)
        vali_x, vali_y, _ = next(txt.rnn_minibatch_sequencer(valitext, bsize, VALI_SEQLEN, 1))  
        vali_nullstate = np.zeros([bsize, INTERNALSIZE*NLAYERS])
        feed_dict = {X: vali_x, Y_: vali_y, Hin: vali_nullstate, pkeep: 1.0,  
                     batchsize: bsize}
        ls, acc, smm = sess.run([batchloss, accuracy, summaries], feed_dict=feed_dict)
        txt.print_validation_stats(ls, acc)
        # save validation data for Tensorboard
        validation_writer.add_summary(smm, step)

    # display a short text generated with the current weights and biases (every 150 batches)
    if step // 3 % _50_BATCHES == 0:
        txt.print_text_generation_header()
        ry = np.array([[txt.convert_from_alphabet(ord("K"))]])
        rh = np.zeros([1, INTERNALSIZE * NLAYERS])
        for k in range(1000):
            ryo, rh = sess.run([Yo, H], feed_dict={X: ry, pkeep: 1.0, Hin: rh, batchsize: 1})
            rc = txt.sample_from_probabilities(ryo, topn=10 if epoch <= 1 else 2)
            print(chr(txt.convert_to_alphabet(rc)), end="")
            ry = np.array([[rc]])
        txt.print_text_generation_footer()

    # save a checkpoint (every 500 batches)
    if step // 10 % _50_BATCHES == 0:
        saved_file = saver.save(sess, '/kaggle/working/checkpoints/rnn_train_' + timestamp, global_step=step)
        print("Saved file: " + saved_file)

    # display progress bar
    progress.step(reset=step % _50_BATCHES == 0)

    # loop state around
    istate = ostate
    step += BATCHSIZE * SEQLEN


# In[ ]:


import tensorflow as tf
import numpy as np
import my_txtutils

ALPHASIZE = my_txtutils.ALPHASIZE
NLAYERS = 3
INTERNALSIZE = 512


# In[ ]:


get_ipython().system('ls /kaggle/working/checkpoints/')


# To see the sample output, run the following code:

# From the above list of files, choose the appropriate path for shakespeareB10 and the respective ".meta" file

# In[ ]:



#shakespeareB10 = "checkpoints/rnn_train_1495440473-102000000"
shakespeareB10="/kaggle/working/checkpoints/rnn_train_1565257844-9000000"

author = shakespeareB10

ncnt = 0
with tf.Session() as sess:
    new_saver = tf.train.import_meta_graph("/kaggle/working/checkpoints/rnn_train_1565257844-0.meta")
    new_saver.restore(sess, author)
    x = my_txtutils.convert_from_alphabet(ord("L"))
    x = np.array([[x]])  # shape [BATCHSIZE, SEQLEN] with BATCHSIZE=1 and SEQLEN=1

    # initial values
    y = x
    h = np.zeros([1, INTERNALSIZE * NLAYERS], dtype=np.float32)  # [ BATCHSIZE, INTERNALSIZE * NLAYERS]
    for i in range(1000000000):
        yo, h = sess.run(['Yo:0', 'H:0'], feed_dict={'X:0': y, 'pkeep:0': 1., 'Hin:0': h, 'batchsize:0': 1})
        c = my_txtutils.sample_from_probabilities(yo, topn=2)
        y = np.array([[c]])  # shape [BATCHSIZE, SEQLEN] with BATCHSIZE=1 and SEQLEN=1
        c = chr(my_txtutils.convert_to_alphabet(c))
        print(c, end="")

        if c == '\n':
            ncnt = 0
        else:
            ncnt += 1
        if ncnt == 100:
            print("")
            ncnt = 0

