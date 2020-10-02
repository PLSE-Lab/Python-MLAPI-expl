import tensorflow as tf
import numpy as np
import pandas as pd
#import pickle

x_train = pd.read_csv(tf.gfile.Open('../input/X_train_sat6.csv'),header=None) #This will kill the available memory on the kaggle machine :\
y_train = pd.read_csv(tf.gfile.Open('../input/y_train_sat6.csv'),header=None)
x_train = x_train.values.reshape(x_train.shape[0],28,28,4).astype(np.float32)
y_train = y_train.values.astype(np.float32)

x_eval = pd.read_csv(tf.gfile.Open('../input/X_test_sat6.csv'),header=None)
y_eval = pd.read_csv(tf.gfile.Open('../input/y_test_sat6.csv'),header=None)
x_eval = x_eval.values.reshape(x_eval.shape[0],28,28,4).astype(np.float32)
y_eval = y_eval.values.astype(np.float32)

#with open('x_train.pickle', 'wb') as pickle_file:
#    pickle.dump(x_train, pickle_file)
#with open('y_train.pickle', 'wb') as pickle_file:
#    pickle.dump(y_train, pickle_file)x_eval = pd.read_csv(tf.gfile.Open('X_test_sat6.csv'),header=None)
#with open('x_eval.pickle', 'wb') as pickle_file:
#    pickle.dump(x_eval, pickle_file)
#with open('y_eval.pickle', 'wb') as pickle_file:
#    pickle.dump(y_eval, pickle_file)

#with open('x_train.pickle', 'rb') as pickle_file:
#    x_train = pickle.load(pickle_file)
#with open('y_train.pickle', 'rb') as pickle_file:
#    y_train = pickle.load(pickle_file)
#with open('x_eval.pickle', 'rb') as pickle_file:
#    x_eval = pickle.load(pickle_file)
#with open('y_eval.pickle', 'rb') as pickle_file:
#    y_eval = pickle.load(pickle_file)

def model_fn(features, labels, mode, params, config):
    
	# Input Layer
    input_layer = tf.reshape(features["x"], [-1,28,28,4])
	tf.summary.image('input_layer',input_layer)
	
	# First Convolutional Layer
    conv1 = tf.layers.conv2d(inputs=input_layer,
                             filters=32,
                             kernel_size=[5,5],
                             padding='same',
                             activation=tf.nn.relu)
	
	print("Shape Conv1:" + str(conv1.shape))
	
	# First MaxPool Layer
    pool1 = tf.layers.max_pooling2d(inputs=conv1, 
                                    pool_size=[2,2], 
                                    strides=2)
    print("Shape Pool1:" + str(pool1.shape))
	
	# Second Convolutional Layer
    conv2 = tf.layers.conv2d(inputs=pool1,
                             filters=64,
                             kernel_size=[5,5],
                             padding='same',
                             activation=tf.nn.relu)
    print("Shape Conv2:" + str(conv2.shape))
	
	# Second MaxPool Layer
    pool2 = tf.layers.max_pooling2d(inputs=conv2, 
                                    pool_size=[2, 2],
                                    strides=2)
    print("Shape Pool2:" + str(pool2.shape))
	
	# Flatten Pool2
    pool2_flat = tf.reshape(pool2, [-1, pool2.shape[1]*pool2.shape[2]*pool2.shape[3]])
    
	# First Dense Layer
    dense1 = tf.layers.dense(inputs=pool2_flat,
                            units=1024,
                            activation=tf.nn.relu)
							
	# Dropout Layer for Training
    dropout = tf.layers.dropout(inputs=dense1,
                                rate=0.5,
                                training=mode == tf.estimator.ModeKeys.TRAIN)

	# Second Dense Layer
    dense2 = tf.layers.dense(inputs=dropout,
                            units=256,
                            activation=tf.nn.relu)
    
	# Output Layer
    logits = tf.layers.dense(inputs=dense2, units=labels.shape[1])
    
    predictions = {
      "classes": tf.argmax(input=logits, axis=1), #Result Classes
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor") #Class Probabilities
    }
    
    # Predict Mode
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Loss Function
    loss = tf.losses.softmax_cross_entropy(onehot_labels=labels,
                                           logits=logits)
    
    ## Classification Metrics
    # accuracy
    acc = tf.metrics.accuracy(labels=tf.argmax(labels,1),
                              predictions=predictions['classes'])
    
    # Precision
    prec = tf.metrics.precision(labels=tf.argmax(labels,1),
                                predictions=predictions['classes'])
    
    # Recall
    rec = tf.metrics.recall(labels=tf.argmax(labels,1),
                            predictions=predictions['classes'])
    
    # F1 Score
    f1 = 2 * acc[1] * rec[1] / ( prec[1] + rec[1] ) #misssing op -> too lazy to create ;)
    
    #TensorBoard Summary
    tf.summary.scalar('Accuracy', acc[1])
    tf.summary.scalar('Precision', prec[1])
    tf.summary.scalar('Recall', rec[1])
    tf.summary.scalar('F1Score', f1)
    tf.summary.histogram('Probabilities', predictions['probabilities'])
    tf.summary.histogram('Classes', predictions['classes'])
    
    summary_hook = tf.train.SummarySaverHook(summary_op=tf.summary.merge_all(),
                                             save_steps=1)
    
    # Learning Rate Decay (Exponential)
    learning_rate = tf.train.exponential_decay(learning_rate=1e-04,
                                               global_step=tf.train.get_global_step(),
                                               decay_steps=10000, 
                                               decay_rate=0.96, 
                                               staircase=True,
                                               name='lr_exp_decay')

    
    # Training Mode (Adam Optimizer)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss=loss,
                                      global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, 
                                          loss=loss, 
                                          train_op=train_op)
                                          #training_hooks=[summary_hook])

    # Evaluation Metrics
    eval_metric_ops = {
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
    }
    
    # Evaluation Mode
    return tf.estimator.EstimatorSpec(mode=mode, 
                                      loss=loss, 
                                      eval_metric_ops=eval_metric_ops)

# Create Custom Classifier
sat6_classifier = tf.estimator.Estimator(model_fn=model_fn,
                                         model_dir='/tmp/cnn_model',
                                         config=tf.estimator.RunConfig(save_summary_steps=1))

# This will create a lot of output ;)
tensors_to_log = {"probabilities": "softmax_tensor"}
logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, 
                                          every_n_iter=1)
										  
# Training input function
train_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": x_train},
                                                    y=y_train,
                                                    batch_size=512,
                                                    num_epochs=1,
                                                    shuffle=True)
# Evaluation input function
eval_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": x_eval},
                                                   y=y_eval,
                                                   num_epochs=1,
                                                   shuffle=False)

# 10 Training Epochs,  Evaluation Step after each Epoch
for i in range(10):
    sat6_classifier.train(input_fn=train_input_fn,
                          hooks=[logging_hook])
    eval_results = sat6_classifier.evaluate(input_fn=eval_input_fn)

#INFO:tensorflow:Loss for final step: 0.01437158.
#INFO:tensorflow:Calling model_fn.
#Shape Conv1:(?, 28, 28, 32)
#Shape Pool1:(?, 14, 14, 32)
#Shape Conv2:(?, 14, 14, 64)
#Shape Pool2:(?, 7, 7, 64)
#INFO:tensorflow:Done calling model_fn.
#INFO:tensorflow:Starting evaluation at 2018-03-11-14:09:55
#INFO:tensorflow:Graph was finalized.
#INFO:tensorflow:Restoring parameters from /tmp/cnn_model/model.ckpt-6330
#INFO:tensorflow:Running local_init_op.
#INFO:tensorflow:Done running local_init_op.
#INFO:tensorflow:Finished evaluation at 2018-03-11-14:11:19
#INFO:tensorflow:Saving dict for global step 6330: Accuracy = 0.98949385, Precision = 0.99994814, Recall = 0.9983826, global_step = 6330, loss = 0.030443478
#F1 Score = 0,98951163