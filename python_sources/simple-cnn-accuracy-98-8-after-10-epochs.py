import tensorflow as tf
import numpy as np
import pandas as pd
import pickle

x4_train = pd.read_csv(tf.gfile.Open('X_train_sat4.csv'),header=None) #Too large for kaggle
y4_train = pd.read_csv(tf.gfile.Open('y_train_sat4.csv'),header=None)
x4_train = x4_train.values.reshape(x4_train.shape[0],28,28,4).astype(np.float32)
#y4_train = y4_train.values.astype(np.float32)with open('x4_train.pickle', 'wb') as pickle_file:
#    pickle.dump(x4_train, pickle_file, protocol=4)
#with open('y4_train.pickle', 'wb') as pickle_file:
#    pickle.dump(y4_train, pickle_file, protocol=4)x4_eval = pd.read_csv(tf.gfile.Open('X_test_sat4.csv'),header=None)
y4_eval = pd.read_csv(tf.gfile.Open('y_test_sat4.csv'),header=None)
x4_eval = x4_eval.values.reshape(x4_eval.shape[0],28,28,4).astype(np.float32)
y4_eval = y4_eval.values.astype(np.float32)with open('x4_eval.pickle', 'wb') as pickle_file:
#    pickle.dump(x4_eval, pickle_file, protocol=4)
#with open('y4_eval.pickle', 'wb') as pickle_file:
#    pickle.dump(y4_eval, pickle_file, protocol=4)

#with open('x4_train.pickle', 'rb') as pickle_file:
#    x4_train = pickle.load(pickle_file)
#with open('y4_train.pickle', 'rb') as pickle_file:
#    y4_train = pickle.load(pickle_file)
#    
#with open('x4_eval.pickle', 'rb') as pickle_file:
#    x4_eval = pickle.load(pickle_file)
#with open('y4_eval.pickle', 'rb') as pickle_file:
#    y4_eval = pickle.load(pickle_file)


def model_fn(features, labels, mode, params, config):
    
    input_layer = tf.reshape(features["x"], [-1,28,28,4])
    
    # First Conv Layer
    conv1 = tf.layers.conv2d(inputs=input_layer,
                             filters=32,
                             kernel_size=[5,5],
                             padding='same',
                             activation=tf.nn.relu)
    print("Shape Conv1:" + str(conv1.shape))
    tf.summary.image('conv1',input_layer)
    
    # First MaxPool Layer
    pool1 = tf.layers.max_pooling2d(inputs=conv1, 
                                    pool_size=[2,2], 
                                    strides=2)
    print("Shape Pool1:" + str(pool1.shape))
    
    # Second Conv Layer
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
    pool2_flat = tf.reshape(pool2, [-1, pool2.shape[1]*pool2.shape[2]*pool2.shape[3]])
    
    # First Dense Layer
    dense1 = tf.layers.dense(inputs=pool2_flat,
                            units=1024,
                            activation=tf.nn.relu)
                            
    # Dropout Layer
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

    # Prediction Mode
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Loss Function
    loss = tf.losses.softmax_cross_entropy(onehot_labels=labels,
                                           logits=logits)
    
    ## Classification Metrics
    # Accuracy
    acc = tf.metrics.accuracy(labels=tf.argmax(labels,1),
                              predictions=predictions['classes'])
    
    # Precision
    prec = tf.metrics.precision(labels=tf.argmax(labels,1),
                                predictions=predictions['classes'])
    
    # Recall
    rec = tf.metrics.recall(labels=tf.argmax(labels,1),
                            predictions=predictions['classes'])
    
    # F1 Score
    f1 = 2 * acc[1] * rec[1] / ( prec[1] + rec[1] ) # Again too lazy to create update op,
                                                    # so you will have no f1 score in tensorboard
    
    # TensorBoard Summary
    tf.summary.scalar('Accuracy', acc[1])
    tf.summary.scalar('Precision', prec[1])
    tf.summary.scalar('Recall', rec[1])
    tf.summary.scalar('F1Score', f1)
    tf.summary.histogram('Probabilities', predictions['probabilities'])
    tf.summary.histogram('Classes', predictions['classes'])
    
    summary_hook = tf.train.SummarySaverHook(summary_op=tf.summary.merge_all(),
                                             save_steps=1)
    
    # Learning Rate Exponential Decay
    learning_rate = tf.train.exponential_decay(learning_rate=1e-04,
                                               global_step=tf.train.get_global_step(),
                                               decay_steps=10000, 
                                               decay_rate=0.96, 
                                               staircase=True,
                                               name='lr_exp_decay')

    
    # Training Mode
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss=loss,
                                      global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, 
                                          loss=loss, 
                                          train_op=train_op,
                                          training_hooks=[summary_hook])

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


sat4_classifier = tf.estimator.Estimator(model_fn=model_fn,
                                         model_dir='/tmp/cnn4_model',
                                         config=tf.estimator.RunConfig(save_summary_steps=1))

tensors_to_log = {"probabilities": "softmax_tensor"}
logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, 
                                          every_n_iter=1)

# Input function for training
train_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": x4_train},
                                                    y=y4_train,
                                                    batch_size=512,
                                                    num_epochs=1,
                                                    shuffle=True)
# Input function for evaluation
eval_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": x4_eval},
                                                   y=y4_eval,
                                                   num_epochs=1,
                                                   shuffle=False)

for i in range(10):
    sat4_classifier.train(input_fn=train_input_fn,
                          hooks=[logging_hook])     # This will produce a lot of output on the console
    eval_results = sat4_classifier.evaluate(input_fn=eval_input_fn)

eval_results['F1 Score:'] = 2 * eval_results['Accuracy'] * eval_results['Recall'] / ( eval_results['Precision'] + eval_results['Recall'] )
print(eval_results)

#'Loss': 0.03589841, 
#'Accuracy': 0.988, 
#'Precision': 0.9932273, 
#'Recall': 0.9934292,
#'F1 Score:': 0.9881004043424464