# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, accuracy_score
import tensorflow as tf
import os
import time


def get_mbs_error(feats, labels, tensors, sess, feed_dict, mbs):
    """
    Return error on given input feats by dividing it into mini-batches of size mbs and feeding it to network.

    I wrote this function because if the size of training or validation sets is too large, can't fed them
    to the network in a single step.


    :param feats: input features
    :param labels: input labels
    :param sess: tensorflow session
    :param feed_dict:  feeding dictionary for running the model
    :param mbs: size of mini-batches
    :return: mean of error, accuracy, AUC
    """
    x = tensors['x']
    y = tensors['y']
    error = tensors['error']
    p_hat = tensors['p_hat']

    errors = []
    predictions = []
    for i in range(int(np.ceil(np.size(feats, 0) / mbs))):
        feed_dict[x] = feats[i * mbs: (i + 1) * mbs, :]
        feed_dict[y] = labels[i * mbs: (i + 1) * mbs, :]
        error_val, p_val = sess.run([error, p_hat], feed_dict=feed_dict)
        errors.append(error_val)
        predictions.extend(p_val)
    binary_predictions = [0 if p < 0.5 else 1 for p in predictions]
    acc = accuracy_score(labels, binary_predictions)
    auc = roc_auc_score(labels, predictions)
    return np.mean(errors), acc, auc


def create_weights(weight_sizes, suffix):
    """
    Create the weights for several fully connected layers with batch-norm and return weights, and weight norms.

    Note: since we are using batch-normalization we don't need to create biases

    :param weight_sizes: [int] array of number of nodes in each layer
    :param suffix: prefix used for naming tensors
    :return: dict containing W, one bias for output layer, weight norms
    """
    weights = []
    weight_norms = 0  # for storing norm of weight
    weight_sizes.append(1)  # adding one as size of last layer

    for i in range(len(weight_sizes) - 1):
        weights.append(tf.get_variable("Wf" + suffix + str(i), [weight_sizes[i], weight_sizes[i + 1]], tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer()))
        weight_norms += tf.nn.l2_loss(weights[-1])

    # only need one bias for last layer where we dont use batch-norm
    b_o = tf.Variable(tf.constant(0.01, shape=[1], dtype=tf.float32), name="bo" + suffix)
    return weights, b_o, weight_norms


def fc_layers(input_data, weights, biases, activation_fun, keep_prob=1.0, is_training=True):
    """
    Return output of a feed forward network.

    :param input_data: input data!
    :param weights: list containing weights of each layer
    :param biases: list of biases for each layer (right-now it has only one bias due to batch-norm)
    :param activation_fun: List of callables, list of activation functions for each layer,
                        if len(layer_func) == 1 then the same activation function is used for all the layers
    :param keep_prob: drop out probability (keeping probability for dropout, this percentage of connections are kept)
    :param is_training: boolean to distinguish between training and inference modes
    :return: Output of the feed forward network.
    """

    input_next = input_data

    for l in range(len(weights) - 1):

        # applying weights to input
        x = tf.matmul(input_next, weights[l])
        # batch-norm (hence no biases)
        x_norm = tf.layers.batch_normalization(x, training=is_training, axis=1)
        # applying activation function
        z = activation_fun(x_norm)
        # applying dropout for regularization
        input_next = tf.nn.dropout(z, keep_prob)

    # we apply last layer separately due to different activation function
    y_hat = tf.add(tf.matmul(input_next, weights[-1]), biases[0], name="y_hat")
    p_hat = tf.nn.sigmoid(y_hat)
    return y_hat, p_hat


def create_graph(input_size, params):
    """
    Create and return tensorflow graph and associated tensors.

    Note : in this version I am not using pooling layers and use strides of 1

    :param input_size: [n_h, n_w, n_c] size of each input sample
    :param output_size:  size of output (Ng + 1)
    :param params:
    :return: graph
    """
    fc_nodes = params["fc_nodes"]
    fc_nodes.insert(0, input_size)

    graph = tf.Graph()

    with graph.as_default():
        # placeholders for input ant output and keep_prob in dropout
        x = tf.placeholder(tf.float32, [None] + [input_size], name="x")
        y = tf.placeholder(tf.float32, [None, 1], name="y")
        # keeping probability in dropout, only used dropout in fully connected layers
        kp = tf.placeholder(tf.float32, name="kp")
        is_training = tf.placeholder(tf.bool, name="is_training")

        # adding fully connected layers and forward pass
        weights, biases, norms = create_weights(fc_nodes, '_fc')
        y_hat, p_hat = fc_layers(x, weights, biases, tf.nn.relu, kp, is_training)

        # computing cross-entropy error and cost function
        error = tf.losses.sigmoid_cross_entropy(y, y_hat)
        cost_func = tf.add(error, params["lambd"] * norms, name="cost_func")

        # Operation block: optimizing the cost function using momentum2
        # adding update ops as dependency to train operation to make sure batch moments are update
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            if hyper_params["optimizer"] == "Adam":
                train = tf.train.AdamOptimizer(learning_rate=params["alpha"]).minimize(cost_func)
            else:
                train = tf.train.MomentumOptimizer(learning_rate=params["alpha"], momentum=0.9).minimize(cost_func)

        # Operation block: initializing all variables
        init = tf.global_variables_initializer()
        saver = tf.train.Saver(name="saver", max_to_keep=None)

        # putting all tensors in a dictionary to be recovered later
        tensors = {"init": init, "cost_func": cost_func,  "y_hat": y_hat, "x" : x, "y" : y,
                   "saver": saver, "train": train, "error": error, "p_hat": p_hat}

    return graph, tensors


def run_training(x_train, y_train, x_valid, y_valid, graph, tensors, hyper_params):
    """
    Run training for neural network and return results

    :param x_train : training features
    :param y_train : training labels
    :param x_valid : validation features
    :param y_valid : validation lables
    :param graph: tensorflow graph object
    :param tensors: dictionary of necessary tensors
    :param hyper_params: dict of hyper parameters
    :return: array of training and validation errors, prediction from final model, tensorflow graph
    """
    mbs = hyper_params["mbs"]
    num_epochs = hyper_params["num_epochs"]
    keep_p = hyper_params["kp"]
    max_validation_check = hyper_params["validation_limit"]

    y_train = np.reshape(y_train, (y_train.shape[0], 1))
    y_valid = np.reshape(y_valid, (y_valid.shape[0], 1))

    init, train, cost_func, saver = tensors["init"], tensors["train"], tensors["cost_func"], tensors["saver"]

    config = tf.ConfigProto()
    sess = tf.Session(graph=graph, config=config)
    validation_check = 0
    sess.run(init)

    # getting placeholders from graph
    kp = graph.get_tensor_by_name("kp:0")
    x = graph.get_tensor_by_name("x:0")
    p_hat = tensors["p_hat"]
    y = graph.get_tensor_by_name("y:0")
    is_training = graph.get_tensor_by_name("is_training:0")

    feed_dict = dict()
    epoch = 0
    min_valid_error = 1e5
    train_error = []
    valid_error = []
    while epoch < num_epochs:

        start_time = time.time()
        # going through training data
        for i in range(int(np.ceil(np.size(x_train, 0) / mbs))):
            feed_dict[kp] = keep_p
            feed_dict[is_training] = True
            x_batch = x_train[i * mbs: (i + 1) * mbs, :]
            y_batch = y_train[i * mbs: (i + 1) * mbs, :]
            # preparing feed dictionary #
            feed_dict[x] = x_batch
            feed_dict[y] = y_batch
            sess.run([train], feed_dict=feed_dict)

        # getting training and validation errors after we went through all minibatches
        feed_dict[kp] = 1
        feed_dict[is_training] = False
        t_err, t_acc, t_auc = get_mbs_error(x_train, y_train, tensors, sess, feed_dict, mbs)
        train_error.append(t_err)
        v_err, v_acc, v_auc = get_mbs_error(x_valid, y_valid, tensors, sess, feed_dict, mbs)
        valid_error.append(v_err)

        end_time = time.time()


        # check for early stopping #
        if epoch >= 1 and valid_error[epoch] > valid_error[epoch - 1] * 1.01:
            validation_check += 1
            print("Validation error increased! counting " + str(validation_check))

        if validation_check == max_validation_check:
            print("Reached max validation, quitting!")
            break

        # printing stats every 5 epochs
        if epoch % 5 == 0:
            print("epoch = %d" % epoch)
            print("elpased time (per epoch):" + str(end_time - start_time))
            print("training error = %f, validation error = %f" % (t_err, v_err))
            print("training accuracy = %f, validation accuracy = %f" % (t_acc, v_acc))
            print("training auc = %f, validation auc = %f" % (t_auc, v_auc))

        # keeping a record of network yielding best validation error
        if v_err < min_valid_error:
            min_valid_error = v_err
            saver.save(sess, "./min_val_error_model.ckpt")
            #print("Updating min validation model!")

        # halving learning rate every 100 epochs
        # if epoch % 100 == 0 and epoch > 0:
        #     train.learning_rate = hyper_params["alpha"] / 2
        #     hyper_params["alpha"] = hyper_params["alpha"] / 2

        epoch += 1

    # restoring min_validation model for final precition
    saver.restore(sess, "min_val_error_model.ckpt")
    predictions = sess.run(p_hat, feed_dict={x : x_valid, is_training: False, kp: 1})
    sess.close()
    return train_error, valid_error, predictions, graph


def one_hot_encoder(df, ohe_cols):
    """
    One-Hot encode ohe_cols columns in df and return the new dataframe.

    :param df : data frame to be encoded
    :param ohe_cols : columns in data frame to be encoded
    :return encoded data frame
    """
    print('Creating OHE features..\nOld df shape:{}'.format(df.shape))
    df = pd.get_dummies(df, columns=ohe_cols)
    print('New df shape:{}'.format(df.shape))
    return df


def get_hyerparams():
    """
    Return hyperparameters for NN training.

    :param state_type: str, linear or square, specifies type of state used for NN training, filter sizes depend on this
    :return:
    """

    params = dict()
    params["lambd"] = 0.0  # coeff of weight penalty
    params["num_epochs"] = 500  # number of epochs for training
    params["kp"] = 0.5  # keeping probability for droupout
    params["optimizer"] = "Momentum"  # optimization method (Adam or Momentum)
    params["fc_nodes"] = [256, 256, 256, 128, 64] # number of nodes in each fully connected layer
    params["alpha"] = 0.0001   # learning rate
    params["mbs"] = 32  # mini-batch size
    params["validation_limit"] = 200  # number of epochs to see validation error increase before breaking training

    return params


if __name__ == "__main__":
    #os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # comment this if there is only one GPU to use
    hyper_params = get_hyerparams()
    ohe = False

    prep_info = np.load("../input/springleaf-competition-eda/prep_info.npz")   # loading info from EDA kernel
    dates, ohe_cols, potentials_to_remove = prep_info["arr_5"], prep_info["arr_8"], prep_info["arr_7"]  
    # do not want to one-hot encode date features since that just blows up feature space
    ohe_cols = [col for col in ohe_cols if col not in dates and col not in potentials_to_remove and len(col) < 9]

    # since size of data is very large we sample the training data 
    train_df = pd.read_csv('../input/springleaf-competition-data-preparation/train_clean.csv').sample(frac=0.1).reset_index(drop=True)
    train_df.drop(potentials_to_remove, axis=1, inplace=True)
    print("{} samples have been loaded!".format(train_df.shape[0]))

    if ohe:
        train_df = one_hot_encoder(train_df, ohe_cols)

    features = train_df.drop(["target"], axis=1)
    labels = train_df["target"]
    features = (features - np.mean(features)) / (np.std(features) + 1e-8)

    # since size of training data is large I am using holdout method instead of k-fold cross validation
    train_feats, valid_feats, train_labels, valid_labels = \
        train_test_split(features, labels, train_size=0.8, test_size=0.2, random_state=0)

    graph, tensors = create_graph(train_feats.shape[1], hyper_params)
    train_err, valid_err, predictions, _ = run_training(train_feats.values, train_labels.values,
                                                        valid_feats.values, valid_labels.values,
                                                        graph, tensors, hyper_params)
    binary_preds = [0 if p < 0.5 else 1 for p in predictions]
    precision, recall, fscore, _ = precision_recall_fscore_support(valid_labels, binary_preds)
    acc = accuracy_score(valid_labels, binary_preds)
    auc = roc_auc_score(valid_labels, predictions)
    res = [auc, acc, precision[0], recall[0], fscore[0]]

    print("AUC = %.3f\nAccuracy = %.3f  Precision = %.3f  Recall = %.3f F-score = %.3f" %
          (res[0], res[1], res[2], res[3], res[4]))
    np.savez('mlp_holdout_results.npz', res, hyper_params)