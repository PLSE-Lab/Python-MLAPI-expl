import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder

#HELPER FUNCTIONS FOR DATA PROCESSING
def split_labels_features(dataset, label_indices):
    tr_labels = dataset.iloc[:, label_indices]
    tr_data = dataset.drop(dataset.columns[label_indices], axis=1)
    return tr_labels, tr_data

def one_hot_encoding(df_cols_to_encode):
    onehotenc = OneHotEncoder()
    df_encoded_cols = pd.DataFrame(onehotenc.fit_transform(df_cols_to_encode).toarray())
    return df_encoded_cols

#GENERATORS FOR BATCH DATA
def generate_batch_data_train(train_labels, train_data, batch_size = 50):
    num_label_cols = train_labels.shape[1]
    train_dataset = pd.concat([train_labels, train_data],axis=1)
    num_samples = train_dataset.shape[0]
    batch_id = -1
    start_id = 0
    end_id = batch_size
    while True:
        batch_id += 1
        if end_id >= num_samples:
            train_dataset = train_dataset.sample(frac=1).reset_index(drop=True)
            start_id = 0
            end_id = batch_size

        train_batch = train_dataset.iloc[start_id:end_id, :]
        train_batch_labels, train_batch_data = split_labels_features(train_batch, list(range(num_label_cols)))
        start_id += batch_size
        end_id += batch_size
        yield batch_id, train_batch_data, train_batch_labels

def generate_batch_data_test(test_data, batch_size = 50):
    num_samples = test_data.shape[0]
    batch_id = 0
    for i in range(0, num_samples, batch_size):
        batch_id += 1
        test_batch = test_data.iloc[i:min(i+batch_size, num_samples), :]
        yield batch_id, test_batch

#HELPER FUNCTIONS TO BUILD CNN USING TENSORFLOW
def tf_init_weights(shape):
    init_random_dist = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(init_random_dist)

def tf_init_bias(shape):
    init_bias_vals = tf.constant(0.1, shape=shape)
    return tf.Variable(init_bias_vals)

def tf_con2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def tf_pool_2by2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

def tf_conv_layer(input_x, shape):
    W = tf_init_weights(shape)
    b = tf_init_bias([shape[3]])
    return tf.nn.relu(tf_con2d(input_x,W)+b)

def tf_full_conn_layer(input_layer, size):
    input_size = int(input_layer.get_shape()[-1])
    W = tf_init_weights([input_size,size])
    b = tf_init_bias([size])
    return tf.matmul(input_layer,W)+b


#MAIN CODE
if __name__ == '__main__':

    dataset = pd.read_csv('../input/train.csv')
    pred_data = pd.read_csv('../input/test.csv')

    train_labels, train_data = split_labels_features(dataset, [0])
    train_labels = one_hot_encoding(train_labels)
    # test_labels, test_data = split_labels_features(dataset.iloc[0:5000,:], [0])
    # test_labels = one_hot_encoding(test_labels)


    # Build CNN
    x = tf.placeholder(tf.float32, shape=[None, 784])
    y_true = tf.placeholder(tf.float32, shape=[None, 10])
    hold_prob = tf.placeholder(tf.float32)

    x_image = tf.reshape(x, [-1, 28, 28, 1])
    conv_1 = tf_conv_layer(x_image, shape=[5, 5, 1, 32])
    conv_1_pool = tf_pool_2by2(conv_1)
    conv_2 = tf_conv_layer(conv_1_pool, shape=[5, 5, 32, 64])
    conv_2_pool = tf_pool_2by2(conv_2)
    conv_2_flat = tf.reshape(conv_2_pool, [-1, 7 * 7 * 64])
    full_1 = tf.nn.relu(tf_full_conn_layer(conv_2_flat, 1024))
    full_1_dropout = tf.nn.dropout(full_1, keep_prob=hold_prob)
    y_pred = tf_full_conn_layer(full_1_dropout, 10)

    cross_ent_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred))

    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    train = optimizer.minimize(cross_ent_loss)

    init = tf.global_variables_initializer()

    num_steps = 2501

    with tf.Session() as sess:
        sess.run(init)

        for batch_id, batch_x, batch_y in generate_batch_data_train(train_labels, train_data):
            if not batch_id < num_steps:
                break
            dict_to_feed_train = {x: batch_x,
                                  y_true: batch_y,
                                  hold_prob: 0.5}
            sess.run(train,feed_dict=dict_to_feed_train)
            if batch_id%100 == 0:
                print('Completed {0} steps'.format(batch_id))
                # print('ACCURACY: ')
                # matches = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
                # acc = tf.reduce_mean(tf.cast(matches, tf.float32))
                # dict_to_feed_acc = {x: test_data,
                #                     y_true: test_labels,
                #                     hold_prob: 1.0}
                # print(sess.run(acc, feed_dict=dict_to_feed_acc))
                # print('\n')

        print('Training Completed')

        y_pred_decoded = tf.argmax(y_pred, 1)

        num_pred = []
        batch_size = 50
        num_test_samples = pred_data.shape[0]

        for test_batch_id, test_batch_x in generate_batch_data_test(pred_data, batch_size):
            dict_to_feed_pred = {x: test_batch_x,
                                 hold_prob: 1.0}
            y_predicted_batch = sess.run(y_pred_decoded, feed_dict=dict_to_feed_pred)
            num_pred.extend(y_predicted_batch)

            if test_batch_id%100 == 0:
                print('Predicting Batch {0}/{1} from Test Set'.format(test_batch_id,num_test_samples/batch_size))
                print(len(num_pred))


    result_list = [[i+1, d] for i, d in enumerate(num_pred)]
    result = pd.DataFrame(result_list, columns=['ImageId','Label'])
    result.to_csv('My_Submission.csv', index=False)