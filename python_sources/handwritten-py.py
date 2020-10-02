import tflearn

data, labels = tflearn.data_utils.load_csv('../input/training.csv', target_column=0, categorical_labels=True, n_classes=10)

train_data = data[:300]
train_labels = labels[:300]

test_data = data[300:]
test_labels = labels[300:]

net = tflearn.input_data(shape=[None, 40])
net = tflearn.fully_connected(net, 40, bias_init='zeros')
net = tflearn.fully_connected(net, 10, bias_init='zeros', activation='softmax', regularizer='L2')
net = tflearn.regression(net, optimizer='sgd', loss='categorical_crossentropy')

model = tflearn.DNN(net, tensorboard_verbose=3)
model.fit(train_data, train_labels, n_epoch=50, batch_size=10, show_metric=True)


