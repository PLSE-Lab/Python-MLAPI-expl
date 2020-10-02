import numpy as np
import keras

k_regular=keras.regularizers
#Input data files are available in the "../input/" directory.
def build_model(train):
    #train should be reshaped as (n, 28, 28, 1)
    model = keras.models.Sequential()
    #First Conv->pool
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu',
                                  input_shape=train.shape[1:]))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    #Seond Conv->pool
    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    #Third Conv->pool
    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
    #Merge into a single shape
    model.add(keras.layers.Flatten())
    #Apply relu with weight decay
    model.add(keras.layers.Dropout(0.4))
    model.add(keras.layers.Dense(256,
                                 kernel_regularizer=k_regular.l2(0.00175),
                                 activation='relu'))
    #Apply Dropout for 50% of nodes
    model.add(keras.layers.Dropout(0.3))
    #Appy sigmoid with weight decay
    model.add(keras.layers.Dense(128,
                                 kernel_regularizer=k_regular.l2(0.00175),
                                 activation='relu'))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(64,
                                 kernel_regularizer=k_regular.l2(0.00175),
                                 activation='relu'))
    #model.add(keras.layers.Dense(64,
    #kernel_regularizer=k_regular.l2(0.001),
    #activation='sigmoid'))
    model.add(keras.layers.Dense(10, activation='softmax'))
    model.compile(
        optimizer=keras.optimizers.RMSprop(lr=.00125, decay=1e-6),
        loss='categorical_crossentropy',#'binary_crossentropy', 'mse',
        metrics=['acc', 'mae'])#'mae'
    return model


def model_train(model, input_tensor, output_tensor, verif_input,
                verif_output, batch=128, epoch=5, vrb=False):
    history = model.fit(input_tensor, output_tensor,
                        epochs=epoch, batch_size=batch,
                        validation_data=(verif_input, verif_output),
                        verbose=vrb)
    print("Completed Training the model")
    return history


def read_from_csv(fname, line_limit=0):
    '''Designed to be modified on demand.'''
    import csv
    file = open(fname, 'r')
    data = csv.reader(file)
    array = []
    if line_limit:
        array = np.array([line for line in data[:min(len(data), line_limmit)]])
    else:
        array = np.array([line for line in data])
    file.close()
    return array


def process_raw_data(array, header=True):
    ######Modifiable Parts
    input_tensor = process_input(array[:,1:], header)
    labels = array[header:, 0]#Get the first element, the output
    #Format the tensors
    output_tensor = np.zeros((labels.shape[0], 10))
    for i in range(len(labels)):
        output_tensor[i][int(labels[i])] = 1
    if header:
        return input_tensor, output_tensor, array[0]
    else:
        return input_tensor, output_tensor, [None, ]


def process_input(array, header=True):
    temp = np.float64(array[header:]) / 255
    return temp.reshape(temp.shape[0], 28, 28, 1)


def decode_output(array):
    ######Modifiable Parts
    return array.argmax(axis=1)


def split_tensors(input_tensor, output_tensor):
    ######Modifiable Parts
    n = int(len(input_tensor) * 0.75)
    test_input, verif_input = input_tensor[:n], input_tensor[n:]
    test_output, verif_output = output_tensor[:n], output_tensor[n:]
    return test_input, test_output, verif_input, verif_output


def save_to_csv(array, fname='predict.csv', label=[]):
    ######Modifiable Parts
    header = ','.join(label) + '\n'
    file = open(fname, 'w')
    file.write(header)
    for i in range(len(array)):
        s = str(i+1) + ',' + str(array[i]) + '\n'
        file.write(s)
    file.close()


def print_history(dic):
    history = dic.history
    keys = [k for k in history.keys()]
    print('\t'.join(map(str, keys)))
    
    for i in range(len(history[keys[0]])):
        for key in keys:
            try:
                print(history[key][i], '\t')
            except:
                print('--\t')
                
print("Starting Network...")
print("Load Training Data...")
fname = '../input/train.csv'
train_input, train_output, header = process_raw_data(read_from_csv(fname))
train_input, train_output, verif_input, verif_output = split_tensors(\
    train_input, train_output)

print('Build Model...')
model = build_model(train_input)

print('Training Model...')
history = model_train(model, train_input, train_output, #Test Set
                      verif_input, verif_output, #Verification Set
                      1024, 25, 0) #Batch, Epochs, verbose

#print_history(history)

print('Loading Test Data..')
fname2 = '../input/test.csv'
test = read_from_csv(fname2)

print('Predicting...')
results = model.predict(process_input(test))

print('Saving...')
pred_head = ['ImageId', 'Label']
save_to_csv(decode_output(results), 'predict.csv', pred_head)

print('End Program.')
