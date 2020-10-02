import tensorflow as tf
import numpy as np

#la prima riga del dataset contiene gli header ed è da togliere
# The competition datafiles are in the directory ../input
dataset = np.genfromtxt('../input/train.csv', delimiter=',', skip_header=1, dtype='float32')  

print ('Dataset shape:', dataset.shape)

labels = dataset[:,0]  # La colonna 0 rappresenta le labels
data = dataset[:,1:]  # I dati (in unica riga delle immagini)

media = np.mean(data)
dev_st = np.std(data)

print ('Data shape: ',data.shape)
print ('Media: ',np.mean(data))
print ('Standard deviation:', np.std(data))

data = data - media
data = data / dev_st

print('Standard Gaussian:')
print ('Data shape: ',data.shape)
print ('Media: ',np.mean(data))
print ('Standard deviation:', np.std(data))

# Costanti per la determinazione dei sottoinsiemi di allenamento, validazione e test
train_subset = 10000
validation_subset = 5000
test_subset = 2000

# Creazione dei sottoinsiemi di allenamento, validazione e test
train = data[:train_subset, :]
validation = data[train_subset + 1 : train_subset + validation_subset + 1, :]
test = data[train_subset + validation_subset + 1 : train_subset + validation_subset + test_subset + 1, :]

# Creazione dei sottoinsiemi LABELS di allenamento, validazione e test
train_labels = labels[:train_subset]
validation_labels = labels[train_subset + 1 : train_subset+validation_subset + 1]
test_labels = labels[train_subset + validation_subset + 1 : train_subset + validation_subset + test_subset + 1]

print ('Training: ', train.shape, train_labels.shape)
print ('Validation: ', validation.shape, validation_labels.shape)
print ('Test: ', test.shape, test_labels.shape)

# Il vettore train_labels deve però essere scritto in forma matriciale:
matrix_train_labels = (np.arange(10) == train_labels[:,None]).astype(np.float32)
print ('Matrix_train_labels: ',matrix_train_labels.shape)

print("Hey, what's up?")

graph = tf.Graph()   
with graph.as_default():
    
    y_ = tf.constant(matrix_train_labels)         # matrice delle labels (conosciute per allenamento)
    
    x = tf.constant(train)                        # ingressi
    W = tf.Variable(tf.zeros([784, 10]))          # matrice dei pesi
    b = tf.Variable(tf.zeros([10]))               # vettore bias
    
    tf_validation = tf.constant(validation)       # matrice immagini di validazione
    tf_test = tf.constant(test)                   # matrice immagini di test
    
    logits = tf.matmul(x, W) + b                                                 # operazione di calcolo   
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, y_))   # calcolo della distanza
    optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(loss)           # ottimizzazione (tramite gradiente)
    
    cross_entropy = -tf.reduce_sum(y_*tf.log(logits))
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
    
    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(tf.matmul(tf_validation, W) + b)
    test_prediction = tf.nn.softmax(tf.matmul(tf_test, W) + b)

num_steps = 140

def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions,1) == np.argmax(labels,1)) / predictions.shape[0])

with tf.Session(graph=graph) as session:
  tf.initialize_all_variables().run()
  print('Inizializzazione effettuata')
  for step in range(num_steps):
    _, l, predictions = session.run([optimizer, loss, train_prediction])
    if (step % 20 == 0):
        print('Loss at step %d: %f' % (step, l))
        print('Training accuracy: %.1f%%' % accuracy(predictions, matrix_train_labels))

# Write to the log:
#print("Training set has {0[0]} rows and {0[1]} columns".format(train.shape))
#print("Test set has {0[0]} rows and {0[1]} columns".format(test.shape))
# Any files you write to the current directory get shown as outputs