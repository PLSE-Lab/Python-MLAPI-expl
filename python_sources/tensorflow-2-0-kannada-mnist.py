#!/usr/bin/env python
# coding: utf-8

# ### Installing TensorFlow 2.0

# In[ ]:


#!pip install --upgrade tensorflow-gpu


# ### Importing the dependencies

# In[ ]:


get_ipython().run_cell_magic('time', '', 'from __future__ import print_function, absolute_import, division, unicode_literals\n\n%matplotlib inline\n%config InlineBackend.figure_format = "retina"\n\nimport numpy as np\nimport pandas as pd\nimport matplotlib.pyplot as plt\nimport seaborn as sns\n\nimport os\n\nimport tensorflow as tf')


# ### TensorFlow setup

# In[ ]:


print(f"TensorFlow version: {tf.__version__}")
print(f"Keras version: {tf.keras.__version__}")
print(f"TensorFlow is executing eagerly: {tf.executing_eagerly()}")
print("GPU is","available." if tf.test.is_gpu_available() else "unavailable.")
print(f"Initializing radom seeds..{tf.random.set_seed(1)}")
print(f"Enabling TensorFlow Device Debugger..")
tf.config.set_soft_device_placement(True)
tf.debugging.set_log_device_placement(True)
print("No of physical devices available: {}".format(len(tf.config.experimental.list_physical_devices())))
get_ipython().run_line_magic('reload_ext', 'tensorboard.notebook')

import warnings; warnings.simplefilter("ignore")
print("Done.!")


# ### Reading the data

# In[ ]:


os.listdir('../input/Kannada-MNIST/')


# In[ ]:


train_df = pd.read_csv('../input/Kannada-MNIST/train.csv')
test_df = pd.read_csv('../input/Kannada-MNIST/test.csv')


# In[ ]:


from IPython.display import display, clear_output
with pd.option_context("display.max_rows",10,"display.max_columns",1000):
    display(train_df.head(5))
    clear_output()
    display(test_df.head(5))


# In[ ]:


train_labels = train_df.iloc[:,0].values
train_features = train_df.iloc[:,1:].values.reshape(-1,28,28,1)
test_features = test_df.iloc[:,1:].values.reshape(-1,28,28,1)


# In[ ]:


print(train_features.shape)
print(train_labels.shape)
print(test_features.shape)


# ### Train and Validation

# In[ ]:


from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(train_features, train_labels,test_size=0.1)


# ### Define distribution strategy

# In[ ]:


strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())


# In[ ]:


print(f"Number of devices in parallel: {strategy.num_replicas_in_sync}")


# ### Setup input pipeline

# In[ ]:


BUFFER_SIZE = 10000
BATCH_SIZE_PER_REPLICA = 64
BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync


# In[ ]:


def scale(image, label):
    image = tf.cast(image, tf.float32)
    image /= 255
    return image, label


# In[ ]:


train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))                 .map(scale, num_parallel_calls=tf.data.experimental.AUTOTUNE)                 .cache()                 .shuffle(BUFFER_SIZE)                 .batch(BATCH_SIZE)                 .repeat()                 .prefetch(tf.data.experimental.AUTOTUNE)

val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))                 .map(scale, num_parallel_calls=tf.data.experimental.AUTOTUNE)                 .batch(BATCH_SIZE)


# In[ ]:


print(train_dataset.element_spec)
print(val_dataset.element_spec)


# In[ ]:


for images, labels in train_dataset.take(1):
    pass

train_features = images.numpy()
train_labels = labels.numpy()


# #### Vizualize

# In[ ]:


sns.set_style("whitegrid")

plt.figure(figsize=(20,18))
for idx in range(36):
    plt.subplot(6,6,idx+1)
    plt.imshow(np.squeeze(train_features[idx]),cmap=plt.cm.binary,interpolation='nearest')
    plt.xticks([])
    plt.yticks([])
    plt.grid(True)
    plt.title(train_labels[idx])
    plt.colorbar()
plt.suptitle("Kannada MNIST Examples")
plt.show()


# ### Create the model

# In[ ]:


activation = 'relu'
padding = 'same'
gamma_initializer = 'uniform'
input_shape=(28,28,1)

with strategy.scope():
    input_layer = tf.keras.layers.Input(shape=input_shape)
    hidden=tf.keras.layers.Conv2D(16, (3,3), strides=1, padding=padding, activation = activation, name="conv1")(input_layer)
    hidden=tf.keras.layers.BatchNormalization(axis =1, momentum=0.1, epsilon=1e-5, gamma_initializer=gamma_initializer, name="batch1")(hidden)
    hidden=tf.keras.layers.Dropout(0.1)(hidden)

    hidden=tf.keras.layers.Conv2D(32, (3,3), strides=1, padding=padding,activation = activation, name="conv2")(hidden)
    hidden=tf.keras.layers.BatchNormalization(axis =1,momentum=0.15, epsilon=1e-5, gamma_initializer=gamma_initializer, name="batch2")(hidden)
    hidden=tf.keras.layers.Dropout(0.15)(hidden)
    hidden=tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding=padding, name="max2")(hidden)

    hidden=tf.keras.layers.Conv2D(64, (3,3), strides=1, padding =padding, activation = activation,  name="conv3")(hidden)
    hidden=tf.keras.layers.BatchNormalization(axis =1,momentum=0.17, epsilon=1e-5, gamma_initializer=gamma_initializer, name="batch3")(hidden)
    hidden=tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding="same", name="max3")(hidden)

    hidden=tf.keras.layers.Conv2D(128, (3,3), strides=1, padding=padding, activation = activation, name="conv4")(hidden)
    hidden=tf.keras.layers.BatchNormalization(axis =1,momentum=0.15, epsilon=1e-5, gamma_initializer=gamma_initializer, name="batch4")(hidden)
    hidden=tf.keras.layers.Dropout(0.15)(hidden)
    
    hidden=tf.keras.layers.Conv2D(256, (3,3), strides=1, padding =padding, activation = activation,  name="conv5")(hidden)
    hidden=tf.keras.layers.BatchNormalization(axis =1,momentum=0.17, epsilon=1e-5, gamma_initializer=gamma_initializer, name="batch5")(hidden)
    hidden=tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding="same", name="max4")(hidden)

    hidden=tf.keras.layers.Flatten()(hidden)
    hidden=tf.keras.layers.Dense(1024,activation = activation, name="Dense1")(hidden)
    hidden=tf.keras.layers.Dropout(0.05)(hidden)
    hidden=tf.keras.layers.Dense(512,activation = activation, name="Dense2")(hidden)
    hidden=tf.keras.layers.Dropout(0.05)(hidden)
    hidden=tf.keras.layers.Dense(256, activation = activation, name="Dense3")(hidden)
    hidden=tf.keras.layers.Dropout(0.03)(hidden)
    output = tf.keras.layers.Dense(10, activation = "softmax")(hidden)

    model = tf.keras.Model(inputs=input_layer, outputs=output)

    model.compile(loss='sparse_categorical_crossentropy',
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                metrics=["accuracy"])


# In[ ]:


model.summary()


# In[ ]:


tf.keras.utils.plot_model(model)


# ### Define the callbacks

# In[ ]:


from IPython.display import clear_output
class PlotLearning(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.loss = []
        self.val_loss = []
        self.acc = []
        self.val_acc=[]
        self.fig = plt.figure()
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):      
        self.logs.append(logs)
        self.x.append(self.i)
        self.loss.append(logs.get('loss'))
        self.val_loss.append(logs.get('val_loss'))
        self.acc.append(logs.get('accuracy'))
        self.val_acc.append(logs.get('val_accuracy'))

        self.i += 1
        f, ax = plt.subplots(1, 2, figsize=(12,4), sharex=True)
        ax = ax.flatten()
        clear_output(wait=True)

        ax[0].plot(self.x, self.loss, label="loss", lw=2)
        ax[0].plot(self.x, self.val_loss, label="val loss")
        ax[0].legend()
        ax[0].grid(True)

        ax[1].plot(self.x, self.acc, label="accuracy", lw=2)
        ax[1].plot(self.x, self.val_acc, label="val accuracy")
        ax[1].legend()
        ax[1].grid(True)

    plt.show();


# In[ ]:


checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")


# In[ ]:


def decay(epoch):
    if epoch < 5:
        return 1e-3
    elif epoch >= 5 and epoch < 10:
        return 1e-4
    else:
        return 1e-5


# In[ ]:


callbacks = [
    tf.keras.callbacks.TensorBoard(log_dir='logs'),
    tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix,
                                       save_weights_only=True,verbose=1),
    tf.keras.callbacks.LearningRateScheduler(decay,verbose=1),
    PlotLearning(),
    tf.keras.callbacks.EarlyStopping(monitor='val_loss',verbose=1,patience=6)
]


# ### Train the model 

# In[ ]:


history = model.fit(train_dataset, 
                    validation_data=val_dataset, 
                    steps_per_epoch=512, 
                    epochs=15, 
                    callbacks=callbacks, 
                    verbose=1)


# In[ ]:


get_ipython().system('ls {checkpoint_dir}')


# ### Evaluate the model

# In[ ]:


model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
eval_loss, eval_acc = model.evaluate(val_dataset,verbose=1,steps=10)
print('Eval loss: {:.2}, Eval Accuracy: {:.2}'.format(eval_loss, eval_acc))


# ### Vizualize in TensorBoard

# In[ ]:


# %tensorboard --logdir=logs


# ### Export to SavedModel

# In[ ]:


path = 'saved_model/'
model.save(path, save_format='tf')


# ### Make Predictions

# In[ ]:


test_data = tf.data.Dataset.from_tensors(tf.cast(test_features,tf.float32))


# In[ ]:


predictions = np.argmax(model.predict(test_data),axis=1)


# ### Submit to Competition

# In[ ]:


submission = pd.read_csv('../input/Kannada-MNIST/sample_submission.csv')
submission.head()


# In[ ]:


submission['label'] = predictions


# In[ ]:


submission.to_csv("submission.csv",index=False)


# In[ ]:


plt.figure(figsize=(8,6))
submission["label"].hist(bins=20)

