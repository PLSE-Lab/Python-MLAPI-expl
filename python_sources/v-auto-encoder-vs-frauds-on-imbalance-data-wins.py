#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


## data analysis for credit_card frauds 


# In[ ]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns


# In[ ]:


df_cred=pd.read_csv("/kaggle/input/creditcardfraud/creditcard.csv")


# In[ ]:


df_cred.shape


# In[ ]:


from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
import warnings
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.figure_factory as ff

warnings.filterwarnings('ignore')

from contextlib import contextmanager

@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))


# In[ ]:


plt.style.use('ggplot') # Using ggplot2 style visuals 

f, ax = plt.subplots(figsize=(11, 15))

ax.set_facecolor('#fafafa')
ax.set(xlim=(-5, 5))
plt.ylabel('Variables')
plt.title("Overview Data Set")
ax = sns.boxplot(data = df_cred.drop(columns=['Amount', 'Class', 'Time']), 
  orient = 'h', 
  palette = 'Set2')


# In[ ]:


fraud = df_cred[(df_cred['Class'] != 0)]
normal = df_cred[(df_cred['Class'] == 0)]

trace = go.Pie(labels = ['Normal', 'Fraud'], values = df_cred['Class'].value_counts(), 
               textfont=dict(size=15), opacity = 0.8,
               marker=dict(colors=['lightskyblue','gold'], 
                           line=dict(color='#000000', width=1.5)))


layout = dict(title =  'Distribution of target variable')
           
fig = dict(data = [trace], layout=layout)
py.iplot(fig)


# ## As we can see fraud cases are negligible and we have to build model to detect anamolies 
# 
# ## so i will be building auto-encoder first in keras then for production level code i will be building it in
# ## tensorflow and finally in tensorflow serving API for it's deployment 
# ## will post github link it it

# ## Feature distribution

# In[ ]:


# Def plot distribution
def plot_distribution(data_select) : 
    figsize =( 15, 8)
    sns.set_style("ticks")
    s = sns.FacetGrid(df_cred, hue = 'Class',aspect = 2.5, palette ={0 : 'lime', 1 :'black'})
    s.map(sns.kdeplot, data_select, shade = True, alpha = 0.6)
    s.set(xlim=(df_cred[data_select].min(), df_cred[data_select].max()))
    s.add_legend()
    s.set_axis_labels(data_select, 'proportion')
    s.fig.suptitle(data_select)
    plt.show()


# In[ ]:


plot_distribution('V4')
plot_distribution('V9')
plot_distribution('V11')
plot_distribution('V12')
plot_distribution('V13')


# ### data preprocessing it requires normalization why normalization 
# ### whenever we are seeing multiple features which are different ranges of distributions
# ### then we should prefer normalization if same range every feature but still lot of within range distribution
# ### we should give it standardization

# ### Now big question why auto-encoder for classification ? 
# ### how it is possible that neural network which is used for recontruction of input values can be used as a classifier for fraud transcations
# 
# ![](https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcQ5aL46qJsIl3AjFoOLyNbn_vdLe2a2tPns9-PikUI8EhpaLTJx&usqp=CAU)
# 

# ### so answer to this curiousity is very simple 
# ### as we know autoencoder is useful for reconstruciton of values 
# ### but if I train it on non-fraudulent transaction then it will be able to contruct non-fraudlent only
# ### so if I pass fraudulent transaction with non-fraudulent one then mse will be high for fraud transaction one
# ### why because it's weight are made on the basis of non-fraudulent transaction 
# ### then at last I will decide a perfect threshold for classify fraudulent vs non-fraudulent transaction 

# # Initial Preprocessing 

# In[ ]:


from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix , roc_auc_score, roc_curve


# In[ ]:


### dropping off unncessary columns


# In[ ]:


min_max_scaler = preprocessing.MinMaxScaler()
df_cred=df_cred.drop("Time",axis=1)
df_cred_scaled = min_max_scaler.fit_transform(df_cred.iloc[:,:-1])
df_cred_normalized = pd.DataFrame(df_cred_scaled)


# In[ ]:


df_cred_normalized["Class"]=df_cred["Class"]


# In[ ]:


df_cred_normalized["Class"].value_counts()


# # Spliting strategy

# 
# ### 20% percent in test set of 0 class then 10% in validation set of class 0 
# ### as we know in training label 1 class will not go but to decide the threshold of mse so that we can 
# ### classify anomaly perfectly so 50% of the 1 class percent we will be in test set rest 50% in validate set 
# (of course class 0 will be there too with class 1 in both sets validation and test)

# In[ ]:


df_cred_normalized_train=df_cred_normalized[df_cred_normalized["Class"]==0]
df_cred_normalized_test=df_cred_normalized[df_cred_normalized["Class"]==1]


# #### splitting dataset as per strategy I have dicussed 
# #### we will train it on non-fraudulent transcation and test on both the classes 
# 

# In[ ]:


df_cred_normalized_test_part_1=df_cred_normalized_train.sample(frac=0.05)
df_cred_normalized_train=df_cred_normalized_train.drop(df_cred_normalized_test_part_1.index)
df_cred_normalized_test_part_2=df_cred_normalized_train.sample(frac=0.05)
df_cred_normalized_train=df_cred_normalized_train.drop(df_cred_normalized_test_part_2.index)


# ### removing of fractional subset from main train set done 
# ### now starting up with making of test and validation set 

# In[ ]:


df_cred_normalized_test_class_1=df_cred_normalized_test.sample(frac=0.5)
df_cred_normalized_validation_class_1=df_cred_normalized_test.drop(df_cred_normalized_test_class_1.index)


# In[ ]:


df_cred_normalized_test_class_1.shape


# ## Merging of test and validation sets 

# In[ ]:


df_cred_normalized_test_set=df_cred_normalized_test_part_1.append(df_cred_normalized_test_class_1)
df_cred_normalized_validation_set=df_cred_normalized_test_part_2.append(df_cred_normalized_validation_class_1)


# ### just re-checking size of train test and validate set 

# In[ ]:


print("train set dimensions :",df_cred_normalized_train.shape)
print("test set dimensions :",df_cred_normalized_test_set.shape)
print("validate set dimensions :",df_cred_normalized_validation_set.shape)


# In[ ]:


df_cred_normalized_validation_set["Class"].value_counts()


# ### still need some small part of training set in testing of autoencoder network for reconstruction of values

# In[ ]:


X_train, X_test = train_test_split(df_cred_normalized_train, test_size=0.2, random_state=2020)
X_train = X_train[X_train.Class == 0]
X_train = X_train.drop(['Class'], axis=1)
y_test = X_test['Class']
X_test = X_test.drop(['Class'], axis=1)
X_train = X_train.values
X_test = X_test.values
X_train.shape


# # Autoencoder here we go 

# In[ ]:


from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.callbacks import EarlyStopping ,ReduceLROnPlateau
from keras.optimizers import Adam

#from keras import regularizers


# # What type of activations ?
# 
# Linear: Autoencoders with a single hidden layer with k hidden neurons and linear activations create equivalent representations to PCA with k principal components.
# 
# Binary: It is often used as introduction to ANN and not in real world applications.
# 
# ReLU: Rectified linear units are widely used in deep learning models. However, they are not suitable for AEs because they distort the decoding process by outputting 0 for negative inputs and consequently, do not lead to faithful representations of the input features.
# 
# SELU: Scaled exponential linear units activation function is a formidable alternative to ReLU as it preserves the advantages of linearly passing the positive inputs while it enables the flow of negative too.
# Sigmoid: The most commonly used activation function for autoencoders.
# 
# Tanh: Hyperbolic tangent is similar to sigmoid with the difference that is symmetric to the origin and its slope is steeper. As a result, it produces stronger gradients than sigmoid and should be preferred.
# 

# In[ ]:


input_dim = X_train.shape[1]
encoding_dim = 20
input_layer = Input(shape=(input_dim, ))
encoder = Dense(encoding_dim*2, activation="sigmoid")(input_layer)
encoder = Dense(encoding_dim, activation="sigmoid")(input_layer)
encoder = Dense(8,activation="sigmoid")(encoder)
decoder = Dense(20, activation='sigmoid')(encoder)
decoder = Dense(40, activation='sigmoid')(encoder)
decoder = Dense(input_dim, activation='sigmoid')(decoder)
autoencoder = Model(inputs=input_layer, outputs=decoder)


# ### Generally MSE as loss function but why , Mae we can use too or not?

# ### after 22 epochs we achieved plateaue and accuracy of 99.23 in reconstruction on test data 

# In[ ]:


nb_epoch = 50
batch_size = 32
autoencoder.compile(optimizer='adam', 
                    loss='mean_squared_error', 
                    metrics=['accuracy'])
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=15)

checkpointer = ModelCheckpoint(filepath="model.h5",
                               verbose=0,
                               save_best_only=True)
history = autoencoder.fit(X_train, X_train,
                    epochs=nb_epoch,
                    batch_size=batch_size,
                    shuffle=True,
                    validation_data=(X_test, X_test),callbacks=[es,checkpointer],
                    verbose=1)


# In[ ]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model acc')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()


# ### reconstruction error on x_test set

# In[ ]:


predictions = autoencoder.predict(X_test)
mse = np.mean(np.power(X_test - predictions, 2), axis=1)
error_df = pd.DataFrame({'reconstruction_error': mse,
                        'true_class': y_test})
error_df.describe()


# ### as you can std deviation is not so much varying while reconstruction of training data 
# ### as far as we are good to go for testing of out main test set and validation set 
# ### after that we can develope our same model in Tensorflow using TF records for mainline production 
# 

# # Evaluation of mse on both classes on test set

# In[ ]:


y_test=df_cred_normalized_test_set["Class"]
df_cred_normalized_test_set=df_cred_normalized_test_set.drop("Class",axis=1)


# In[ ]:


predictions = autoencoder.predict(df_cred_normalized_test_set)
mse = np.mean(np.power(df_cred_normalized_test_set - predictions, 2), axis=1)
error_df_test = pd.DataFrame({'reconstruction_error': mse,
                        'true_class': y_test})
error_df_test.describe()


# ### now checking how much reconstruction error present in class 0 and class 1

# In[ ]:


fig = plt.figure()
ax = fig.add_subplot(111)
normal_error_df = error_df_test[(error_df_test['true_class']== 0) & (error_df_test['reconstruction_error'] < 10)]
_ = ax.hist(normal_error_df.reconstruction_error.values, bins=200)


# ### very small mse will be present we are concerned for deciding the threshold in test set

# In[ ]:


fig = plt.figure()
ax = fig.add_subplot(111)
fraud_error_df = error_df_test[error_df_test['true_class'] == 1]
_ = ax.hist(fraud_error_df.reconstruction_error.values, bins=100)


# ### as we can see that from both graphs MSE for fraudulent cases is x10 times > Non-fraudulent cases 

# In[ ]:


fraud_error_df.describe() ### frauds cases 


# In[ ]:


normal_error_df.describe() ### non fraud cases


# #### selection of threshold as you can see max is 0.02 but if you observed 3rd quartile range it is really very small in comparison to max one which indicates that in selection of threshold we should not take max into the account because mean value of mse in frauds cases is 0.012 with std of 0.013 
# #### even minimum value of mse in normal transaction is range of minima of fraud cases but it's mse approx 30% is higher from normal transaction
# 
# #### so these cases which i saw are very much corner cases means and at extreme points to be get classified correctly

# In[ ]:


error_df_test["predicted_class"]=[1 if x > 0.001 else 0 for x in error_df_test["reconstruction_error"]]


# In[ ]:


error_df_test


# #### let's present our evaluation metrics over the threshold we have decided

# In[ ]:


from sklearn.metrics import (confusion_matrix, precision_recall_curve, auc,
                             roc_curve, recall_score, classification_report, f1_score,
                             precision_recall_fscore_support)


# In[ ]:


error_df_test["predicted_class"]=[1 if x > 0.001 else 0 for x in error_df_test["reconstruction_error"]]


# In[ ]:


fpr, tpr, thresholds = roc_curve(error_df_test.true_class, error_df_test.reconstruction_error)
roc_auc = auc(fpr, tpr)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, label='AUC = %0.4f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.001, 1])
plt.ylim([0, 1.001])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show();


# In[ ]:


print(classification_report(error_df_test["true_class"],error_df_test["predicted_class"]))


# ### now question comes up why precision is so low while recall is high 
# ### see we are actually testing imbalance test but in real world frauds cases will be 
# ### like this so what should we do ?
# ### focus on precision why see we can't catch every fraud but what we can catch as fraud case should be fraud to save company's money and customer as well

# In[ ]:


LABELS = ["Normal", "Fraud"]
y_pred = [1 if e > 0.004 else 0 for e in error_df_test.reconstruction_error.values]
conf_matrix = confusion_matrix(error_df_test.true_class,error_df_test.predicted_class)
plt.figure(figsize=(8, 8))
sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");
plt.title("Confusion matrix")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()


# ### so conclusion that out of 246 fraud cases 209 we were able to classify correctly 
# ### and normal transaction 352 cases model declaring them as fraud out of 14k cases
# ### so here threshold is making final model is like that 85 percent cases model can detect but it not precise so much , means model yes or no has no value , only what model can do is put those cases in suspect but can't bring final conclusions
# 
# ## *Stats can might change after running the kernel but it will approximate to those which were stated earlier

# ## now thershold is changed to focus on precision as primary
# 

# In[ ]:


error_df_test["predicted_class"]=[1 if x > 0.0039888 else 0 for x in error_df_test["reconstruction_error"]]


# In[ ]:


print(classification_report(error_df_test["true_class"],error_df_test["predicted_class"]))


# In[ ]:


LABELS = ["Normal", "Fraud"]
y_pred = [1 if e >  0.0039888 else 0 for e in error_df_test.reconstruction_error.values]
conf_matrix = confusion_matrix(error_df_test.true_class,error_df_test.predicted_class)
plt.figure(figsize=(8, 8))
sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");
plt.title("Confusion matrix")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()


# In[ ]:


fpr, tpr, thresholds = roc_curve(error_df_test.true_class, error_df_test.reconstruction_error)
roc_auc = auc(fpr, tpr)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, label='AUC = %0.4f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.001, 1])
plt.ylim([0, 1.001])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show();


# ### our roc-curve is telling that our model is doing really great in classifying both the classes
# ### but one should never forget sample size of test cases are imbalanced so always precision and recall 
# ### before deploying the model in real world 

# ### now same for final evaluation set that is our validation-set

# In[ ]:


y_test=df_cred_normalized_validation_set["Class"]
df_cred_normalized_validation_set=df_cred_normalized_validation_set.drop("Class",axis=1)
predictions = autoencoder.predict(df_cred_normalized_validation_set)
mse = np.mean(np.power(df_cred_normalized_validation_set - predictions, 2), axis=1)
error_df_test = pd.DataFrame({'reconstruction_error': mse,
                        'true_class': y_test})
error_df_test.describe()


# In[ ]:


error_df_test["predicted_class"]=[1 if x > 0.003 else 0 for x in error_df_test["reconstruction_error"]]


# In[ ]:


print(classification_report(error_df_test["true_class"],error_df_test["predicted_class"]))


# ### our model is performing really well even on validaiton dataset on fraud cases better than test set

# In[ ]:


fpr, tpr, thresholds = roc_curve(error_df_test.true_class, error_df_test.reconstruction_error)
roc_auc = auc(fpr, tpr)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, label='AUC = %0.4f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.001, 1])
plt.ylim([0, 1.001])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show();


# In[ ]:


LABELS = ["Normal", "Fraud"]
y_pred = [1 if e >  0.00398888 else 0 for e in error_df_test.reconstruction_error.values]
conf_matrix = confusion_matrix(error_df_test.true_class,error_df_test.predicted_class)
plt.figure(figsize=(8, 8))
sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");
plt.title("Confusion matrix")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()


# ### So guys here I expalined my strategy why I have chosen auto-encoder as for classifying fraud cases but as you can see that training of this model requires lot of computation time atleast 45 minson kaggle's gpu.
# ### we have move towards some faster solution not to save our training time but to save company resources as well 
# ### sometimes even in kaggle competition time contsraint issues can be solved if training can be done much more faster ways 
# ### but is just not about winining kaggle competition it is about real time working (model deployment should be done in Tensorflow)

# ![](https://www.cartoonmotivators.com/images/D/SocketOverload-01.jpg)

# ### so why keras as framework and it's model file should not be sent in model deployment (in production)?

# ### Tensorflow offers advance methods for managing and devlopement Neural Nets :
# #### 1> highly efficient data pipeline using tf records
# #### 2> allows you to use tf data pipeline api for feeding the model more faster 
# #### 3> allow you to control cpu gpu parrallel scheduling so that data preprocessing and training can be done much more faster but in keras everything is explicityly done by TF which takes time for processing data and training it (cpu gpu schedulling is not able to take place in keras)
# #### 4> model file is very small compared to keras so it faster process , deployable and it high scalability if deployed on aws EKS cluster (performance on images response or any input data will increase to significant scale)
#     

# #### so hardware best utilization we have to make it feel tired 
# #### utlimately apart from data scientist we need the best engineering skills to deploy our solution 
# #### with faster process if we can't then what is the use building such stacked and blend models

# ![](https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcTpdOSZuDXbWZdIaoSJRT2LvruTlATWgrEkgtKZhLaQpY6Sj60i&usqp=CAU)

# # So let's start with Tensorflow what is first we need to do 
# #### data exploration is done already 
# 
# ## Basic understanding :
# ### tensorflow session ,dataflow graphs , placeholders , variables , training  
# 
# ### so that you can come to know how in tf neural nets are structured first why the need of advance methods arises up 
# 
# 

# In[ ]:


### what are sessions and data flow graphs ??


# In[ ]:


# Import the TensorFlow library
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# Create an empty graph, that will be filled with operation nodes later
main_graph=tf.Graph()


# Register this graph as default graph. 
# All operations within this context will become operation nodes in this graph
with main_graph.as_default():
    
    # Create two constants of value 5
    a=tf.constant(5.0)
    b=tf.constant(5.0)
    
    # Multiply the constants with each other
    c=tf.multiply(a,b)
    

# Create a session to execute the dataflow graph
with tf.Session(graph=main_graph) as session:
    
    # Perform the calculation defined in the dataflow graph and get the result
    output=session.run(c)
    print('Result of the multiplication: %d '%output)


# In[ ]:


## what are placeholders ??


# In[ ]:



# Create an empty graph, that will be filled with operation nodes later
main_graph=tf.Graph()

# Register this graph as default graph. 
# All operations within this context will become operation nodes in this graph
with main_graph.as_default():
    
    # Define the placeholders that will feed python arrays into the dataflow graph
    a=tf.placeholder(name='a', shape=[5], dtype=tf.float32)
    b=tf.placeholder(name='b', shape=[5], dtype=tf.float32)
    
    c=tf.multiply(a,b)
    
# Create a session to execute the dataflow graph
with tf.Session(graph=main_graph) as session:
    
    # Perform the calculation defined in the dataflow graph and get the result.
    # We must provide the values for the placeholders with "feed_dict" dictionary
    output=session.run(c, feed_dict={a: [5.0,7.0,3.0,9.0,2.0],
                                     b: [1.0,2.0,4.0,8.0,4.0],
                                     })
    print(output)

   


# # Basic example NN as classfication using TF placeholders

# In[ ]:


import numpy as np
from random import shuffle

# list that will contain our training set
training_data=[]

# How many digits should this binary number have?
n=4
    
#Mini-batch size
batch_size=8

print('\n\nGeneration of Data...\n') 
for i in np.arange(0, 10):
    
    # Create a binary number of type string
    b = bin(i)[2:]
    l = len(b)
    b = str(0) * (n - l) + b  

    # Convert binary string number to type float
    features=np.array(list(b)).astype(float)
    # Create the corresponding binary label / class
    label=i
    
    # Put the feature-label pair into the list
    training_data.append([features, label])

    print('binary number: %s, decimal number: %d' %(b, i))
        
    
#%%
# shuffle the data
shuffle(training_data)  

training_data=training_data*1000

# convert the list to np.array     
training_data=np.array(training_data)



#%%

# Get the next mini-batch of training samples
def get_next_batch(n_batch):
    
    # Get the next mini-batch of training samples from the dataset
    features=training_data[n_batch*batch_size:(batch_size*(1+n_batch)),0]
    labels=training_data[n_batch*batch_size:(batch_size*(1+n_batch)),1]
    
    # Reshape the list of arrays into a nxn np.array
    features = np.concatenate(features).reshape([batch_size, 4])  
    # Reshape the labels 
    labels=np.reshape(labels, [batch_size])
    
    return features, labels
    
features, labels=get_next_batch(n_batch=1)

print('\n\nMini-batch of features: \n')
print(features)
print('\n\nMini-batch of labels: \n')
print(labels)

#%%  

# Create the training graph
main_graph=tf.Graph()

with main_graph.as_default():
    
    # Define the placeholders for the features and the labels
    x=tf.placeholder(dtype=tf.float32,shape=[batch_size, 4], name='features')
    y=tf.placeholder(dtype=tf.int32, shape=[batch_size], name='labels')
           
    # Create the weight matrices and the bias vectors 
    initializer=initializer=tf.random_normal_initializer(mean=0.0, stddev=0.25)
   
    W1=tf.get_variable('W1',shape=[4,50], initializer=initializer)
    W2=tf.get_variable('W2',shape=[50,25], initializer=initializer)
    W3=tf.get_variable('W3',shape=[25,10], initializer=initializer)
    
    b1=tf.get_variable('b1',shape=[50], initializer=initializer)
    b2=tf.get_variable('b2',shape=[25], initializer=initializer)

    ### Define the forward propagation step ###
    
    # First hidden layer
    z1=tf.matmul(x,W1)+b1
    a1=tf.nn.tanh(z1)
    
    # Second hidden layer
    z2=tf.matmul(a1,W2)+b2
    a2=tf.nn.tanh(z2)
    
    # Outputlayer, without an activation function (input for the loss function)
    logits=tf.matmul(a2,W3)
       
    # Compute the probability scores after the training)
    probs=tf.nn.softmax(logits)
    
    # Define the loss function
    loss_op=tf.losses.sparse_softmax_cross_entropy(labels=y, logits=logits)
       
    # Perform a gradient descent step
    optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.001)
    trainable_parameters = tf.trainable_variables()
    gradients = tf.gradients(loss_op,trainable_parameters)
    update_step = optimizer.apply_gradients(zip(gradients, trainable_parameters))


print('\n\nStart of the training...\n')
with tf.Session(graph=main_graph) as sess:
    
    sess.run(tf.global_variables_initializer())

    # How many mini-batches in total?
    num_batches=int(10000/batch_size)
    
    loss=0

    #Iterate over the entire training set for 10 times
    for epoch in range(10):
            
        # Iterate over the number of mini-batches
        for n_batch in range(num_batches-1):
            
            # Get the next mini-batches of samples for the training set
            features, labels=get_next_batch(n_batch)
              
            # Perform the gradient descent step on that mini-batch and compute the loss value
            _, loss_=sess.run((update_step, loss_op), feed_dict={x:features, y:labels})   
            
            loss+=loss_
             
        print('epoch_nr.: %i, loss: %.3f' %(epoch,(loss/num_batches)))
        loss=0 
    
    
    print('\n\nTesting the neural network:\n')
    # Compute the probability scores for the last mini-batch
    prob_scores=sess.run(probs, feed_dict={x:features, y:labels})
    
    # Iterate over the features and labels from the last mini-batch as well as
    # the predicitons made by the network, and compare them to check the performance
    for f, l, p in zip(features, labels, prob_scores):
    
        # Get the class with the highest probability score
        predicted_class=np.argmax(p)
        # Get the actual probability score
        predicted_class_score=np.max(p)
     
        print('Binary number: %s, decimal number: %i, predicted_class: %i, predicted_prob_score: %.3f' 
              %(str(f), l, predicted_class, predicted_class_score))


# ## Advanced methods
# ### 1> step - conversion of data to tf records (binary format data)
# ### 2> step - use tf.data input pipeline to connect tf records and send to base model
# ### 3> step - advance methods to schedule cpu and gpu processing (Parrallelize Data Transformation)
# ### 4> step - tf flags, namescopes are really important for documenting your model
# 

# In[ ]:




