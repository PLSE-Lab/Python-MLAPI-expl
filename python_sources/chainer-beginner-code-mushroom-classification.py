#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import chainer as ch
from chainer import datasets
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# [Chainer reference](https://docs.chainer.org/en/stable/glance.html)    
# Chainer has a "[Define-by-run](https://docs.chainer.org/en/stable/guides/define_by_run.html)" philosophy which lets us build flexible neural architectures.   
# This means that the network can be defined dynamically using the forward computation. 
# 
# Chainer has a *trainer* which is used to set up the neural network  and data for training. The trainer has a structure that's hierarchical and looks like this  
# ![](http://gdurl.com/Hu7m)  
# 
# Each of the components is fed information from the components within it.   
# **Therefore in order to set up the trainer we start with the inner most components first and move outwards.** (with the extensions as an exception which are added after the trainer is made.)  
# 
# So we  are going to follow this particular order so as to set up a trainer:  
# 
# 1. format the dataset  
# 2. configure the iterator to step through the dataset for training and validation
# 3. define the neural network to include in the model
# 4. pick an optimizer and setup the model to use it
# 5. create an updater using the optimizer. The updater is called after the training batches 
# 6. Set up the trainer using the updater.
# 7. The extensions are added to the trainer once the latter is built. an e.g. of an extension to the trainier is the [Evaluator](https://docs.chainer.org/en/stable/reference/generated/chainer.training.extensions.Evaluator.html#chainer.training.extensions.Evaluator) which provides test scores.   
# 
# The following set of images   describes how we go about building a trainer.(steps 1->7)
# ![](http://gdurl.com/cYDc)
# 
# 
# 
# 
# 
# Ok so we are now going to move through these steps one at a time.
# 
# 
# **[Datasets](https://docs.chainer.org/en/stable/reference/datasets.html)**   
# First lets transform the dataset into a type of [Chainers General Datasets](https://docs.chainer.org/en/stable/reference/datasets.html#module-chainer.dataset) called the [TupleDataset](https://docs.chainer.org/en/stable/reference/generated/chainer.datasets.TupleDataset.html#chainer.datasets.TupleDataset)  
# The TupleDataset basically does what the zip() function does i.e it takes in two equal length datasets d1=[1,2,3,4] and d2=[5,6,7,8] and returns a tuple dataset (1,5), (2,6), (3,7), (4,8)  
# We are now going to load the raw mushrooms csv file into a Chainer TupleDataset

# In[ ]:


mfile='../input/mushrooms.csv'
data_array=np.genfromtxt(mfile,delimiter=',',dtype=str,skip_header=True)
df=pd.read_csv('../input/mushrooms.csv')
df.head()


# In[ ]:


df.shape


# In[ ]:


df.describe()


#  The first column is the label. The rest are nominal features. We can use LabelEncoder for the converting label values to integers ${y \in \{0,1\}}$ depending on whether ${label=p}$ or ${label=e}$.  
#  However the same can't be said for the 22 nominal features. Applying the LabelEncoder on these features will convert the nominal values to integers , so if ${k=1}$ and ${n=2}$ for the feature gill color, it implies ${k}$ which represents the color black  is less than ${n}$ which stands for  the color brown i.e black is less than brown which is obviously an incorrect assumption since there is no ordering amongst the colors . Such an approach will affect the learning of our algorithm.  
#  Here I will be using the One-Hot-Encoding (ohe) which will end up creating dummy features for each new value that the original feature takes. So if the stalk_shape feature takes 2 values (e,t) then the ohe of this feature will produce two more "dummy" features stalk_shape_e and stalk_shape_t , so for a sample where ${stalk\_shape=e \implies stalk\_shape\_e=1}$ and ${stalk\_shape\_t=0 }$ for this sample .  
#  For a dataframe we will be using the *get_dummies()*  method implemented in  pandas to get ohe done.

# In[ ]:


df.describe().iloc[1][1:].sum()   # summing along the "unique" row . all columns except the first one which corresponds to the class of the sample.


# Expecting a resulting 117-dimensional feature space at the end of ohe

# In[ ]:


l=[x for x in df.columns[1:]]   # make a list of all the column names , all except the class column
print(l)
data_df=pd.get_dummies(df[l],drop_first=False)
data_df.shape


# In[ ]:


data_df.describe()


# Now we perform LabelEncoding on the class labels

# In[ ]:


from sklearn.preprocessing import LabelEncoder
class_le=LabelEncoder()
y=class_le.fit_transform(df['class'].values)
y=y.reshape(y.shape[0],1)
assert(y.size==data_df.shape[0])              # I do this because I intend to create tuples of (data_df,y), so both the structures need to be of equal length
print(y)


# 
# 
# Next to convert the data to TupleDataset and split into training and testing sets using [datasets.split_dataset_random](https://docs.chainer.org/en/stable/reference/generated/chainer.datasets.split_dataset_random.html#chainer.datasets.split_dataset_random).
# This function creates two instances of [SubDataset](https://docs.chainer.org/en/stable/reference/generated/chainer.datasets.SubDataset.html#chainer.datasets.SubDataset). These instances do not share any examples, and they together cover all examples of the original dataset.

# In[ ]:


TRAIN_SIZE_PERC=0.7

tuple_dataset=datasets.TupleDataset(data_df.values.astype(np.float32),y)
train,test=datasets.split_dataset_random(tuple_dataset,first_size=int(TRAIN_SIZE_PERC*len(tuple_dataset)))  # 70% of the data is used for training
print(len(train))
print(len(test))


# Now to the next step  
# 
# **[The Iterator](https://docs.chainer.org/en/stable/reference/iterators.html#module-chainer.iterators)**  
# The iterator is used to implement strategies to create mini-batches by iterating over the datasets. I will be using the [SerialIterator](https://docs.chainer.org/en/stable/reference/generated/chainer.iterators.SerialIterator.html#chainer.iterators.SerialIterator). and a batch size of 120 .  
# For the training iterator, we set the shuffle and repeat parameters to True  while for the testing iterator we set them to False (This is to basically prevent overfitting of the training data . The  variance between the mini batches aka the sampling variance is reduced each time there's a shuffle at the beginning of the epoch , [further logic answered on stackoverflow](https://datascience.stackexchange.com/questions/24511/why-should-the-data-be-shuffled-for-machine-learning-tasks))
# 

# In[ ]:


BATCH_SIZE=120
train_iter=ch.iterators.SerialIterator(train,BATCH_SIZE)
test_iter=ch.iterators.SerialIterator(test,BATCH_SIZE,repeat=False,shuffle=False)


# **The Model**  
# This is where we create a neural network .
# There is an easy way to create the neural network using Chainer's [Sequential](https://docs.chainer.org/en/stable/reference/generated/chainer.Sequential.html#chainer.Sequential) class.  
# 
# But I will be building from more basic classes  
# 
# Chainer uses [Link](https://docs.chainer.org/en/stable/guides/links.html) as it's building block of neural network. You can use this class to define your own implementation of a neural network.   
# As an example consider we want to make our own implementation of a Chainers connected  [Linear](https://docs.chainer.org/en/stable/reference/generated/chainer.links.Linear.html#chainer.links.Linear)  where each layer implements its forward propogation as ${y_{i}=Wx_{i}+b}$ but instead of initializing the weights as Linears default Normal distribution, ours will be a [He-initialization](https://docs.chainer.org/en/stable/reference/generated/chainer.initializers.HeNormal.html#chainer.initializers.HeNormal)  this is how we go about doing this.  
# 
# 
# 
# 
# 
# 
# 

# In[ ]:


import chainer
from chainer import initializers
import chainer.functions as F

class CustomLinearLayer(chainer.Link):
    
    def __init__(self,n_in,n_out):
        super(CustomLinearLayer,self).__init__()
        with self.init_scope():
            self.W=chainer.Parameter(
                                        initializers.HeNormal(),      # He-initialization
                                        (n_out,n_in)                # W matrix is (n_out X n_in)
                                        
                                    )
            self.b=chainer.Parameter(
                                        initializers.Zero(),        # initialized to zero
                                        (n_out,)                   # bias is of shape (n_out,)
                                    )
            
    #forward propogation implementation:
    def forward(self,x):
        return F.linear(x,self.W,self.b)
    


# This above Link can be linked together to form a  [Chain](https://docs.chainer.org/en/stable/reference/generated/chainer.Chain.html#chainer.Chain).( i.e a neural network architecture ).  Here are some of Chainer's [links](https://docs.chainer.org/en/stable/reference/links.html#module-chainer.links).  
# 
# Let's create a CustomMultiLayerPerceptron Chain  using the above CustomLinearLayer Link .  
# Our architecture will have 1 input layer, 3 hidden layers and an output layer.    
# The hidden layers will have a [relu activation](https://docs.chainer.org/en/stable/reference/functions.html#activation-functions) function while the output layer will have a sigmoid.

# In[ ]:


import chainer
import chainer.functions as F
import chainer.links as L

class CustomMultiLayerPerceptron(chainer.Chain):
    
    def __init__(self,n_in,n_hidden,n_out):
        super(CustomMultiLayerPerceptron,self).__init__()
        with self.init_scope():
            self.layer1 = CustomLinearLayer(n_in,n_hidden)                                     # input layer
            self.layer2 = CustomLinearLayer(n_hidden,n_hidden)                                 # hidden layer
            self.layer3 = CustomLinearLayer(n_hidden,n_hidden)                                 # hidden layer
            self.layer4 = CustomLinearLayer(n_hidden,n_hidden)                                 # hidden layer
            self.layer5 = CustomLinearLayer(n_hidden,n_hidden)                                 # hidden layer
            self.layer6 = CustomLinearLayer(n_hidden,n_out)                                    # output layer
        
        #forward propagation
    def forward(self,*args):
        x=args[0]
        h1=F.relu(self.layer1(x))        # implements the  CustomLinearLayer link's forward propogation on x. i.e. h1=relu(x.W_1+b_1)
        h2=F.relu(self.layer2(h1))       # h2= relu( h1.W_2 + b_2)
        h3=F.relu(self.layer3(h2))       # h3= relu( h2.W_3 + b_3)
        h4=F.relu(self.layer4(h3))       # h4= relu( h3.W_4 + b_4)
        h5=F.relu(self.layer5(h4))       # h5= relu( h4.W_5 + b_5)
        #h6=F.sigmoid(self.layer6(h5))    # h6= sigmoid( h5.W_6 + b_6)
        h6=self.layer6(h5)    # h6=  h5.W_6 + b_6
        #print(h6)
        return h6
    


# Ok so now we have a predictor network. i.e one that does this ${x \implies h_{1}\implies h_{2}\implies h_{3} \implies \hat{y } }$   where ${\hat{y}}$  is the predicted output.   
# But we want our network to calculate a loss and also output a metric that shows how well our network is performing at the task of classifying the mushrooms as either edible or deadly.  
# We create a new Chain, a CustomClassifier which is our implementation of Chainer's [Classifier](https://docs.chainer.org/en/stable/reference/generated/chainer.links.Classifier.html#chainer.links.Classifier). 
# 
# This CustomClassifier will wrap the predictor network chain . It will then compute the loss and metric(accuracy) based on a given input/truth pair
# 
# 
# 

# In[ ]:


from chainer.functions.evaluation import accuracy
from chainer.functions.loss import softmax_cross_entropy
from chainer import link
from chainer import reporter

class CustomClassifier(link.Chain):
    def __init__(self,
                    predictor,                                            #predictor network that this classifier wraps
                    lossfun=softmax_cross_entropy.softmax_cross_entropy,  #the lossfunction it uses        
                    accfun=accuracy.accuracy,                              #the performance metric used
                    label_key=-1                                          #the location of the label in the input minibatch. (defaulted to the rightmost column)
                ):
        super(CustomClassifier,self).__init__()
        self.lossfun = lossfun
        self.accfun  = accfun
        self.y       = None                                               # the prediction from the last minibatch  y_hat
        self.loss    = None                                               #loss value for the last minibatch
        self.accuracy= None                                               #accuracy for the last minibatch
        self.label_key=label_key                                         # the location of the label in the input minibatch
        with self.init_scope():                                          # creates an initialization scope. See documentation for details.
            self.predictor = predictor
    
    
    def forward(self,*args,**kwargs):
        """
            Computes loss value for an input /label pair
            Computes accuracy 
            
            Args:
                args  : Input minibatch  
                kwargs: Input minibatch
            
        """
        self.y = None
        self.loss = None
        self.accuracy = None
        
        t=args[self.label_key]                                              #ground truth for the minibatch
    
        self.y = self.predictor(*args)                                 #get the output from the predictor
        self.loss=self.lossfun(self.y,t)                               #calculate the loss for this minibatch
        reporter.report({'loss':self.loss},self)
        self.accuracy = self.accfun(self.y,t)                          #the performance metric
        reporter.report({'accuracy':self.accuracy},self)
        
        return self.loss


# Ok now to create the model   
# We pass the Predictor chain to the Classifier  along with a loss function and performance metric  
# The predictor has 3 hidden layers as defined in the class, each parameterized to have 234 nodes in each of the hidden layers and a single node in the output layer  
# 
# 

# In[ ]:


model=CustomClassifier(CustomMultiLayerPerceptron(n_in=data_df.shape[1],n_hidden=data_df.shape[1]*3,n_out=1),
                       lossfun=F.sigmoid_cross_entropy,
                       accfun=F.binary_accuracy
                      )


# **[Optmizer](https://docs.chainer.org/en/stable/reference/generated/chainer.Optimizer.html#chainer.Optimizer)**  
# This is the class responsible for all numerical optimizers and  provides basic features for all optimization methods.   
# We will use the [SGD](https://docs.chainer.org/en/stable/reference/generated/chainer.optimizers.SGD.html#chainer.optimizers.SGD) optimizer provided in Chainer which represents the vanilla Stochastic Gradient Descent Algorithm.  
# 
# The Optimizer optimizes parameters (weights and biases) of a target link (our model from the previous step).   
# The target link which is our model is registered via the setup() method of the Optimizer, and then the parameter update is taken care of by the [update()](https://docs.chainer.org/en/stable/reference/generated/chainer.Optimizer.html#chainer.Optimizer.update) method which does the update based on a given loss function which in our case is the sigmoid cross entropy function  
# 
# We will be using Chainers [SGD](https://docs.chainer.org/en/stable/reference/generated/chainer.optimizers.SGD.html#chainer.optimizers.SGD) which is its implementation of the vanilla Stochastic Gradient Descent

# In[ ]:


optimizer=ch.optimizers.SGD(lr=0.001).setup(model)


# At this point we have the dataset , the training iterator  , the model and now the optimizer.  
# Now to the next step which is setting up the updater.  
# 
# **[Updater](https://docs.chainer.org/en/stable/reference/generated/chainer.training.Updater.html#chainer.training.Updater)**  
# The updater uses the minibatch from the training iterator and  does the forward and backward processing of the model.  Based on the optimizer we chose this class updates the  weights and biases of the network.   More specifically 
# If the training is to be done on a GPU , then set the device parameter to the number of the GPU device (usually device=0) while for a CPU use device=-1  
# The Updater is responsible for   
# * getting the minibatch from  the dataset via the iterator.  
# * run the forward and backward process of the Chain by using the Optimizer (which in turn calls its own [update()](https://docs.chainer.org/en/stable/reference/generated/chainer.Optimizer.html#chainer.Optimizer.update) method to do these tasks).  
# * Update parameters according to their [UpdateRule](https://docs.chainer.org/en/stable/reference/generated/chainer.UpdateRule.html#chainer.UpdateRule). This too is handled up the Optimizer.update() method.   
# The UpdateRule is a class that is implements how to update a parameter variable using the gradient of the loss function  
# If you wish to write your own implementation of the update rule, then this can be done by overriding the [Updater.update()](https://docs.chainer.org/en/stable/reference/generated/chainer.training.Updater.html#chainer.training.Updater.update) method 
# 
# We will be using the [StandardUpdater](https://docs.chainer.org/en/stable/reference/generated/chainer.training.updaters.StandardUpdater.html)
# 

# In[ ]:


updater=training.StandardUpdater(iterator=train_iter,optimizer=optimizer,device=-1) # set up the updater using 
                                                                          #the iterator and the optimizer


# Once the updater is made, time to create the [**Trainer**](https://docs.chainer.org/en/stable/reference/generated/chainer.training.Trainer.html#chainer.training.Trainer)  
# The trainer represents a training loop  and consists of two parts. The Updater which actually updates the parameters and the [Extension](https://docs.chainer.org/en/stable/reference/generated/chainer.training.Extension.html#chainer.training.Extension) for arbitary functionalities viz producing test scores.  
# 
# Each iteration of this loop does :  
# * Update of the parameters by the Updater. This includes loading of the mini-batch using the training iterator,  the backward and forward propagations and the update of the parameters.   
# * Call any extensions attached to the trainer (We havent attached any yet. But we will once our trainer is built)
# 
# 
# The Trainer takes as arguments:  
# The updater  
# A [Trigger](https://docs.chainer.org/en/stable/reference/training.html#triggers) which is a callable that decides when to process some specific event within the training loop.   
# 
# In our example we will be doing 50 Sweeps across the entire training set. i.e 50 epochs which is specified  here as a tuple  (period,unit) where period is the length of an interval and the unit of measurement is  'epoch'. see this [link](https://docs.chainer.org/en/stable/reference/generated/chainer.training.triggers.IntervalTrigger.html#chainer.training.triggers.IntervalTrigger)  
# 

# In[ ]:


PERIOD=50                           
trainer=training.Trainer(updater,(PERIOD,'epoch'),out='result')


# **[Extensions](https://docs.chainer.org/en/stable/guides/extensions.html)**  
# We have the option to attach any Extensions to the Trainer using the extend() method of the trainer.  
# In addition we can write our own Extensions as simple functions and attach them to the trainer.   
# In the example we will be using [PlotReport](https://docs.chainer.org/en/stable/reference/generated/chainer.training.extensions.PlotReport.html#chainer.training.extensions.PlotReport) and [PrintReport](https://docs.chainer.org/en/stable/reference/generated/chainer.training.extensions.PrintReport.html#chainer.training.extensions.PrintReport) to plot our loss and accuracy.
# 

# In[ ]:


trainer.extend(extensions.Evaluator(test_iter, model, device=-1))
trainer.extend(extensions.dump_graph('main/loss'))
trainer.extend(extensions.snapshot(), trigger=(20, 'epoch'))
trainer.extend(extensions.LogReport())

if extensions.PlotReport.available():
    trainer.extend(
        extensions.PlotReport(['main/loss', 'validation/main/loss'],
                              'epoch', file_name='loss.png'))
    trainer.extend(
        extensions.PlotReport(
            ['main/accuracy', 'validation/main/accuracy'],
            'epoch', file_name='accuracy.png'))

    
trainer.extend(extensions.PrintReport(
    ['epoch', 'main/loss', 'validation/main/loss',
     'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))
trainer.run()


# **Save the model.**

# In[ ]:


from chainer import serializers
serializers.save_hdf5('mushroom_model_01.hdf5',model)


# **Inference**  
# First load the saved model and check the model on the test data

# In[86]:


#Load the trained model
nmod=CustomClassifier(CustomMultiLayerPerceptron(n_in=data_df.shape[1],n_hidden=data_df.shape[1]*3,n_out=1),
                       lossfun=F.sigmoid_cross_entropy,
                       accfun=F.binary_accuracy
                      )
serializers.load_hdf5('mushroom_model_01.hdf5',nmod)


# In[109]:


#Check on test data
xtest=[x[0] for x in test]  # extract the test features
ytest=[x[1] for x in test]  # get the test labels
probs=nmod.predictor(np.array(xtest)).data # see Variable class. predictor(x) does the forward prop returning a Variable object h6. The .data is a member of the class


preds=np.where(probs>0,1,0)             # predictions thresholded at 0


# Next some performance [metrics](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html#sklearn.metrics.roc_curve)

# In[119]:


from sklearn.metrics import roc_curve,auc,roc_auc_score
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go

false_pos_rates,true_pos_rates,thresholds=roc_curve(ytest,probs)                       # we need to plot the roc . the y-axis are all true positives and the x-axis are all false positives.
print(false_pos_rates)                                                                 # we use the probabilities here while plotting the roc and NOT the predictions.

roc_auc=auc(false_pos_rates,true_pos_rates)
print(roc_auc)

roc_auc_sc=roc_auc_score(ytest,probs)
print(roc_auc_sc)

#plotting the roc 
trace=go.Scatter(
    x=false_pos_rates,
    y=true_pos_rates,
    mode='lines',
    name='ROC',
    
)
randomGuess=go.Line(
    x=[0,1],
    y=[0,1],
    line=dict(dash='dash'),
    name='random guess'
)

layout=go.Layout(
    annotations=[
        dict(
        x=0.2,
        y=0.6,
        text='AUC: \n '+str(roc_auc_sc),
        showarrow=False
        )
    ]
)
data=[trace,randomGuess]
fig=go.Figure(data=data,layout=layout)
py.iplot(fig)


# This seems like a pretty decent classifier. 

# In[ ]:




