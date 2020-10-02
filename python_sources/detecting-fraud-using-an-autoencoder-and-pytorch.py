#!/usr/bin/env python
# coding: utf-8

# In[43]:


import os

import pandas as pd
import numpy as np
import seaborn as sns
import collections
from tqdm import tqdm, tqdm_notebook

# PyTorch Packages
import torch.nn as nn
from torch.autograd import Variable as V
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader

# SKLearn Packages
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import (confusion_matrix, precision_recall_curve, auc,
                             roc_curve, recall_score, classification_report, f1_score)
from sklearn.metrics import accuracy_score, precision_score

# Plotting packages
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from bokeh.plotting import figure, output_notebook, show, ColumnDataSource
from bokeh.models import HoverTool, NumeralTickFormatter
from bokeh.palettes import Set3_12
from bokeh.transform import jitter


# In[44]:


output_notebook()


# ## Approach
# 
# The general approach to this notebook is based on this awesome [post.](https://medium.com/@curiousily/credit-card-fraud-detection-using-autoencoders-in-keras-tensorflow-for-hackers-part-vii-20e0c85301bd) However, this implementation utilizes Pytorch as well as several modifications the architecture and input data.
# This notebook investigates using an autoencoder for anomaly and fraud detection. Autoencoders aren't new to machine learning, but they are not as prolific as some of the newer techniques in deep learning. An autoencoder's goal is to compress an input and then reconstruct the original input from the compressed data.
# ![deep_autoencoder.png](https://deeplearning4j.org/img/deep_autoencoder.png)
# 
# It is an unsupervised algorithm since the target variable is the input. There are many use cases and interesting applications of autoencoders which can be seen [here](https://towardsdatascience.com/autoencoders-bits-and-bytes-of-deep-learning-eaba376f23ad).
# 
# 
# In the case of detecting fraud, the approach is to build a model that will learn the characteristics of a normal transaction by training the model on non-fraudulent data points only. The idea is to get the model as accurate as possible at recreating normal transaction from the filtered data set. Once you have a model that can recreate normal transactions with a given amount of error, you then use the model to predict a set of data that has both normal and fraudulent transaction. If everything goes well, you will have a model that given a normal transaction will predict the original inputs with a small margin of error and a much larger error for transactions that are fraudulent.

# In[45]:


df = pd.read_csv('../input/creditcard.csv')


# Since the *time* column comes in as seconds from the start of the 2 day period. I first normalized the time to a normal 24 hour clock as well as the time of every transaction and the amount to see if there is any pattern in the timing of fraudulent charges. For this notebook I will be using a mix of Bokeh and Seaborn for visualization.

# In[29]:


df['TimeNorm'] = df['Time'].apply(lambda x: np.ceil(float(x)/3600) % 24.)


# In[30]:


def format_plot(p, x_label, y_label):
    p.grid.grid_line_color = None
    p.background_fill_color = "whitesmoke"
    p.axis.minor_tick_line_color = None
    p.title.align = 'center'
    p.title.text_font_size = "18px"
    p.xaxis.axis_label = x_label
    p.yaxis.axis_label = y_label
    p.xaxis.axis_label_text_font_size = "14px"
    p.yaxis.axis_label_text_font_size = "14px"
    p.yaxis.axis_line_color = None
    p.yaxis.major_tick_line_color = None
    p.axis.major_label_text_font_size = "12px"
    return p


# In[31]:


# Get only fraud transactions
fraud_df = df[df['Class'] == 1]
f_source = ColumnDataSource(data = dict(x = fraud_df['TimeNorm'].values,
                                    y = fraud_df['Amount'].values))
# get only normal transactions
non_fraud_df = df[df['Class'] == 0]
# Limit amount of data in plot
sample_non_fraud = df.sample(frac=0.01, replace=False)
norm_source = ColumnDataSource(data = dict(x = sample_non_fraud['TimeNorm'].values,
                                    y = sample_non_fraud['Amount'].values))
# create Bokeh figure
p = figure(plot_width = 800, 
           toolbar_location = None, 
           title = 'Transactions by Time and Amount')

# plot Normal Transactions
p.circle(x=jitter('x', width=0.9, range=p.x_range), 
         y='y', color = Set3_12[4], 
         fill_alpha = 0.1, 
         source = norm_source)

# plot fraud transactions
p.circle(x=jitter('x', width=0.9,range=p.x_range), 
         y='y', color = Set3_12[3], 
         fill_alpha = 0.7, 
         source = f_source)

#function to format plot
p = format_plot(p, "Time", "Amount")

p.yaxis[0].formatter = NumeralTickFormatter(format="$0,0")

show(p)


# In[32]:


df.loc[df['Class'] == 1, 'Amount'].max()


# From the chart above, it appears that there is no clear time window when fraud occurs more frequently. You can also see that all fraudulent charges fall below $2,126. While there may not be a lot of information gained by a visual inspection of time, I decided to leave it in the dataset as there may be some interactions with other variables that help detect these charges.
# 
# ### Data Normalization
# 
# Next I normalized the time and amount variables to ease with model learning,and split the data into a training and testing dataset. 

# In[33]:


scl = StandardScaler()
df['TimeNorm'] = scl.fit_transform(df['TimeNorm'].values.reshape(-1,1))
scl = StandardScaler()
df['NormAmt'] = scl.fit_transform(df['Amount'].values.reshape(-1,1))


# In[34]:


df = df.drop(['Time', 'Amount'], axis = 1)


# In[35]:


x_train, x_test = train_test_split(df, test_size = 0.2, random_state = 42)


# Since we are training an autoencoder that will learn the features of a normal transaction, we filter the training set to remove all transactions that were fraudulent from the training set . We leave the test set as is, containing both types of transactions.

# In[36]:


x_train = x_train[x_train['Class'] == 0]
x_train = x_train.drop('Class', axis = 1)

y_test = x_test['Class'].values
x_test = x_test.drop('Class', axis = 1)
x_test = x_test.values
x_train = x_train.values

x_train.shape, x_test.shape


# Next I transformed the numpy arrays as a PyTorch Float Tensor and loaded them into a DataLoader object. The dataloader is a generator that feeds the data to the training loop at the specified batch size.

# In[37]:


xt = torch.FloatTensor(x_train)
xtr = torch.FloatTensor(x_test)
xdl = DataLoader(xt,batch_size = 1000)
tdl = DataLoader(xtr, batch_size = 1000)


# ### Autoencoder Model
# 
# The architecture of the autoencoder consist of 4 fully connected layers starting at an input size of 30 (30 variables) reducing down to 10 and then decoding back up to 30.

# In[38]:


class Net1(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(30,20)
        self.lin2 = nn.Linear(20,10)
        self.lin7 = nn.Linear(10,20)
        self.lin8 = nn.Linear(20,30)
        
        self.drop2 = nn.Dropout(0.05)
        
        self.lin1.weight.data.uniform_(-2,2)
        self.lin2.weight.data.uniform_(-2,2)
        self.lin7.weight.data.uniform_(-2,2)
        self.lin8.weight.data.uniform_(-2,2)

    def forward(self, data):
        x = F.tanh(self.lin1(data))
        x = self.drop2(F.tanh(self.lin2(x)))
        x = F.tanh(self.lin7(x))
        x = self.lin8(x)
        
        return (x)


# This function utilizes the model's loss function to calculate the loss for the validation test set. In this instance of using an autoencoder for anomaly detection, the validation score doesn't mean much since it will have both fraud and non-fraud scores. What we really care about is having the model learn as much as it can about the features of a normal transaction.

# In[39]:


def score(x):
    y_pred = model(V(x))
    x1 = V(x)
    return loss(y_pred,x1).data[0]


# Next is the training loop. This function takes in the number of epochs you want to run as well as a model_loss variable that is used to keep the history (model and validation loss) at each epoch. 
# This is a fairly standard PyTorch training loop, except that instead of taking in a X and a Y variable, it takes in X from the dataloader, gets the prediction from the model, and determines the loss between the original X and the model prediction. It then goes on to change the gradients through back propagation based on the determined loss.

# In[40]:


# Utilize a named tuple to keep track of scores at each epoch
model_hist = collections.namedtuple('Model','epoch loss val_loss')
model_loss = model_hist(epoch = [], loss = [], val_loss = [])


# In[41]:


def train(epochs, model, model_loss):
    try:c = model_loss.epoch[-1]
    except: c = 0
    for epoch in tqdm_notebook(range(epochs),position=0, total = epochs):
        losses=[]
        dl = iter(xdl)
        for t in range(len(dl)):
            # Forward pass: compute predicted y and loss by passing x to the model.
            xt = next(dl)
            y_pred = model(V(xt))
            
            l = loss(y_pred,V(xt))
            losses.append(l)
            optimizer.zero_grad()

            # Backward pass: compute gradient of the loss with respect to model parameters
            l.backward()

            # Calling the step function on an Optimizer makes an update to its parameters
            optimizer.step()
    
        val_dl = iter(tdl)
        val_scores = [score(next(val_dl)) for i in range(len(val_dl))]
        
        model_loss.epoch.append(c+epoch)
        model_loss.loss.append(l.data[0])
        model_loss.val_loss.append(np.mean(val_scores))
        print(f'Epoch: {epoch}   Loss: {l.data[0]:.4f}    Val_Loss: {np.mean(val_scores):.4f}')


# ### Model Training
# 
# With the Pytorch structure set up we can now train the model.

# In[42]:


model = Net1()
loss=nn.MSELoss()
learning_rate = 1e-2
optimizer=torch.optim.Adam(model.parameters(), lr=learning_rate)


# In[18]:


train(model = model, epochs = 30,model_loss= model_loss)


# ### Training Evaluation

# In[ ]:


# Define a plot source
source = ColumnDataSource(data=dict(
    x=model_loss.epoch,
    loss=model_loss.loss,
    val_loss = model_loss.val_loss
))

p = figure(plot_height = 500, 
           plot_width = 800,
           toolbar_location = None, 
           title = "Model L1 Loss")

ml = p.line(x = "x", y = "loss", 
           color=Set3_12[4], 
           line_width = 2, 
           legend = "Training Loss", 
           source=source)

p.line(x = "x", y = "val_loss", 
       color=Set3_12[5], 
       line_width = 2, 
       legend = "Validation Loss",
       source=source)


tips = [
    ("Epoch","@x"),
    ("Model Loss","@loss{0.000}"),
    ("Val Loss","@val_loss{0.000}")
]
p.add_tools(HoverTool(tooltips=tips, 
                      renderers= [ml], 
                      mode='vline'))


p = format_plot(p, 'Epoch', 'Loss')

show(p);


# From the chart above you can see that the the model loss is still going down. It is likely that there could have been additional performance gains from running a few more epochs to allow the loss to flatten out.
# 
# Now it's time to calculate the reconstruction error on the test set to see how well it predicts normal transactions vs fraud.
# 
# ### Predictions
# 
# Next I utilized the trained model to make predictions on the test set which. unlike the training set, has fraudulent transactions. I fed in the test dataloader to the model to get predictions,and then calculated the reconstruction error, which is the amount of error between the original input and the model predictions against the test set. The error for each row will be used to determine if each transaction is normal or fraud.

# In[ ]:


# Iterate through the dataloader and get predictions for each batch of the test set.
p = iter(tdl)
preds = np.vstack([model(V(next(p))).cpu().data.numpy() for i in range(len(p))])

# Create a pandas DF that shows the Autoencoder MSE vs True Labels
error = np.mean(np.power((x_test - preds),2), axis = 1)
error_df = pd.DataFrame(data = {'error':error,'true':y_test})

error_df.groupby('true')['error'].describe().reset_index()


# By stacking the error on the test set next to the actual target value (1,0) for ease of use I created a pandas DataFrame and grouped the data by the two types of transactions. From the summary table above we can see that the statistics for the two groups look quite different. However we have to evaluate further to see if they are in fact different enough for accurate final predictions. I started by looking at the ROC AUC.

# In[ ]:


fpr, tpr, thresholds = roc_curve(error_df.true, error_df.error)
roc_auc = auc(fpr, tpr)

source = ColumnDataSource(data=dict(
    fpr = fpr,
    tpr = tpr,
    x = np.linspace(0,1,len(fpr)),
    y = np.linspace(0,1,len(fpr))
))

p = figure(plot_height = 500, plot_width = 500,
           toolbar_location = None, 
           title = "Receiver Operating Characteristic")

j = p.line(x = "x", y = "y", 
           color=Set3_12[3], 
           line_width = 2, 
           line_dash = 'dashed', 
           source=source)

k = p.line(x = "fpr", y = "tpr", 
           color=Set3_12[4], 
           line_width = 2, 
           legend = f'AUC = {roc_auc:0.4f}',
           source=source)

tips= [
    ("False-Pos", "@fpr{00.0%}"),
    ("True-Pos", "@tpr{00.0%}"),
    ]
p.add_tools(HoverTool(tooltips=tips, renderers=[k], mode='vline'))

p = format_plot(p, 'False Positive Rate', 'True Positive Rate')
p.legend.location = 'bottom_right'

show(p);


# ### Model Perfomance
# 
# ROC AUC plots the False Positive Rate against the True Positive rate. The dashed red line represents a random guess of a 50/50 chance of predicting the correct outcome. Ideally the ROC line would fit perfectly in the upper left hand corner which would represent a perfect model. While the ROC AUC is generally a good thing to use to check model performance, it only looks at 2 of the possible 4 outcomes of binary classification. False Negatives and True negatives are not directly observed in this plot. To get the rest of the story it is necessary to look at the confusion matrix. 
# 
# When performing binary classification there are 4 types of outcomes:
# 
# ![confusion_matrix_1.png](https://rasbt.github.io/mlxtend/user_guide/evaluate/confusion_matrix_files/confusion_matrix_1.png)
# 
# In the case of this dataset:
# * ***True Positive:*** The number of normal transactions did we correctly predict as normal
# * ***False Negatives:*** The number of normal transactions that the model incorrectly classifies as fraud
# * ***False Positives:*** The number of fraudulent transactions the model incorrectly classifies as normal
# * ***True Negatives:*** The number of fraudlent transactions the model accurately classifies as fraud
# 
# Since our predictions are based on the error between the original and predicted values of the input, it is necessary to find a cut-off point in the error where anything above the threshold is considered fraud, and anything below is considered to be a normal transaction. To start I decided to set the threshold as the sum of the mean and standard deviation of the error of the normal transactions.

# In[ ]:


temp_df = error_df[error_df['true'] == 0]
threshold = temp_df['error'].mean() + temp_df['error'].std()
print(f'Threshold: {threshold:.3f}')


# In[ ]:


y_pred = [1 if e > threshold else 0 for e in error_df.error.values]
print(classification_report(error_df.true.values,y_pred))


# Above is the classification report, but I found these metrics to be a bit deceiving and hard to conceptualize. I found that in the case of this unbalanced dataset it is easier to understand by looking at the actual values for number of fraudulent transactions missed as well as the number of normal transactions that were misclassified in the confusion matrix.

# In[ ]:


conf_matrix = confusion_matrix(error_df.true, y_pred)

sns.set(font_scale = 1.2)
plt.figure(figsize=(10, 10))
sns.heatmap(conf_matrix, xticklabels=['Not Fraud','Fraud'], yticklabels=['Not Fraud','Fraud'], annot=True, fmt="d");
plt.title("Confusion matrix")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()


# At first look, the model seems to catch all but about 15% of the fraudulent transactions, but at that threshold the model incorrectly identifies around
# 4000 normal transactions as fraud (False Negatives).
# 
# This is where the benefit of having a threshold pays off. It allows the flexibility of determining the amount of normal transactions you are willing to accept in order to catch the targeted amount of Fraud. The False Negatives and False Positives exhibit inverse behavior; as the amount of missed fraudulent transactions decreases, the normal transactions that are incorrectly predicted as fraud increases.
# 
# Next I iterated through a range of thresholds and plotted the confusion matrix at each threshold to get a better idea of what the decision space looks like.

# In[ ]:


plt.figure(figsize=(12, 12))
m = []
for thresh in np.linspace(0.2,2,9):
    y_pred = [1 if e > thresh else 0 for e in error_df.error.values]
    conf_matrix = confusion_matrix(error_df.true, y_pred)
    m.append((conf_matrix,thresh))
    
count = 0
for i in range(3):
    for j in range(3):
        plt.subplot2grid((3, 3), (i, j))
        sns.heatmap(m[count][0], xticklabels=['Not Fraud','Fraud'], yticklabels=['Not Fraud','Fraud'], annot=True, fmt="d");
        plt.title(f"Threshold - {m[count][1]:.3f}")
        plt.ylabel('True class')
        plt.xlabel('Predicted class')
        plt.tight_layout()
        count += 1
plt.show()


# In[ ]:


thr = []
tp = []
fn = []
fp = []
tn = []
for thresh in np.linspace(0.2,3,200):
    y_pred = [1 if e > thresh else 0 for e in error_df.error.values]
    conf = confusion_matrix(error_df.true, y_pred)
    tp.append(conf[0][0])
    fp.append(conf[0][1])
    fn.append(conf[1][0])
    tn.append(conf[1][1])
    thr.append(thresh)

conf_df = pd.DataFrame(data = {'fp':fp,'fn':fn,'threshold':thr})
cdf = conf_df.drop_duplicates(subset='fn',keep='last')
print(cdf)


# In[ ]:


xx = [str(x) for x in cdf['fn'].values]
source = ColumnDataSource(data=dict(
    fn = xx,
    fp = cdf['fp'].values,
))

p = figure(plot_width = 800,
           toolbar_location = None, x_range = xx,
           title = "False Negative vs False Positive")

p.vbar(x = "fn", top = "fp", width = 0.9,
           color=Set3_12[3],  
           source=source)

p = format_plot(p, 'False Negatives', 'False Positive')

show(p);


# In the plot above you can see the trade-space for a given reconstruction error threshold. There is a large dropoff after 2 False Negatives. Ideally this plot would be in percentages to show overall model performance, since these numbers would change depending on your test set size. However actual number of fraudulent transactions is a bit more intuitive.
# 
# ### Conclusion and Further Work
# 
# Since the data we were given represents the principal components, it is difficult to fully understand the value of each of the model inputs. In this notebook I did not attempt to do any feature engineering or general dimensional reduction (although the autoencoder does) to try to improve the model.
# 
# In general the model performs well, but there is still a small number of fraudulent transactions that can't be cleanly separated from normal transactions. The model provides a decision space to do tradeoff analysis and determine the best threshold given how much a fraudulent transaction costs the business vs how much it costs to incorrectly label normal transactions (which is difficult to quantify).
# 
# Next, I would like to look into ensembling this with other models as well as taking the weights of the pre-trained autoencoder and using additonal layers to perform the final binary classification.
# 
# Thanks for reading.
