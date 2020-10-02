#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))


# # Intro

# In[3]:


#data reading
train_df = pd.read_csv('../input/fashion-mnist_train.csv')
test_df = pd.read_csv('../input/fashion-mnist_train.csv')

#Nrows, Ncols @ train
train_df.shape


# In[4]:


train_df.shape


# These are the labels in the dataset:
# * 0 T-shirt/top
# * 1 Trouser
# * 2 Pullover
# * 3 Dress
# * 4 Coat
# * 5 Sandal
# * 6 Shirt
# * 7 Sneaker
# * 8 Bag
# * 9 Ankle boot 

# In[5]:


label_dict = {0: 'tshirt',
              1: 'trouser',
              2: 'pullover',
              3: 'dress',
              4: 'coat',
              5: 'sandal',
              6: 'shirt',
              7: 'sneaker',
              8: 'bag',
              9: 'boot'}


# In[6]:


#header
train_df.head()


# We can see the label column, and then the 784 pixel colums (we are working with 28x28 images)

# # Data Visualization

# In[7]:


#plot an image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# Let's now look at the first image in the train dataset.

# In[8]:


def get_pixel_cols():
    """
    This function returns the pixel column names
    """
    return ['pixel' + str(i) for i in range(1, 785)]


# In[9]:


def idx_to_pixseries(df, idx):
    """
    Given a pandas dataframe, and an index, it returns the pixel series for that index
    """
    return df.iloc[idx][get_pixel_cols()]


# In[10]:


def plot_image_pd(pixels_series):
    """
    This functions plots an image, given a series with all the pixels
    """
    pix_mat = pixels_series.values.reshape(28, 28)
    imgplot = plt.imshow(pix_mat, cmap='gray')


# In[11]:


plot_image_pd(idx_to_pixseries(train_df, 3))


# It seems to be a t-shirt...it has some texture, but is very hard to see the details, as we are working with quite low-res images. Let's now look at one example of each label in the dataset.

# In[12]:


labels = train_df.label.value_counts().index.values.tolist()
labels = sorted(labels)


# In[13]:


plt.figure(figsize=(10,10))
plt.plot([4, 3, 11])
for lab in labels:
    ax = plt.subplot(4, 3, lab+1)
    ax.set_title(str(lab) + " - " + label_dict[lab])
    plt.axis('off')
    plot_image_pd(idx_to_pixseries(train_df, train_df[train_df.label == lab].index[0]))


# Looking at first sight, it looks like non trivial task, as for example, shirt, coat, and pullover can be very similar at this resolution. Sandal, sneaker, and boot also seem very similar.

# Lets now look at some more examples of each class, to see if there are big intra-class variances

# In[14]:


#N images per row
N_im_lab = 6
N_labs = len(labels)
plt.figure(figsize=(11,11))
plt.plot([N_labs, N_im_lab, (N_im_lab * N_labs) + 1])

#for every label
for lab in labels:
    #show N_im_lab first samples
    for i in range(N_im_lab):
        ax = plt.subplot(N_labs, N_im_lab, 1 + (i + (lab*N_im_lab)))
        plt.axis('off')
        plot_image_pd(idx_to_pixseries(train_df, train_df[train_df.label == lab].index[i]))


# How does each average label image look like?

# In[15]:


plt.figure(figsize=(10,10))
plt.plot([4, 3, 11])
for lab in labels:
    ax = plt.subplot(4, 3, lab+1)
    ax.set_title("Avg. " + str(lab) + " - " + label_dict[lab])
    plt.axis('off')
    avg_pixels = train_df.loc[train_df.label == lab][get_pixel_cols()].mean()
    plot_image_pd(avg_pixels)


# # Modeling

# ## Baseline model
# We will train a logistic regression as a baseline model

# In[16]:


#normalize data, so we get values between 0 and 1
train_df[get_pixel_cols()] = train_df[get_pixel_cols()] / 255.
test_df[get_pixel_cols()] = test_df[get_pixel_cols()] / 255.


# In[17]:


#split train data in train-val
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(train_df[get_pixel_cols()], train_df.label, test_size=0.25, random_state=4)


# In[ ]:


#train a logistic regression model
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(C = 0.1, solver = 'sag')
get_ipython().run_line_magic('time', 'lr.fit(X_train, y_train)')


# ### Baseline model evaluation

# In[ ]:


#class estimation
lr_y_val_pred = lr.predict(X_val)


# In[ ]:


#prints accuracy score
def print_acc(y_true, y_pred, set_str):
    print ("This model has a {0:.2f}% acc. score @ {1}".format(100*accuracy_score(y_true, y_pred), set_str))


# In[26]:


#compute confusion matrix
import seaborn as sn
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc

def plot_conf_matrix(y_true, y_pred, set_str):
    """
    This function plots a basic confusion matrix, and also shows the model
    accuracy score
    """
    conf_mat = confusion_matrix(y_true, y_pred)
    df_conf = pd.DataFrame(conf_mat, index = [str(l) + '-' + label_dict[l] for l in labels],
                           columns = [str(l) + '-' + label_dict[l] for l in labels])

    plt.figure(figsize = (12, 12))
    sn.heatmap(df_conf, annot=True, cmap="YlGnBu")
    
    print_acc(y_true, y_pred, set_str)

plot_conf_matrix(y_val, lr_y_val_pred, 'Validation')


# This baseline model has a 85.13% accuracy score. We can see in the confusion matrix, that the main errors come from the tshirt, coat, pullover and tshirt classes. Let's now look at some of this errors closer.

# In[ ]:


print_acc(y_train, lr.predict(X_train), 'Train')


# This baseline model has a 86.64 percent accuracy in the train set, slightly better than at validation set, but it does not seem to be any overfit

# ### visual error inspection 

# In[63]:


def visual_err_inspection(y_true, y_pred, lab_eval, N_samples=6):
    """
    This function runs a visual error inspection. It plots two rows of images,
    the first row shows true positive predictions, while the second one shows
    flase positive predictions
    """
    
    df_y = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred})
    idx_y_eval_tp = df_y.loc[(df_y.y_true == lab_eval) & (df_y.y_pred == lab_eval)].index.values[:N_samples]
    idx_y_eval_fp = df_y.loc[(df_y.y_true != lab_eval) & (df_y.y_pred == lab_eval)].index.values[:N_samples]
    
    plt.figure(figsize=(12,5))
    plt.plot([2, N_samples, 2*N_samples + 1])

    for i in range(N_samples):
        ax = plt.subplot(2, N_samples, i+1)
        ax.set_title("OK: " + str(lab_eval) + " - " + label_dict[lab_eval])
        plt.axis('off')
        plot_image_pd(idx_to_pixseries(train_df, idx_y_eval_tp[i]))

        ax2 = plt.subplot(2, N_samples, i+N_samples+1)
        lab_ = train_df.iloc[idx_y_eval_fp[i]].label
        ax2.set_title("KO: " + str(int(lab_)) + " - " + label_dict[lab_])
        plt.axis('off')
        plot_image_pd(idx_to_pixseries(train_df, idx_y_eval_fp[i]))


# In[ ]:


#run visual inspection for class 6 - shirts
visual_err_inspection(y_val, lr_y_val_pred, 6, 6)


# The previous output shows some true shirt predictions, in the first row, and in the second row we can see some mispredictions, and its true label. We can see that for example, the second and third mispredictions are a pullover, and a tshirt, but at first sight might seem like a shirt, at least in this image resolution. 

# In[ ]:


#run visual inspection for class 4 - coats
visual_err_inspection(y_val, lr_y_val_pred, 4, 6)


# One issue related to this dataset, is that we do not have a huge data set, as we are talking of around 40K images. Let's take a look to the 'learning curve' to assess if in the scenario of having more data the performance would improve.

# # Ensemble models

# In this section we are going to train ensemble models, to see if we can improve the baseline model performance using ensemble models, specifically RandomForest and XGboost.

# ## RF model

# In[ ]:


#train a RF model on this data


# In[ ]:


from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=500, min_samples_leaf=25, n_jobs=4)
#train the model
get_ipython().run_line_magic('time', 'rf.fit(X_train, y_train)')

plot_conf_matrix(y_val, rf.predict(X_val), 'Validation')


# In[ ]:


print_acc(y_train, rf.predict(X_train), 'Train')


# The RandomForest model achieves just 1 percentage point better accuracy at validation set compared to the baseline model. It might be improved as, it might have a little overfit, as the train set accuracy is around 4 points better.

# ## XGboost

# In[ ]:


#train a xgboost model on this data
from xgboost import  XGBClassifier
xgb = XGBClassifier(max_depth=5, n_estimators=300, learning_rate=0.03, n_jobs=4)
get_ipython().run_line_magic('time', 'xgb.fit(X_train, y_train)')


# In[ ]:


plot_conf_matrix(y_val, xgb.predict(X_val), 'Validation')


# In[ ]:


print_acc(y_train, xgb.predict(X_train), 'Train')


# ## Basic model ensemble

# Let's now combine the output of these three models, and see if there is any improvement. Now let's see how the prediction probabilities look like. 

# In[ ]:


lr_val_probs, rf_val_probs, xgb_val_probs = lr.predict_proba(X_val), rf.predict_proba(X_val), xgb.predict_proba(X_val)


# In[ ]:


def show_probs_dist(prob_pred, label_dictionary = label_dict):
    '''
    Given the probabilities prediction of a model, for a set of labels, show how all look together 
    '''
    for i in range(prob_pred.shape[1]):
        plt.hist(prob_pred[:,i], alpha=0.4, label = label_dictionary[i])
    plt.legend(loc='upper right')


# In[ ]:


show_probs_dist(lr_val_probs)


# In[ ]:


show_probs_dist(rf_val_probs)


# In[ ]:


show_probs_dist(xgb_val_probs)


# We cannot see a big difference in shape looking at all labels at the same. Let's see what happens if we focus in some specific labels

# In[ ]:


def show_probs_dist_label(prob_pred_list, preds_names_list, label_picked, label_dictionary = label_dict):
    '''
    Given a list of probablities prediction for different models, select one label an plot all models 
    probabilities prediction for that label
    '''
    for i in range(len(prob_pred_list)):
        plt.hist(prob_pred_list[i][:,label_picked], alpha=0.5, label = preds_names_list[i])
    plt.legend(loc='upper right')
    plt.title('Probability distribution for ' + label_dict[label_picked])
    plt.show()


# In[ ]:


list_probs = [lr_val_probs, rf_val_probs, xgb_val_probs]
list_names = ['LR', 'RF', 'XGB']
for i in label_dict.keys():
    show_probs_dist_label(list_probs, list_names, i)


# Looking label by label, we can see that all label probabilities shapes look pretty much the same, so no normalizations will be done to compare these probabilities

# In[ ]:


#average all predictions
comb_val_probs = (lr_val_probs + rf_val_probs + xgb_val_probs) / 3
#pick the column idx with the max probability
comb_val_pred = comb_val_probs.argmax(axis=1)
plot_conf_matrix(y_val, comb_val_pred, 'Validation')


# In this case, the average of all scores gives no better result. Let's try to set some weights, giving a bit more importance to the xgb prediction, as it was the model with the best performance

# In[ ]:


weighted_val_probs = (0.2*lr_val_probs + 0.3*rf_val_probs + 0.5*xgb_val_probs) / 3
weighted_val_pred = weighted_val_probs.argmax(axis=1)
plot_conf_matrix(y_val, weighted_val_pred, 'Validation')


# Well, in this case we get a 0.1% accuracy improvement, that performance boost can result in jumping some positions in a Kaggle ladder!

# ## Dimensionality reduction

# Even if this dataset contains low resolution images (resulting in a non too high dimensional problem), let's use PCA and see if there is any accuracy improvement in the resulting lower dimensional space

# In[ ]:


from sklearn import decomposition
N_PCA_COMP = 50
pca_est = decomposition.PCA(n_components=N_PCA_COMP, svd_solver='randomized', whiten=True, random_state=42)


# In[ ]:


pca_est.fit(X_val)


# Let's take a look at the explained variance of the 50 components we used to fit this PCA

# In[ ]:


pca_est.explained_variance_ratio_


# In[ ]:


plt.scatter([i for i in range(N_PCA_COMP)], pca_est.explained_variance_ratio_)
plt.plot([i for i in range(N_PCA_COMP)], pca_est.explained_variance_ratio_)


# We can see that the curve has an 'elbow' shape, meaning that the first components (first 10) explain most of the variance, while the rest explain much less variance

# In[ ]:


plt.scatter([i for i in range(N_PCA_COMP)], pca_est.explained_variance_ratio_.cumsum())
plt.plot([i for i in range(N_PCA_COMP)], pca_est.explained_variance_ratio_.cumsum())


# Here we can see the cumulative explained variance of the components. Let's now visualize the first 8 components of the transformation in the original image space:

# In[ ]:


plt.figure(figsize=(12,5))
plt.plot([2, 4])

for i in range(4):
    ax = plt.subplot(2, 4, i + 1)
    ax.set_title("Comp # " + str(i+1))
    plt.axis('off')
    plt.imshow(pca_est.components_[i, :].reshape(28, 28), cmap= 'gray')

    ax2 = plt.subplot(2, 4, i + 4 + 1)
    ax2.set_title("Comp # " + str(i + 4 + 1))
    plt.axis('off')
    plt.imshow(pca_est.components_[i + 4, :].reshape(28, 28), cmap= 'gray')


# Looking at the components, we can see that for example, the first component shows a mix of sneaker, and shirt/pullover. The sneaker takes much lower values, while the shirt/pullover take positive values.

# In[ ]:


#transorm the original images in the PCA space
X_val_pca = pca_est.transform(X_val)


# In[ ]:


#work with just a sample
N_samps = 1000
y_val_ = y_val.reset_index(drop=True)
sample = X_val_pca[:N_samps, :3]


# In[ ]:


plt.figure(figsize=(12,5))
plt.subplot(111)
for lab in np.unique([a for a in label_dict.keys()]):
    ix = np.where(y_val_[:N_samps] == lab)
    plt.scatter(sample[:, 0][ix], sample[:, 1][ix], label = label_dict[lab])
plt.legend(bbox_to_anchor=(1.05, 1), loc=2)
plt.xlabel('1st PCA component')
plt.ylabel('2nd PCA component')
plt.show()


# I really like this visualization. We can see how on the top left side we have mainly sandals, sneakers, and boots. On the very bottom side we have trousers, that in these first two components are very similar to dresses. Bags are spread on the top side of the plot, some of them having collisions with some boots. On these two first components, t-shirts, pullovers, coats and shirts live in the same region.
# 
# Looking at the component visualization, the second component had a 'negative' trouser on it. We can see now in this scatter plot, that trousers are in a region where the 2nd PCA component takes negative values, so that the 'reconstruction' from these PCA components will result in a 'positive' trouser.
# 
# What about a 3D visualization of the first three components?

# In[ ]:


import plotly.offline as py
import plotly.graph_objs as go
get_ipython().run_line_magic('matplotlib', 'inline')
py.init_notebook_mode(connected=True)

get_ipython().run_line_magic('matplotlib', 'inline')
trace1 = go.Scatter3d(
    x= sample[:, 0],
    y= sample[:, 1],
    z= sample[:, 2],
    text= [label_dict[k] for k in y_val_[:N_samps].values],
    mode= 'markers',
    marker= dict(
        color=y_val_[:N_samps], 
        opacity=0.8
    )
)

data = [trace1]

layout = go.Layout(
    scene = dict(
    xaxis = dict(
        title='1st PCA component'),
    yaxis = dict(
        title='2nd PCA component'),
    zaxis = dict(
        title='3rd PCA component'),)
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)


# ## CNN model

# Let's now train a CNN model, using keras.

# In[18]:


import keras
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, ZeroPadding2D, Activation
from keras.layers.normalization import BatchNormalization

#CONV-> BatchNorm-> RELU block
def conv_bn_relu_block(X, n_channels, kernel_size=(3, 3)):
    X = Conv2D(n_channels, kernel_size)(X)
    X = BatchNormalization(axis = 3)(X)
    X = Activation('relu')(X)
    return X

#simple keras model
def fashion_cnn_model(input_shape):
    X_input = Input(input_shape)
    #zeropad
    X = ZeroPadding2D((1, 1))(X_input)
    #run a CONV -> BN -> RELU block
    X = conv_bn_relu_block(X, 32)
    #Maxpooling and dropout
    X = MaxPooling2D((2, 2))(X)
    X = Dropout(0.3)(X)
    #run another CONV -> BN -> RELU block
    X = ZeroPadding2D((1, 1))(X)
    X = conv_bn_relu_block(X, 64)
    #Maxpooling and dropout
    X = MaxPooling2D((2, 2))(X)
    X = Dropout(0.4)(X)
    #run another CONV -> BN -> RELU block
    #X = ZeroPadding2D((1, 1))(X)
    X = conv_bn_relu_block(X, 128)
    #dropout
    X = Dropout(0.3)(X)
    #flatten
    X = Flatten()(X)
    #dense layer
    X = Dense(len(label_dict.keys()), activation='softmax')(X)
    #output model
    model = Model(inputs = X_input, outputs = X, name='fashion_cnn_model')

    return model


# In[19]:


fashionModel = fashion_cnn_model((28, 28, 1,))


# In[20]:


#show the model architecture summary
fashionModel.summary()


# In[21]:


fashionModel.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])


# In[22]:


#reshape the input data
X_train_ = X_train.values.reshape(X_train.shape[0], 28, 28, 1)
X_val_ = X_val.values.reshape(X_val.shape[0], 28, 28, 1)


# In[23]:


from keras.utils import to_categorical
#fit the model
fashionModel.fit(x = X_train_ , y = to_categorical(y_train) ,epochs = 50, batch_size = 64)


# In[24]:


#evaluate the model performance in the validation set
evs = fashionModel.evaluate(x = X_val_, y = to_categorical(y_val))
#show the accuracy metric
print(evs[1])


# This model, without much adjustment reaches a 93.1% accuracy socore in the validation set, 5.3 points better than the XGboost model, and almost 8 points better than the baseline model!

# ### CNN model inspection

# In[27]:


#label prediction
cnn_y_val_pred = fashionModel.predict(X_val_).argmax(axis=-1)
#plot confusion matrix
plot_conf_matrix(y_val, cnn_y_val_pred, 'Validation')


# Looking at the confusion matrix, we can see that there much less errors in all labels. Most of the errors belong to the shirt/tshirt/pullover/coat classes.

# In[28]:


from sklearn.metrics import classification_report
print(classification_report(y_val, cnn_y_val_pred, target_names=[v for v in label_dict.values()]))


# Looking at the classification report, we can see that trouser, bag, and boot classes have almost a perfect performance. We can also confirm, that the tshirt, coat and shirt classes have the worst performance (but always above a 0.81 f1-score, which is not bad either!). Let's now run a visual inspection for all classes.

# In[64]:


#run visual inspection for all classes
for i in label_dict.keys():
    visual_err_inspection(y_val, cnn_y_val_pred, i, 6)


# Looking at the error inspection, we can see that some of the errors would be really hard to avoid, with the data we currently have. For example, looking at the errors in the boot label, we can see some sneakers and sandals that look quite much to a boot (well, at least to me!)

# ### Recommendation generation using an intermediate layer as descriptor

# In[32]:


#get the output of the last conv layer
#we will use it as description vector
intermediate_layer_model = Model(inputs = fashionModel.input,
                                 outputs = fashionModel.get_layer('conv2d_3').output)


# In[33]:


#generate the output for all observations in validation set
intermediate_output = intermediate_layer_model.predict(X_val_)
#the cosine distance is applied to 1D vectors
intermediate_output_res = intermediate_output.reshape(15000, 128*5*5)


# In[107]:


import scipy

def get_N_rec(idx_eval, N_recoms):
    eval_row = X_val_[idx_eval, :, :, :].reshape((1, 28, 28, 1))
    eval_activ =  intermediate_layer_model.predict(eval_row)
    #apply the cosine distance to all rows
    distance = np.apply_along_axis(scipy.spatial.distance.cosine, 1, intermediate_output_res, eval_activ.reshape(128*5*5))
    #get the N_recoms with the lowest distance
    #drop the first, as it is the row to be evaluated
    idx_recoms = distance.argsort()[1:N_recoms+1]
    #pass this idx to the idx space of the original datset
    out = [X_val.index[i] for i in idx_recoms]
    #also convert the original idx
    original = X_val.index[idx_eval]
    return out, original

#give me 6 recommendations for idx 50
idx_rec, orig = get_N_rec(50, 6)


# In[113]:


import math
def plot_recommendations(idx_eval, N_recoms):
    idx_rec, orig = get_N_rec(idx_eval, N_recoms)
    fig = plt.figure(figsize=(10,10))
    N_cols_rec = math.ceil(1 + len(idx_rec) / 2)
    ax1 = fig.add_subplot(2, N_cols_rec,1)
    ax1.set_title('Original item')
    plot_image_pd(idx_to_pixseries(train_df, orig))
    for i in range(len(idx_rec)):
        ax_ = fig.add_subplot(2, N_cols_rec, i+2)
        ax_.set_title('Recomendation #' + str(i+1))
        plot_image_pd(idx_to_pixseries(train_df, idx_rec[i]))

#draw 6 recommendations for idx 50
plot_recommendations(50, 6)


# In[115]:


#draw 6 recommendations for idx 4
plot_recommendations(4, 6)


# In[117]:


#draw 6 recommendations for idx 4242
plot_recommendations(4242, 6)


# In[122]:


#draw 6 recommendations for idx 101
plot_recommendations(101, 6)


# Woah, I must admit theese recommendations look quite good! (It has been fun playing with some idx to see what's the recommendation output!)

# In[ ]:




