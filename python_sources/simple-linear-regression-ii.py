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


train = pd.read_csv(r'/kaggle/input/random-linear-regression/train.csv', encoding='iso-8859-1')
test = pd.read_csv(r'/kaggle/input/random-linear-regression/test.csv', encoding='iso-8859-1')


# In[ ]:


train.head(2)


# In[ ]:


test.head(2)


# In[ ]:


from bokeh.plotting import *
from bokeh.models import *
from bokeh.layouts import *
from bokeh.io import *
from bokeh.embed import *
from bokeh.resources import *
import pandas as pd


def bokeh_regg_plot(x,y):
    output_notebook()
    regg_plot = figure(plot_width=800,plot_height=400,title="Data Visualization Plot")
    regg_plot.circle(x, y)   
    html_name = 'regression.html'
    output_file(html_name,mode='inline')
    show(regg_plot)
    save(regg_plot)


# In[ ]:


bokeh_regg_plot(train['x'],train['y'])


# In[ ]:


from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score


# Create linear regression object
regg = linear_model.LinearRegression()

m = train['x'].shape[0]
#nan_To_num will handle nan and large numbers
#hsstack will add another column to train['x'] , np.newaxis increases the dimention of Y_train from (700,) to (700,1)
X_train = np.nan_to_num(np.hstack((np.ones((m,1)), train['x'][:,np.newaxis])))
Y_train = np.nan_to_num(train['y'].to_numpy()[:,np.newaxis])

#Fit the model to X and Y training set
regg.fit(X_train,Y_train)


# In[ ]:


# Make predictions using the testing set
n = test['x'].shape[0]
X_test = np.nan_to_num(np.hstack((np.ones((n,1)), test['x'][:,np.newaxis])))
Y_test = np.nan_to_num(test['y'].to_numpy()[:,np.newaxis])


Y_predict = regg.predict(X_test)


# In[ ]:


# The coefficients
print('Coefficients: \n', regg.coef_)

# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(Y_test, Y_predict))

# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(Y_test, Y_predict))


# *We can see very low variance (underfit model) because of np.nan_to_num() , which assigns 0 to nan and inf to very large values,
# Thus the line shifts or deviates largly because of this.
# Better option would be to remove the nans and large values from data*

# In[ ]:


from bokeh.plotting import *
from bokeh.models import *
from bokeh.layouts import *
from bokeh.io import *
from bokeh.embed import *
from bokeh.resources import *
import pandas as pd


def bokeh_regg_model_plot(x,y, x_test,y_pred):
    output_notebook()
#     df = pd.DataFrame(list(zip(x,y,x_test,y_pred)),columns =['X','Y','X_Test','Y_Pred']) 
#     source = ColumnDataSource(data=df)
    regg_plot = figure(plot_width=800,plot_height=400,title="Regression Model Plot")
    regg_plot.circle(x, y)
    
    regg_plot.line(x=x_test, y=y_pred, line_width=4, line_color='red')


    html_name = 'regression_model.html'
    output_file(html_name,mode='inline')
    show(regg_plot)
    save(regg_plot)


# In[ ]:


bokeh_regg_model_plot(train['x'],train['y'], test['x'],Y_predict[:,0])


# **Since there are some values which are very large , the Liniear regression gets heavily deviated. 
# Below is the graph where nan and other large values are filled with 0 and inf**

# In[ ]:


bokeh_regg_plot(np.nan_to_num(train['x'].to_numpy()),np.nan_to_num(train['y'].to_numpy()))


# **To overcome this underfitting because of extremities , let us see if we can use Logistic regression. 
# NOTE: Logistic regression is actually a Classification , it draws a desicion boundary
# Below is the code for Logistic regression**

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


logmodel = LogisticRegression()

#Fit the data 
#Since Logistic Regression is a Classifier, it accepts only Categorical Data (i.e int and not float)
logmodel.fit(X_train,Y_train.astype(int))

Y_predict = logmodel.predict(X_test)


# In[ ]:


#Since Logistic Regression is a Classifier, it accepts only Categorical Data (i.e int and not float)
#These are the metrics for the Classifier
print(classification_report(Y_test.astype(int),Y_predict.astype(int)))


# In[ ]:


bokeh_regg_model_plot(train['x'],train['y'], test['x'],Y_predict)


# **The Logistic Regression has actually done a One vs All Classification for 300 classes!!!.
# So dont be surprised that the Logistic regression did not work**

# **Let us now try to use Nueral Network to solve this problem**
# 
# Resource - https://towardsdatascience.com/deep-neural-networks-for-regression-problems-81321897ca33

# In[ ]:


from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten

NN_model = Sequential()

# The Input Layer :
NN_model.add(Dense(1, kernel_initializer='normal',input_dim = X_train.shape[1], activation='relu'))

# The Hidden Layers :
NN_model.add(Dense(5, kernel_initializer='normal',activation='relu'))

# The Output Layer :
NN_model.add(Dense(1, kernel_initializer='normal',activation='linear'))

# Compile the network :
NN_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
NN_model.summary()


# In[ ]:


#Define a checkpoint callback :

checkpoint_name = 'Weights-{epoch:03d}--{val_loss:.5f}.hdf5' 
checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose = 1, save_best_only = True, mode ='auto')
callbacks_list = [checkpoint]


# In[ ]:


#Train the model :
NN_model.fit(X_train, Y_train, epochs=500, batch_size=32, validation_split = 0.2, callbacks=callbacks_list)


# In[ ]:


# Load wights file of the best model :
# wights_file = 'Weights-478--18738.19831.hdf5' # choose the best checkpoint 
# NN_model.load_weights(wights_file) # load it
NN_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])


# In[ ]:


#Predict Y with NN_model
Y_predict = NN_model.predict(X_test)


# In[ ]:


bokeh_regg_model_plot(train['x'],train['y'], test['x'],Y_predict[:,0])


# **As you can see, the Nueral Model does a much better job at regression**

# In[ ]:


bokeh_regg_model_plot(np.nan_to_num(train['x'].to_numpy()),np.nan_to_num(train['y'].to_numpy()), np.nan_to_num(test['x'].to_numpy()),Y_predict[:,0])


# **Above is the graph of the data with extremities**
# 
# 
# *Thank you!!! , if you found this useful please UPVOTE*

# In[ ]:




