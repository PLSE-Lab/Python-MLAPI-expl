#!/usr/bin/env python
# coding: utf-8

# <h2>Script-level imports</h2>

# In[112]:


import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# <h2>Reading the input</h2>

# In[199]:


df = pd.read_csv('../input/diabetes.csv')
print(df.info())


# <h2>Data exploration</h2>
# 
# Steps:
# <ul>
#     <li> Check class frequency.</li>
#     <li> Check feature dependency.</li>
# </ul>

# <h2>Analyzing class label frequencies</h2>

# In[200]:


# Plotting the counts for the 'Outcome' column (class labels)
sb.countplot(x='Outcome', data=df)


# In[201]:


# We replace the 0s in each column by the columnar mean.
for column in set(df.columns).difference({'Pregnancies', 'Outcome'}):
    df[column] = df[column].replace(0, df[column].mean())


# <h2>Heatmaps for feature correlation</h2>
# 
# Inter-feature correlation analysis is an extremely useful tool that allows us to analyze which features are 'bonded' with each other.

# In[202]:


# Displaying the heatmap.
sb.heatmap(df.corr())


# <h3>Analysis result</h3>
# 
# From the heatmap above, we can see that the following features have a high correlational strength:
# 1. **BMI** & **Skin Thickness**
# 2. ** Pregnancies** & **Age**

# In[203]:


print(df.head())


# In[204]:


# Converting the dataframe into a numpy matrix.
df_values = df.values

# Shuffling rows of the matrix.
np.random.shuffle(df_values)


# <h2>Splitting the data into Input and Output and computing class weights</h2>
# 
# Here, We split the numpy matrix into input and output.
# 
# We also compute class weights, i.e., the ratio of instances in each class. Performing this step is **extremely** crucial in imbalanced problems.

# In[205]:


# Splitting the first N-1 columns as X.
x = df_values[:,:-1]

# Splitting the last column as Y.
y = df_values[:, -1].reshape(x.shape[0], 1)

print(x.shape)
print(y.shape)

from sklearn.utils import class_weight

# Computing the class weights.
# Note: This returns an ndarray.
weights = class_weight.compute_class_weight('balanced', np.unique(y), y.ravel()).tolist()

# Converting the ndarray to a dict.
weights_dict = {
    i: weights[i] for i in range(len(weights))
}

print("Class weights: ", weights_dict)


# <h2>Keras-specific imports</h2>

# In[206]:


from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Activation, Input
import keras.regularizers


# <h2>Building the model</h2>
# 
# We use a simple logistic regression model with a few tweaks added in:
# * Batch Normalization: This causes covariate shift, resulting in accelerated learning.
# * L2 regularization: By imposing penalties on the weights, we ensure that the model doesn't overfit.

# In[207]:


# Instantiate the model.
model = Sequential()

# Add the input layer and the output layer.
# The '1' indicates the number of output units.
# The 'input_shape' is where we specify the dimensionality of our input instances.
# The 'kernel_regularizer' specifies the strength of the L2 regularization.
model.add(Dense(1, input_shape=(x.shape[1], ), kernel_regularizer=keras.regularizers.l2(0.017)))

# Adding the BatchNorm layer.
model.add(BatchNormalization())

# Adding the final activation, i.e., sigmoid.
model.add(Activation('sigmoid'))

# Printing the model summary.
print(model.summary())


# <h2>Normalizing the inputs</h2>
# 
# Any ML model prefers its inputs to be scaled, i.e, with 0 mean and unit standard deviation. This makes learning **much faster**.
# 
# To achieve this, we:
# 1. Obtain the mean of the input.
# 2. Obtain the std. deviation of the input.
# 3. Subtract the mean from the input and divide the result by the standard deviation

# In[208]:


# Mean, columnar axis.
x_mean = np.mean(x, axis=0, keepdims=True)

# Std. Deviation, columnar axis.
x_std = np.std(x, axis=0, keepdims=True)

# Normalizing.
x = (x - x_mean)/x_std

print(x[:5, :])


# In[209]:


from sklearn.model_selection import train_test_split

# Split the model into a 0.9-0.1 train-test split.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=5)

print("Shape of x_train: ", x_train.shape)
print("Shape of y_train: ", y_train.shape)
print("Shape of x_test: ", x_test.shape)
print("Shape of y_test: ", y_test.shape)


# <h2>Compiling the model</h2>
# 
# We now compile our model with our preferred metrics and optimizer before fitting the data to it (the model).
# 
# Since we're performing logistic regression, the obvious choice for the loss function would be *binary crossentropy*.
# For the optimizer, I've chosen AdaMax. You may use any one you wish.

# In[210]:


model.compile(loss='binary_crossentropy', optimizer='adam')
print('Model compiled!')


# <h2>Fitting data to the model</h2>
# 
# 
# We harness a technique called Early Stopping that stops training immediately as soon as it notices that the validation loss is increasing, which indicates overfitting.
# 
# 
# We fit the data to the model with the following parameters:
# * Input: x_train.
# * Output: y_train.
# * Batch size: 128.
# * Callbacks: The Early Stopper.
# * Class weights: The dictionary we had created before. This ensures that each label instance is weighted according to its frequency in the dataset.
# * Epochs: Any high number (The Early Stopping technique will abort training as soon as the model starts showing signs of overfitting).

# In[211]:


from keras.callbacks import EarlyStopping

# Initialize the Early Stopper.
stopper = EarlyStopping(monitor='val_loss', mode='min', patience=3)

# Fit the data to the model and get the per-batch metric history.
history = model.fit(x_train, y_train, validation_split=0.1, 
                    batch_size=128, epochs=700, 
                    callbacks=[stopper], class_weight=weights_dict, verbose=1)


# <h2>Plotting the losses</h2>
# 
# Plotting the losses is a great way to see how your model is performing. For every batch, we:
# 1. Compute the loss for the batch.
# 2. Back-propagate the error derivative.
# 3. Update each weight,
# 4. Get the next batch and repeat step 1.

# In[212]:


# Plot the training loss.
plt.plot(history.history['loss'], 'r-')

# Plot the validation loss.
plt.plot(history.history['val_loss'], 'b-')

# X-axis label.
plt.xlabel('Epochs')

# Y-axis label.
plt.ylabel('Cost')

# Graph legend.
plt.legend(["Training loss", "Validation loss"])

# Graph title.
plt.title('Loss Graph')

plt.show()


# <h2>Are we done?</h2>
# 
# 
# <h3> Absolutely NOT. </h3> <br />
# 
# 
# A common problem with imbalanced binary classification problems is that the classifier opts for a simple way out and starts predicting each input  as an instance belonging to the output class with a large volume. This way, not only does the classifier satisfy the optimization objective, but also gets a high *accuracy*. Therefore, it is **strongly** recommended to use an alternative scoring metric, for instance, the F1-score.

# In[213]:


# Initialize variables.
tp = 0
fp = 0
fn = 0
tn = 0

# Get the predictions for the test inputs.
# One critical thing to note here is that, unlike scikit-learn,
# Keras will return the non-rounded prediction confidence
# probabilities.Therefore, rounding-off is critical.
predictions = model.predict(x_test)

# The hyperparameter that controls the tradeoff between how
# 'precise' the model is v/s how 'safe' the model is.
pr_hyperparameter = 0.5

# Rounding-off the predictions.
predictions[predictions > pr_hyperparameter] = 1
predictions[predictions <= pr_hyperparameter] = 0

# Computing the precision and recall.
for i in range(predictions.shape[0]):
    if y_test[i][0] == 1 and predictions[i][0] == 1:
        tp += 1
    elif y_test[i][0] == 1 and predictions[i][0] == 0:
        fn += 1
    elif y_test[i][0] == 0 and predictions[i][0] == 1:
        fp += 1
    else:
        tn += 1

pr_positive = tp/(tp + fp + 1e-8)
re_postive = tp/(tp + fn + 1e-8)
pr_negative = tn/(tn + fn + 1e-8)
re_negative = tn/(tn + fp + 1e-8)

# Computing the F1 scores.
f1 = (2*pr_positive*re_postive)/(pr_positive + re_postive + 1e-8)
f1_neg = (2*pr_negative*re_negative)/(pr_negative + re_negative + 1e-8)

print("F1 score (y=1): {}".format(f1))
print("F1 score (y=0): {}".format(f1_neg))


# In[214]:


from sklearn import metrics

# Print the detailed classification report.
print(metrics.classification_report(y_true=y_test, y_pred=predictions))

# Compute the confusion matrix.
conf_matrix = metrics.confusion_matrix(y_true=y_test, y_pred=predictions)

# Print the confusion matrix.
print(conf_matrix)


# In[215]:


# Display the heatmap for the confusion matrix.
sb.heatmap(conf_matrix)


# In[ ]:




