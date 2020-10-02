#!/usr/bin/env python
# coding: utf-8

# ## Using Tensorflow for Classification: ##
# 
# Tensorflow is Deep Learning Framework from Google. It has become quite popular in recent years and help in putting Neural Network Architecture together
# 
# This kernel is about doing a classification task using Tensorflow. This example uses a DNNClassifier for performing Binary Classification. The data used is **Breast Cancer data from Wisconsin.** This has 32 Columns and 569 rows.
# 
# The classifier used is DNNClassifier.
#  

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pandas.api.types as ptypes
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# In[ ]:


df_wisconsin = pd.read_csv('../input/data.csv')
df_wisconsin.columns = df_wisconsin.columns.str.replace('\s+', '_')  # Replacing column names by _ whereever space id found
len(df_wisconsin.columns)


# **Determining numeric columns from the dataset.**
# Following is a strategy to find out which are numeric columns. This is done because, later on when building feature columns, this information will be required.

# In[ ]:


numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
df_wisconsin_numeric = df_wisconsin.select_dtypes(include=numerics) # exclude is another keyword.
# Looking at the columns of the dataset
len(df_wisconsin_numeric.columns)


# We observe that there is difference of only one column, which is the **diagnosis** column. This will also be the response or target column. Out of these 32 columns, *id* and *Unnamed:_32* will be ignored.
# 
# Numeric columns are for further consideration are:
# 
# 'radius_mean',
# 'texture_mean', 
# 'perimeter_mean',
# 'area_mean',
# 'smoothness_mean',
# 'compactness_mean', 
# 'concavity_mean',
# 'concave points_mean', 
# 'symmetry_mean',
# 'fractal_dimension_mean',
# 'radius_se',
#  'texture_se', 
# 'perimeter_se', 
#  'area_se',
#  'smoothness_se',
#  'compactness_se',
#   'concavity_se',
#   'concave points_se',
#   'symmetry_se',
#   'fractal_dimension_se', 
#   'radius_worst',
#   'texture_worst',
#   'perimeter_worst',
#   'area_worst', 
#    'smoothness_worst',
#   'compactness_worst', 
#    'concavity_worst', 
#    'concave points_worst',
#    'symmetry_worst', 
#    'fractal_dimension_worst'
#    
#    
#    Above mentioned numeric columns will be normalized.

# 

# In[ ]:


normalize_columns = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
       'smoothness_mean', 'compactness_mean', 'concavity_mean',
       'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean',
       'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
       'compactness_se', 'concavity_se', 'concave_points_se', 'symmetry_se',
       'fractal_dimension_se', 'radius_worst', 'texture_worst',
       'perimeter_worst', 'area_worst', 'smoothness_worst',
       'compactness_worst', 'concavity_worst', 'concave_points_worst',
       'symmetry_worst', 'fractal_dimension_worst']


# We observe that Diagnosis is the column which we need to predict on unseen data. 
# Data seems to be pretty well balanced and possible values are **Benign** and **Malignant**.

# In[ ]:


x_values = df_wisconsin.drop(['diagnosis','id','Unnamed:_32'],axis=1) # Getting Predictors
y_val = df_wisconsin['diagnosis'] # getting response


# In[ ]:


x_values.head() # Examining x_values


# In[ ]:


y_val.value_counts() # Checking if the classes are balanced. Seems pretty good.


# In[ ]:


# Converting Labels to integer form. 'B' and 'M; are represented as 0,1. Since Tensorflow does not accept categorical variables in text form, we are converting to integers.
def label_numeric(label):
    if (label == 'B'):
        return(0)
    else:
        return(1)


# In[ ]:


y_val_numeric =y_val.apply(label_numeric)
y_val_numeric.value_counts()


# A quick check above confirms that things are happening as per the expectation.

# ## Performing Training and Testing split ##
# Preparing data for model building and checking. 10 percent of data is kept aside for validation purposes. We will also be performing scaling of data using sklearn's *MinMaxScaler*.

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(x_values,y_val_numeric,test_size=0.1,random_state=1234)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = pd.DataFrame(data=scaler.transform(X_train),columns = X_train.columns,index=X_train.index)
#X_train.head()
# Have a look at the before and after scaling step to understand what transformation has been done to the data.


# In[ ]:


# Similarly, scaling is performed on test data as well.
X_test = pd.DataFrame(data=scaler.transform(X_test),columns = X_test.columns,index=X_test.index)
X_test.head()


# ### Building feature columns for Tensorflow framework ###
# Tensorflow expects feature columns to be built using its api calls. Following is a lengthy way of creating them. Later on, an efficient mechanism is also used. Following is retained as it makes the understanding easier.

# In[ ]:


# Now we have 30 features which will be built in tensorflow framewor. Tensorflow requires these features to be defined as feature columns.
fc_radius_mean = tf.feature_column.numeric_column('radius_mean')
fc_texture_mean = tf.feature_column.numeric_column('texture_mean')
fc_perimeter_mean = tf.feature_column.numeric_column('perimeter_mean')
fc_area_mean = tf.feature_column.numeric_column('area_mean')
fc_smoothness_mean = tf.feature_column.numeric_column('smoothness_mean')
fc_compactness_mean = tf.feature_column.numeric_column('compactness_mean')
fc_concavity_mean = tf.feature_column.numeric_column('concavity_mean')
fc_concave_points_mean = tf.feature_column.numeric_column('concave points_mean')
fc_symmetry_mean = tf.feature_column.numeric_column('symmetry_mean')
fc_fractal_dimension_mean = tf.feature_column.numeric_column('fractal_dimension_mean')
fc_radius_se = tf.feature_column.numeric_column('radius_se')
fc_texture_se = tf.feature_column.numeric_column('texture_se')
fc_perimeter_se = tf.feature_column.numeric_column('perimeter_se')
fc_area_se = tf.feature_column.numeric_column('area_se')
fc_smoothness_se = tf.feature_column.numeric_column('smoothness_se')
fc_compactness_se = tf.feature_column.numeric_column('compactness_se')
fc_concavity_se = tf.feature_column.numeric_column('concavity_se')
fc_concave_points_se = tf.feature_column.numeric_column('concave points_se')
fc_symmetry_se = tf.feature_column.numeric_column('symmetry_se')
fc_fractal_dimension_se = tf.feature_column.numeric_column('fractal_dimension_se')
fc_radius_worst = tf.feature_column.numeric_column('radius_worst')
fc_texture_worst = tf.feature_column.numeric_column('texture_worst')
fc_perimeter_worst = tf.feature_column.numeric_column('perimeter_worst')
fc_area_worst = tf.feature_column.numeric_column('area_worst')
fc_smoothness_worst = tf.feature_column.numeric_column('smoothness_worst')
fc_compactness_worst = tf.feature_column.numeric_column('compactness_worst')
fc_concavity_worst = tf.feature_column.numeric_column('concavity_worst')
fc_concave_points_worst = tf.feature_column.numeric_column('concave points_worst')
fc_symmetry_worst = tf.feature_column.numeric_column('symmetry_worst')
fc_fractal_dimension_worst = tf.feature_column.numeric_column('fractal_dimension_worst')
#feat_cols = [fc_radius_mean, ..., fc_concave_points_worst, fc_symmetry_worst]


# In[ ]:


# Efficient way of building feature columns. 
# Please notice that Categorical Columns and Numerical Columns are treated differently.
feat_cols = []
df = X_train
for col in df.columns:
  if ptypes.is_string_dtype(df[col]): #is_string_dtype is pandas function
    feat_cols.append(tf.feature_column.categorical_column_with_hash_bucket(col, 
        hash_bucket_size= len(df[col].unique())))

  elif ptypes.is_numeric_dtype(df[col]): #is_numeric_dtype is pandas function
    feat_cols.append(tf.feature_column.numeric_column(col))


# Having a look at the Feature Columns as below:

# In[ ]:


print(feat_cols)


# ### Choosing Classification model, input function and training steps. ###
# *  Classification model -  tf.estimator.DNNClassifier is chosen as the model for training the data.
# *  Input function - Since we have loaded data using Pandas input function, we will use pandas_input_fn. Tensorflow also provides numpy based input function.
# *  epocs and steps - Since these terms are used frequently, it is important to understand the difference between them.
#     Following explanation is taken from Stackoverflow:
#     
# >     In the neural network terminology:
# >         one epoch = one forward pass and one backward pass of all the training examples
# >         batch size = the number of training examples in one forward/backward pass. The higher the batch size, the more memory space you'll need.
# >         number of iterations = number of passes, each pass using [batch size] number of examples. To be clear, one pass = one forward pass + one backward pass (we do not count the forward pass and backward pass as two different passes).

# In[ ]:


classifier = tf.estimator.DNNClassifier(
        feature_columns=feat_cols,
        # Two hidden layers of 20 nodes each.
        hidden_units=[20, 20],
        # The model must choose between 2 classes.
        n_classes=2)


# In[ ]:


input_func = tf.estimator.inputs.pandas_input_fn(x=X_train,y=y_train ,batch_size=10,num_epochs=500, shuffle=True)


# ### Now it is time to train the Classifier ###

# In[ ]:


classifier.train(input_fn=input_func, steps=10000)


# Creating an input function for prediction, will be applied on test data. Please note that shuffle parameter is set to False here.

# In[ ]:


pred_fn = tf.estimator.inputs.pandas_input_fn(x=X_test,batch_size=len(X_test),shuffle=False) # Shuffle should be set as false as we are interested in comparing with actual results.


# In[ ]:


# Prediction is done here now.
predictions = list(classifier.predict(input_fn=pred_fn))
predictions[0]


# Since prediction API outputs a number of information, we are taking the required stuff here which is class_ids.

# In[ ]:


final_preds = []
for pred in predictions:
    #info = "{} {} {}".format(pred['class_ids'][0], pred['probabilities'][0] , pred['probabilities'][1])
    final_preds.append(pred['class_ids'][0])
    #final_preds.append(info)


# ### Evaluating model performance ###
# Once we have prediction data available with us, we can now compare with the 10% data we have kept aside. As we have kept aside the unseen data, we can be reasonably confident that our model has not overfit. This is important step in machine learning to not to use evaluation data for training a model.
# We have used classification report, accuracy and confusion matrix to evaluate the model performance.

# In[ ]:


print(classification_report(y_test,final_preds))


# In[ ]:


print(confusion_matrix(y_test,final_preds))


# In[ ]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test,final_preds, normalize=True, sample_weight=None)


# We see that out of 57 observations, only 1 is misclassified. This concludes this article of using Tensorflow for classification. ** Happy Model Building! #**
# 
# *Note: This has been a good experience using Tensorflow for NN. Hope you liked!*
