#!/usr/bin/env python
# coding: utf-8

# **Introduction**

# 
# Shopping Malls assign to each customer a spending score between 0 and 100. We can imagine that the more the customer spends money in the Mall, the more its score increases. The objective of this paper is to predict a spending score of a consumer based on demographic information like age, gender and personal annual income.

# **Data visualisation**

# let's first import the needed librairies and the data set.

# In[101]:


import math
from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset
tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format


# In[102]:


dataframe = pd.read_csv("../input/Mall_Customers_Data.csv", sep=",")
dataframe = dataframe.reindex(np.random.permutation(dataframe.index))
dataframe.head()


# The available dataset contains only 200 records. However, the quality of data is good since the dataset doesn't contain NULL values and the values seems to be correct.
# For example, the gender can be either Male or Female, the age is between 18 and 70 and the annual income is between 15.000 dollars and 137.000 dollars.

# In[103]:


dataframe.describe()


# In[104]:


#Get general info about data
print ("data shape :",dataframe.shape,"\n")
print ("data info  :",dataframe.info())
print ("\ncolumns  :",dataframe.columns)
print ("\nmissing values :",dataframe.isnull().sum())


# The spending score and the age are highly correlated (33%) which is expected since the shopping behavior can be different from one generation to another. The annual income in the other hand is only 1% correlated to to the spending score. this may be counter intuitive since we can think that the more we gain, the more we spend.
# The correlation matrix bellow does not show the differences between male and female since it is a categorical feature.

# In[105]:


import seaborn as sns
import warnings
import itertools
warnings.filterwarnings("ignore")
get_ipython().run_line_magic('matplotlib', 'inline')
correlation = dataframe[["Annual Income (k$)", "Age", 'Spending Score (1-100)']].corr()
plt.figure(figsize=(9,7))
sns.heatmap(correlation,annot=True,cmap="coolwarm",linewidth=2,edgecolor="k")
plt.title("CORRELATION BETWEEN VARIABLES")


# To visualize the the differences between male and female, we do the following plot. We can see that spending score of men is slightly lower than the one for women. Other than this there is no other obvious differences to be noticed.

# In[106]:


import seaborn as sns
sns.set(style="ticks")
sns.pairplot(dataframe, vars=["Annual Income (k$)", "Age", 'Spending Score (1-100)'],hue="Gender", markers=["o", "s"])


# Now let's do a heat map with age, annual income and spending score. We can see that generally that people who have the higher spending scores are the following :
# - very young pepole with less than 25 years old and with relatively low annual income (less than 40k dolars)
# - people betwenn 27 and 40 years old and with relatively high annual income (more than 70k dolars)

# In[107]:


plt.scatter(dataframe['Age'],dataframe['Annual Income (k$)'], cmap="coolwarm", c=dataframe['Spending Score (1-100)'])
plt.ylabel("Annual Income (k$)")
plt.xlabel("Age")
plt.title("Spending Score (1-100)")
plt.show()

dataframe_female = dataframe[dataframe["Gender"]=="Female"]
plt.scatter(dataframe_female['Age'],dataframe_female['Annual Income (k$)'], cmap="coolwarm", c=dataframe_female['Spending Score (1-100)'])
plt.ylabel("Annual Income (k$)")
plt.xlabel("Age")
plt.title("Female Spending Score (1-100)")
plt.show()

dataframe_male = dataframe[dataframe["Gender"]=="Male"]
plt.scatter(dataframe_male['Age'],dataframe_male['Annual Income (k$)'], cmap="coolwarm", c=dataframe_male['Spending Score (1-100)'])
plt.ylabel("Annual Income (k$)")
plt.xlabel("Age")
plt.title("Male Spending Score (1-100)")
plt.show()


# **Create a linear prediction model**

# let's build a linear regression model to predict the spending score based on the age, the gender and the annual income and the gender

# We preprocess the features and we normalize them so our model converge more easly. We also split our dataset into a trainning subset (70%) and a validation subset (30%).

# In[108]:


def preprocess_features(dataframe):
  selected_features = dataframe[
    ["Gender",          
     "Age",           
     "Annual Income (k$)"      
    ]]
  processed_features = selected_features
  return processed_features

def preprocess_targets(dataframe):
  output_targets = pd.DataFrame()
  output_targets["Spending Score (1-100)"] = dataframe["Spending Score (1-100)"]/100
  return output_targets

def linear_scale(series):
  min_val = series.min()
  max_val = series.max()
  scale = (max_val - min_val) / 2.0
  return series.apply(lambda x:((x - min_val) / scale) - 1.0)

def normalize(dataframe):
  processed_features = pd.DataFrame()
  processed_features["Gender"] = dataframe["Gender"]
  processed_features["Age"] = linear_scale(dataframe["Age"])
  processed_features["AnnualIncome"] = linear_scale(dataframe["Annual Income (k$)"])
  return processed_features

def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):    
    features = {key:np.array(value) for key,value in dict(features).items()}                                           
    ds = Dataset.from_tensor_slices((features,targets)) # warning: 2GB limit
    ds = ds.batch(batch_size).repeat(num_epochs)
    if shuffle:
      ds = ds.shuffle(10000)
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels


# In[109]:


total_records = dataframe.shape[0]
training_percentage = 0.7
validation_percentage = 1 - training_percentage

normalized_preprocessed_features = normalize(preprocess_features(dataframe))
preprocessed_targets = preprocess_targets(dataframe)
# Choose the first 80% examples for training.
training_examples = normalized_preprocessed_features.head(int(total_records*training_percentage))
training_targets = preprocessed_targets.head(int(total_records*training_percentage))
# Choose the last 20% examples for validation.
validation_examples = normalized_preprocessed_features.tail(int(total_records*validation_percentage))
validation_targets = preprocessed_targets.tail(int(total_records*validation_percentage))


# In[110]:


feature_columns = [tf.feature_column.numeric_column("Age"),
             tf.feature_column.numeric_column("AnnualIncome"),
             tf.feature_column.indicator_column(
                   tf.feature_column.categorical_column_with_vocabulary_list("Gender",["Male","Female"]))
                  ]


# We define then our model.

# In[111]:


def train_model(
    learning_rate,
    steps,
    batch_size,
    training_examples,
    training_targets,
    validation_examples,
    validation_targets):

  periods = 10
  steps_per_period = steps / periods

  # Create a linear regressor object.
  my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
  my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
  linear_regressor = tf.estimator.LinearRegressor(
      feature_columns=feature_columns,
      optimizer=my_optimizer
  )
    
  # Create input functions
  training_input_fn = lambda: my_input_fn(training_examples, 
                                          training_targets, 
                                          batch_size=batch_size)
  predict_training_input_fn = lambda: my_input_fn(training_examples, 
                                                  training_targets, 
                                                  num_epochs=1, 
                                                  shuffle=False)
  predict_validation_input_fn = lambda: my_input_fn(validation_examples, 
                                                    validation_targets, 
                                                    num_epochs=1, 
                                                    shuffle=False)

  # Train the model, but do so inside a loop so that we can periodically assess
  # loss metrics.
  print("Training model...")
  print("RMSE (on training data):")
  training_rmse = []
  validation_rmse = []
  for period in range (0, periods):
    # Train the model, starting from the prior state.
    linear_regressor.train(
        input_fn=training_input_fn,
        steps=steps_per_period,
    )
    # Take a break and compute predictions.
    training_predictions = linear_regressor.predict(input_fn=predict_training_input_fn)
    training_predictions = np.array([item['predictions'][0] for item in training_predictions])
    
    validation_predictions = linear_regressor.predict(input_fn=predict_validation_input_fn)
    validation_predictions = np.array([item['predictions'][0] for item in validation_predictions])
    
    # Compute training and validation loss.
    training_root_mean_squared_error = math.sqrt(
        metrics.mean_squared_error(training_predictions, training_targets))
    validation_root_mean_squared_error = math.sqrt(
        metrics.mean_squared_error(validation_predictions, validation_targets))
    # Occasionally print the current loss.
    print("  period %02d : %0.2f" % (period, training_root_mean_squared_error))
    # Add the loss metrics from this period to our list.
    training_rmse.append(training_root_mean_squared_error)
    validation_rmse.append(validation_root_mean_squared_error)
  print("Model training finished.")
  
  # Output a graph of loss metrics over periods.
  plt.ylabel("RMSE")
  plt.xlabel("Periods")
  plt.title("Root Mean Squared Error vs. Periods")
  plt.tight_layout()
  plt.plot(training_rmse, label="training")
  plt.plot(validation_rmse, label="validation")
  plt.legend()

  return linear_regressor


# Now let's train it:

# In[112]:


linear_regressor = train_model(
    learning_rate=0.001,
    steps=100,
    batch_size=100,
    training_examples=training_examples,
    training_targets=training_targets,
    validation_examples=validation_examples,
    validation_targets=validation_targets)


# The loss function seems to converge to a value of 26% wich a little bit high. The limits of this model are related to the few data we possess (only 200 records). If we had more data, we can imagine training the model on a deep neural network or do more feature preprocessing.

# We can see that all the values of spending score predicted are between 30 and 60 in a scale of 0 to 100.

# In[113]:


predict_training_input_fn = lambda: my_input_fn(training_examples, 
                                                  training_targets, 
                                                  num_epochs=1, 
                                                  shuffle=False)
training_predictions = linear_regressor.predict(input_fn=predict_training_input_fn)
training_predictions = np.array([item['predictions'][0] for item in training_predictions]) 
training_predictions = pd.DataFrame(training_predictions)

x2 = np.linspace(0.0, 1.0)
plt.scatter(training_targets,training_predictions)
plt.plot(x2, x2)
plt.show()


# ****Linear model with advanced feature preprocessing****

# Let's bucketize our features into small intervalls. This wll lead us to a wide network. this implies that the model will more "learn" the data that "generalize" it. This means that we may face some overfitting.

# In[114]:


Age = tf.feature_column.numeric_column("Age")
AnnualIncome = tf.feature_column.numeric_column("AnnualIncome")
Gender = tf.feature_column.categorical_column_with_vocabulary_list("Gender",["Male","Female"])
Age_buckets = tf.feature_column.bucketized_column(Age, np.arange(-1.0,1.0,0.05).tolist())
AnnualIncome_buckets = tf.feature_column.bucketized_column(AnnualIncome, np.arange(-1.0,1.0,0.05).tolist())
feature_columns = [Age_buckets, AnnualIncome_buckets, Gender]


# In[115]:


linear_regressor = train_model(
    learning_rate=0.01,
    steps=100,
    batch_size=100,
    training_examples=training_examples,
    training_targets=training_targets,
    validation_examples=validation_examples,
    validation_targets=validation_targets)


# As expected, the train loss is relatively low at 20% but the validation loss is in the other hand high, wich means that we are overfitting our data.

# In[116]:


predict_training_input_fn = lambda: my_input_fn(training_examples, 
                                                  training_targets, 
                                                  num_epochs=1, 
                                                  shuffle=False)
training_predictions = linear_regressor.predict(input_fn=predict_training_input_fn)
training_predictions = np.array([item['predictions'][0] for item in training_predictions]) 
training_predictions = pd.DataFrame(training_predictions)

x2 = np.linspace(0.0, 1.0)
plt.scatter(training_targets,training_predictions)
plt.plot(x2, x2)
plt.show()


# In[117]:


predict_validation_input_fn = lambda: my_input_fn(validation_examples, 
                                                  validation_targets, 
                                                  num_epochs=1, 
                                                  shuffle=False)

validation_predictions = linear_regressor.predict(input_fn=predict_validation_input_fn)
validation_predictions = np.array([item['predictions'][0] for item in validation_predictions])
validation_predictions = pd.DataFrame(validation_predictions)

x1 = np.linspace(0.0, 1.0)
plt.scatter(validation_targets,validation_predictions)
plt.plot(x1, x1)
plt.show()


# **Conclusion**

# The small dataset was a real limit in developping a large and more complex model.
# With a larer dataset, we can train un Deep Neural Network or even a Wide and Deep Neural Network
