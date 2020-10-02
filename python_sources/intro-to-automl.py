#!/usr/bin/env python
# coding: utf-8

# **[Introduction to Machine Learning Home Page](https://www.kaggle.com/learn/intro-to-machine-learning)**
# 
# ---
# 

# # Introduction
# 
# When applying machine learning to real-world data, there are a lot of steps involved in the process -- starting with collecting the data and ending with generating predictions.  (*We work with the seven steps of machine learning, as defined by Yufeng Guo **[here](https://towardsdatascience.com/the-7-steps-of-machine-learning-2877d7e5548e)**.*)
# 
# ![](https://i.imgur.com/mqTCqBR.png)
# 
# It all begins with **Step 1: Gather the data**.  In industry, there are important considerations you need to take into account when building a dataset, such as **[target leakage](https://www.kaggle.com/alexisbcook/data-leakage)**. When participating in a Kaggle competition, this step is already completed for you.
# 
# In the **[Intro to Machine Learning](https://www.kaggle.com/learn/intro-to-machine-learning)** and the **[Intermediate Machine Learning](https://www.kaggle.com/learn/intermediate-machine-learning)** courses, you can learn how to:
# - **Step 2: Prepare the data** - Deal with [missing values](https://www.kaggle.com/alexisbcook/missing-values) and [categorical data](https://www.kaggle.com/alexisbcook/categorical-variables).  ([Feature engineering](https://www.kaggle.com/learn/feature-engineering) is covered in a separate course.)
# - **Step 4: Train the model** - Fit [decision trees](https://www.kaggle.com/dansbecker/your-first-machine-learning-model) and [random forests](https://www.kaggle.com/dansbecker/random-forests) to patterns in training data.
# - **Step 5: Evaluate the model** - Use a [validation set](https://www.kaggle.com/dansbecker/model-validation) to assess how well a trained model performs on unseen data.
# - **Step 6: Tune parameters** - Tune parameters to get better performance from [XGBoost](https://www.kaggle.com/alexisbcook/xgboost) models.
# - **Step 7: Get predictions** - Generate predictions with a trained model and [submit your results to a Kaggle competition](https://www.kaggle.com/kernels/fork/1259198).
# 
# That leaves **Step 3: Select a model**.  There are _a lot_ of different types of models.  Which one should you select for your problem?  When you're just getting started, the best option is just to try everything and build your own intuition - there aren't any universally accepted rules.  There are also many useful Kaggle notebooks (like **[this one](https://www.kaggle.com/vbmokin/data-science-for-tabular-data-advanced-techniques)**) where you can see how and when other Kagglers used different models.
# 
# Mastering the machine learning process involves a lot of time and practice.  While you're still learning, you can turn to **automated machine learning (AutoML) tools** to generate intelligent predictions.

# # Automated machine learning (AutoML)
# 
# In this notebook, you'll learn how to use [**Google Cloud AutoML Tables**](https://cloud.google.com/automl-tables/docs/beginners-guide) to automate the machine learning process.  While Kaggle has already taken care of the data collection, AutoML Tables will take care of all remaining steps.
# 
# ![](https://i.imgur.com/5SekA3O.png)

# AutoML Tables is a **paid service**.  In the exercise that follows this tutorial, we'll show you how to claim $300 of free credits that you can use to train your own models!
# 
# <div class="alert alert-block alert-info">
# <b>Note</b>: This lesson is <b>optional</b>. It is not required to complete the <b><a href="https://www.kaggle.com/learn/intro-to-machine-learning">Intro to Machine Learning</a></b> course.
# </div><br> 

# # Code
# 
# We'll work with data from the **[New York City Taxi Fare Prediction](https://www.kaggle.com/c/new-york-city-taxi-fare-prediction)** competition.  In this competition, we want you to predict the fare amount (inclusive of tolls) for a taxi ride in New York City, given the pickup and dropoff locations, number of passengers, and the pickup date and time.
# 
# To do this, we'll use a **[Python class](https://www.kaggle.com/alexisbcook/automl-tables-wrapper)** that calls on AutoML Tables.  To use this code, you need only define the following variables:
# - `PROJECT_ID` - The name of your [Google Cloud project](https://cloud.google.com/resource-manager/docs/creating-managing-projects).  All of the work that you'll do in Google Cloud is organized in "projects".  
# - `BUCKET_NAME` - The name of your [Google Cloud storage bucket](https://cloud.google.com/storage/docs/creating-buckets).  In order to work with AutoML, we'll need to create a storage bucket, where we'll upload the Kaggle dataset.
# - `DATASET_DISPLAY_NAME` - The name of your dataset.  
# - `TRAIN_FILEPATH` - The filepath for the training data (`train.csv` file) from the competition.
# - `TEST_FILEPATH` - The filepath for the test data (`test.csv` file) from the competition.
# - `TARGET_COLUMN` - The name of the column in your training data that contains the values you'd like to predict.
# - `ID_COLUMN` - The name of the column containing IDs.
# - `MODEL_DISPLAY_NAME` - The name of your model.
# - `TRAIN_BUDGET` - How long you want your model to train (use 1000 for 1 hour, 2000 for 2 hours, and so on).
# 
# All of these variables will make more sense when you run your own code in the following exercise!

# In[ ]:



# Save CSV file with first 2 million rows only
import pandas as pd
train_df = pd.read_csv("../input/new-york-city-taxi-fare-prediction/train.csv", nrows = 2_000_000)
train_df.to_csv("train_small.csv", index=False)


# In[ ]:


PROJECT_ID = 'kaggle-playground-170215'
BUCKET_NAME = 'automl-tutorial-alexis'

DATASET_DISPLAY_NAME = 'taxi_fare_dataset'
TRAIN_FILEPATH = "../working/train_small.csv" 
TEST_FILEPATH = "../input/new-york-city-taxi-fare-prediction/test.csv"

TARGET_COLUMN = 'fare_amount'
ID_COLUMN = 'key'

MODEL_DISPLAY_NAME = 'tutorial_model'
TRAIN_BUDGET = 4000

# Import the class defining the wrapper
from automl_tables_wrapper import AutoMLTablesWrapper

# Create an instance of the wrapper
amw = AutoMLTablesWrapper(project_id=PROJECT_ID,
                          bucket_name=BUCKET_NAME,
                          dataset_display_name=DATASET_DISPLAY_NAME,
                          train_filepath=TRAIN_FILEPATH,
                          test_filepath=TEST_FILEPATH,
                          target_column=TARGET_COLUMN,
                          id_column=ID_COLUMN,
                          model_display_name=MODEL_DISPLAY_NAME,
                          train_budget=TRAIN_BUDGET)


# Next, we train a model and use it to generate predictions on the test dataset.

# In[ ]:


# Create and train the model
amw.train_model()

# Get predictions
amw.get_predictions()


# After completing these steps, we have a file that we can submit to the competition!  In the code cell below, we load this submission file and view the first several rows.

# In[ ]:


submission_df = pd.read_csv("../working/submission.csv")
submission_df.head()


# 
# And how well does it perform?  Well, the competition provides a **[starter notebook](https://www.kaggle.com/dster/nyc-taxi-fare-starter-kernel-simple-linear-model)** with a simple linear model that predicts a fare amount based on the distance between the pickup and dropoff locations.  This approach outperforms that notebook, and it ranks better than roughly half of the total submissions to the competition.
# 
# # Keep going
# 
# Run your own code using AutoML Tables to **[make a submission to a Kaggle competition](https://www.kaggle.com/kernels/fork/10027938)**!

# ---
# **[Introduction to Machine Learning Home Page](https://www.kaggle.com/learn/intro-to-machine-learning)**
# 
# 
# 
# 
# 
# *Have questions or comments? Visit the [Learn Discussion forum](https://www.kaggle.com/learn-forum/161285) to chat with other Learners.*
