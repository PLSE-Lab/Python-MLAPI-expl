#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# This Notebook will produce a Deep Neural Network Regression model that can predict the salary of a person working in the field of Data Science by analyzing the Kaggle ML and Data Science Survey results.

# In[ ]:


from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import tensorflow as tf


# # Data Preparation
# 
# ## Load Multiple Choice Responses CSV file
# The multipleChoiceResponses.csv file will be loaded with the ```pandas.read_csv``` method and stored into a pandas DataFrame. Pandas is a great library for importing, analyzing, and manipulating data in python.
# 
# Make sure the csv file is in the same directory as this Notebook. If it's not, you will have to update the ```csv_filepath``` variable with the correct path.
# 
# ```dtype=object``` means that the data type is generic. It can be used to represent any type such as ```string```, ```int```, ```float```, ```datetime``` or ```boolean```. For columns with numeric values, we would typically use data types of ```int``` or ```float``` but there are some values in the numeric columns that are of type ```string``` with a value of NA. For example, if we tried to set the ```Age``` column with a data type of ```int8```, the loading of the csv file will fail because the rows with NA are not of type ```int8```.

# In[ ]:


responses_csv_filepath = "../input/kaggle-survey-2017/multipleChoiceResponses.csv"
all_responses = pd.read_csv(responses_csv_filepath, dtype=object, encoding="iso-8859-1")


# Calling the ```info()``` method on the DataFrame will display the total number of rows and columns.

# In[ ]:


all_responses.info()


# ## Feature Selection
# 
# There are 228 columns in the entire dataset but we only want to keep a subset. 
# 
# To start with, we will keep the ```CompensationAmount``` column. This is the **label**, sometimes referred to as the **target**. ```CompensationAmount``` is the value that our Regression model will ultimately try to predict.
# 
# Now, we need to choose which **features** to keep. Only the features that I think are relevant to predicting the ```CompensationAmount``` value will be kept.

# In[ ]:


column_names = ["CompensationAmount", "CompensationCurrency", "GenderSelect", "Country", 
                "Age", "EmploymentStatus","CodeWriter", "CurrentJobTitleSelect", 
                "FormalEducation", "MajorSelect", "Tenure", "ParentsEducation", 
                "EmployerIndustry", "EmployerSize"]

responses = all_responses[column_names]


# Let's call the ```info()``` method again and take a high level look of the columns that we are going to keep.

# In[ ]:


print(responses.info())


# ## Data Cleansing
# 
# As you may have noticed in the previous cell, we have a lot of null values in our data set. We need to delete all the rows that have missing or invalid values.
# 
# For example, there are 5224 rows that have a non-null ```CompensationAmount``` value. This means that there are 11492 rows that have a null value for the ```CompensationAmount``` field. These 11492 rows will be deleted. 

# In[ ]:


responses = responses.dropna(axis=0, how='any')


# In[ ]:


print(responses.info())


# ### Keep Countries with at least 100 samples
# 
# Some countries are not well represented in the dataset. I only want to keep the countries that have at least 100 occurences to ensure that the sample size for each country is large enough.

# In[ ]:


country_vc = responses["Country"].value_counts()

min_samples_threshold = 100
country_names = country_vc[country_vc >= min_samples_threshold].index
print("Countries with at least 100 samples: " + str(country_names.tolist()))


# The next line of code may seem complicated at first glance. 
# 
# ```responses[responses["Country"].isin(country_names)]``` will look at each row in the ```Country``` column and will only keep the rows that have a value in the ```country_names``` variable.

# In[ ]:


responses = responses[responses["Country"].isin(country_names)]

print("There are now " + str(responses.shape[0]) + " rows left.")


# ### Keep Full-Time Workers

# In[ ]:


responses = responses[responses["EmploymentStatus"] == "Employed full-time"]
responses = responses.drop("EmploymentStatus", axis=1)

print("There are now " + str(responses.shape[0]) + " rows left.")


# ### Remove outliers in Age
# 
# There are two age values in the dataset that are outliers. The first outlier is age 1. This is obviously not a truthful answer given by the person filling out the survey. The second outlier is age 100. This may be a truthful answer but it's the only age above 72 and will skew the mean. Both outliers will be removed.
# 
# Notice that the ```"1"``` and ```"100"``` are string values and not integers. This is because we specified all the column data types to be ```object```. Later on in the Notebook, we will convert this column to type ```int8```.

# In[ ]:


age_outliers = ["1", "100"]
responses = responses[responses["Age"].isin(age_outliers) == False]

print("There are now " + str(responses.shape[0]) + " rows left.")


# ### Drop CodeWriter
# 
# The ```CodeWriter``` column only has one type of value in the remaining dataset, "Yes". The entire column will be dropped because it will not be useful when training the model later on. 

# In[ ]:


print(responses["CodeWriter"].value_counts())

responses = responses.drop("CodeWriter", axis=1)


# ### Remove invalid values from CompensationAmount
# 
# After inspecting the various ```CompensationAmount``` values, I noticed there were several invalid values. They were "-99", "-1", "0", and "-". Rows with these values will be removed. There was a specific instance where there was a comma instead of a decimal. This intance will be fixed so the row can be saved. 

# In[ ]:


invalid_comp_amt = ["-99", "-1", "0", "-"]
responses = responses[responses["CompensationAmount"].isin(invalid_comp_amt) == False]

responses["CompensationAmount"] = responses["CompensationAmount"].replace("140000,00", "140000.00")

print("There are now " + str(responses.shape[0]) + " rows left.")


# ### Convert CompensationAmount to float
# 
# It's time to convert the data type for the ```CompensationAmount``` column to ```float64```. Before we can do the data type conversion, we need to remove all the commas. e.g. 100,000 -> 100000
# 
# If you look carefully, the ```replace``` method below is from ```str``` and not directly from ```responses["CompensationAmount"]``` as before. In the previous example, the entire string value in a particular column has to match before it can be replaced with the new string. In the example below, all occurrences within each string will be replaced with the new string.

# In[ ]:


responses["CompensationAmount"] = responses["CompensationAmount"].str.replace(",", "")
responses["CompensationAmount"] = responses["CompensationAmount"].astype(np.float64)


# ### Convert Age to int
# 
# Convert the data type for the ```Age``` column to ```int8```.

# In[ ]:


responses["Age"] = responses["Age"].astype(np.int8)


# ## Feature Transformation
# 
# The next step is to transform the features to numeric values and scale them. Currently, the vast majority of the features are not numeric, they are categorical.
# 
# ### Convert CompensationAmount to USD
# Although the ```CompensationAmount``` feature is already numeric, each value has a different currency. Comparing these values would be like comparing apples to oranges. For example, a person working in the United States may list 1.0 as the ```CompensationAmount``` and another person working in India may list 64.02 as the ```CompensationAmount```. At first glace, you might assume the person working in India has a much greater amount than the person working in the United States. After converting 64.02 (INR) to USD, the person working in India also makes 1.0, exactly the same as the person working in the United States.
# 
# First, let's load the conversionRates.csv file into a DataFrame.

# In[ ]:


conversionRates_csv_filepath = "../input/kaggle-survey-2017/conversionRates.csv"
column_names = ["originCountry", "exchangeRate"]
dtype = {"originCountry": object, "exchangeRate": np.float16}

conversionRates = pd.read_csv(conversionRates_csv_filepath, usecols=column_names, dtype=dtype)


# We need to merge/join the two columns from ```conversionRates``` into ```responses```. They will be joined by the ```CompensationCurrency``` column from ```responses``` and ```originCountry``` from ```conversionRates```. I'm doing an inner join because if the ```CompensationCurrency``` doesn't exist in the ```originCountry``` then we can't convert the rate. There is one occurrence of this. There is a ```CompensationCurrency``` value of "SPL". This value doesn't exist in ```originCountry```. The inner join will naturally exclude this row when doing the merge.

# In[ ]:


responses = responses.merge(conversionRates, left_on="CompensationCurrency", 
                                  right_on="originCountry", how="inner")

print("There are now " + str(responses.shape[0]) + " rows left.")


# Now it's time to convert all the values in ```CompensationAmount``` to USD. To avoid confusion, we will create a new feature called ```CompensationAmountUSD```, which will hold the converted compensation values.

# In[ ]:


responses["CompensationAmountUSD"] = responses["CompensationAmount"] * responses["exchangeRate"]


# In[ ]:


column_names_to_drop = ["CompensationAmount", "CompensationCurrency", "originCountry", "exchangeRate"]
responses = responses.drop(column_names_to_drop, axis=1)


# In[ ]:


responses.info()


# ### Visualize Data
# 
# Before we continue transforming the data and removing more rows, let's look at the current dataset with graphs and charts to get a better understanding of the distributions.
# 
# You will need the current state of our dataset to be locally stored on your computer.
# 
# I have already exported our current dataset to a CSV and it's uploaded on Kaggle. It's called ```multipleChoiceResponsesCleaned.csv```. You can download it from Kaggle or you can uncomment the two lines of code below and run the cell. The CSV file will be created in the same directory as this Jupyter Notebook.
# 
# Go to https://pair-code.github.io/facets/
# 
# Look for a button that says ```LOAD YOUR OWN DATA``` and click it. Note: There are actually two buttons that say ```LOAD YOUR OWN DATA```, either one will work. When the prompt opens for you to select the ```multipleChoiceResponsesCleaned.csv``` file.
# 
# There are two different types of visualizations offered in Facets, Facets Overview and Facets Dive.

# In[ ]:


# filename = "multipleChoiceResponsesCleaned.csv"
# responses.to_csv(filename, index=False, encoding='utf-8')


# #### Facets Overview
# 
# This section is located in the top half of the website page. Facets Overview looks at each feature individually.
# 
# To make each graph larger, click the ```expand``` checkbox.
# 
# Below is a screenshot that I took of the Age bar graph. The age values are bucketed in groups of 5.3. For example, the first group is 19 - 24.3, the second group is 24.3 - 29.6, etc. Even though the mean age is almost 34, the most common age group is 24.3 - 29.6. The older ages in the right tail end of the distribution skews the mean.

# ![Bar graph of Age Distribution](https://storage.googleapis.com/kaggle-datasets/4417/6791/age_bar_graph.png?GoogleAccessId=datasets@kaggle-161607.iam.gserviceaccount.com&Expires=1510691007&Signature=oAxZKwnd2tCn%2Ff3GQs6jwOSDPLBVTpf4zOAcm29VlrwRG%2FRR3Yv4oNz6Vad24xh91g2IoXCPwB82frSE3anFC4L5HbIYPdptS3JtVaVnCdw3QN1O7vuWCrubMBHLY5Qubg%2FetfaijdlLClA1iz31WlRh%2F6sPjuK4eEqbB5JDYxdBdpapZWySlUvytetocYUMFOTjFMVJnZkB3s9dIEF3zGvsxCgDGYhkpoU4O3RgQX6p9FKcBJAro3efVy11zNvs8MiSWeO6ACHH3UvddUas3%2BGnilpFdzoWT%2BBBI1PbjU5Dmrj3Ja9YPNHC3Rw%2BL8jxQW6hKMqnWPLo4ayjYixicw%3D%3D "Age Distribution")

# #### Facets Dive
# 
# This section is located in the bottom half, under Facets Overview. Facets Dive allows you to look at the relationships between features. After selecting which features you want, every data item (row) will be  grouped accordingly on the visual. You can select any data point on the visual to see all of its values. 
# 
# In the screenshot below, I selected ```CompensationAmountUSD``` for the Row-Based Faceting. The blue slider bar below the ```CompensationAmountUSD``` on the far left determines how many buckets (groups) of compensation amounts there should be. Since I selected ```100``` and the min/max of the ```CompensationAmountUSD``` is 0.259 and 2500000 respectively, the interval scale for the buckets is 25000. This was computed by ```2500000 / 100 = 25000```.
# 
# The Column-Based Faceting selected was ```Country```. All 9 countries in our dataset are being displayed.
# 
# The third feature is ```Tenure```. The question for ```Tenure``` is *How long have you been writing code to analyze data?* This feature is represented by colors.
# 
# I randomly selected a data point in the visual board from the United States column. The details for that data point is displayed on the far right.
# 
# The majority of the countries have a nice Normal (Gaussian) distribution. For the United States, the distribution is centered around 75-100k and 100-125k. United Kingdom is centered at 50-75k. India is not displaying a Normal distribution in this visual. The vast majority of the data points are in the 0.159-25k bucket. To see a Normal distribution for India, the bucket interval scale needs to be much smaller so we can have a more granular view.
# 

# ![Compensation Amount USD by Country](https://storage.googleapis.com/kaggle-datasets/4417/6791/comp_country_facet.png?GoogleAccessId=datasets@kaggle-161607.iam.gserviceaccount.com&Expires=1510691230&Signature=DQKycu9TPPLVpKVbTI4mUHZib%2BkSeugn2uJDEW0ND2wWIctroaQOFuU2YFwLgpZQSanA2NsB1dPQiOPdyOWBM3TuNuMve7%2Bo8KI5MMBuYfasS2aoazWo6gbTdG6xxprn7kOLbH8UT1WjOltnrf0IjbOkBksbfbxt2LoaGs%2BiT8nyoIA9qc3IvFWcD6UIKvp9Jw1t%2FIl4EcdAOEaA%2FW3pq7tL9ZGiwNohJxrfGLGAstTwibeJWcdET6LHWZNYk%2B2ftdudmy9IzR8XBCtZPboH5iv3yd4Vu33JLFE%2Bq%2BlLhCoS4oeSY6GsCEgud9UDqUUftvlo1JAJ0M4wjxAL2pwGPg%3D%3D "Compensation Amount (USD) by Country")

# I noticed that the United States seems to have a lot of data points where the ```Tenure``` was 6 or more years, (```6 to 10 years``` and ```More than 10 years```). India seems to have the opposite.
# 
# Let's use the pandas ```value_counts``` method to find the exact percentage. 53% of the data points in the United States have a ```Tenure``` value of 6 or more years. India has 19%.

# In[ ]:


united_states = responses[responses["Country"] == "United States"]
united_states["Tenure"].value_counts(normalize=True)


# In[ ]:


india = responses[responses["Country"] == "India"]
india["Tenure"].value_counts(normalize=True)


# ### Remove outliers in CompensationAmountUSD
# 
# The average compensation amount for the remaining dataset is about $82000 (USD).
# Amounts that are less than 10000 and greater than 500000 will be removed.

# In[ ]:


print("Before removing outliers: mean=" + str(np.mean(responses["CompensationAmountUSD"])))

comp_amnt = responses["CompensationAmountUSD"]
responses = responses[(comp_amnt >= 30000) & (comp_amnt <= 300000)]

print("After removing outliers: mean=" + str(np.mean(responses["CompensationAmountUSD"])))

print("There are now " + str(responses.shape[0]) + " rows left.\n")

print(responses["Country"].value_counts())


# ### One Hot Encoding
# 
# We will convert the 9 columns of type ```object``` to a numeric type by using the one hot encoding. In one hot encoding, a new feature of type ```int``` is created for each distinct value. Let's perform one hot encoding on ```GenderSelect``` and view the new features that were created.

# In[ ]:


gender_select_one_hot = pd.get_dummies(responses["GenderSelect"])

print(gender_select_one_hot.info())
print("First row values: " + str(gender_select_one_hot.values[0]))


# Now that we have seen how one-hot encoding works, let's perform it on the 9 columns and join the newly created columns to ```responses```. Afterwards, the 9 columns can be dropped.

# In[ ]:


column_names_to_encode = ["GenderSelect", "Country", "CurrentJobTitleSelect", "FormalEducation", 
                          "MajorSelect", "Tenure", "ParentsEducation", "EmployerIndustry", "EmployerSize"]

columns_encoded = pd.get_dummies(responses[column_names_to_encode])
responses = responses.join(columns_encoded)

responses = responses.drop(column_names_to_encode, axis=1)


# ### Update Column Names
# 
# Many of the newly created features contains characters in their names that Tensorflow does not allow. e.g. spaces and commas. We need to replace these types of characters with an underscore.

# In[ ]:


responses.columns = responses.columns.str.replace(" ", "_")
responses.columns = responses.columns.str.replace(",", "_")
responses.columns = responses.columns.str.replace("'", "_")
responses.columns = responses.columns.str.replace("(", "_")
responses.columns = responses.columns.str.replace(")", "_")


# ### Scale Dataset by Standardization
# 
# Standardization rescales the dataset to have a mean value of zero and a standard deviation of 1. Essentially, the data is proportionately "shrunk". The regression model will perform better with the smaller range and more dense values. The label ```CompensationAmountUSD``` will not be standardized.

# In[ ]:


feature_names = responses.columns[responses.columns != "CompensationAmountUSD"]

scaler = StandardScaler()
responses[feature_names] = scaler.fit_transform(responses[feature_names])


# ## Split Dataset
# 
# ### Training and Test Sets
# 
# 80% will be used for training, 20% for testing.

# In[ ]:


np.random.seed(0) # Seed is hard coded to 0 so that the results are reproducible

total_rows = responses.shape[0]
shuffled_indices = np.random.permutation(total_rows)
test_set_size = int(total_rows * 0.2)
test_indices = shuffled_indices[:test_set_size]
train_indices = shuffled_indices[test_set_size:]

train_set = responses.iloc[train_indices]
test_set = responses.iloc[test_indices]


# ### Features and Labels
# 
# ```X``` denotes features and ```y``` denotes the label.

# In[ ]:


X_train_set = train_set.drop("CompensationAmountUSD", axis=1)
y_train_set = train_set["CompensationAmountUSD"]

X_test_set = test_set.drop("CompensationAmountUSD", axis=1)
y_test_set = test_set["CompensationAmountUSD"]


# # TensorFlow
# 
# ### DNN Regressor Model
# 
# The DNNRegressor model will need all the feature names and the number of nodes in each hidden layer.

# In[ ]:


feature_names = X_train_set.columns
feature_columns = [tf.feature_column.numeric_column(feature_name) for feature_name in feature_names]

model_dir = "tmp"
regressor = tf.estimator.DNNRegressor(feature_columns=feature_columns,
                                      hidden_units=[1000, 1000, 1000],
                                      model_dir=model_dir)


# ### Input Function

# In[ ]:


input_fn_train = tf.estimator.inputs.pandas_input_fn(x=X_train_set, y=y_train_set, shuffle=True)


# ### Train Model

# In[ ]:


regressor.train(input_fn=input_fn_train, steps=1000)


# ### Evaluate Model with Test Set
# 
# Tensorflow offers an ```evaluation``` method from the DNNRegressor class. This method returns the average loss value. I will not be using this method. Instead, I'll use the ```predict``` method from the DNNRegressor class and calculate a different type of error with some simple arithmetic.
# 
# Each predicted compensation amount will be subtracted from the actual compensation amount for that row in the test set. Then, the difference will be divided by the predicted compensation amount. Since there is randomness in the prediction process, the mean and median will vary each time. After calling the train and prediction methods several times, the mean is usually between 0.23 to 0.27 and the median is usually between 0.18 to 0.22.
# 
# What do these numbers mean?
# 
# Let's say the mean is 0.24. If a predicted compensation amount is \$100000, on average, the true compensation amount will be  +-\$24000 of \$100000. i.e. Between \$76000 and \$124000. 
# 
# For a median of 0.19, there is a 50% chance that the compensation amount will be between \$81000 and \$119000.
# 
# This is all based on the assumption that our test data is representative of the general population.

# In[ ]:


input_fn_test = tf.estimator.inputs.pandas_input_fn(x=X_test_set, y=y_test_set, shuffle=False, num_epochs=1)

prediction_generators = regressor.predict(input_fn=input_fn_test)
y_predictions = [pg["predictions"] for pg in prediction_generators]

errors = []
for i in range(len(y_predictions)):
    y_prediction = y_predictions[i]
    y_actual = y_test_set.values[i]
    
    error = abs(y_prediction - y_actual) / y_prediction
    errors.append(error)
    
print("test set size: " + str(len(y_predictions)))
print("mean:" + str(np.mean(errors)))
print("median:" + str(np.median(errors)))


# In[ ]:


for i in range(len(y_predictions)):
    y_prediction = y_predictions[i]
    y_actual = y_test_set.values[i]
    
    error = abs(y_prediction - y_actual) / y_prediction
    print("Predicted: " + str(int(y_prediction)) 
          + ", Actual: " + str(int(y_actual)) 
          + ", error: " + str(round(float(error), 2)))


# ## scikit-learn LinearRegression
# 
# I wanted to compare the results of the DDNRegression model from Tensorflow to another model. I tried various models from scikit-learn such as SVM (Support Vector Machine), MLPRegressor (Multi-layer Perceptron regressor), and LinearRegression. LinearRegression had the best results. The results were very similar to the DDNRegression model.

# In[ ]:


from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train_set, y_train_set)
y_predictions = lr.predict(X_test_set)

errors = []
for i in range(len(y_predictions)):
    y_prediction = y_predictions[i]
    y_actual = y_test_set.values[i]
    
    error = abs(y_prediction - y_actual) / y_prediction
    errors.append(error)
    
print("test set size: " + str(len(y_predictions)))
print("mean:" + str(np.mean(errors)))
print("median:" + str(np.median(errors)))


# In[ ]:


for i in range(len(y_predictions)):
    y_prediction = y_predictions[i]
    y_actual = y_test_set.values[i]
    
    error = abs(y_prediction - y_actual) / y_prediction
    print("Predicted: " + str(int(y_prediction)) 
          + ", Actual: " + str(int(y_actual)) 
          + ", error: " + str(round(float(error), 2)))

