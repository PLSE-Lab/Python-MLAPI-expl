#!/usr/bin/env python
# coding: utf-8

# 

# In[ ]:



import pandas as pd
import statsmodels.formula.api as smf
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
from statsmodels.api import add_constant

# Some appearance options.
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('pylab', 'inline')
pylab.rcParams['figure.figsize'] = (10, 6)
pd.set_option('display.max_rows', 21)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Load data.
data = pd.read_csv("../input/car_data.csv")
data = data.dropna()
print(len(data))


# In[ ]:


# Print some data
display(data.head())


# In[ ]:


# Clean up the data
category_col = ['make', 'fuel_type', 'aspiration', 'num_of_doors',
                'body_style', 'drive_wheels', 'engine_location', 
                'engine_type', 'num_of_cylinders','fuel_system']
numeric_col = ['wheel_base', 'length','width', 'height', 'curb_weight',
              'engine_size','compression_ratio', 'horsepower',
               'peak_rpm', 'city_mpg', 'highway_mpg', 'price']
for i in category_col:
    data[i] = data[i].astype("category")
for i in numeric_col:
    data[i] = pd.to_numeric(data[i],errors= "coerce")
for i in data.columns:
    print("%20s : %10s" %(i,data[i].dtype), ",", data.at[0,i])

new_data = data.dropna()
print(len(new_data))
print(len(data))
drop_indices = [i for i in data.index if i not in new_data.index]
used_indices = [i for i in new_data.index]


# In[ ]:


# Plot a histogram to see the frequencies of car prices
plt.style.use('ggplot')
plt.hist(new_data["price"])
plt.xlabel("Price")
plt.ylabel("Frequency")


# In[ ]:


# Plot a histogram to see the frequencies of car prices
plt.hist(new_data["price"])
plt.xlabel("Price")
plt.ylabel("Frequency")


# In[ ]:


# Choose test data
test_indices = used_indices
random.seed(773)
shuffle(test_indices)
test_indices = test_indices[0:31]
test_data = data.loc[test_indices]
train_validation_indices = [i for i in used_indices if i not in test_indices]
print(test_indices)


# In[ ]:


# Overall summary
formula = " price ~ C(make) + C(fuel_type)+C(aspiration)+C(num_of_doors)+C(body_style)+C(drive_wheels)+C(engine_location)+wheel_base+length+width+height+curb_weight+C(engine_type)+C(num_of_cylinders)+engine_size+C(fuel_system)+compression_ratio+horsepower+peak_rpm+city_mpg+highway_mpg"
model = smf.ols(formula=formula, data = data).fit()
model.summary()


# In[ ]:


# Car brands are believed to affect car price significantly. 
# A boxplot was plotted to see the different price ranges for different car manufacturers.

data.boxplot(column="price", by="make", rot=90, grid=False)


# In[ ]:


# Car brands are believed to affect car prices significantly.
# Plot a boxplot to see different price ranges for different car manufacturers.
new_data.boxplot(column="price", by="make", rot=90, grid=False)


# In[ ]:


# Calculate training error, validation error, test error

average_train_error = []
average_validation_error = []
average_test_error = []

formula_list = ["engine_location","peak_rpm","curb_weight","fuel_type","wheel_base","width",
               "length","engine_size","height","aspiration"]
formula = " price ~ make"
feature_list = ["make"]
for feature in formula_list:
    print(feature)
    formula += '+'
    formula += feature
    feature_list.append(feature)
    train_error = []
    validation_error = []
    test_error = []
    # Choose validation index (Leave-One-Out Cross Validation)
    for validation_index in train_validation_indices:
            train_indices = [i for i in train_validation_indices if i != validation_index]
            train_data = data.loc[train_indices]
            validation_data = data.loc[[validation_index]]
            # Train model
            model = smf.ols(formula=formula, data = train_data).fit()
            # Compute training error
            train_predict = model.predict()
            square_error_sum = 0
            count = 0
            for i in train_indices:
                if not np.isnan(train_data.loc[i,"price"]):
                    square_error_sum += pow((train_predict[count]-train_data.loc[i,"price"]),2)
                    count += 1
            train_error.append(sqrt(square_error_sum/count))
            # Compute validation error
            validation_predict = model.predict(validation_data[feature_list])
            validation_error.append(sqrt(pow(validation_data.loc[validation_index,"price"]-                                             validation_predict[0],2)))
            # Compute test error
            square_error_sum = 0
            count = 0
            test_predict = model.predict(test_data[feature_list])
            for i in test_indices:
                if not np.isnan(test_data.loc[i,"price"]):
                    square_error_sum += pow((test_predict[count]-test_data.loc[i,"price"]),2)
                    count += 1
            test_error.append(sqrt(square_error_sum/count))
    average_train_error.append(sum(train_error)/len(train_error))
    average_validation_error.append(sum(validation_error)/len(validation_error))
    average_test_error.append(sum(test_error)/len(test_error))
    


# In[ ]:


print(average_train_error)
print(average_validation_error)
print(average_test_error)


# In[ ]:


print(average_train_error)
print(average_validation_error)
print(average_test_error)


# In[ ]:


# Plot errors against complexity

l = [i for i in range(1,len(average_validation_error)+1)]
plt.plot(l,average_train_error,label = "train",c = "r")
plt.plot(l,average_validation_error, label = "validation", c = "g")
plt.plot(l,average_test_error,label = "test",c = "b")


# In[ ]:


predicted_price = model.predict()
predicted_price = pd.Series(predicted_price, name="PredictedPrice")
plt.scatter(new_data["price"], predicted_price)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")


# In[ ]:





# In[ ]:


# Compute the predicted price and convert the results to a pandas series.
predicted_price = model.predict(new_data)
predicted_price = pd.Series(predicted_price, name="PredictedPrice")


# In[ ]:


# Plot the relationship between predicted price and actual price
plt.scatter(new_data["price"], predicted_price)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")

