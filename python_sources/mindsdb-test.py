# Install the necessary modules
# !pip3 install mindsdb --user

# Import the necessary packages
import pandas as pd
import os
from mindsdb import *

df = pd.DataFrame(pd.read_csv("../input/home_rentals.csv"))
print(df.head())

print(len(df))
print(df.columns.values)

# Train the model
MindsDB().learn(
    from_data="../input/home_rentals.csv", # the path to the file where we can learn from, (note: can be url)
    predict='rental_price', # the column we want to learn to predict given all the data in the file
    model_name='home_rentals' # the name of this model
)

# Use the model to make predictions
result = MindsDB().predict(predict='rental_price', when={'number_of_rooms': 2,'number_of_bathrooms':1, 'sqft': 1190}, model_name='home_rentals')
print(result.predicted_values)