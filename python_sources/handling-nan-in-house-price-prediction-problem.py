import numpy as np
import pandas as pd

#Read Data into a dataset
input_data = pd.read_csv("../input/train.csv",index_col = "Id")
input_data.head()

#Returns back the shape of the datadset in the form (Rows, Columns)
print(input_data.shape)

#Get the number of missing values from each column
input_data_missing_values = input_data.isnull().sum()
input_data_missing_values

#Get the total number of values in the original dataset
input_data_original_rows_total = np.product(input_data.shape)

#Get the total number of missing values in the original dataset
input_data_missing_rows = input_data_missing_values.sum()

#Get the percentage of missing data
percent_missing = (input_data_missing_rows / input_data_original_rows_total) * 100
print("Percent Missing: %f\n" %(percent_missing))

#Get the number of rows in the original dataset
input_data_original_rows_count = input_data.shape[1]

#Drop the columns which have NaN in them
input_data_modified_rows = input_data.dropna(axis = 1)

#Get the number of rows in the dataset left
input_data_modified_rows_count = input_data_modified_rows.shape[1]

#Print the results
print("Original Number of Columns: %d\n" %(input_data_original_rows_count))
print("Modified Number of Columns: %d\n" %(input_data_modified_rows_count))

#Better than using dropna is by using the fillna function to fill the missing values
input_data_imputed = input_data.fillna(method = "bfill" , axis = 0).fillna(0)
input_data_imputed.head()

#Get the shape of the imputed dataset in form of (Rows, Columns).
#This clearly says that imputation is better than dropping columns in critical data science,
#since there is a chane of data loss.
input_data_imputed.shape