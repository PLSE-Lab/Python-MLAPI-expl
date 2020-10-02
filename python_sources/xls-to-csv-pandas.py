import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))

# This is a simple script that converts xls files to csv files (xls to csv) using pandas' to_csv() function
# Its better to create file in xls in data collection (dataset creation) in research


# name = pd.read_excel('my_file', sheetname='NAME OF THE SHEET') # Second parameter is optional here

# name.to_csv('output_file_name', index = False), index=False prevents pandas to write row index

# OR IN OTHER FORM, IT CAN BE DONE AS -

# pd.read_excel('my_file', sheetname='my_sheet_name').to_csv('output_file_name', index=False)