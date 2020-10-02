import csv as csv 
import numpy as np

# Open up the csv file in to a Python object
csv_file_object = csv.reader(open('../input/CLIWOC15.csv', 'rU')) 
header = next(csv_file_object)  # The next() command just skips the 
                                 # first line which is a header
print(header)