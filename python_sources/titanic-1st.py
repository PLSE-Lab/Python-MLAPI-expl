import csv as csv
import numpy as np

csv_file_object = csv.reader(open('../input/train.csv'))
header = next(csv_file_object)
