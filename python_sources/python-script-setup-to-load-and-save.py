# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
# Any results you write to the current directory are saved as output.
# from subprocess import check_output
# print(check_output(["ls", "../input"]).decode("utf8"))


# The competition datafiles are in the directory ../input
# Read competition data files:
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
# Write to the log:
print("Training set has %d rows and %d columns\n"%(train.shape[0], train.shape[1]) )
print("Test set has %d rows and %d columns\n"%(test.shape[0], test.shape[1]))

# train a classifier here
#TODO

# Generate output files csv, plot
# Any files you write to the current directory get shown as outputs
out = open('output_1.csv', "w")
out.write("ImageId,Label\n")

rows = ['']*test.shape[0] # predefine or use append
for num in range(0, test.shape[0]):
    label = 0 #TODO; classify here
    rows[num] = "%d,%d\n"%(num+1,label)

out.writelines(rows)
out.close()