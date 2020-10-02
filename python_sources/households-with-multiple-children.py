import pandas as pd
import numpy as np
import pandas


import matplotlib.pyplot as plt
#%matplotlib inline

from sklearn.neighbors import KernelDensity

# print folders available
# import subprocess
# p = subprocess.Popen(["ls", "-lh", "../input"], stdout=subprocess.PIPE)
# output, err = p.communicate()
# print ("*** Running ls -l command ***\n", output)

housing_a = '../input/pums/ss13husa.csv'
housing_b = '../input/pums/ss13husb.csv'
population_a = '../input/pums/ss13pusa.csv'
population_b = '../input/pums/ss13pusb.csv'

housing_data = pandas.DataFrame.from_csv(housing_a, header=0)
alabama_data = housing_data[housing_data['ST'] == 1]
print(alabama_data.head())
#temp = pandas.DataFrame.from_csv(housing_b, header=0)
#housing_data = housing_data.append(temp, ignore_index=True)

#print(housing_data.columns.tolist())
#print(housing_data.head())
#alabama_data = housing_data.loc[housing_data['ST'] == 1].values
#print(alabama_data.head())