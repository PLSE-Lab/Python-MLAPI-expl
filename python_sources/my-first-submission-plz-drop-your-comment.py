# This R environment comes with all of CRAN preinstalled, as well as many other helpful packages
# The environment is defined by the kaggle/rstats docker image: https://github.com/kaggle/docker-rstats
# For example, here's several helpful packages to load in 
import pandas as pd # Data visualization
import numpy as np 
import scipy 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split , cross_val_score
from sklearn.linear_model import  LinearRegression
from sklearn.preprocessing import StandardScaler # CSV file I/O, e.g. the read_csv function

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#importing file 
df = pd.read_csv("../input/kc_house_data.csv")


#defining test size and train size 
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


df.info()


np.round(df.describe())

df.isnull().any()

df.columns
df.corr()["price"].sort_values(ascending = False)

correlation = df.corr()

plt.figure(figsize =(14,12))

