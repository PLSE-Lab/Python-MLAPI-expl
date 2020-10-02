# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# Load the data.
df_train = pd.read_csv("../input/train.csv")
df_test = pd.read_csv("../input/test.csv")

# Remove the outliers with low sale prices and large living areas
df_train.drop(df_train[df_train["GrLivArea"] > 4000].index, inplace=True)


#missing data for training samples
total = df_train.isnull().sum().sort_values(ascending=False)
percent = ((df_train.isnull().sum()/(df_train.count()+df_train.isnull().sum()))*100).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent Missing'])
print("Missing training data" + str(missing_data))


total = df_test.isnull().sum().sort_values(ascending=False)
percent = ((df_test.isnull().sum()/(df_test.count()+df_test.isnull().sum()))*100).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent Missing'])
print("Missing test data" + str(missing_data))



from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
