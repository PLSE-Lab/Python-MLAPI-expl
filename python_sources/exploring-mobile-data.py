# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
inputFiles = check_output(["ls", "../input"]).decode("utf8").splitlines()
print(inputFiles)
df = None
counter = 0
for file in inputFiles:
    df_file = pd.read_csv("../input/{}".format(file))
    if counter == 0:
        df = df_file
    else:
        match = ""
        cols1 = list(df.columns.values)
        cols2 = list(df_file.columns.values)
        for col1 in cols1:
            for col2 in cols2:
                if col1 == col2:
                    match = col1

        print("joining tables on {}".format(match))        
        df.merge(df_file,on=match,how='left')
        print("done joining")
        
    counter = counter + 1
print(df.head)
