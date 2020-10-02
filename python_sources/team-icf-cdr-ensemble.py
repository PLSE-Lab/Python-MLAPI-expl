# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

a=pd.read_csv("../input/submission_37_cleaned_data_2.csv")
b=pd.read_csv("../input/submission_30_xgboost1_cleaned_data_2.csv")
c=pd.read_csv("../input/submission_24_reducing_categories.csv")
d=pd.read_csv("../input/submission_32_xgboost_cleaned_data.csv")

b["is_female"]=(a["is_female"]+b["is_female"]+c["is_female"]+d["is_female"])/4

b[["test_id","is_female"]].to_csv("submission_38_ensemble.csv",index=False)