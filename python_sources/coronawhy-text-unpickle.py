# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

PATH = "/kaggle/input/coronawhy/v6_text/"
# df = pd.concat([pd.read_pickle(PATH + p, compression="gzip") for p in os.listdir(PATH)]) 
for p in os.listdir(PATH):
    if p.endswith(".pkl"):
        print("unpickling: " + PATH + p)
        p_df = pd.read_pickle(PATH + p, compression="gzip")
        p_df.to_csv(p + ".tsv", sep="\t", encoding="utf-8", index=False)
        print("wrote file: " + p + ".csv")
# Any results you write to the current directory are saved as output.