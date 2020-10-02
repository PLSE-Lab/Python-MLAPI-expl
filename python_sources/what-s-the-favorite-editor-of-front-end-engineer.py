# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

df = pd.read_excel("../input/State Of JavaScript 2016 (clean).xlsx")

df['Favorite Text Editor'].fillna(df['Other'], inplace=True)
editor = df['Favorite Text Editor'].str.lower()
editor.replace(regex='.*code.*', value='visual studio [code]', inplace=True)
editor.replace(regex='.*visual.*', value='visual studio [code]', inplace=True)
editor.replace(regex='.*vs.*', value='visual studio [code]', inplace=True)
editor.replace(regex='.*idea.*', value='intellij idea', inplace=True)
editor.replace(regex='.*intellij.*', value='intellij idea', inplace=True)

df['Favorite Text Editor'] = editor
editor.value_counts()[0:10].plot.bar()

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# from subprocess import check_output
# print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.