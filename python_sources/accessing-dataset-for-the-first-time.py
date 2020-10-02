#!/usr/bin/env python
# coding: utf-8
import numpy as np # linear algebra
import pandas as pd # import in pandas
from subprocess import check_output
import os
#print(os.listdir("C:/Users/ALYUS/Downloads/input")) Just to make sure where the files are located
#D = "C:/Users/ALYUS/Downloads/input" Consider this line to find out the path of your PC
#df = pd.read_csv("test.csv")
dg = pd.read_csv("../input/titanic/kaggle-titanic-master.zip")
#df.columns
#print(os.listdir('../input/test.csv'))
#x = os.listdir('../input/titanic/test.csv')
#x
#print(check_output(["ls", "../input/"]).decode("utf8"))
print(os.listdir("../input/titanic/"))

