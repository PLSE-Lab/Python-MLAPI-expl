"""
IDENTIFY AND CHART THE 30 NATIONS WITH POPULATION OVER ONE MILLION WHOSE GDPs/PerCapita
GREW THE MOST ON AVERAGE IN THE THREE DECADES THAT ELAPSED FROM 1980 TO 2010
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

df1=pd.read_csv('../input/Country.csv',index_col='CountryCode')
print(df1)
