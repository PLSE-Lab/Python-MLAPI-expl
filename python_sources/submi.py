# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import sys
import re
import pandas as pd


def clean(file):
    df=pd.read_csv(file)
    df["Tweet_Text"]=df["Tweet_Text"].str.replace("$",'')
    df["Tweet_Text"]=df["Tweet_Text"].str.replace("--",'')
    df["Tweet_Text"]=df["Tweet_Text"].str.replace("?",'')
    df["Tweet_Text"]=df["Tweet_Text"].str.lower()
    print(df)
    sys.stdout.flush()
    return df
    
"""def count():"""
    

def main():
    cl=clean("C:/Users/Akshay/Desktop/RS/Project/Cleaned.csv")