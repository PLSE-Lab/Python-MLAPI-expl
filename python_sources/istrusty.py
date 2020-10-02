# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


#Dataframe
df = pd.read_csv("/kaggle/input/popular-websites-across-the-globe/Web_Scrapped_websites.csv", usecols=['Website', 'Trustworthiness'], encoding='ISO-8859-1')

#Input
choice = input("Check the website trustworthiness(e.g. www.google.com) = ")

#Query loop
while choice != "exit":
        choice = choice.lower()
        for index, row in df.iterrows():
            if choice in row['Website']:
                print("Trustworthiness of website {} is".format(choice), row['Trustworthiness'])
                choice = input("Check the website trustworthiness(e.g. www.google.com) = ")
        
        if choice not in row['Website']:
            print("The website {} could not be found".format(choice))
            choice = input("Check the website trustworthiness(e.g. www.google.com) = ")