# %% [code]
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

#import numpy as np # linear algebra
#import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

#import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
#    for filename in filenames:


#        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

name = input("Hello my name is Jazmine, but you can call me jay. What is your name?  \n")

statis = input(f"  Hello {name}, how are you? \n")
if (statis == 'good' or statis=='great'):
    why = input('Great! why are you good?\n')
else:
  why = input ("What's wrong?\n")
if (why == 'meh' or why =='don\'t know'):
    print('You can\'t see me but i am rolling my eyes')
    A23dys9 = input("Fine. What do you want to talk about now?\n")
else:
    A23dys9 = input("thanks for sharing. What do you want to talk about now?\n")
if A23dys9=='stuff':
    print ('Stuff is for stuffies. Oh look at the time got to go bye!')
else:
    print(f'{A23dys9} is cool. Oh look at the time got to go bye!')


# %% [code]
